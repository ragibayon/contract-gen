from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

TestSplit = Literal["auto", "public", "private", "all"]


@dataclass(frozen=True)
class DatasetTask:
    dataset: str
    source_path: Path
    row_index: int
    task_id: str
    task_key: str
    prompt: str
    display_id: str
    problem_id: str | int | None
    source_uid: str | None
    entry_point: str
    witnesses: dict[str, list[Any]]
    metadata: dict[str, Any]
    source_row: dict[str, Any]


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_dataset_rows(path: str | Path) -> Iterator[dict[str, Any]]:
    dataset_path = Path(path)

    if dataset_path.suffix.lower() == ".jsonl":
        with dataset_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at {dataset_path}:{line_number}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(
                        f"Expected a JSON object at {dataset_path}:{line_number}"
                    )
                yield row
        return

    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("rows", "data", "tasks", "problems", "items"):
            rows = payload.get(key)
            if isinstance(rows, list):
                break
        else:
            rows = [payload]
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError(
            f"Dataset must contain an object or list of objects: {dataset_path}"
        )

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Expected a JSON object at {dataset_path} row {index}")
        yield row


def dataset_name(path: str | Path) -> str:
    name = Path(path).stem



    name = re.sub(r"(?i)(?:_specine)?_update(?:_row\d+)?$", "", name)
    name = re.sub(r"(?i)_row\d+$", "", name)
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._-")
    lower = name.lower()
    if lower in {"xcodeeval", "xcode_eval", "xcode-eval"}:
        return "XcodeEval"
    if lower in {"code_contests", "code-contests", "codecontest", "code_contest"}:
        return "code_contest"
    return name or "dataset"


def build_programming_prompt(row: dict[str, Any]) -> str:
    direct_prompt = row.get("prompt") or row.get("instruction")
    if direct_prompt:
        return str(direct_prompt).strip()

    parts: list[str] = []

    title = row.get("name") or row.get("title")
    if title:
        parts.append(f"Title:\n{title}")

    statement = (
        row.get("description")
        or row.get("question")
        or row.get("statement")
        or row.get("problem")
        or row.get("text")
    )
    if statement:
        parts.append(f"Problem:\n{statement}")

    if row.get("input_spec"):
        parts.append(f"Input specification:\n{row['input_spec']}")

    if row.get("output_spec"):
        parts.append(f"Output specification:\n{row['output_spec']}")

    if row.get("notes"):
        parts.append(f"Notes:\n{row['notes']}")

    if row.get("input_from") or row.get("output_to"):
        parts.append(
            "Execution format:\n"
            f"Input from: {row.get('input_from', 'unknown')}\n"
            f"Output to: {row.get('output_to', 'unknown')}"
        )

    if row.get("time_limit") or row.get("memory_limit"):
        parts.append(
            "Limits:\n"
            f"Time: {row.get('time_limit', 'unknown')}\n"
            f"Memory: {row.get('memory_limit', 'unknown')}"
        )

    if row.get("memory_limit_bytes") and not row.get("memory_limit"):
        parts.append(f"Memory limit: {row['memory_limit_bytes']} bytes")

    starter_code = row.get("starter_code")
    if isinstance(starter_code, str) and starter_code.strip():
        parts.append(f"Starter code:\n{starter_code.strip()}")

    io = _maybe_json_loads(row.get("input_output"))
    if isinstance(io, dict) and io.get("fn_name"):
        parts.append(
            "Execution protocol for this pipeline:\n"
            f"The original task is function-style with target function `{io.get('fn_name')}`. "
            "Implement a complete Python program with solve(...) and main(). "
            "For evaluation, stdin contains one compact JSON value. If that value is a JSON array, "
            "treat its elements as positional arguments to solve(...); otherwise treat it as one argument. "
            "Print the return value as compact JSON using json.dumps(..., separators=(',', ':'))."
        )

    return "\n\n".join(parts).strip()


def get_task_id(row: dict[str, Any], fallback: str = "task") -> str:
    if row.get("task_id"):
        return str(row["task_id"])
    if row.get("src_uid"):
        return str(row["src_uid"])
    if row.get("id") is not None:
        return str(row["id"])
    if row.get("problem_id") is not None:
        return f"problem_{row['problem_id']}"
    return fallback


def _maybe_json_loads(value: Any) -> Any:
    """Decode JSON strings used by APPS/CodeContests metadata when possible."""
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return value
        if text[0] in "[{\"" or text in {"true", "false", "null"}:
            try:
                return json.loads(text)
            except Exception:
                return value
    return value


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _case_value_to_stdin(value: Any, *, function_style: bool = False) -> str:
    """Convert dataset input value into the stdin string consumed by our generated program.

    Programming contest rows already carry stdin strings. APPS/specine rows often carry
    Python/function-call arguments in input_output; for those, we expose a deterministic
    JSON stdin protocol so the agent can implement main() without seeing private values.
    """
    if function_style:
        return _compact_json(value) + "\n"
    if isinstance(value, str):
        return value if value.endswith("\n") else value + "\n"
    if isinstance(value, (list, tuple)):

        if all(not isinstance(item, (list, tuple, dict)) for item in value):
            return "\n".join(str(item) for item in value) + "\n"
    return str(value) + "\n"


def _case_value_to_stdout(value: Any, *, function_style: bool = False) -> Any:
    if function_style:
        return _compact_json(value)
    if isinstance(value, list):
        return [str(item).strip() for item in value]
    return str(value).strip()


def _normalize_tests(value: Any, *, function_style: bool = False) -> dict[str, list[Any]]:
    value = _maybe_json_loads(value)

    if isinstance(value, dict):
        inputs = value.get("inputs")
        if inputs is None:
            inputs = value.get("input")
        if inputs is None:
            inputs = value.get("stdin")
        outputs = value.get("outputs")
        if outputs is None:
            outputs = value.get("output")
        if outputs is None:
            outputs = value.get("stdout")
        if outputs is None:
            outputs = value.get("expected")
        if not isinstance(inputs, list):
            inputs = [] if inputs is None else [inputs]
        if not isinstance(outputs, list):
            outputs = [] if outputs is None else [outputs]
        total = min(len(inputs), len(outputs))
        return {
            "inputs": [_case_value_to_stdin(inputs[i], function_style=function_style) for i in range(total)],
            "outputs": [_case_value_to_stdout(outputs[i], function_style=function_style) for i in range(total)],
        }

    if isinstance(value, list):
        inputs: list[Any] = []
        outputs: list[Any] = []
        for case in value:
            case = _maybe_json_loads(case)
            if isinstance(case, dict):
                case_input = (
                    case.get("input")
                    if "input" in case
                    else case.get("stdin", case.get("inputs"))
                )
                case_output = (
                    case.get("output")
                    if "output" in case
                    else case.get("stdout", case.get("expected", case.get("outputs")))
                )
            elif isinstance(case, (list, tuple)) and len(case) >= 2:
                case_input, case_output = case[0], case[1]
            else:
                continue
            if case_input is not None and case_output is not None:
                inputs.append(_case_value_to_stdin(case_input, function_style=function_style))
                outputs.append(_case_value_to_stdout(case_output, function_style=function_style))
        return {"inputs": inputs, "outputs": outputs}

    return {"inputs": [], "outputs": []}


def _merge_bundles(*bundles: dict[str, list[Any]]) -> dict[str, list[Any]]:
    inputs: list[Any] = []
    outputs: list[Any] = []
    seen: set[tuple[str, str]] = set()
    for bundle in bundles:
        for inp, out in zip(bundle.get("inputs", []), bundle.get("outputs", [])):
            key = (str(inp), json.dumps(out, sort_keys=True, ensure_ascii=False, default=str))
            if key in seen:
                continue
            seen.add(key)
            inputs.append(inp)
            outputs.append(out)
    return {"inputs": inputs, "outputs": outputs}


def _first_nonempty_bundle(row: dict[str, Any], keys: tuple[str, ...], *, function_style: bool = False) -> dict[str, list[Any]]:
    for key in keys:
        if key in row and row.get(key) is not None:
            bundle = _normalize_tests(row.get(key), function_style=function_style)
            if _test_count(bundle):
                return bundle
    return {"inputs": [], "outputs": []}


def _test_count(bundle: dict[str, list[Any]]) -> int:
    return min(len(bundle.get("inputs", [])), len(bundle.get("outputs", [])))


def _has_function_style_tests(row: dict[str, Any]) -> bool:
    io = _maybe_json_loads(row.get("input_output"))
    return isinstance(io, dict) and bool(io.get("fn_name"))


def _input_output_bundle(row: dict[str, Any]) -> dict[str, list[Any]]:
    io = _maybe_json_loads(row.get("input_output"))
    if not isinstance(io, dict):
        return {"inputs": [], "outputs": []}
    return _normalize_tests(io, function_style=bool(io.get("fn_name")))


def get_tests(
    row: dict[str, Any],
    split: TestSplit = "public",
) -> dict[str, list[Any]]:
    """Return the requested evaluation split without leaking hidden tests.

    Important policy:
    - public means only explicitly public/sample fields; if absent, return empty.
    - private means explicitly private/hidden fields; APPS input_output is treated as
      private-only because there is usually no public/private split.
    - we do NOT fall back from public to all_test_cases_raw/all_test_cases.
    """
    function_style = _has_function_style_tests(row)
    public = _first_nonempty_bundle(
        row,
        (
            "public_test_cases",
            "public_tests",
            "sample_tests",
            "samples",
            "examples",
        ),
        function_style=False,
    )
    private = _first_nonempty_bundle(
        row,
        (
            "private_test_cases",
            "private_tests",
            "hidden_tests",
        ),
        function_style=False,
    )
    input_output = _input_output_bundle(row)
    if not _test_count(private) and _test_count(input_output):
        private = input_output

    all_bundle = _merge_bundles(
        public,
        private,
        _first_nonempty_bundle(
            row,
            (
                "all_test_cases",
                "all_tests",
                "test_cases",
                "tests",
                "all_test_cases_raw",
            ),
            function_style=function_style,
        ),
        input_output,
    )

    if split == "public":
        return public
    if split == "private":
        return private
    if split == "all":
        return all_bundle
    if split == "auto":
        return public if _test_count(public) else {"inputs": [], "outputs": []}
    raise ValueError(f"Unsupported test split: {split!r}")


def task_metadata(
    row: dict[str, Any],
    *,
    dataset: str,
    source_path: str | Path,
    row_index: int,
) -> dict[str, Any]:
    excluded = {
        "description",
        "question",
        "statement",
        "problem",
        "text",
        "prompt",
        "instruction",
        "input_spec",
        "output_spec",
        "notes",
        "starter_code",
        "hidden_tests",
        "input_output",
        "solutions",
        "incorrect_solutions",
        "public_test_cases",
        "private_test_cases",
        "all_test_cases",
        "all_test_cases_raw",
        "public_tests",
        "private_tests",
        "all_tests",
        "test_cases",
        "tests",
    }
    metadata = {
        key: value
        for key, value in row.items()
        if key not in excluded and not isinstance(value, (dict, list))
    }
    metadata.update(
        {
            "Dataset": dataset,
            "Source file": str(Path(source_path)),
            "Row index": row_index,
        }
    )
    return metadata


def build_dataset_task(
    row: dict[str, Any],
    *,
    source_path: str | Path,
    row_index: int,
    test_split: TestSplit = "auto",
) -> DatasetTask:
    source = Path(source_path)
    name = dataset_name(source)
    prompt = build_programming_prompt(row)
    if not prompt:
        raise ValueError(
            "Could not build a prompt; expected one of prompt, instruction, "
            "description, question, statement, problem, or text."
        )

    fallback_hash = hashlib.sha256(
        json.dumps(row, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()[:16]
    task_id = get_task_id(row, fallback=f"row_{row_index}_{fallback_hash}")
    task_key = f"{name}/{task_id}"

    problem_id = row.get("problem_id")
    if problem_id is not None:
        display_id = f"{name}_problem_{problem_id}"
    else:
        display_id = f"{name}_{row_index + 1}"
    source_uid = str(row["src_uid"]) if row.get("src_uid") is not None else None
    return DatasetTask(
        dataset=name,
        source_path=source,
        row_index=row_index,
        task_id=task_id,
        task_key=task_key,
        prompt=prompt,
        entry_point=str(row.get("entry_point") or "solve"),
        display_id=display_id,
        problem_id=problem_id,
        source_uid=source_uid,
        witnesses=get_tests(row, test_split),
        metadata=task_metadata(
            row,
            dataset=name,
            source_path=source,
            row_index=row_index,
        ),
        source_row=row,
    )
