from __future__ import annotations

import argparse
import ast
import copy
from datetime import datetime, timezone
import difflib
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import tempfile
import time

from contract.utils.dataset import build_dataset_task, get_tests, iter_dataset_rows


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def backend():
    try:
        from mutmut.mutation.file_mutation import create_mutations
        from mutmut.mutation.mutators import mutation_operators
        import libcst
        from libcst.metadata import MetadataWrapper, PositionProvider
    except ImportError as error:
        raise RuntimeError(
            "Mutation support requires the locked development dependencies. Run: uv sync --frozen"
        ) from error
    version = metadata.version("mutmut")
    if version != "3.6.0":
        raise RuntimeError(
            f"This adapter supports locked mutmut 3.6.0; found {version}. Run uv sync --frozen."
        )
    return (
        create_mutations,
        mutation_operators,
        libcst,
        MetadataWrapper,
        PositionProvider,
    )


def registered_operators() -> list[dict]:
    _, operators, _, _, _ = backend()
    return [
        {"node_type": node.__name__, "operator": function.__name__}
        for node, function in operators
    ]


def arithmetic_change(change, cst) -> bool:
    before, after = change.original_node, change.mutated_node
    if type(before) is not type(after):
        return False
    arithmetic = (
        cst.Add,
        cst.Subtract,
        cst.Multiply,
        cst.Divide,
        cst.FloorDivide,
        cst.Modulo,
        cst.Power,
    )
    augmented = (
        cst.AddAssign,
        cst.SubtractAssign,
        cst.MultiplyAssign,
        cst.DivideAssign,
        cst.FloorDivideAssign,
        cst.ModuloAssign,
        cst.PowerAssign,
    )
    unary = (cst.Plus, cst.Minus)
    allowed = (
        isinstance(before, cst.BinaryOperation)
        and isinstance(before.operator, arithmetic)
        and isinstance(after.operator, arithmetic)
    )
    allowed = allowed or (
        isinstance(before, cst.AugAssign)
        and isinstance(before.operator, augmented)
        and isinstance(after.operator, augmented)
    )
    allowed = allowed or (
        isinstance(before, cst.UnaryOperation)
        and isinstance(before.operator, unary)
        and isinstance(after.operator, unary)
    )
    return bool(
        allowed
        and before.with_changes(operator=after.operator).deep_equals(after)
        and not before.deep_equals(after)
    )


def candidate_pool(
    source: str,
    *,
    filename: str = "seed.py",
    pool: str = "all",
    operators: list[str] | None = None,
    scope: str = "all",
    target: str = "solve",
) -> tuple[list[dict], dict]:
    create_mutations, registered, cst, MetadataWrapper, PositionProvider = backend()
    valid_operators = {function.__name__ for _, function in registered}
    if set(operators or []) - valid_operators:
        raise ValueError(
            f"Unknown mutation operators: {sorted(set(operators) - valid_operators)}"
        )
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                function = (
                    decorator.func if isinstance(decorator, ast.Call) else decorator
                )
                if (
                    isinstance(function, ast.Attribute)
                    and function.attr in {"ensure", "require", "invariant"}
                ) or (
                    isinstance(function, ast.Name)
                    and function.id in {"ensure", "require", "invariant"}
                ):
                    raise ValueError(
                        "Mutate unannotated code, not executable contracts or decorated code."
                    )
    bodies = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            scope == "functions" or (scope == "target" and node.name == target)
        ):
            bodies.append((node.body[0].lineno, node.end_lineno))
    module, changes, _, _ = create_mutations(filename, source, covered_lines=None)
    positions = MetadataWrapper(module, unsafe_skip_copy=True).resolve(PositionProvider)
    seed_ast = ast.dump(tree, include_attributes=False)
    candidates = {}
    counts = {
        "enumerated": len(changes),
        "filtered": 0,
        "invalid": 0,
        "ast_identical": 0,
        "duplicate_ast": 0,
    }
    for index, change in enumerate(changes):
        position = positions.get(change.original_node)
        line = position.start.line if position else -1
        column = position.start.column if position else -1
        if scope != "all" and not any(start <= line <= end for start, end in bodies):
            counts["filtered"] += 1
            continue
        families = []
        for node_type, operator in registered:
            if isinstance(change.original_node, node_type):
                try:
                    if any(
                        node.deep_equals(change.mutated_node)
                        for node in operator(change.original_node)
                    ):
                        families.append(operator.__name__)
                except Exception:
                    continue
        if (pool == "arithmetic" and not arithmetic_change(change, cst)) or (
            operators and not set(operators).intersection(families)
        ):
            counts["filtered"] += 1
            continue
        text = module.deep_replace(change.original_node, change.mutated_node).code
        try:
            mutated_ast = ast.parse(text)
            compile(mutated_ast, "<mutmut-candidate>", "exec")
        except (SyntaxError, ValueError, TypeError):
            counts["invalid"] += 1
            continue
        normalized = ast.dump(mutated_ast, include_attributes=False)
        if normalized == seed_ast:
            counts["ast_identical"] += 1
            continue
        fingerprint = sha256(normalized)
        if fingerprint in candidates:
            counts["duplicate_ast"] += 1
            continue
        candidates[fingerprint] = {
            "enumeration_index": index,
            "id": f"mutmut-api-{index:06d}",
            "line": line,
            "column": column,
            "node_type": type(change.original_node).__name__,
            "before": module.code_for_node(change.original_node),
            "after": module.code_for_node(change.mutated_node),
            "operators": sorted(set(families)),
            "ast_sha256": fingerprint,
            "source_sha256": sha256(text),
            "source": text,
        }
    ordered = sorted(
        candidates.values(),
        key=lambda value: (
            value["line"],
            value["column"],
            value["node_type"],
            value["before"],
            value["after"],
            value["ast_sha256"],
        ),
    )
    counts["eligible"] = len(ordered)
    return ordered, counts


def select_mutations(
    candidates: list[dict], *, limit: int = 10, seed: int = 42
) -> list[dict]:
    if limit < 1:
        raise ValueError("Mutation limit must be positive")
    ranks = sorted(
        random.Random(seed).sample(range(len(candidates)), min(limit, len(candidates)))
    )
    return [{**candidates[rank], "pool_rank": rank} for rank in ranks]


def merge_contract(source: str, contract: str, target: str = "solve") -> str:
    suffix = "\ndef __contract_target__(*args, **kwargs):\n    pass\n"
    contract_tree = ast.parse(contract.rstrip() + suffix)
    template = contract_tree.body[-1]
    if not isinstance(template, ast.FunctionDef) or not template.decorator_list:
        raise ValueError(
            "Contract input must be an import/helper section followed by decorators, like ec.py."
        )
    tree = ast.parse(source)
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == target
    ]
    if len(functions) != 1:
        raise ValueError(
            f"Mutant does not have exactly one top-level {target} function"
        )
    functions[0].decorator_list = (
        copy.deepcopy(template.decorator_list) + functions[0].decorator_list
    )
    body = copy.deepcopy(contract_tree.body[:-1]) + tree.body
    futures = [
        node
        for node in body
        if isinstance(node, ast.ImportFrom) and node.module == "__future__"
    ]
    tree.body = futures + [node for node in body if node not in futures]
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def run_program(
    source: str,
    stdin: str,
    *,
    target: str = "solve",
    trace: bool = False,
    timeout: float = 5,
) -> dict:
    try:
        import resource
    except ImportError as error:
        raise RuntimeError(
            "Mutation execution requires POSIX CPU/file limits (macOS or Linux)."
        ) from error

    def limits():
        resource.setrlimit(
            resource.RLIMIT_CPU, (max(1, int(timeout)), max(1, int(timeout)) + 1)
        )
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        if sys.platform != "darwin":
            resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="contract-mutant-") as tmp:
        root = Path(tmp)
        program = root / "program.py"
        program.write_text(source)
        request = root / "request.json"
        request.write_text(
            json.dumps(
                {
                    "program": str(program),
                    "input": stdin,
                    "target": target,
                    "trace_contracts": trace,
                }
            )
        )
        worker = Path(__file__).with_name("_mutation_worker.py")
        process = subprocess.Popen(
            [sys.executable, "-I", "-B", str(worker), str(request)],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            preexec_fn=limits,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            return {
                "state": "timeout",
                "stdout": stdout,
                "stderr": stderr,
                "elapsed_seconds": round(time.perf_counter() - started, 6),
            }
        try:
            result = json.loads(stdout)
            if (
                process.returncode != 0
                or not isinstance(result, dict)
                or "state" not in result
            ):
                raise ValueError("Worker failed")
        except (ValueError, TypeError):
            result = {
                "state": (
                    "resource_failure"
                    if process.returncode == -signal.SIGXCPU
                    else "worker_failure"
                ),
                "stdout": stdout,
                "stderr": stderr,
                "worker_exit_code": process.returncode,
            }
        result["elapsed_seconds"] = round(time.perf_counter() - started, 6)
        return result


def output_outcome(result: dict, expected, oracle: str) -> str:
    if result["state"] != "normal":
        return result["state"]
    options = expected if isinstance(expected, list) else [expected]
    actual = result["stdout"].strip()
    if any(actual == str(option).strip() for option in options):
        return "correct_output"
    if oracle == "yes-no":
        normalized = actual.upper().split()
        if any(normalized == str(option).upper().split() for option in options):
            return "correct_output"
    return (
        "confirmed_wrong_output"
        if oracle in {"yes-no", "exact"}
        else "unresolved_output_difference"
    )


def save(path: Path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def study_inputs(args):
    if args.results:
        record = None
        for candidate in iter_dataset_rows(args.results):
            if candidate.get("task_key") == args.task_key:
                record = candidate
        if record is None:
            raise ValueError(f"No saved record matches task key {args.task_key!r}")
        outputs = record.get("outputs") or {}
        source = (outputs.get("code") or {}).get("code")
        ec = outputs.get("executable_contract") or {}
        if not source or not ec.get("decorators"):
            raise ValueError(
                "Selected record lacks generated code or executable contracts"
            )
        contract = "\n\n".join([ec.get("imports", ""), ec["decorators"]]).strip() + "\n"
        for index, row in enumerate(iter_dataset_rows(args.dataset)):
            task = build_dataset_task(row, source_path=args.dataset, row_index=index)
            if task.task_key == args.task_key:
                return (
                    source,
                    contract,
                    row,
                    index,
                    {
                        "results": str(args.results.resolve()),
                        "results_sha256": file_sha256(args.results),
                        "task_key": args.task_key,
                        "recorded_run": record.get("run"),
                        "response_models": sorted(
                            {
                                attempt["model_name"]
                                for stage in (
                                    (record.get("llm_metrics") or {}).get("by_stage")
                                    or {}
                                ).values()
                                for attempt in (stage or {}).get("attempts") or []
                                if attempt.get("model_name")
                            }
                        ),
                    },
                )
        raise ValueError(
            "Selected task key was not found in the supplied matching benchmark dataset"
        )
    for index, row in enumerate(iter_dataset_rows(args.dataset)):
        if index == args.task_index:
            return (
                args.code.read_bytes().decode("utf-8"),
                args.contract.read_bytes().decode("utf-8"),
                row,
                index,
                None,
            )
    raise ValueError(f"Dataset has no row {args.task_index}")


def run_study(args) -> dict:
    registry = registered_operators()
    unknown = set(args.operator or []) - {item["operator"] for item in registry}
    if unknown:
        raise ValueError(
            f"Unknown mutation operators: {sorted(unknown)}. Use --list-operators."
        )
    if not args.prepare_only and os.name != "posix":
        raise RuntimeError(
            "Mutation execution requires POSIX process and resource limits; use --prepare-only to enumerate without execution."
        )
    source, contract, row, task_index, saved_record = study_inputs(args)
    bundle = get_tests(row, "private")
    inputs, expected = bundle["inputs"], bundle["outputs"]
    if not inputs or len(inputs) != len(expected):
        raise ValueError(
            "Mutation testing requires nonempty, paired private inputs and expected outputs"
        )
    cases = [
        {"id": f"private:{index}", "input": stdin, "expected": answer}
        for index, (stdin, answer) in enumerate(zip(inputs, expected))
    ]
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    oracle = args.oracle
    if oracle == "auto":
        oracle = (
            "yes-no"
            if row.get("problem_id") == 782
            and row.get("src_uid") == "ed9a763362abc6ed40356731f1036b38"
            else "unverified"
        )
    if oracle == "yes-no" and any(
        not str(value).split()
        or any(token.upper() not in {"YES", "NO"} for token in str(value).split())
        for case in cases
        for value in (
            case["expected"]
            if isinstance(case["expected"], list)
            else [case["expected"]]
        )
    ):
        raise ValueError("The yes-no oracle requires YES/NO benchmark outputs")
    root = args.output or Path("results/mutations") / datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%S_%fZ")
    root.mkdir(parents=True, exist_ok=False)
    (root / "seed.py").write_text(source)
    (root / "ec.py").write_text(contract)
    save(root / "cases.json", cases)
    configuration = {
        "code": (
            str(args.code.resolve())
            if not saved_record
            else "selected saved record: outputs.code.code"
        ),
        "code_sha256": sha256(source),
        "contract": (
            str(args.contract.resolve())
            if not saved_record
            else "selected saved record: outputs.executable_contract"
        ),
        "contract_sha256": sha256(contract),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": file_sha256(args.dataset),
        "task_index": task_index,
        "saved_record": saved_record,
        "task_uid": row.get("src_uid"),
        "problem_id": row.get("problem_id"),
        "pool": args.pool,
        "operators": args.operator,
        "registered_operators": registry,
        "mutmut_version": metadata.version("mutmut"),
        "scope": args.scope,
        "target": args.target,
        "seed": args.seed,
        "max_mutants": args.max_mutants,
        "selected_private_cases": len(cases),
        "available_private_cases": len(inputs),
        "oracle": oracle,
        "prepare_only": args.prepare_only,
        "timeout_seconds": args.timeout,
        "execution_boundary": "Temporary child processes, network disabled in workers, CPU/file/descriptor limits, process-group wall timeouts; not an OS security sandbox.",
    }
    save(root / "config.json", configuration)
    reference = []
    if not args.prepare_only:
        for case in cases:
            result = run_program(
                source, case["input"], target=args.target, timeout=args.timeout
            )
            reference.append(
                {
                    "case_id": case["id"],
                    "outcome": output_outcome(result, case["expected"], oracle),
                    "execution": result,
                }
            )
        save(root / "reference.json", reference)
        if any(item["outcome"] != "correct_output" for item in reference):
            summary = {
                "status": "reference_gate_failed",
                "selected_private_cases": len(cases),
                "mutants_executed": 0,
                "output": str(root),
            }
            save(root / "summary.json", summary)
            return summary
    pool, counts = candidate_pool(
        source,
        filename=str(args.code) if not saved_record else str(args.results) + ".py",
        pool=args.pool,
        operators=args.operator,
        scope=args.scope,
        target=args.target,
    )
    chosen = select_mutations(pool, limit=args.max_mutants, seed=args.seed)
    (root / "mutants").mkdir()
    (root / "diffs").mkdir()
    manifest = []
    for mutation in chosen:
        (root / "mutants" / f"{mutation['id']}.py").write_text(mutation["source"])
        (root / "diffs" / f"{mutation['id']}.diff").write_text(
            "".join(
                difflib.unified_diff(
                    source.splitlines(True),
                    mutation["source"].splitlines(True),
                    fromfile="seed.py",
                    tofile=f"{mutation['id']}.py",
                )
            )
        )
        manifest.append(
            {key: value for key, value in mutation.items() if key != "source"}
        )
    save(
        root / "selection.json",
        {
            "policy": "AST-valid and AST-unique pool, stable location/node/change/hash sort, random.Random(seed).sample without replacement before any mutant outcomes.",
            "counts": counts,
            "mutants": manifest,
        },
    )
    summary = {
        "status": "prepared" if args.prepare_only else "completed",
        "candidate_counts": counts,
        "selected_mutants": len(chosen),
        "mutants_executed": 0,
        "private_test_survivors": 0,
        "mutants_with_confirmed_wrong_outputs": 0,
        "program_failure_only_mutants": 0,
        "unresolved_mutants": 0,
        "raw_outcomes": {},
        "contract_outcomes": {},
        "reference_eligible_wrong_output_executions": 0,
        "reference_controlled_postcondition_violations": 0,
        "output": str(root),
        "interpretation": "Finite private-test survival is not equivalence. Crashes, timeouts, precondition rejection, integration/worker failures and unresolved differences are not postcondition detections. No semantic-recall claim.",
    }
    if args.prepare_only:
        save(root / "summary.json", summary)
        return summary
    controls = {}
    try:
        reference_contract = merge_contract(source, contract, args.target)
    except (SyntaxError, ValueError) as error:
        reference_contract = None
        controls = {
            case["id"]: {"state": "integration_failure", "error": str(error)}
            for case in cases
        }
    for case in cases:
        if reference_contract is not None:
            controls[case["id"]] = run_program(
                reference_contract,
                case["input"],
                target=args.target,
                trace=True,
                timeout=args.timeout,
            )
    save(root / "reference_controls.json", controls)
    outcomes_path = root / "outcomes.jsonl"
    outcomes_path.touch(exist_ok=False)
    for mutation in chosen:
        raw_states = []
        for case in cases:
            raw = run_program(
                mutation["source"],
                case["input"],
                target=args.target,
                timeout=args.timeout,
            )
            outcome = output_outcome(raw, case["expected"], oracle)
            raw_states.append(outcome)
            summary["raw_outcomes"][outcome] = (
                summary["raw_outcomes"].get(outcome, 0) + 1
            )
            item = {
                "mutant_id": mutation["id"],
                "case_id": case["id"],
                "raw_outcome": outcome,
                "raw_execution": raw,
            }
            if outcome == "confirmed_wrong_output":
                control = controls[case["id"]]
                eligible = (
                    output_outcome(control, case["expected"], oracle)
                    == "correct_output"
                    and control.get("predicate_calls", {}).get("ensure", 0) > 0
                )
                try:
                    combined = merge_contract(mutation["source"], contract, args.target)
                    checked = run_program(
                        combined,
                        case["input"],
                        target=args.target,
                        trace=True,
                        timeout=args.timeout,
                    )
                except (SyntaxError, ValueError) as error:
                    checked = {"state": "integration_failure", "error": str(error)}
                state = checked["state"]
                if state == "normal":
                    state = (
                        "fault_not_reaching_postcondition"
                        if not checked.get("predicate_calls", {}).get("ensure", 0)
                        else (
                            "accepted_faulty_output"
                            if checked["stdout"].strip() == raw["stdout"].strip()
                            else "integration_output_changed"
                        )
                    )
                detection = eligible and state == "postcondition_violation"
                summary["reference_eligible_wrong_output_executions"] += int(eligible)
                summary["reference_controlled_postcondition_violations"] += int(
                    detection
                )
                summary["contract_outcomes"][state] = (
                    summary["contract_outcomes"].get(state, 0) + 1
                )
                item.update(
                    contract_execution=checked,
                    contract_outcome=state,
                    reference_control_eligible=eligible,
                    reference_controlled_detection=detection,
                )
            with outcomes_path.open("a") as handle:
                handle.write(json.dumps(item) + "\n")
        summary["mutants_executed"] += 1
        if all(state == "correct_output" for state in raw_states):
            summary["private_test_survivors"] += 1
        elif "confirmed_wrong_output" in raw_states:
            summary["mutants_with_confirmed_wrong_outputs"] += 1
        elif any(
            state in {"unresolved_output_difference", "worker_failure"}
            for state in raw_states
        ):
            summary["unresolved_mutants"] += 1
        else:
            summary["program_failure_only_mutants"] += 1
    save(root / "summary.json", summary)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="contract mutate",
        description="Prepare or execute an offline mutmut study; defaults to the bundled Football case.",
    )
    parser.add_argument("--code", type=Path, default=Path("motivating_example/code.py"))
    parser.add_argument(
        "--contract", type=Path, default=Path("motivating_example/ec.py")
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("motivating_example/xCodeEval_update.json")
    )
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument(
        "--results",
        type=Path,
        help="Select embedded code and contracts from a saved result JSONL file.",
    )
    parser.add_argument(
        "--task-key",
        help="Exact task_key to select from --results; the dataset must be the matching benchmark.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pool", choices=["all", "arithmetic"], default="all")
    parser.add_argument(
        "--operator",
        action="append",
        help="Restrict to a registered operator name; repeat to select several.",
    )
    parser.add_argument(
        "--scope", choices=["all", "functions", "target"], default="all"
    )
    parser.add_argument("--target", default="solve")
    parser.add_argument("--max-mutants", type=int, default=10)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=5)
    parser.add_argument(
        "--oracle",
        choices=["auto", "yes-no", "exact", "unverified"],
        default="auto",
        help="auto certifies only the bundled Football identity; exact asserts a unique exact-output oracle for a custom task.",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--list-operators", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.results) != bool(args.task_key):
        parser.error("Use --results and --task-key together.")
    if (
        args.max_mutants < 1
        or args.task_index < 0
        or args.timeout <= 0
        or (args.max_cases is not None and args.max_cases < 1)
    ):
        parser.error(
            "Mutation/case limits and timeout must be positive; task index must be non-negative."
        )
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.list_operators:
            print(json.dumps(registered_operators(), indent=2))
            return
        summary = run_study(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Mutation study failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(json.dumps(summary, indent=2))
    if summary["status"] == "reference_gate_failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
