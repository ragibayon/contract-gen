from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ProgramRunResult:
    stdout: str
    stderr: str
    returncode: int | None
    timed_out: bool = False
    timeout_sec: float | None = None
    elapsed_sec: float = 0.0
    failure_kind: str | None = None
    error_type: str | None = None


@dataclass
class WitnessCaseResult:
    index: int
    input: str
    expected: Any
    actual: str
    stderr: str = ""
    returncode: int | None = None
    timed_out: bool = False
    timeout_sec: float | None = None
    elapsed_sec: float = 0.0
    passed: bool = False
    completed: bool = False
    failure_kind: str | None = None
    error_type: str | None = None


@dataclass
class WitnessFailure:
    index: int
    input: str
    expected: Any
    actual: str
    stderr: str = ""
    returncode: int | None = None
    timed_out: bool = False
    timeout_sec: float | None = None
    elapsed_sec: float = 0.0
    failure_kind: str | None = None
    error_type: str | None = None


@dataclass
class ExecutionWitnessReport:
    passed: int
    total: int
    pass_ratio: float
    failures: list[WitnessFailure] = field(default_factory=list)
    cases: list[WitnessCaseResult] = field(default_factory=list)
    completed: int = 0
    completeness: float = 0.0
    contract_violations: int = 0
    runtime_errors: int = 0
    timeouts: int = 0
    wrong_outputs: int = 0
    nonzero_exits: int = 0


def normalize_stdin(input_text: str) -> str:
    return str(input_text).replace("\r\n", "\n").replace("\r", "\n")


def _looks_like_contract_violation(stderr: str) -> bool:
    markers = [
        "icontract.errors.ViolationError",
        "ViolationError",
        "icontract",
        "Precondition",
        "Postcondition",
        "precondition",
        "postcondition",
        "contract",
        "AssertionError",
    ]
    return any(marker in (stderr or "") for marker in markers)


def _extract_error_type(stderr: str) -> str | None:
    stderr = stderr or ""
    if not stderr:
        return None
    for line in reversed(stderr.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        if ":" in stripped:
            candidate = stripped.split(":", 1)[0].strip()
        else:
            candidate = stripped
        if candidate.endswith("Error") or candidate.endswith("Exception") or candidate in {"SystemExit", "KeyboardInterrupt"}:
            return candidate
    return None


def classify_run_result(run: ProgramRunResult, *, expected: Any | None = None, actual_matches_expected: bool | None = None) -> str | None:
    if run.timed_out:
        return "timeout"
    if run.returncode is None:
        return "runtime_error"
    if run.returncode != 0:
        if _looks_like_contract_violation(run.stderr):
            return "contract_violation"
        error_type = _extract_error_type(run.stderr)
        if error_type:
            return "runtime_error"
        return "nonzero_exit"
    if actual_matches_expected is False:
        return "wrong_output"
    return None


def run_python_program_detailed(
    code: str,
    input_text: str,
    timeout_sec: float = 10.0,
) -> ProgramRunResult:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "solution.py"
        path.write_text(code, encoding="utf-8")

        try:
            proc = subprocess.run(
                [sys.executable, str(path)],
                input=normalize_stdin(input_text),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_sec,
            )
            elapsed = time.perf_counter() - started
            run = ProgramRunResult(
                stdout=proc.stdout.strip(),
                stderr=proc.stderr.strip(),
                returncode=proc.returncode,
                timed_out=False,
                timeout_sec=timeout_sec,
                elapsed_sec=round(elapsed, 6),
            )
            run.failure_kind = classify_run_result(run)
            run.error_type = _extract_error_type(run.stderr)
            return run
        except subprocess.TimeoutExpired as exc:
            elapsed = time.perf_counter() - started
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            return ProgramRunResult(
                stdout=str(stdout).strip(),
                stderr=str(stderr).strip() or f"TIMEOUT after {timeout_sec} seconds",
                returncode=None,
                timed_out=True,
                timeout_sec=timeout_sec,
                elapsed_sec=round(elapsed, 6),
                failure_kind="timeout",
                error_type="TimeoutExpired",
            )


def run_python_program(
    code: str, input_text: str, timeout_sec: float = 10.0
) -> tuple[str, str, int]:
    """Backward-compatible wrapper used by older call sites."""
    run = run_python_program_detailed(code=code, input_text=input_text, timeout_sec=timeout_sec)
    return run.stdout, run.stderr, -1 if run.returncode is None else run.returncode


def output_matches(actual: str, expected: Any) -> bool:
    actual = actual.strip()

    if isinstance(expected, list):
        return actual in {str(x).strip() for x in expected}

    return actual == str(expected).strip()


def evaluate_execution_witnesses(
    code: str,
    witnesses: dict[str, list[Any]],
    timeout_sec: float = 5.0,
) -> ExecutionWitnessReport:
    inputs = witnesses.get("inputs", [])
    outputs = witnesses.get("outputs", [])

    total = min(len(inputs), len(outputs))
    passed = 0
    completed = 0
    failures: list[WitnessFailure] = []
    cases: list[WitnessCaseResult] = []
    counts = {
        "contract_violation": 0,
        "runtime_error": 0,
        "timeout": 0,
        "wrong_output": 0,
        "nonzero_exit": 0,
    }

    for i in range(total):
        input_text = inputs[i]
        expected = outputs[i]

        run = run_python_program_detailed(
            code=code,
            input_text=str(input_text),
            timeout_sec=timeout_sec,
        )
        matches = run.returncode == 0 and output_matches(run.stdout, expected)
        failure_kind = classify_run_result(
            run,
            expected=expected,
            actual_matches_expected=matches,
        )
        is_completed = (not run.timed_out) and run.returncode == 0
        if matches:
            passed += 1
        if is_completed:
            completed += 1
        if failure_kind in counts:
            counts[failure_kind] += 1

        case = WitnessCaseResult(
            index=i,
            input=str(input_text),
            expected=expected,
            actual=run.stdout,
            stderr=run.stderr,
            returncode=run.returncode,
            timed_out=run.timed_out,
            timeout_sec=run.timeout_sec,
            elapsed_sec=run.elapsed_sec,
            passed=matches,
            completed=is_completed,
            failure_kind=failure_kind,
            error_type=run.error_type,
        )
        cases.append(case)

        if not matches:
            failures.append(
                WitnessFailure(
                    index=i,
                    input=str(input_text),
                    expected=expected,
                    actual=run.stdout,
                    stderr=run.stderr,
                    returncode=run.returncode,
                    timed_out=run.timed_out,
                    timeout_sec=run.timeout_sec,
                    elapsed_sec=run.elapsed_sec,
                    failure_kind=failure_kind,
                    error_type=run.error_type,
                )
            )

    return ExecutionWitnessReport(
        passed=passed,
        total=total,
        pass_ratio=(passed / total) if total else 0.0,
        failures=failures,
        cases=cases,
        completed=completed,
        completeness=(completed / total) if total else 0.0,
        contract_violations=counts["contract_violation"],
        runtime_errors=counts["runtime_error"],
        timeouts=counts["timeout"],
        wrong_outputs=counts["wrong_output"],
        nonzero_exits=counts["nonzero_exit"],
    )
