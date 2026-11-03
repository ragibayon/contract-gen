from __future__ import annotations

import re
from typing import Any

from contract.agent.schema.contract_probe import ContractProbe, ContractProbeReport
from contract.eval.execution import (
    classify_run_result,
    output_matches,
    run_python_program_detailed,
)


def evaluate_contract_probes(
    code: str,
    probes: list[ContractProbe],
    timeout_sec: float = 10.0,
    executable_requirement_ids: set[str] | list[str] | tuple[str, ...] | None = None,
) -> ContractProbeReport:
    """Evaluate generated contract probes and store full execution traces.

    Important negative-probe policy:
    - A violation probe passes only when the merged contracted program fails with an
      actual contract violation, e.g. icontract.errors.ViolationError.
    - A ValueError, parser crash, timeout, or arbitrary non-zero exit does NOT count
      as successful contract evidence. It is surfaced as a failed probe case with a
      probe_failure_kind explaining what happened.
    """

    passed = 0
    failures: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    counts = {
        "contract_violation": 0,
        "invalid_violation_probe": 0,
        "runtime_error": 0,
        "timeout": 0,
        "wrong_output": 0,
        "predicate_failed": 0,
        "relation_failed": 0,
        "nonzero_exit": 0,
        "supported_violation_probe": 0,
        "unsupported_violation_probe": 0,
    }
    covered_requirement_ids = {str(item) for item in (executable_requirement_ids or [])}
    uncovered_violation_requirement_ids: set[str] = set()

    for index, probe in enumerate(probes):
        ok, detail = _evaluate_one_probe(
            code=code,
            probe=probe,
            timeout_sec=timeout_sec,
        )

        target_requirement_ids = {str(item) for item in (probe.requirement_ids or [])}
        covered_targets = sorted(target_requirement_ids & covered_requirement_ids)
        uncovered_targets = sorted(target_requirement_ids - covered_requirement_ids)
        violation_supported = None
        if probe.type == "violation":





            violation_supported = bool(covered_targets) if target_requirement_ids else None
            if violation_supported is True:
                counts["supported_violation_probe"] += 1
            elif violation_supported is False:
                counts["unsupported_violation_probe"] += 1
                uncovered_violation_requirement_ids.update(uncovered_targets)

        case = {
            "index": index,
            "type": probe.type,
            "input": probe.input,
            "paired_input": probe.paired_input,
            "expected": probe.expected,
            "relation": probe.relation,
            "requirement_ids": probe.requirement_ids,
            "covered_requirement_ids": covered_targets,
            "uncovered_requirement_ids": uncovered_targets,
            "violation_supported_by_lowered_contract": violation_supported,
            "reason": probe.reason,
            "priority": getattr(probe, "priority", "low"),
            "passed": ok,
            **detail,
        }
        cases.append(case)

        probe_failure_kind = case.get("probe_failure_kind") or case.get("failure_kind")
        failure_kind = case.get("failure_kind")

        if failure_kind == "contract_violation":
            counts["contract_violation"] += 1
        if probe_failure_kind == "invalid_violation_probe":
            counts["invalid_violation_probe"] += 1
        elif probe_failure_kind == "predicate_failed":
            counts["predicate_failed"] += 1
        elif probe_failure_kind == "relation_failed":
            counts["relation_failed"] += 1
        elif probe_failure_kind == "wrong_output" or failure_kind == "wrong_output":
            counts["wrong_output"] += 1
        elif failure_kind == "runtime_error":
            counts["runtime_error"] += 1
        elif failure_kind == "timeout":
            counts["timeout"] += 1
        elif failure_kind == "nonzero_exit":
            counts["nonzero_exit"] += 1

        if ok:
            passed += 1
        else:
            failures.append(case)

    total = len(probes)
    return ContractProbeReport(
        passed=passed,
        total=total,
        pass_ratio=(passed / total) if total else 0.0,
        failures=failures,
        cases=cases,
        contract_violations=counts["contract_violation"],
        invalid_violation_probes=counts["invalid_violation_probe"],
        runtime_errors=counts["runtime_error"],
        timeouts=counts["timeout"],
        wrong_outputs=counts["wrong_output"],
        predicate_failures=counts["predicate_failed"],
        relation_failures=counts["relation_failed"],
        nonzero_exits=counts["nonzero_exit"],
        supported_violation_probes=counts["supported_violation_probe"],
        unsupported_violation_probes=counts["unsupported_violation_probe"],
        uncovered_violation_requirement_ids=sorted(uncovered_violation_requirement_ids),
    )


def _evaluate_one_probe(
    code: str,
    probe: ContractProbe,
    timeout_sec: float,
) -> tuple[bool, dict[str, Any]]:
    if probe.type == "oracle":
        return _evaluate_oracle(code, probe, timeout_sec)

    if probe.type == "violation":
        return _evaluate_violation(code, probe, timeout_sec)

    if probe.type == "metamorphic":
        return _evaluate_metamorphic(code, probe, timeout_sec)

    if probe.type == "predicate":
        return _evaluate_predicate(code, probe, timeout_sec)

    return False, {
        "error": f"Unsupported probe type: {probe.type}",
        "probe_failure_kind": "unsupported_probe_type",
    }


def _run_detail(code: str, input_text: str, timeout_sec: float) -> dict[str, Any]:
    run = run_python_program_detailed(code=code, input_text=input_text, timeout_sec=timeout_sec)
    return {
        "actual": run.stdout,
        "stderr": run.stderr,
        "returncode": run.returncode,
        "timed_out": run.timed_out,
        "timeout_sec": run.timeout_sec,
        "elapsed_sec": run.elapsed_sec,
        "failure_kind": run.failure_kind,
        "error_type": run.error_type,
    }


def _evaluate_oracle(
    code: str,
    probe: ContractProbe,
    timeout_sec: float,
) -> tuple[bool, dict[str, Any]]:
    run = run_python_program_detailed(code=code, input_text=probe.input, timeout_sec=timeout_sec)
    ok = run.returncode == 0 and output_matches(run.stdout, probe.expected)
    failure_kind = classify_run_result(run, expected=probe.expected, actual_matches_expected=ok)
    return ok, {
        "actual": run.stdout,
        "stderr": run.stderr,
        "returncode": run.returncode,
        "timed_out": run.timed_out,
        "timeout_sec": run.timeout_sec,
        "elapsed_sec": run.elapsed_sec,
        "failure_kind": failure_kind,
        "probe_failure_kind": None if ok else (failure_kind or "wrong_output"),
        "error_type": run.error_type,
    }


def _evaluate_violation(
    code: str,
    probe: ContractProbe,
    timeout_sec: float,
) -> tuple[bool, dict[str, Any]]:
    run = run_python_program_detailed(code=code, input_text=probe.input, timeout_sec=timeout_sec)
    failure_kind = classify_run_result(run)





    ok = failure_kind == "contract_violation"

    if ok:
        probe_failure_kind = None
    elif run.returncode == 0 and not run.timed_out:
        probe_failure_kind = "violation_not_triggered"
    elif failure_kind in {"runtime_error", "nonzero_exit"}:
        probe_failure_kind = "invalid_violation_probe"
    elif failure_kind == "timeout":
        probe_failure_kind = "timeout"
    else:
        probe_failure_kind = failure_kind or "violation_not_triggered"

    return ok, {
        "actual": run.stdout,
        "stderr": run.stderr,
        "returncode": run.returncode,
        "timed_out": run.timed_out,
        "timeout_sec": run.timeout_sec,
        "elapsed_sec": run.elapsed_sec,
        "failure_kind": failure_kind,
        "probe_failure_kind": probe_failure_kind,
        "error_type": run.error_type,
    }


def _evaluate_metamorphic(
    code: str,
    probe: ContractProbe,
    timeout_sec: float,
) -> tuple[bool, dict[str, Any]]:
    if not probe.paired_input:
        return False, {
            "error": "Metamorphic probe missing paired_input.",
            "probe_failure_kind": "invalid_probe_definition",
        }

    run_a = run_python_program_detailed(code=code, input_text=probe.input, timeout_sec=timeout_sec)
    run_b = run_python_program_detailed(code=code, input_text=probe.paired_input, timeout_sec=timeout_sec)

    failure_a = classify_run_result(run_a)
    failure_b = classify_run_result(run_b)

    if run_a.returncode != 0 or run_b.returncode != 0:
        return False, {
            "actual": run_a.stdout,
            "paired_actual": run_b.stdout,
            "stderr": run_a.stderr,
            "paired_stderr": run_b.stderr,
            "returncode": run_a.returncode,
            "paired_returncode": run_b.returncode,
            "timed_out": run_a.timed_out,
            "paired_timed_out": run_b.timed_out,
            "timeout_sec": run_a.timeout_sec,
            "paired_timeout_sec": run_b.timeout_sec,
            "elapsed_sec": run_a.elapsed_sec,
            "paired_elapsed_sec": run_b.elapsed_sec,
            "failure_kind": failure_a,
            "paired_failure_kind": failure_b,
            "probe_failure_kind": failure_a or failure_b or "metamorphic_execution_failed",
            "error_type": run_a.error_type,
            "paired_error_type": run_b.error_type,
        }

    relation = probe.relation.strip()

    if relation == "same_output":
        ok = run_a.stdout.strip() == run_b.stdout.strip()
    else:
        ok = False

    return ok, {
        "actual": run_a.stdout,
        "paired_actual": run_b.stdout,
        "stderr": run_a.stderr,
        "paired_stderr": run_b.stderr,
        "returncode": run_a.returncode,
        "paired_returncode": run_b.returncode,
        "timed_out": run_a.timed_out,
        "paired_timed_out": run_b.timed_out,
        "timeout_sec": run_a.timeout_sec,
        "paired_timeout_sec": run_b.timeout_sec,
        "elapsed_sec": run_a.elapsed_sec,
        "paired_elapsed_sec": run_b.elapsed_sec,
        "relation": relation,
        "failure_kind": None if ok else "relation_failed",
        "probe_failure_kind": None if ok else "relation_failed",
        "error_type": run_a.error_type,
        "paired_error_type": run_b.error_type,
    }



def _stdout_as_int(stdout: str) -> int | None:
    text = stdout.strip()
    if not re.fullmatch(r"[-+]?\d+", text):
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _parse_numeric_args(relation: str) -> tuple[str, list[int]]:
    parts = [part.strip() for part in relation.split(":")]
    name = parts[0]
    args: list[int] = []
    for part in parts[1:]:
        if not part:
            continue
        try:
            args.append(int(part))
        except ValueError:
            return name, []
    return name, args


def _check_stdout_predicate(stdout: str, relation: str, probe: ContractProbe) -> bool:
    text = stdout.strip()
    name, args = _parse_numeric_args(relation.strip())
    value = _stdout_as_int(text)

    if name == "stdout_nonempty":
        return bool(text)
    if name == "stdout_length_2":
        return len(text) == 2
    if name == "stdout_is_uppercase":
        return text.isupper()
    if name == "stdout_in_input":
        return text in probe.input
    if name == "stdout_one_of_expected":
        return output_matches(stdout, probe.expected)
    if name == "stdout_int":
        return value is not None
    if name == "stdout_positive_int":
        return value is not None and value > 0
    if name == "stdout_int_ge" and len(args) == 1:
        return value is not None and value >= args[0]
    if name == "stdout_int_gt" and len(args) == 1:
        return value is not None and value > args[0]
    if name == "stdout_int_le" and len(args) == 1:
        return value is not None and value <= args[0]
    if name == "stdout_int_lt" and len(args) == 1:
        return value is not None and value < args[0]
    if name == "stdout_int_between" and len(args) == 2:
        low, high = args
        return value is not None and low <= value <= high
    if name == "stdout_int_strict_between" and len(args) == 2:
        low, high = args
        return value is not None and low < value < high
    return False


def _evaluate_predicate(
    code: str,
    probe: ContractProbe,
    timeout_sec: float,
) -> tuple[bool, dict[str, Any]]:
    run = run_python_program_detailed(code=code, input_text=probe.input, timeout_sec=timeout_sec)

    if run.returncode != 0:
        failure_kind = classify_run_result(run)
        return False, {
            "actual": run.stdout,
            "stderr": run.stderr,
            "returncode": run.returncode,
            "timed_out": run.timed_out,
            "timeout_sec": run.timeout_sec,
            "elapsed_sec": run.elapsed_sec,
            "failure_kind": failure_kind,
            "probe_failure_kind": failure_kind,
            "error_type": run.error_type,
        }

    relation = probe.relation.strip()

    ok = _check_stdout_predicate(run.stdout, relation, probe)

    return ok, {
        "actual": run.stdout,
        "stderr": run.stderr,
        "returncode": run.returncode,
        "timed_out": run.timed_out,
        "timeout_sec": run.timeout_sec,
        "elapsed_sec": run.elapsed_sec,
        "relation": relation,
        "failure_kind": None if ok else "predicate_failed",
        "probe_failure_kind": None if ok else "predicate_failed",
        "error_type": run.error_type,
    }

def compare_probe_reports(
    *,
    probes: list[ContractProbe],
    code_report: ContractProbeReport,
    contract_report: ContractProbeReport,
) -> dict[str, Any]:
    """Compare identical probe executions without mis-scoring negative probes.

    A violation probe is expected to fail the code-only evidence check because no
    executable contract exists there. Its comparison classification therefore uses
    execution outcomes, not the two report pass booleans.
    """

    code_cases = {int(case["index"]): case for case in code_report.cases}
    contract_cases = {int(case["index"]): case for case in contract_report.cases}
    counts: dict[str, int] = {}
    cases: list[dict[str, Any]] = []

    for index, probe in enumerate(probes):
        code_case = code_cases.get(index, {})
        contract_case = contract_cases.get(index, {})
        code_passed = code_case.get("passed") is True
        contract_passed = contract_case.get("passed") is True

        if probe.type == "violation":
            code_failure = code_case.get("failure_kind")
            contract_failure = contract_case.get("failure_kind")
            if contract_failure == "contract_violation":
                classification = (
                    "preexisting_contract_violation"
                    if code_failure == "contract_violation"
                    else "contract_added_rejection"
                )
            elif code_failure in {"runtime_error", "nonzero_exit", "timeout"}:
                classification = "invalid_probe_on_code"
            elif contract_failure in {"runtime_error", "nonzero_exit", "timeout"}:
                classification = "contract_execution_failure"
            else:
                classification = "missing_contract_enforcement"
        elif code_passed and contract_passed:
            classification = "unchanged_pass"
        elif code_passed and not contract_passed:
            classification = "contract_regression"
        elif not code_passed and contract_passed:
            classification = "contract_changed_to_pass"
        elif (
            code_case.get("failure_kind") == contract_case.get("failure_kind")
            and code_case.get("probe_failure_kind")
            == contract_case.get("probe_failure_kind")
        ):
            classification = "shared_failure"
        else:
            classification = "changed_failure"

        counts[classification] = counts.get(classification, 0) + 1
        cases.append(
            {
                "index": index,
                "type": probe.type,
                "requirement_ids": probe.requirement_ids,
                "covered_requirement_ids": contract_case.get("covered_requirement_ids"),
                "uncovered_requirement_ids": contract_case.get("uncovered_requirement_ids"),
                "violation_supported_by_lowered_contract": contract_case.get("violation_supported_by_lowered_contract"),
                "classification": classification,
                "code_passed": code_passed,
                "contract_passed": contract_passed,
                "code_failure_kind": code_case.get("failure_kind"),
                "code_probe_failure_kind": code_case.get("probe_failure_kind"),
                "contract_failure_kind": contract_case.get("failure_kind"),
                "contract_probe_failure_kind": contract_case.get(
                    "probe_failure_kind"
                ),
            }
        )

    return {
        "total": len(probes),
        "code_passed": code_report.passed,
        "contract_passed": contract_report.passed,
        "code_pass_ratio": code_report.pass_ratio,
        "contract_pass_ratio": contract_report.pass_ratio,
        "counts": counts,
        "cases": cases,
    }
