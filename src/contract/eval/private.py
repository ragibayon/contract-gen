from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable, Mapping

from contract.eval.execution import evaluate_execution_witnesses


def _case_count(tests: dict[str, list[Any]] | None) -> int:
    tests = tests or {"inputs": [], "outputs": []}
    return min(len(tests.get("inputs") or []), len(tests.get("outputs") or []))


def _empty_eval(*, artifact: str, private_tests: dict[str, list[Any]] | None, reason: str) -> dict[str, Any]:
    total = _case_count(private_tests)
    return {
        "split": "private",
        "artifact": artifact,
        "passed": 0,
        "total": total,
        "pass_ratio": 0.0 if total else None,
        "pass_at_1": False,
        "completed": 0,
        "completeness": 0.0 if total else None,
        "contract_violations": 0,
        "runtime_errors": 0,
        "timeouts": 0,
        "wrong_outputs": 0,
        "nonzero_exits": 0,
        "cases": [],
        "failures": [],
        "error": reason,
    }


def evaluate_private_cases(
    *,
    code: str | None,
    private_tests: dict[str, list[Any]] | None,
    timeout_sec: float = 10.0,
    artifact: str = "contracted_code",
) -> dict[str, Any]:
    private_tests = private_tests or {"inputs": [], "outputs": []}

    if not code:
        return _empty_eval(
            artifact=artifact,
            private_tests=private_tests,
            reason=f"No {artifact} artifact available; counted as 0 for outside evaluation.",
        )

    report = evaluate_execution_witnesses(
        code=code,
        witnesses=private_tests,
        timeout_sec=timeout_sec,
    )
    return {
        "split": "private",
        "artifact": artifact,
        "passed": report.passed,
        "total": report.total,
        "pass_ratio": report.pass_ratio,
        "pass_at_1": report.total > 0 and report.pass_ratio == 1.0,
        "completed": report.completed,
        "completeness": report.completeness,
        "contract_violations": report.contract_violations,
        "runtime_errors": report.runtime_errors,
        "timeouts": report.timeouts,
        "wrong_outputs": report.wrong_outputs,
        "nonzero_exits": report.nonzero_exits,
        "cases": [asdict(case) for case in report.cases],
        "failures": [asdict(failure) for failure in report.failures],
        "error": None,
    }


def summarize_private_evaluations(
    records: list[dict[str, Any]],
    *,
    key: str = "private_eval",
) -> dict[str, Any]:
    evaluated = [r for r in records if ((r.get(key) or {}).get("total") or 0) > 0]
    passed_cases = sum(int((r.get(key) or {}).get("passed") or 0) for r in evaluated)
    total_cases = sum(int((r.get(key) or {}).get("total") or 0) for r in evaluated)
    completed_cases = sum(int((r.get(key) or {}).get("completed") or 0) for r in evaluated)
    pass_at_1 = sum(1 for r in evaluated if (r.get(key) or {}).get("pass_at_1") is True)
    ratios = [float((r.get(key) or {}).get("pass_ratio") or 0.0) for r in evaluated]
    completeness_values = [float((r.get(key) or {}).get("completeness") or 0.0) for r in evaluated]

    def _sum(field: str) -> int:
        return sum(int((r.get(key) or {}).get(field) or 0) for r in evaluated)

    return {
        "evaluated_instances": len(evaluated),
        "pass_at_1": pass_at_1,
        "pass_at_1_rate": round(pass_at_1 / len(evaluated), 4) if evaluated else None,
        "passed_cases": passed_cases,
        "total_cases": total_cases,
        "apr_micro": round(passed_cases / total_cases, 4) if total_cases else None,
        "apr_macro": round(sum(ratios) / len(ratios), 4) if ratios else None,
        "completed_cases": completed_cases,
        "completeness_micro": round(completed_cases / total_cases, 4) if total_cases else None,
        "completeness_macro": round(sum(completeness_values) / len(completeness_values), 4) if completeness_values else None,
        "contract_violations": _sum("contract_violations"),
        "runtime_errors": _sum("runtime_errors"),
        "timeouts": _sum("timeouts"),
        "wrong_outputs": _sum("wrong_outputs"),
        "nonzero_exits": _sum("nonzero_exits"),
    }


def compare_private_evaluations(record: Mapping[str, Any]) -> dict[str, Any]:
    code_eval = record.get("code_private_eval") or {}
    contract_eval = record.get("contract_private_eval") or {}
    return {
        "code_private_passed": int(code_eval.get("passed") or 0),
        "code_private_total": int(code_eval.get("total") or 0),
        "contract_private_passed": int(contract_eval.get("passed") or 0),
        "contract_private_total": int(contract_eval.get("total") or 0),
        "pre_correctness": code_eval.get("pass_ratio"),
        "post_correctness": contract_eval.get("pass_ratio"),
        "pre_completeness": code_eval.get("completeness"),
        "post_completeness": contract_eval.get("completeness"),
        "contract_hurt_correctness": int(contract_eval.get("passed") or 0) < int(code_eval.get("passed") or 0),
        "contract_hurt_completeness": int(contract_eval.get("completed") or 0) < int(code_eval.get("completed") or 0),
        "contract_caused_violations": int(contract_eval.get("contract_violations") or 0),
        "contract_caused_runtime_errors": int(contract_eval.get("runtime_errors") or 0),
        "contract_caused_timeouts": int(contract_eval.get("timeouts") or 0),
    }
