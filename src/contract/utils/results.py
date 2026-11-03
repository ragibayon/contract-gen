from __future__ import annotations

import json
import re
import traceback
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

from contract.eval.private import compare_private_evaluations, summarize_private_evaluations


def _dump(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if is_dataclass(obj):
        return asdict(obj)
    return obj


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sanitize_path_part(value: Any) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "__", text).strip("._-")
    return text[:180] or "item"


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def append_jsonl(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _bool_score(value: bool | None) -> float | None:
    if value is None:
        return None
    return 1.0 if value else 0.0


def _report_ratio(report: Any) -> float | None:
    if report is None:
        return None
    return getattr(report, "pass_ratio", None)


def pipeline_passed(result: Any) -> bool:
    """Hard-gate diagnostic for pipeline completion only.

    Public witnesses, generated probes, and private tests are behavioral evidence
    and are scored separately.  A pipeline run passes this gate when the pipeline
    produced static-valid artifacts and generated/evaluated at least one contract
    probe.  Probe pass ratio is intentionally not a hard gate.
    """
    if result.specification is None or result.specification.output is None:
        return False
    if result.interface is None or result.interface.output is None:
        return False
    if result.code is None or result.code.output is None:
        return False
    if not result.code.static_validation_passed:
        return False
    if result.executable_contract is None or result.executable_contract.output is None:
        return False
    if not result.executable_contract.static_validation_passed:
        return False
    if result.merged_contract is None or result.merged_contract.output is None:
        return False
    if not result.merged_contract.static_validation_passed:
        return False
    if result.contract_probe is None or result.contract_probe.output is None:
        return False
    if result.contract_probe.probe_report is None:
        return False
    if result.contract_probe.probe_report.total == 0:
        return False
    return True


def dbc_gate_report(result: Any) -> dict[str, Any]:
    return {
        "specification_exists": bool(result.specification is not None and result.specification.output is not None),
        "interface_exists": bool(result.interface is not None and result.interface.output is not None),
        "code_exists": bool(result.code is not None and result.code.output is not None),
        "code_static_validation_passed": result.code.static_validation_passed if result.code else None,
        "code_execution_witness_passed": result.code.execution_witness_passed if result.code else None,
        "executable_contract_exists": bool(result.executable_contract is not None and result.executable_contract.output is not None),
        "executable_contract_static_validation_passed": result.executable_contract.static_validation_passed if result.executable_contract else None,
        "merged_contract_exists": bool(result.merged_contract is not None and result.merged_contract.output is not None),
        "merged_contract_static_validation_passed": result.merged_contract.static_validation_passed if result.merged_contract else None,
        "merged_contract_execution_witness_passed": result.merged_contract.execution_witness_passed if result.merged_contract else None,
        "public_witnesses_are_behavioral_evidence": True,
        "contract_probe_exists": bool(result.contract_probe is not None and result.contract_probe.output is not None),
        "contract_probe_report_exists": bool(result.contract_probe is not None and result.contract_probe.probe_report is not None),
        "contract_probe_total": result.contract_probe.probe_report.total if result.contract_probe and result.contract_probe.probe_report else None,
        "contract_probe_pass_ratio": result.contract_probe.probe_report.pass_ratio if result.contract_probe and result.contract_probe.probe_report else None,
        "contract_probe_pass_ratio_is_behavioral_evidence": True,
    }



def probe_availability(result: Any) -> dict[str, Any]:
    """Explain whether self-probes were actually produced/evaluated.

    This is intentionally separate from pipeline_passed: failed earlier stages may
    legitimately prevent probe execution, but the record should still say *where*
    probes are missing and why.
    """
    reasons: list[str] = []
    required = [
        ("specification", result.specification),
        ("interface", result.interface),
        ("code_generation", result.code),
        ("executable_contract", result.executable_contract),
        ("merge_contract", result.merged_contract),
    ]
    for name, stage in required:
        if stage is None:
            reasons.append(f"{name} stage was not run")
            break
        if getattr(stage, "output", None) is None:
            error = getattr(stage, "error", None)
            reasons.append(f"{name} produced no output" + (f": {error}" if error else ""))
            break
        if getattr(stage, "static_validation_passed", None) is False:
            reasons.append(f"{name} failed static validation")
            break

    probe_stage = result.contract_probe
    plan = getattr(probe_stage, "output", None) if probe_stage else None
    code_report = getattr(probe_stage, "code_probe_report", None) if probe_stage else None
    contract_report = getattr(probe_stage, "contract_probe_report", None) if probe_stage else None
    legacy_report = getattr(probe_stage, "probe_report", None) if probe_stage else None

    if probe_stage is None:
        if not reasons:
            reasons.append("contract_probe stage was not run")
    else:
        if plan is None:
            reasons.append(getattr(probe_stage, "error", None) or "contract_probe produced no probe plan")
        if code_report is None:
            reasons.append("code-only probe report is missing")
        if contract_report is None and legacy_report is None:
            reasons.append("contracted-code probe report is missing")

    return {
        "status": "available" if not reasons else "missing",
        "missing": bool(reasons),
        "missing_reasons": reasons,
        "plan_exists": plan is not None,
        "plan_total": len(getattr(plan, "probes", []) or []) if plan is not None else 0,
        "code_probe_report_exists": code_report is not None,
        "contract_probe_report_exists": (contract_report is not None or legacy_report is not None),
        "comparison_exists": bool(getattr(probe_stage, "probe_comparison", None)) if probe_stage else False,
        "source": getattr(probe_stage, "source", None) if probe_stage else None,
        "stage_error": getattr(probe_stage, "error", None) if probe_stage else None,
    }

def contract_lowering_coverage(result: Any) -> dict[str, Any]:
    spec = result.specification.output if result.specification else None
    executable_contract = result.executable_contract.output if result.executable_contract else None
    contracts = list(getattr(spec, "contracts", []) or [])
    coverage = list(getattr(executable_contract, "coverage", []) or [])
    coverage_by_id = {item.contract_id: item for item in coverage}

    items = []
    lowered = 0
    omitted = 0
    missing = 0
    for contract in contracts:
        item = coverage_by_id.get(contract.id)
        if item is None:
            missing += 1
            items.append(
                {
                    "contract_id": contract.id,
                    "contract_type": contract.type,
                    "contract_text": contract.text,
                    "lowered_to": "missing",
                    "expression": "",
                    "reason": "No coverage item was reported for this NLContract.",
                }
            )
            continue
        if item.lowered_to == "omitted":
            omitted += 1
        else:
            lowered += 1
        items.append(
            {
                "contract_id": contract.id,
                "contract_type": contract.type,
                "contract_text": contract.text,
                "lowered_to": item.lowered_to,
                "expression": item.expression,
                "reason": item.reason,
            }
        )

    total = len(contracts)
    return {
        "total": total,
        "lowered": lowered,
        "omitted": omitted,
        "missing": missing,
        "lowering_ratio": round(lowered / total, 4) if total else None,
        "items": items,
    }


def contract_probe_evidence_score(result: Any) -> dict[str, Any]:
    probe_stage = result.contract_probe
    if probe_stage is None or probe_stage.output is None or probe_stage.probe_report is None:
        return {
            "score": None,
            "total": 0,
            "passed": 0,
            "pass_ratio": None,
            "source": None,
            "by_type": {},
            "covered_requirement_ids": [],
            "covered_requirement_count": 0,
            "probes": [],
            "case_counts": {},
        }

    probes = probe_stage.output.probes
    report = probe_stage.probe_report
    report_cases = getattr(report, "cases", None) or []
    cases_by_index: dict[int, dict[str, Any]] = {}
    for case in report_cases:
        dumped = _dump(case)
        index = dumped.get("index")
        if isinstance(index, int):
            cases_by_index[index] = dumped
        elif isinstance(index, str) and index.isdigit():
            cases_by_index[int(index)] = dumped

    failed_indexes: set[int] = set()
    if cases_by_index:
        for index, case in cases_by_index.items():
            if case.get("passed") is not True:
                failed_indexes.add(index)
    else:
        for failure in report.failures:
            index = failure.get("index") if isinstance(failure, dict) else getattr(failure, "index", None)
            if isinstance(index, int):
                failed_indexes.add(index)
            elif isinstance(index, str) and index.isdigit():
                failed_indexes.add(int(index))

    weights = {"oracle": 1.0, "violation": 1.0, "metamorphic": 0.8, "predicate": 0.6}
    by_type: dict[str, dict[str, Any]] = {}
    by_priority: dict[str, dict[str, Any]] = {}
    covered: set[str] = set()
    weighted_total = 0.0
    weighted_passed = 0.0
    probe_rows = []

    for index, probe in enumerate(probes):
        ptype = str(probe.type)
        case = cases_by_index.get(index, {})
        passed = bool(case.get("passed")) if case else index not in failed_indexes
        priority = str(getattr(probe, "priority", "low") or "low").lower()
        weight = weights.get(ptype, 0.5)
        by_type.setdefault(ptype, {"total": 0, "passed": 0, "weight": weight, "pass_rate": None})
        by_priority.setdefault(priority, {"total": 0, "passed": 0, "pass_rate": None})
        by_type[ptype]["total"] += 1
        by_priority[priority]["total"] += 1
        if passed:
            by_type[ptype]["passed"] += 1
            by_priority[priority]["passed"] += 1
        covered.update(getattr(probe, "requirement_ids", []) or [])
        weighted_total += weight
        if passed:
            weighted_passed += weight
        row = _dump(probe)
        row["index"] = index
        row["passed"] = passed


        for key in (
            "actual",
            "stderr",
            "returncode",
            "timed_out",
            "timeout_sec",
            "elapsed_sec",
            "failure_kind",
            "probe_failure_kind",
            "error_type",
            "paired_actual",
            "paired_stderr",
            "paired_returncode",
            "paired_timed_out",
            "paired_failure_kind",
            "paired_error_type",
            "covered_requirement_ids",
            "uncovered_requirement_ids",
            "violation_supported_by_lowered_contract",
        ):
            if key in case:
                row[key] = case.get(key)
        probe_rows.append(row)

    for item in by_type.values():
        item["pass_rate"] = item["passed"] / item["total"] if item["total"] else None
    for item in by_priority.values():
        item["pass_rate"] = item["passed"] / item["total"] if item["total"] else None

    return {
        "score": round(weighted_passed / weighted_total, 4) if weighted_total else None,
        "total": report.total,
        "passed": report.passed,
        "pass_ratio": report.pass_ratio,
        "source": getattr(probe_stage, "source", "generated"),
        "by_type": by_type,
        "by_priority": by_priority,
        "covered_requirement_ids": sorted(covered),
        "covered_requirement_count": len(covered),
        "probes": probe_rows,
        "case_counts": {
            "contract_violations": getattr(report, "contract_violations", 0),
            "invalid_violation_probes": getattr(report, "invalid_violation_probes", 0),
            "runtime_errors": getattr(report, "runtime_errors", 0),
            "timeouts": getattr(report, "timeouts", 0),
            "wrong_outputs": getattr(report, "wrong_outputs", 0),
            "predicate_failures": getattr(report, "predicate_failures", 0),
            "relation_failures": getattr(report, "relation_failures", 0),
            "nonzero_exits": getattr(report, "nonzero_exits", 0),
            "supported_violation_probes": getattr(report, "supported_violation_probes", 0),
            "unsupported_violation_probes": getattr(report, "unsupported_violation_probes", 0),
            "uncovered_violation_requirement_ids": getattr(report, "uncovered_violation_requirement_ids", []),
        },
    }


def _public_witness_ratio(merged_stage: Any) -> float | None:
    if merged_stage is None:
        return None
    report = getattr(merged_stage, "witness_report", None)
    if report is not None and getattr(report, "total", 0):
        return getattr(report, "pass_ratio", None)
    return _ratio(getattr(merged_stage, "execution_witness_passed", None)) if "_ratio" in globals() else _bool_score(getattr(merged_stage, "execution_witness_passed", None))


def _priority_probe_ratio(report: Any, probes: list[Any], priority: str) -> float | None:
    if report is None or not probes:
        return None
    cases = getattr(report, "cases", None) or []
    cases_by_index: dict[int, dict[str, Any]] = {}
    for case in cases:
        dumped = _dump(case)
        index = dumped.get("index")
        if isinstance(index, int):
            cases_by_index[index] = dumped
        elif isinstance(index, str) and index.isdigit():
            cases_by_index[int(index)] = dumped

    selected = []
    for index, probe in enumerate(probes):
        probe_priority = str(getattr(probe, "priority", "low") or "low").lower()
        if probe_priority != priority:
            continue
        case = cases_by_index.get(index)
        if case is not None:
            selected.append(bool(case.get("passed")))
        else:
            selected.append(index not in {f.get("index") for f in getattr(report, "failures", []) if isinstance(f, dict)})
    if not selected:
        return None
    return sum(1.0 for passed in selected if passed) / len(selected)


def behavioral_contract_score(result: Any) -> dict[str, Any]:
    """Behavioral evidence score from public witnesses and self-probes only.

    This is intentionally not the same as pipeline_passed. Static validation,
    lowering coverage, and stage-gate success remain diagnostics elsewhere; they
    must not inflate or deflate behavioral evidence. Public witnesses receive the
    largest weight because they are dataset-provided examples. Self-probes are
    secondary evidence and are split into high/low priority groups.
    """
    merged_contract = result.merged_contract
    contract_probe = result.contract_probe
    plan = getattr(contract_probe, "output", None) if contract_probe else None
    probes = list(getattr(plan, "probes", []) or [])
    report = getattr(contract_probe, "contract_probe_report", None) if contract_probe else None
    report = report or (getattr(contract_probe, "probe_report", None) if contract_probe else None)

    components = {
        "merged_contract_public_witness": {
            "weight": 0.70,
            "value": _public_witness_ratio(merged_contract),
        },
        "high_priority_probe_satisfaction": {
            "weight": 0.20,
            "value": _priority_probe_ratio(report, probes, "high"),
        },
        "low_priority_probe_satisfaction": {
            "weight": 0.10,
            "value": _priority_probe_ratio(report, probes, "low"),
        },
    }
    usable = {name: item for name, item in components.items() if item["value"] is not None}
    available_weight = sum(item["weight"] for item in usable.values())
    if available_weight == 0:
        return {"score": None, "available_weight": 0.0, "components": components}
    score = sum(item["weight"] * item["value"] for item in usable.values()) / available_weight
    return {
        "score": round(score, 4),
        "available_weight": round(available_weight, 4),
        "components": components,
        "note": "Behavioral score uses only merged public witnesses plus high/low priority self-probes; pipeline_passed is a separate completion gate; behavioral pass ratios do not hard-fail it.",
    }

def stage_timings(result: Any) -> dict[str, Any]:
    stages = {
        "specification": result.specification,
        "interface": result.interface,
        "code_generation": result.code,
        "executable_contract": result.executable_contract,
        "merge_contract": result.merged_contract,
        "contract_probe": result.contract_probe,
    }
    values = {name: float(getattr(stage, "elapsed_sec", 0.0) or 0.0) for name, stage in stages.items()}
    values["total_stage_elapsed_sec"] = round(sum(values.values()), 3)
    return values


def _stage_record(stage: Any) -> dict[str, Any]:
    if stage is None:
        return {"ok": False, "error": "not run", "attempts": 0, "elapsed_sec": 0.0}
    return {
        "ok": getattr(stage, "output", None) is not None,
        "error": getattr(stage, "error", None),
        "attempts": getattr(stage, "attempts", 0),
        "elapsed_sec": getattr(stage, "elapsed_sec", 0.0),
        "llm_metrics": getattr(stage, "llm_metrics", {}),
        "refinements": [_dump(item) for item in getattr(stage, "refinements", [])],
        "static_validation_passed": getattr(stage, "static_validation_passed", None),
        "execution_witness_passed": getattr(stage, "execution_witness_passed", None),
    }


def _aggregate_result_llm_metrics(result: Any) -> dict[str, Any]:
    stages = [result.specification, result.interface, result.code, result.executable_contract, result.contract_probe]
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0, "calls": 0}
    saw = False
    call_count = 0
    latency = 0.0
    by_stage: dict[str, Any] = {}
    for name, stage in zip(("specification", "interface", "code_generation", "executable_contract", "contract_probe"), stages):
        metrics = getattr(stage, "llm_metrics", {}) or {}
        by_stage[name] = metrics
        call_count += int(metrics.get("call_count") or 0)
        latency += float(metrics.get("latency_seconds") or 0.0)
        usage = metrics.get("token_usage")
        if isinstance(usage, Mapping):
            saw = True
            for key in totals:
                totals[key] += int(usage.get(key) or 0)
    return {
        "call_count": call_count,
        "latency_seconds": round(latency, 3),
        "token_usage": totals if saw else None,
        "by_stage": by_stage,
    }


def result_to_record(result: Any) -> dict[str, Any]:
    spec = result.specification.output if result.specification else None
    interface = result.interface.output if result.interface else None
    code = result.code.output if result.code else None
    executable_contract = result.executable_contract.output if result.executable_contract else None
    merged_contract = result.merged_contract.output if result.merged_contract else None
    contract_probe = result.contract_probe.output if result.contract_probe else None
    code_witness_report = result.code.witness_report if result.code else None
    merged_witness_report = result.merged_contract.witness_report if result.merged_contract else None
    probe_report = result.contract_probe.probe_report if result.contract_probe else None
    code_probe_report = (
        result.contract_probe.code_probe_report if result.contract_probe else None
    )
    contract_probe_report = (
        result.contract_probe.contract_probe_report if result.contract_probe else None
    ) or probe_report

    return {
        "task_id": result.task_id,
        "pipeline_passed": pipeline_passed(result),
        "dbc_gates": dbc_gate_report(result),
        "dbc_behavioral_score": behavioral_contract_score(result),
        "probe_availability": probe_availability(result),
        "dbc_probe_evidence": contract_probe_evidence_score(result),
        "contract_lowering_coverage": contract_lowering_coverage(result),
        "timings": stage_timings(result),
        "llm_metrics": _aggregate_result_llm_metrics(result),
        "probe_eval_comparison": _dump(result.contract_probe.probe_comparison) if result.contract_probe else None,
        "stages": {
            "specification": _stage_record(result.specification),
            "interface": _stage_record(result.interface),
            "code_generation": _stage_record(result.code),
            "executable_contract": _stage_record(result.executable_contract),
            "merge_contract": _stage_record(result.merged_contract),
            "contract_probe": _stage_record(result.contract_probe),
        },
        "outputs": {
            "specification": _dump(spec),
            "interface": _dump(interface),
            "code": _dump(code),
            "executable_contract": _dump(executable_contract),
            "merged_contract": _dump(merged_contract),
            "contract_probe": _dump(contract_probe),
        },
        "reports": {
            "code_execution_witness": _dump(code_witness_report),
            "merged_contract_execution_witness": _dump(merged_witness_report),
            "code_probe": _dump(code_probe_report),
            "contract_probe": _dump(contract_probe_report),
            "contract_probe_legacy": _dump(probe_report),
        },
        "raw_outputs": {
            "specification": result.specification.raw_output if result.specification else "",
            "interface": result.interface.raw_output if result.interface else "",
            "code": result.code.raw_output if result.code else "",
            "executable_contract": result.executable_contract.raw_output if result.executable_contract else "",
            "contract_probe": result.contract_probe.raw_output if result.contract_probe else "",
        },
    }


def _write_artifacts(run_dir: Path, display_id: str, record: dict[str, Any]) -> dict[str, str]:
    artifact_dir = run_dir / "artifacts" / sanitize_path_part(display_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    outputs = record.get("outputs", {})
    raw_outputs = record.get("raw_outputs", {})
    artifacts: dict[str, str] = {}

    def write_text(name: str, text: str) -> None:
        path = artifact_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        artifacts[name] = str(path)

    def write_obj(name: str, obj: Any) -> None:
        path = artifact_dir / name
        write_json(path, obj)
        artifacts[name] = str(path)

    write_obj("record.json", record)
    problem = record.get("problem") or {}
    if problem.get("statement") or problem.get("prompt"):
        write_text("problem_statement.txt", str(problem.get("statement") or problem.get("prompt") or ""))
    if problem.get("public_test_cases") is not None:
        write_obj("public_test_cases.json", problem.get("public_test_cases"))
    if outputs.get("specification") is not None:
        write_obj("specification.json", outputs["specification"])
    if outputs.get("interface") is not None:
        write_text("interface.py", outputs["interface"].get("skeleton", ""))
    if outputs.get("code") is not None:
        write_text("generated.py", outputs["code"].get("code", ""))
    if outputs.get("executable_contract") is not None:
        contract = outputs["executable_contract"]
        write_text("executable_contract.py", "\n\n".join([contract.get("imports", ""), contract.get("decorators", "")]).strip() + "\n")
        write_obj("contract_lowering_coverage.json", contract.get("coverage", []))
    if outputs.get("merged_contract") is not None:
        write_text("contracted.py", outputs["merged_contract"].get("code", ""))
    if outputs.get("contract_probe") is not None:
        write_obj("contract_probes.json", outputs["contract_probe"])
    reports = record.get("reports", {})
    if reports.get("code_probe") is not None:
        write_obj("code_probe_report.json", reports["code_probe"])
    if reports.get("contract_probe") is not None:
        write_obj("contract_probe_report.json", reports["contract_probe"])
    if record.get("probe_availability") is not None:
        write_obj("probe_availability.json", record["probe_availability"])
    if record.get("probe_eval_comparison") is not None:
        write_obj("probe_comparison.json", record["probe_eval_comparison"])
    if record.get("code_private_eval") is not None:
        write_obj("code_private_eval.json", record["code_private_eval"])
    if record.get("contract_private_eval") is not None:
        write_obj("contract_private_eval.json", record["contract_private_eval"])
    if record.get("private_eval") is not None:
        write_obj("private_eval.json", record["private_eval"])
    if raw_outputs:
        for stage, raw in raw_outputs.items():
            if raw:
                write_text(f"raw_outputs/{stage}.txt", str(raw))
    return artifacts


def persist_result(
    result: Any,
    *,
    run_dir: str | Path | None = None,
    run_config: Mapping[str, Any] | None = None,
    task: Any | None = None,
    started_at: str | None = None,
    elapsed_seconds: float | None = None,
    private_eval: Mapping[str, Any] | None = None,
    code_private_eval: Mapping[str, Any] | None = None,
    contract_private_eval: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record = result_to_record(result)
    record["status"] = "completed"
    record["started_at"] = started_at
    record["finished_at"] = utc_now()
    if elapsed_seconds is not None:
        record["elapsed_seconds"] = round(float(elapsed_seconds), 3)
    if code_private_eval is not None:
        record["code_private_eval"] = dict(code_private_eval)
    if contract_private_eval is not None:
        record["contract_private_eval"] = dict(contract_private_eval)
    if private_eval is not None:
        record["private_eval"] = dict(private_eval)
    elif contract_private_eval is not None:
        record["private_eval"] = dict(contract_private_eval)
    if record.get("code_private_eval") is not None or record.get("contract_private_eval") is not None:
        record["private_eval_comparison"] = compare_private_evaluations(record)

    if task is not None:
        record.update(
            {
                "task_key": task.task_key,
                "display_id": task.display_id,
                "dataset": task.dataset,
                "row_index": task.row_index,
                "problem_id": task.problem_id,
                "source_uid": task.source_uid,
                "entry_point": task.entry_point,
                "metadata": task.metadata,
                "problem": {
                    "statement": task.prompt,
                    "prompt": task.prompt,
                    "entry_point": task.entry_point,
                    "public_test_cases": _dump(task.witnesses),
                    "metadata": dict(task.metadata),
                },
            }
        )
    else:
        record.setdefault("task_key", record["task_id"])
        record.setdefault("display_id", record["task_id"])

    if run_config is not None:
        record["run"] = {
            "run_name": run_config.get("run_name"),
            "provider": run_config.get("provider"),
            "model": run_config.get("model"),
            "temperature": run_config.get("temperature"),
            "max_tokens": run_config.get("max_tokens"),
        }

    if run_dir is not None:
        run_dir = Path(run_dir)
        record["artifacts"] = _write_artifacts(run_dir, record["display_id"], record)

        write_json(Path(record["artifacts"]["record.json"]), record)
        append_jsonl(run_dir / "results.jsonl", record)
    return record


def persist_exception(
    *,
    run_dir: str | Path,
    run_config: Mapping[str, Any],
    task: Any,
    started_at: str,
    elapsed_seconds: float,
    error: BaseException,
    traceback_text: str | None = None,
    private_eval: Mapping[str, Any] | None = None,
    code_private_eval: Mapping[str, Any] | None = None,
    contract_private_eval: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "status": "exception",
        "task_id": task.task_id,
        "task_key": task.task_key,
        "dataset": task.dataset,
        "row_index": task.row_index,
        "entry_point": task.entry_point,
        "display_id": task.display_id,
        "metadata": task.metadata,
        "pipeline_passed": False,
        "probe_availability": {
            "status": "missing",
            "missing": True,
            "missing_reasons": ["task raised an exception before self-probes could be recorded"],
            "plan_exists": False,
            "plan_total": 0,
            "code_probe_report_exists": False,
            "contract_probe_report_exists": False,
            "comparison_exists": False,
            "source": None,
            "stage_error": None,
        },
        "problem_id": task.problem_id,
        "source_uid": task.source_uid,
        "started_at": started_at,
        "finished_at": utc_now(),
        "elapsed_seconds": round(float(elapsed_seconds), 3),
        "problem": {
            "statement": task.prompt,
            "prompt": task.prompt,
            "entry_point": task.entry_point,
            "public_test_cases": _dump(task.witnesses),
            "metadata": dict(task.metadata),
        },
        "error": f"{type(error).__name__}: {error}",
        "traceback": traceback_text or traceback.format_exc(),
        "run": {
            "run_name": run_config.get("run_name"),
            "provider": run_config.get("provider"),
            "model": run_config.get("model"),
            "temperature": run_config.get("temperature"),
            "max_tokens": run_config.get("max_tokens"),
        },
    }
    if code_private_eval is not None:
        record["code_private_eval"] = dict(code_private_eval)
    if contract_private_eval is not None:
        record["contract_private_eval"] = dict(contract_private_eval)
    if private_eval is not None:
        record["private_eval"] = dict(private_eval)
    elif contract_private_eval is not None:
        record["private_eval"] = dict(contract_private_eval)
    if record.get("code_private_eval") is not None or record.get("contract_private_eval") is not None:
        record["private_eval_comparison"] = compare_private_evaluations(record)

    artifact_dir = Path(run_dir) / "artifacts" / sanitize_path_part(task.display_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / "record.json"
    write_json(path, record)
    availability_path = artifact_dir / "probe_availability.json"
    write_json(availability_path, record.get("probe_availability"))
    record["artifacts"] = {"record.json": str(path), "probe_availability.json": str(availability_path)}
    write_json(path, record)
    append_jsonl(Path(run_dir) / "results.jsonl", record)
    return record


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def latest_records(results_path: str | Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(results_path):
        key = row.get("task_key") or row.get("task_id")
        if key:
            latest[str(key)] = row
    return latest


def completed_task_keys(results_path: str | Path, retry_failed: bool = False) -> set[str]:
    completed: set[str] = set()
    for key, record in latest_records(results_path).items():
        if retry_failed and (record.get("status") != "completed" or not record.get("pipeline_passed")):
            continue
        completed.add(key)
    return completed


def _avg(values: Iterable[Any]) -> float | None:
    numeric = [float(v) for v in values if isinstance(v, (int, float))]
    return round(mean(numeric), 4) if numeric else None


def _sum_token_usage(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0, "calls": 0}
    for record in records:
        usage = (((record.get("llm_metrics") or {}).get("token_usage")) or {})
        if isinstance(usage, Mapping):
            for key in totals:
                totals[key] += int(usage.get(key) or 0)
    return totals


def _failure_stage(record: Mapping[str, Any]) -> str:
    if record.get("status") != "completed":
        return "exception"
    stages = record.get("stages") or {}
    for name in ("specification", "interface", "code_generation", "executable_contract", "merge_contract", "contract_probe"):
        stage = stages.get(name) or {}
        if not stage.get("ok"):
            return name
        if stage.get("static_validation_passed") is False:
            return name
        if stage.get("execution_witness_passed") is False:
            return name
    if not record.get("pipeline_passed"):
        return "pipeline_gate"
    return "passed"

def _summarize_probe_comparisons(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    comparisons = [
        record.get("probe_eval_comparison") or {}
        for record in records
        if record.get("probe_eval_comparison")
    ]
    counts: dict[str, int] = {}
    for comparison in comparisons:
        for name, value in (comparison.get("counts") or {}).items():
            counts[str(name)] = counts.get(str(name), 0) + int(value or 0)
    return {
        "evaluated_tasks": len(comparisons),
        "total_probes": sum(int(item.get("total") or 0) for item in comparisons),
        "code_passed": sum(int(item.get("code_passed") or 0) for item in comparisons),
        "contract_passed": sum(
            int(item.get("contract_passed") or 0) for item in comparisons
        ),
        "counts": counts,
    }


def _summarize_probe_availability(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    missing_items: list[dict[str, Any]] = []
    for record in records:
        availability = record.get("probe_availability") or {}
        status = str(availability.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        if availability.get("missing"):
            item = {
                "task_key": record.get("task_key") or record.get("task_id"),
                "display_id": record.get("display_id"),
                "dataset": record.get("dataset"),
                "row_index": record.get("row_index"),
                "problem_id": record.get("problem_id"),
                "reasons": list(availability.get("missing_reasons") or []),
            }
            missing_items.append(item)
            for reason in item["reasons"]:
                reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1
    return {
        "status_counts": status_counts,
        "available_tasks": status_counts.get("available", 0),
        "missing_tasks": sum(1 for r in records if (r.get("probe_availability") or {}).get("missing")),
        "reason_counts": reason_counts,
        "missing_records": missing_items[:200],
        "missing_records_truncated": max(0, len(missing_items) - 200),
    }


def summarize_by_dataset(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        dataset = str(record.get("dataset") or "unknown")
        grouped.setdefault(dataset, []).append(record)

    summaries: dict[str, dict[str, Any]] = {}
    for dataset, dataset_records in sorted(grouped.items()):
        completed = [
            record
            for record in dataset_records
            if record.get("status") == "completed"
        ]
        passed = sum(
            1 for record in completed if record.get("pipeline_passed") is True
        )
        summaries[dataset] = {
            "recorded_tasks": len(dataset_records),
            "completed_tasks": len(completed),
            "failed_tasks": len(dataset_records) - len(completed),
            "pipeline_passed": passed,
            "pipeline_pass_rate": (
                round(passed / len(completed), 4) if completed else 0.0
            ),
            "code_private_eval": summarize_private_evaluations(
                dataset_records,
                key="code_private_eval",
            ),
            "contract_private_eval": summarize_private_evaluations(
                dataset_records,
                key="contract_private_eval",
            ),
        }
    return summaries


def write_summary(
    run_dir: str | Path,
    *,
    run_config: Mapping[str, Any] | None = None,
    selected_tasks: int | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    results_path = run_dir / "results.jsonl"
    records = list(latest_records(results_path).values())
    completed = [r for r in records if r.get("status") == "completed"]
    pipeline_passed_records = [r for r in completed if r.get("pipeline_passed") is True]
    code_private_summary = summarize_private_evaluations(records, key="code_private_eval")
    contract_private_summary = summarize_private_evaluations(records, key="contract_private_eval")
    private_summary = contract_private_summary
    probe_comparison_summary = _summarize_probe_comparisons(completed)
    probe_availability_summary = _summarize_probe_availability(records)
    dataset_summaries = summarize_by_dataset(records)

    stage_elapsed: dict[str, list[float]] = {}
    for record in completed:
        for stage, info in (record.get("stages") or {}).items():
            stage_elapsed.setdefault(stage, []).append(float((info or {}).get("elapsed_sec") or 0.0))

    failures_by_stage: dict[str, int] = {}
    for record in records:
        stage = _failure_stage(record)
        failures_by_stage[stage] = failures_by_stage.get(stage, 0) + 1

    summary = {
        "schema_version": "2026-06-contract-summary-v3",
        "run_config": dict(run_config or {}),
        "selected_tasks": selected_tasks,
        "recorded_tasks": len(records),
        "completed_tasks": len(completed),
        "failed_tasks": len(records) - len(completed),
        "pipeline_passed": len(pipeline_passed_records),
        "pipeline_failed": len(completed) - len(pipeline_passed_records),
        "pipeline_pass_rate": round(len(pipeline_passed_records) / len(completed), 4) if completed else 0.0,
        "avg_dbc_behavioral_score": _avg((r.get("dbc_behavioral_score") or {}).get("score") for r in completed),
        "avg_probe_evidence_score": _avg((r.get("dbc_probe_evidence") or {}).get("score") for r in completed),
        "avg_contract_lowering_ratio": _avg((r.get("contract_lowering_coverage") or {}).get("lowering_ratio") for r in completed),
        "code_private_eval": code_private_summary,
        "contract_private_eval": contract_private_summary,
        "private_eval": private_summary,
        "datasets": dataset_summaries,
        "probe_availability": probe_availability_summary,
        "probe_eval_comparison": probe_comparison_summary,
        "private_eval_comparison": {
            "pre_correctness": code_private_summary.get("apr_micro"),
            "post_correctness": contract_private_summary.get("apr_micro"),
            "pre_completeness": code_private_summary.get("completeness_micro"),
            "post_completeness": contract_private_summary.get("completeness_micro"),
            "contract_correctness_delta": (
                round((contract_private_summary.get("apr_micro") or 0.0) - (code_private_summary.get("apr_micro") or 0.0), 4)
                if code_private_summary.get("apr_micro") is not None and contract_private_summary.get("apr_micro") is not None
                else None
            ),
            "contract_completeness_delta": (
                round((contract_private_summary.get("completeness_micro") or 0.0) - (code_private_summary.get("completeness_micro") or 0.0), 4)
                if code_private_summary.get("completeness_micro") is not None and contract_private_summary.get("completeness_micro") is not None
                else None
            ),
        },
        "timing": {
            "total_elapsed_seconds": round(sum(float(r.get("elapsed_seconds") or 0.0) for r in records), 3),
            "avg_elapsed_seconds": _avg(r.get("elapsed_seconds") for r in records),
            "avg_stage_elapsed_seconds": {stage: _avg(values) for stage, values in stage_elapsed.items()},
        },
        "llm_metrics": {
            "token_usage": _sum_token_usage(completed),
        },
        "failures_by_stage": failures_by_stage,
        "results_path": str(results_path),
        "summary_path": str(run_dir / "summary.json"),
        "updated_at": utc_now(),
    }
    write_json(run_dir / "summary.json", summary)
    return summary
