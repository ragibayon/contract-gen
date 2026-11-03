from __future__ import annotations

from dataclasses import asdict
from typing import Any

from contract.agent.contract import ContractAgentResult
from contract.eval.execution import ExecutionWitnessReport
from contract.utils.logger import logger
from contract.utils.results import (
    behavioral_contract_score,
    contract_lowering_coverage,
    contract_probe_evidence_score,
    pipeline_passed,
)


def show_witness_report(label: str, report: ExecutionWitnessReport | None) -> None:
    if report is None:
        return
    message = f"{label}: {report.passed}/{report.total} passed"
    if report.pass_ratio == 1.0:
        logger.success(message)
    else:
        logger.error(message)
    if report.failures:
        logger.records(
            f"{label} failures",
            [asdict(failure) for failure in report.failures],
            {
                "index": "Case",
                "input": "Input",
                "expected": "Expected",
                "actual": "Actual",
                "stderr": "Stderr",
            },
        )


def _attempts(stage: Any) -> int | str:
    if stage is None:
        return "not run"
    if getattr(stage, "attempts", 0) == 0 and getattr(stage, "output", None) is not None:
        return "cached"
    return getattr(stage, "attempts", 0)


def _validation(stage: Any) -> str:
    if stage is None:
        return "not run"
    value = getattr(stage, "static_validation_passed", None)
    if value is None:
        return "not applicable"
    return "passed" if value else "failed"


def _elapsed(stage: Any) -> str:
    if stage is None:
        return "not run"
    return f"{float(getattr(stage, 'elapsed_sec', 0.0) or 0.0):.3f}s"


def show_probe_report(result: ContractAgentResult) -> None:
    probe_stage = result.contract_probe
    if probe_stage is None:
        return
    plan = probe_stage.output
    report = probe_stage.probe_report
    if plan is None or report is None:
        logger.error(f"Contract probes unavailable: {probe_stage.error or 'no output'}")
        return

    evidence = contract_probe_evidence_score(result)
    logger.kv(
        "Contract probe summary",
        {
            "Source": evidence.get("source") or getattr(probe_stage, "source", "generated"),
            "Probes": f"{report.passed}/{report.total} passed",
            "Pass ratio": f"{report.pass_ratio:.3f}",
            "Evidence score": evidence.get("score"),
            "Covered requirement IDs": ", ".join(evidence.get("covered_requirement_ids", [])) or "none",
        },
    )
    if plan.probes:
        rows = []
        failed = {failure.get("index") for failure in report.failures if isinstance(failure, dict)}
        for index, probe in enumerate(plan.probes):
            item = probe.model_dump()
            item["index"] = index
            item["passed"] = index not in failed
            rows.append(item)
        logger.records(
            "Generated contract probes",
            rows,
            {
                "index": "#",
                "type": "Type",
                "input": "Input",
                "expected": "Expected",
                "paired_input": "Paired input",
                "relation": "Relation",
                "requirement_ids": "NLC IDs",
                "passed": "Passed",
                "reason": "Reason",
            },
        )
    if report.failures:
        logger.records("Contract probe failures", report.failures, {})


def show_contract_lowering(result: ContractAgentResult) -> None:
    coverage = contract_lowering_coverage(result)
    if not coverage.get("items"):
        return
    logger.kv(
        "Requirement-to-executable-contract coverage",
        {
            "Lowered": f"{coverage['lowered']}/{coverage['total']}",
            "Omitted": coverage["omitted"],
            "Missing": coverage["missing"],
            "Lowering ratio": coverage["lowering_ratio"],
        },
    )
    logger.records(
        "Contract lowering map",
        coverage["items"],
        {
            "contract_id": "ID",
            "contract_type": "Type",
            "lowered_to": "Lowered to",
            "expression": "Expression",
            "reason": "Reason",
        },
    )


def show_pipeline_result(result: ContractAgentResult) -> None:
    logger.rule("Pipeline artifacts")
    specification = result.specification.output
    if specification is not None:
        logger.text_panel(specification.summary, "Specification summary")
        if specification.contracts:
            logger.records(
                "Natural-language contracts",
                [contract.model_dump() for contract in specification.contracts],
                {
                    "id": "ID",
                    "type": "Type",
                    "text": "Contract",
                    "evidence": "Evidence",
                },
            )
        else:
            logger.muted("No natural-language contracts were produced.")
    else:
        logger.error(f"Specification unavailable: {result.specification.error or 'no output'}")

    interface = result.interface.output if result.interface else None
    if interface is not None:
        logger.code(interface.skeleton, title="Program interface", line_numbers=True)
    elif result.interface is not None:
        logger.error(f"Interface unavailable: {result.interface.error or 'no output'}")

    code = result.code.output if result.code else None
    if code is not None:
        logger.code(code.code, title="Generated code", line_numbers=True)
    elif result.code is not None:
        logger.error(f"Generated code unavailable: {result.code.error or 'no output'}")

    executable_contract = result.executable_contract.output if result.executable_contract else None
    if executable_contract is not None:
        logger.code(
            "\n\n".join([executable_contract.imports, executable_contract.decorators]),
            title="Executable contract",
            line_numbers=True,
        )
    elif result.executable_contract is not None:
        logger.error(f"Executable contract unavailable: {result.executable_contract.error or 'no output'}")

    merged_contract = result.merged_contract.output if result.merged_contract else None
    if merged_contract is not None:
        logger.code(merged_contract.code, title="Merged contract code", line_numbers=True)
    elif result.merged_contract is not None:
        logger.error(f"Merged contract unavailable: {result.merged_contract.error or 'no output'}")

    show_contract_lowering(result)
    show_probe_report(result)

    logger.rule("Validation report")
    logger.kv(
        "Stage validation and time",
        {
            "Specification attempts": _attempts(result.specification),
            "Specification time": _elapsed(result.specification),
            "Interface attempts": _attempts(result.interface),
            "Interface time": _elapsed(result.interface),
            "Code attempts": _attempts(result.code),
            "Code static validation": _validation(result.code),
            "Code time": _elapsed(result.code),
            "Executable contract attempts": _attempts(result.executable_contract),
            "Executable contract static validation": _validation(result.executable_contract),
            "Executable contract time": _elapsed(result.executable_contract),
            "Merged contract static validation": _validation(result.merged_contract),
            "Merged contract time": _elapsed(result.merged_contract),
            "Contract probe attempts": _attempts(result.contract_probe),
            "Contract probe time": _elapsed(result.contract_probe),
            "Pipeline passed": pipeline_passed(result),
            "DBC score": behavioral_contract_score(result).get("score"),
        },
    )
    show_witness_report("Generated code witnesses", result.code.witness_report if result.code else None)
    show_witness_report("Merged contract witnesses", result.merged_contract.witness_report if result.merged_contract else None)
