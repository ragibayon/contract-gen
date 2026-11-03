from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import ast

from contract.agent.schema.code import CodeArtifact
from contract.agent.schema.contract_probe import (
    ContractProbe,
    ContractProbePlan,
    ContractProbeReport,
    RawContractProbePlan,
)
from contract.agent.schema.interface import ProgramInterface
from contract.agent.schema.merged_contract import MergedContractArtifact
from contract.agent.schema.specification import Specification
from contract.eval.probes import compare_probe_reports, evaluate_contract_probes
from contract.llm.callbacks.metrics import (
    LLMRunMetrics,
    invoke_with_metrics,
    merge_llm_metrics,
)
from contract.llm.client import LLMClient
from contract.utils.json_extract import extract_json_object
from contract.utils.logger import logger
from contract.utils.prompt_renderer import render_prompt


@dataclass
class ContractProbeStageResult:
    name: str
    output: ContractProbePlan | None = None
    raw_output: str = ""
    error: str | None = None
    attempts: int = 0
    elapsed_sec: float = 0.0
    probe_report: ContractProbeReport | None = None
    code_probe_report: ContractProbeReport | None = None
    contract_probe_report: ContractProbeReport | None = None
    probe_comparison: dict[str, Any] | None = None
    source: str = "generated"
    llm_metrics: dict[str, Any] = field(default_factory=dict)
    refinements: list[dict[str, Any]] = field(default_factory=list)


OutputStructure = Literal["raw", "structured"]


class ContractProbeStage:
    def __init__(
        self,
        llm: LLMClient,
        max_attempts: int = 2,
        output_structure: OutputStructure = "raw",
        prompt_variant: str = "contract",
    ):
        self.llm = llm
        self.max_attempts = max(1, max_attempts)
        self.output_structure = self._normalize_output_structure(output_structure)
        self.prompt_variant = prompt_variant

    def run(
        self,
        task_prompt: str,
        specification: Specification,
        interface: ProgramInterface,
        generated_code: CodeArtifact,
        merged_contract: MergedContractArtifact,
        public_witnesses: dict[str, list[Any]] | None = None,
    ) -> ContractProbeStageResult:
        started = time.time()
        last_error: str | None = None
        last_raw = ""
        feedback = ""
        metrics_items: list[dict[str, Any]] = []
        refinements: list[dict[str, Any]] = []
        last_raw_probe_count = 0
        last_valid_probe_count = 0
        last_dropped_probe_count = 0
        last_drop_reasons: dict[str, int] = {}

        public_plan = self._public_oracle_plan(public_witnesses)
        if public_plan.probes:
            logger.info(
                "contract_probe: public examples are available and remain the "
                f"highest-weight behavioral evidence ({len(public_plan.probes)} examples); "
                "still generating independent LLM contract probes."
            )
        else:
            logger.info(
                "contract_probe: no structured public examples available; "
                "generating independent LLM contract probes."
            )

        contracts_json = json.dumps(
            [contract.model_dump() for contract in specification.contracts],
            indent=2,
            ensure_ascii=False,
        )
        executable_coverage = self._executable_coverage_summary(merged_contract)
        executable_coverage_json = json.dumps(
            executable_coverage,
            indent=2,
            ensure_ascii=False,
        )
        public_witnesses_json = json.dumps(
            public_witnesses or {},
            indent=2,
            ensure_ascii=False,
        )

        interface_shape_json = json.dumps(
            self._compact_interface_shape(interface),
            indent=2,
            ensure_ascii=False,
        )

        logger.info(
            "contract_probe: using "
            f"output_structure={self.output_structure}, prompt_variant={self.prompt_variant}"
        )
        system_prompt = render_prompt(
            "contract_probe/system.j2",
            output_structure=self.output_structure,
            prompt_variant=self.prompt_variant,
        )

        for attempt in range(1, self.max_attempts + 1):
            user_prompt = render_prompt(
                "contract_probe/user.j2",
                task_prompt=task_prompt,
                contracts_json=contracts_json,
                interface_json=interface_shape_json,
                public_witnesses_json=public_witnesses_json,
                executable_coverage_json=executable_coverage_json,
                output_structure=self.output_structure,
                prompt_variant=self.prompt_variant,
            )
            if feedback:
                user_prompt += "\n\n" + render_prompt(
                    "contract_probe/repair.j2",
                    attempt=attempt,
                    max_attempts=self.max_attempts,
                    previous_error=last_error or "Previous probe plan was invalid.",
                    previous_raw_output=last_raw or "<unavailable>",
                    repair_feedback=feedback,
                    output_structure=self.output_structure,
                    prompt_variant=self.prompt_variant,
                )

            try:
                raw, raw_text, metric, parser_name = self._invoke_llm(system_prompt, user_prompt)
                metrics_items.append(metric)
                last_raw = raw_text

                raw_probes = list(raw.probes or [])
                probes, drop_diagnostics = self._normalize_probes_with_diagnostics(raw_probes)
                dropped_count = len(drop_diagnostics["dropped"])
                drop_reasons = drop_diagnostics["drop_reasons"]
                last_raw_probe_count = len(raw_probes)
                last_valid_probe_count = len(probes)
                last_dropped_probe_count = dropped_count
                last_drop_reasons = dict(drop_reasons)

                logger.info(
                    "contract_probe: LLM probe plan attempt "
                    f"{attempt}/{self.max_attempts} | parser={parser_name} | "
                    f"raw={len(raw_probes)} | valid={len(probes)} | "
                    f"dropped={dropped_count} | drop_reasons={drop_reasons}"
                )
                logger.text_panel(
                    last_raw,
                    title=f"contract_probe raw LLM output attempt {attempt}",
                    border_style="blue",
                )

                if not probes:
                    last_error = (
                        "Probe planner returned zero valid probes "
                        f"(raw={len(raw_probes)}, dropped={dropped_count})."
                    )
                    feedback = self._zero_probe_feedback(
                        raw_count=len(raw_probes),
                        drop_reasons=drop_reasons,
                    )
                    refinements.append(
                        {
                            "attempt": attempt,
                            "error": last_error,
                            "raw_output": last_raw,
                            "raw_probe_count": len(raw_probes),
                            "valid_probe_count": 0,
                            "dropped_probe_count": dropped_count,
                            "drop_reasons": drop_reasons,
                            "dropped_probes": drop_diagnostics["dropped"],
                            "repair_feedback": feedback,
                        }
                    )
                    logger.warning(
                        "contract_probe: zero valid LLM probes; retrying with contract-probe repair guidance"
                    )
                    continue

                plan = ContractProbePlan(probes=probes)
                logger.info(
                    "contract_probe: generated valid probe plan; executing once as behavioral evidence "
                    "without repairing based on code/contract outcomes"
                )

                return self._evaluate_plan(
                    plan=plan,
                    generated_code=generated_code,
                    merged_contract=merged_contract,
                    source="generated_spec_probes",
                    started=started,
                    attempts=attempt,
                    raw_output=last_raw,
                    error=None,
                    metrics_items=metrics_items,
                    refinements=refinements,
                )

            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                feedback = self._repair_feedback(last_error)
                refinements.append(
                    {
                        "attempt": attempt,
                        "error": last_error,
                        "raw_output": last_raw,
                        "repair_feedback": feedback,
                    }
                )
                logger.warning(
                    "contract_probe: probe planning attempt failed; retrying with repair guidance "
                    f"({last_error})"
                )

        logger.stage_finished(
            "contract_probe",
            False,
            round(time.time() - started, 3),
            {
                "Error": last_error or "No probes generated.",
                "LLM raw probes": last_raw_probe_count,
                "LLM valid probes": last_valid_probe_count,
                "LLM dropped probes": last_dropped_probe_count,
                "Drop reasons": last_drop_reasons,
            },
        )
        return ContractProbeStageResult(
            name="contract_probe",
            output=None,
            raw_output=last_raw,
            error=last_error or "No probes generated.",
            attempts=self.max_attempts,
            elapsed_sec=round(time.time() - started, 3),
            probe_report=None,
            source="none",
            llm_metrics=merge_llm_metrics(metrics_items),
            refinements=refinements,
        )

    @staticmethod
    def _evaluate_plan(
        *,
        plan: ContractProbePlan,
        generated_code: CodeArtifact,
        merged_contract: MergedContractArtifact,
        source: str,
        started: float,
        attempts: int,
        raw_output: str,
        error: str | None,
        metrics_items: list[dict[str, Any]],
        refinements: list[dict[str, Any]],
    ) -> ContractProbeStageResult:
        executable_ids = ContractProbeStage._lowered_requirement_ids(merged_contract)
        contract_probe_report = evaluate_contract_probes(
            code=merged_contract.code,
            probes=plan.probes,
            executable_requirement_ids=executable_ids,
        )
        code_probe_report = evaluate_contract_probes(
            code=generated_code.code,
            probes=plan.probes,
            executable_requirement_ids=executable_ids,
        )
        probe_comparison = compare_probe_reports(
            probes=plan.probes,
            code_report=code_probe_report,
            contract_report=contract_probe_report,
        )
        ok = contract_probe_report.total > 0

        logger.stage_finished(
            "contract_probe",
            ok,
            round(time.time() - started, 3),
            {
                "Source": source,
                "Probes": f"{contract_probe_report.passed}/{contract_probe_report.total} passed",
                "Pass ratio": f"{contract_probe_report.pass_ratio:.3f}",
                "Invalid violation probes": contract_probe_report.invalid_violation_probes,
                "Runtime errors": contract_probe_report.runtime_errors,
                "Contract violations": contract_probe_report.contract_violations,
                "Supported violation probes": contract_probe_report.supported_violation_probes,
                "Unsupported violation probes": contract_probe_report.unsupported_violation_probes,
            },
        )
        return ContractProbeStageResult(
            name="contract_probe",
            output=plan,
            raw_output=raw_output,
            error=error,
            attempts=attempts,
            elapsed_sec=round(time.time() - started, 3),
            probe_report=contract_probe_report,
            code_probe_report=code_probe_report,
            contract_probe_report=contract_probe_report,
            probe_comparison=probe_comparison,
            source=source,
            llm_metrics=merge_llm_metrics(metrics_items),
            refinements=refinements,
        )

    @staticmethod
    def _lowered_requirement_ids(merged_contract: MergedContractArtifact) -> set[str]:
        return {
            str(item.contract_id)
            for item in (getattr(merged_contract, "coverage", None) or [])
            if getattr(item, "lowered_to", "omitted") != "omitted"
        }

    @staticmethod
    def _executable_coverage_summary(
        merged_contract: MergedContractArtifact,
    ) -> dict[str, Any]:
        coverage = list(getattr(merged_contract, "coverage", None) or [])
        lowered_ids = ContractProbeStage._lowered_requirement_ids(merged_contract)
        omitted_ids = sorted(
            str(item.contract_id)
            for item in coverage
            if getattr(item, "lowered_to", "omitted") == "omitted"
        )
        groups: dict[tuple[str, str], dict[str, Any]] = {}
        for item in coverage:
            if getattr(item, "lowered_to", "omitted") == "omitted":
                continue
            expression = str(getattr(item, "expression", "") or "").strip()
            lowered_to = str(getattr(item, "lowered_to", ""))
            key = (lowered_to, expression)
            group = groups.setdefault(
                key,
                {
                    "lowered_to": lowered_to,
                    "expression": expression,
                    "covers": [],
                    "reasons": [],
                },
            )
            group["covers"].append(str(item.contract_id))
            reason = str(getattr(item, "reason", "") or "").strip()
            if reason and reason not in group["reasons"]:
                group["reasons"].append(reason)

        return {
            "lowered_contract_ids": sorted(lowered_ids),
            "omitted_contract_ids": omitted_ids,
            "merged_decorator_groups": [
                {**group, "covers": sorted(group["covers"])}
                for group in groups.values()
            ],
            "note": (
                "Each merged_decorator_group may cover multiple NLContract IDs. "
                "Violation probes should name the precondition IDs they intend to check; "
                "the evaluator will record covered/uncovered IDs for traceability."
            ),
        }

    @staticmethod
    def _compact_interface_shape(interface: ProgramInterface) -> dict[str, Any]:
        """Return a compact execution-shape view for probe planning.

        The probe planner only needs invocation and stdin/stdout structure.  It
        should not receive the generated implementation, and it also does not
        need the full NLContract docstring embedded in the interface skeleton.
        """
        skeleton = interface.skeleton.strip()
        shape: dict[str, Any] = {
            "execution_mode": "stdin_stdout",
            "solve_signature": "solve(...)",
            "interface_shape": skeleton,
        }

        try:
            tree = ast.parse(skeleton)
            solve_node = next(
                (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "solve"),
                None,
            )
            main_node = next(
                (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"),
                None,
            )
            if solve_node is not None:
                params = [arg.arg for arg in solve_node.args.args]
                shape["solve_parameters"] = params
                shape["solve_signature"] = f"solve({', '.join(params)})"

            compact_nodes: list[ast.stmt] = []
            if solve_node is not None:
                compact_solve = ast.FunctionDef(
                    name="solve",
                    args=solve_node.args,
                    body=[ast.Pass()],
                    decorator_list=[],
                    returns=None,
                    type_comment=None,
                )
                compact_nodes.append(compact_solve)
            if main_node is not None:
                compact_main = ast.FunctionDef(
                    name="main",
                    args=main_node.args,
                    body=ContractProbeStage._strip_docstring(list(main_node.body)),
                    decorator_list=[],
                    returns=None,
                    type_comment=None,
                )
                compact_nodes.append(compact_main)

            if compact_nodes:
                compact_module = ast.Module(body=compact_nodes, type_ignores=[])
                ast.fix_missing_locations(compact_module)
                shape["interface_shape"] = ast.unparse(compact_module)
        except Exception:



            pass

        return shape

    @staticmethod
    def _strip_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            return body[1:] or [ast.Pass()]
        return body or [ast.Pass()]

    def _invoke_llm(
        self, system_prompt: str, user_prompt: str
    ) -> tuple[RawContractProbePlan, str, dict[str, Any], str]:
        llm = self._bound_probe_llm()
        messages = [("system", system_prompt), ("human", user_prompt)]

        if self.output_structure == "structured":
            try:
                structured_llm = llm.with_structured_output(RawContractProbePlan)
                metric = LLMRunMetrics()
                response, metric = invoke_with_metrics(
                    structured_llm,
                    messages,
                    metric,
                )
                if isinstance(response, RawContractProbePlan):
                    return (
                        response,
                        response.model_dump_json(),
                        metric.as_dict(),
                        "langchain_structured",
                    )
                raw = RawContractProbePlan.model_validate(response)
                return (
                    raw,
                    json.dumps(response, ensure_ascii=False, default=str),
                    metric.as_dict(),
                    "langchain_structured",
                )
            except Exception as exc:
                logger.warning(
                    "contract_probe: structured output failed; "
                    "falling back to raw manual JSON parsing "
                    f"({type(exc).__name__}: {exc})"
                )

        metric = LLMRunMetrics()
        response, metric = invoke_with_metrics(llm, messages, metric)
        raw_text = self._response_to_text(response)
        if not raw_text:
            raise ValueError("Model output is empty; no contract probe plan found.")
        data = extract_json_object(raw_text)
        raw = RawContractProbePlan.model_validate(data)
        return raw, raw_text, metric.as_dict(), "manual_json"

    def _bound_probe_llm(self) -> Any:
        llm = self.llm.llm
        probe_max_tokens = self._probe_max_tokens()
        base_max_tokens = int(
            getattr(getattr(self.llm, "config", None), "max_tokens", 0) or 0
        )
        if probe_max_tokens and probe_max_tokens > base_max_tokens:
            provider = str(getattr(getattr(self.llm, "config", None), "provider", ""))
            bind_kwargs = (
                {"num_predict": probe_max_tokens}
                if provider == "ollama"
                else {"max_tokens": probe_max_tokens}
            )
            llm = llm.bind(**bind_kwargs)
        return llm

    @staticmethod
    def _response_to_text(response: Any) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    if "text" in item:
                        parts.append(str(item["text"]))
                    elif "content" in item:
                        parts.append(str(item["content"]))
                    else:
                        parts.append(json.dumps(item, ensure_ascii=False, default=str))
                else:
                    parts.append(str(item))
            return "\n".join(parts).strip()
        return str(content).strip()

    @staticmethod
    def _normalize_output_structure(value: str) -> OutputStructure:
        normalized = (value or "raw").strip().lower().replace("-", "_")
        if normalized in {"raw", "text", "manual"}:
            return "raw"
        if normalized in {"structured", "json", "schema"}:
            return "structured"
        raise ValueError(
            "output_structure must be either 'raw' or 'structured'; "
            f"got {value!r}."
        )

    @staticmethod
    def _probe_max_tokens() -> int:
        for name in ("LLM_PROBE_MAX_OUTPUT_TOKENS", "PROBE_MAX_TOKENS"):
            value = os.getenv(name)
            if value and value.strip():
                try:
                    return max(0, int(value))
                except ValueError:
                    continue
        return 4096

    @staticmethod
    def _normalize_probes(probes: list[ContractProbe]) -> list[ContractProbe]:
        normalized, _ = ContractProbeStage._normalize_probes_with_diagnostics(probes)
        return normalized

    @staticmethod
    def _normalize_probes_with_diagnostics(
        probes: list[ContractProbe],
    ) -> tuple[list[ContractProbe], dict[str, Any]]:
        normalized: list[ContractProbe] = []
        dropped: list[dict[str, Any]] = []
        drop_reasons: dict[str, int] = {}

        def drop(index: int, probe: ContractProbe, reason: str) -> None:
            drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
            dropped.append(
                {
                    "index": index,
                    "type": getattr(probe, "type", None),
                    "reason": reason,
                    "probe": probe.model_dump() if hasattr(probe, "model_dump") else str(probe),
                }
            )

        for index, probe in enumerate(probes):
            if not probe.input.strip():
                drop(index, probe, "empty_input")
                continue
            if probe.type == "oracle" and probe.expected is None:
                drop(index, probe, "oracle_missing_expected")
                continue
            if probe.type == "violation" and probe.relation != "contract_violation":
                probe.relation = "contract_violation"
            if probe.type == "metamorphic" and not probe.paired_input:
                drop(index, probe, "metamorphic_missing_paired_input")
                continue
            if probe.type == "predicate" and not probe.relation.strip():
                drop(index, probe, "predicate_missing_relation")
                continue
            normalized.append(probe)

        return normalized, {"dropped": dropped, "drop_reasons": drop_reasons}

    @staticmethod
    def _zero_probe_feedback(
        *,
        raw_count: int = 0,
        drop_reasons: dict[str, int] | None = None,
    ) -> str:
        return "\n".join(
            [
                "The previous contract-probe plan produced zero valid probes.",
                f"Raw probes returned: {raw_count}.",
                f"Drop reasons: {drop_reasons or {}}.",
                "",
                "This is invalid. The next response must contain a non-empty probes array.",
                "",
                "Repair requirements:",
                "- Generate 4 to 6 compact probes when possible; never return an empty list.",
                "- Include at least one high-priority valid positive probe.",
                "- Include at least one high-priority should-violate probe for a precondition if possible.",
                "- Include lower-bound and upper-bound violations when bounds exist.",
                "- Include at least one predicate or metamorphic probe for a postcondition if exact stdout is uncertain.",
                "- Preserve exact stdin shape for every probe, including violation probes.",
                "- Use oracle only when the exact output is certain.",
                "- Return only the structured response.",
            ]
        )

    @staticmethod
    def _repair_feedback(error: str) -> str:
        return "\n".join(
            [
                "The previous probe plan failed validation.",
                "",
                "Validation feedback:",
                error,
                "",
                "Repair requirements:",
                "- Generate small valid stdin probes.",
                "- Do not return an empty probe list.",
                "- Use only oracle, violation, metamorphic, or predicate.",
                "- Return only the structured response.",
            ]
        )

    @staticmethod
    def _public_oracle_plan(
        public_witnesses: dict[str, list[Any]] | None,
    ) -> ContractProbePlan:
        if not public_witnesses:
            return ContractProbePlan(probes=[])
        inputs = public_witnesses.get("inputs") or []
        outputs = public_witnesses.get("outputs") or []
        probes: list[ContractProbe] = []
        for input_data, expected in list(zip(inputs, outputs))[:3]:
            probes.append(
                ContractProbe(
                    type="oracle",
                    input=str(input_data),
                    expected=expected,
                    relation="",
                    requirement_ids=[],
                    reason=(
                        "Primary oracle probe from user-provided public example."
                    ),
                    priority="high",
                )
            )
        return ContractProbePlan(probes=probes)
