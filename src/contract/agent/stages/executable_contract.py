from __future__ import annotations

import ast
import json
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from contract.agent.schema.executable_contract import (
    ContractLoweringItem,
    ExecutableContractArtifact,
    RawExecutableContractArtifact,
)
from contract.agent.schema.interface import ProgramInterface
from contract.agent.schema.specification import Specification
from contract.llm.callbacks.metrics import (
    LLMRunMetrics,
    invoke_with_metrics,
    merge_llm_metrics,
)
from contract.llm.client import LLMClient
from contract.utils.json_extract import extract_json_object
from contract.utils.logger import logger
from contract.utils.prompt_renderer import render_prompt

OutputStructure = Literal["raw", "structured"]


@dataclass
class ExecutableContractStageResult:
    name: str
    output: Any = None
    raw_output: str = ""
    error: str | None = None
    attempts: int = 0
    elapsed_sec: float = 0.0
    static_validation_passed: bool = False
    llm_metrics: dict[str, Any] = field(default_factory=dict)
    refinements: list[dict[str, Any]] = field(default_factory=list)


class ExecutableContractStage:
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
    ) -> ExecutableContractStageResult:
        started = time.time()
        last_error: str | None = None
        last_raw = ""
        last_valid_artifact: ExecutableContractArtifact | None = None
        repair_feedback = ""
        metrics_items: list[dict[str, Any]] = []
        refinements: list[dict[str, Any]] = []

        contracts_json = json.dumps(
            [contract.model_dump() for contract in specification.contracts],
            indent=2,
            ensure_ascii=False,
        )
        expected_contract_ids = [contract.id for contract in specification.contracts]
        logger.info(
            "executable_contract: preparing decorators "
            f"for {len(specification.contracts)} contracts "
            f"(output_structure={self.output_structure}, prompt_variant={self.prompt_variant})"
        )

        system_prompt = render_prompt(
            "executable_contract/system.j2",
            output_structure=self.output_structure,
            prompt_variant=self.prompt_variant,
        )
        base_user_prompt = render_prompt(
            "executable_contract/user.j2",
            task_prompt=task_prompt,
            contracts_json=contracts_json,
            skeleton=interface.skeleton,
            output_structure=self.output_structure,
            prompt_variant=self.prompt_variant,
        )

        for attempt in range(1, self.max_attempts + 1):
            logger.stage_attempt(
                "executable_contract",
                attempt,
                self.max_attempts,
                repairing=bool(repair_feedback),
            )
            user_prompt = base_user_prompt
            if repair_feedback:
                user_prompt += "\n\n" + render_prompt(
                    "executable_contract/repair.j2",
                    feedback=repair_feedback,
                    output_structure=self.output_structure,
                    prompt_variant=self.prompt_variant,
                )

            try:
                logger.info("executable_contract: invoking LLM")
                raw, raw_text, metric, parser_name = self._invoke_llm(
                    system_prompt,
                    user_prompt,
                )
                metrics_items.append(metric)
                last_raw = raw_text

                imports = raw.imports.strip()
                decorators = raw.decorators.strip()
                coverage = self._normalize_coverage(raw.coverage, expected_contract_ids)

                self._validate_decorators(imports=imports, decorators=decorators)
                self._validate_contains_icontract(imports=imports, decorators=decorators)
                self._validate_has_decorators(decorators=decorators)

                last_valid_artifact = ExecutableContractArtifact(
                    imports=imports,
                    decorators=decorators,
                    coverage=coverage,
                )

                result = ExecutableContractStageResult(
                    name="executable_contract",
                    output=last_valid_artifact,
                    raw_output=last_raw,
                    attempts=attempt,
                    elapsed_sec=round(time.time() - started, 3),
                    static_validation_passed=True,
                    llm_metrics=merge_llm_metrics(metrics_items),
                    refinements=refinements,
                )
                logger.stage_finished(
                    "executable_contract",
                    True,
                    result.elapsed_sec,
                    {
                        "Attempts": attempt,
                        "Parser": parser_name,
                        "Static validation": "passed",
                        "Decorator lines": len(decorators.splitlines()),
                        "Coverage": f"{sum(1 for c in coverage if c.lowered_to != 'omitted')}/{len(coverage)} lowered",
                    },
                )
                return result

            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                repair_feedback = self._format_repair_feedback(last_error)
                refinements.append(
                    {
                        "attempt": attempt,
                        "error": last_error,
                        "repair_feedback": repair_feedback,
                    }
                )
                logger.warning(
                    f"executable_contract: attempt {attempt} failed: {last_error}"
                )

        result = ExecutableContractStageResult(
            name="executable_contract",
            output=last_valid_artifact,
            raw_output=last_raw,
            error=last_error or "Unknown error.",
            attempts=self.max_attempts,
            elapsed_sec=round(time.time() - started, 3),
            static_validation_passed=last_valid_artifact is not None,
            llm_metrics=merge_llm_metrics(metrics_items),
            refinements=refinements,
        )
        logger.stage_finished(
            "executable_contract",
            False,
            result.elapsed_sec,
            {
                "Attempts": result.attempts,
                "Static validation": result.static_validation_passed,
                "Error": result.error,
            },
        )
        return result

    def _invoke_llm(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[RawExecutableContractArtifact, str, dict[str, Any], str]:
        messages = [("system", system_prompt), ("human", user_prompt)]

        if self.output_structure == "structured":
            try:
                structured_llm = self.llm.llm.with_structured_output(
                    RawExecutableContractArtifact
                )
                metric = LLMRunMetrics()
                response, metric = invoke_with_metrics(structured_llm, messages, metric)
                if isinstance(response, RawExecutableContractArtifact):
                    return (
                        response,
                        response.model_dump_json(),
                        metric.as_dict(),
                        "langchain_structured",
                    )
                raw = RawExecutableContractArtifact.model_validate(response)
                return (
                    raw,
                    json.dumps(response, ensure_ascii=False, default=str),
                    metric.as_dict(),
                    "langchain_structured",
                )
            except Exception as exc:
                logger.warning(
                    "executable_contract: structured output failed; "
                    "falling back to raw manual JSON parsing "
                    f"({type(exc).__name__}: {exc})"
                )

        metric = LLMRunMetrics()
        response, metric = invoke_with_metrics(self.llm.llm, messages, metric)
        raw_text = self._response_to_text(response)
        if not raw_text:
            raise ValueError("Model output is empty; no executable contract artifact found.")

        data = extract_json_object(raw_text)
        raw = RawExecutableContractArtifact.model_validate(data)
        return raw, raw_text, metric.as_dict(), "manual_json"

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
    def _normalize_coverage(
        coverage: list[ContractLoweringItem],
        expected_contract_ids: list[str],
    ) -> list[ContractLoweringItem]:
        by_id = {item.contract_id: item for item in coverage}
        normalized: list[ContractLoweringItem] = []
        for contract_id in expected_contract_ids:
            item = by_id.get(contract_id)
            if item is None:
                item = ContractLoweringItem(
                    contract_id=contract_id,
                    lowered_to="omitted",
                    expression="",
                    reason="The model did not provide a lowering decision for this NLContract.",
                )
            normalized.append(item)
        return normalized

    @staticmethod
    def _validate_decorators(imports: str, decorators: str) -> None:
        snippet = "\n".join(
            [
                imports.strip(),
                "",
                decorators.strip(),
                "def solve(*args, **kwargs):",
                "    return None",
                "",
            ]
        )
        tree = ast.parse(snippet)
        compiled = compile(tree, "<contract_decorators>", "exec")
        exec(compiled, {"__name__": "__contract_validation__"})

    @staticmethod
    def _validate_contains_icontract(imports: str, decorators: str) -> None:
        combined = f"{imports}\n{decorators}"
        if "icontract" not in combined:
            raise ValueError("Contract artifact must import/use icontract.")

    @staticmethod
    def _validate_has_decorators(decorators: str) -> None:
        if "@icontract." not in decorators:
            raise ValueError("Contract artifact must contain icontract decorators.")

    @staticmethod
    def _format_repair_feedback(error: str) -> str:
        return "\n".join(
            [
                "The previous executable contract decorators failed validation.",
                "",
                "Validation feedback:",
                error,
                "",
                "Repair guidance:",
                "- Return syntactically valid Python imports and decorators.",
                "- Use icontract decorators only.",
                "- Do not generate solve(...).",
                "- Do not generate main().",
                "- Do not generate implementation code.",
                "- Decorators must be valid when placed immediately above solve(...).",
                "- Include coverage for every NLContract ID.",
                "- Omit predicates that are too expensive for the stated limits.",
            ]
        )

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
