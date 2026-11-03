from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from contract.agent.schema.normalizer import normalize_spec
from contract.agent.schema.parser import parse_raw_specification
from contract.agent.schema.specification import RawSpecification
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
class AgentStageResult:
    name: str
    output: Any = None
    raw_output: str = ""
    error: str | None = None
    attempts: int = 0
    elapsed_sec: float = 0.0
    llm_metrics: dict[str, Any] = field(default_factory=dict)
    refinements: list[dict[str, Any]] = field(default_factory=list)


class SpecificationStage:
    def __init__(
        self,
        llm: LLMClient,
        max_attempts: int = 1,
        output_structure: OutputStructure = "raw",
        prompt_variant: str = "contract",
    ):
        self.llm = llm
        self.max_attempts = max(1, max_attempts)
        self.output_structure = self._normalize_output_structure(output_structure)
        self.prompt_variant = prompt_variant

    def run(
        self,
        task_id: str,
        prompt: str,
        entry_point: str | None = None,
    ) -> AgentStageResult:
        started = time.time()
        last_raw_output = ""
        last_error: str | None = None
        repair_feedback = ""
        metrics_items: list[dict[str, Any]] = []
        refinements: list[dict[str, Any]] = []

        logger.info(
            "specification: preparing prompts "
            f"(entry_point={entry_point or 'not specified'}, "
            f"output_structure={self.output_structure}, "
            f"prompt_variant={self.prompt_variant})"
        )

        schema_json = self._compact_schema_json()
        system_prompt = render_prompt(
            "specification/system.j2",
            output_structure=self.output_structure,
            prompt_variant=self.prompt_variant,
        )
        base_user_prompt = render_prompt(
            "specification/user.j2",
            task_id=task_id,
            entry_point_json=json.dumps(entry_point),
            task_prompt=prompt,
            schema_json=schema_json,
            output_structure=self.output_structure,
            prompt_variant=self.prompt_variant,
        )

        for attempt in range(1, self.max_attempts + 1):
            user_prompt = base_user_prompt
            if repair_feedback:
                user_prompt += "\n\n" + render_prompt(
                    "specification/repair.j2",
                    error=repair_feedback,
                    schema_json=schema_json,
                    output_structure=self.output_structure,
                    prompt_variant=self.prompt_variant,
                )

            logger.stage_attempt(
                "specification",
                attempt,
                self.max_attempts,
                repairing=bool(repair_feedback),
            )

            try:
                logger.info("specification: invoking LLM")
                raw_spec, raw_text, metric, parser_name = self._invoke(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
                metrics_items.append(metric)
                last_raw_output = raw_text

                spec = normalize_spec(
                    raw_spec,
                    task_id=task_id,
                    entry_point=entry_point,
                )

                if not spec.contracts:
                    raise ValueError("Specification contains zero contracts.")

                result = AgentStageResult(
                    name="specification",
                    output=spec,
                    raw_output=raw_text,
                    attempts=attempt,
                    elapsed_sec=round(time.time() - started, 3),
                    llm_metrics=merge_llm_metrics(metrics_items),
                    refinements=refinements,
                )

                logger.stage_finished(
                    "specification",
                    True,
                    result.elapsed_sec,
                    {
                        "Attempts": attempt,
                        "Contracts": len(spec.contracts),
                        "Output structure": self.output_structure,
                        "Parser": parser_name,
                    },
                )
                return result

            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                refinements.append(
                    {
                        "attempt": attempt,
                        "error": last_error,
                        "repair_feedback": last_error,
                        "output_structure": self.output_structure,
                    }
                )
                repair_feedback = last_error
                logger.warning(f"specification: attempt {attempt} failed: {last_error}")

        result = AgentStageResult(
            name="specification",
            output=None,
            raw_output=last_raw_output,
            error=last_error or "Unknown error.",
            attempts=self.max_attempts,
            elapsed_sec=round(time.time() - started, 3),
            llm_metrics=merge_llm_metrics(metrics_items),
            refinements=refinements,
        )

        logger.stage_finished(
            "specification",
            False,
            result.elapsed_sec,
            {
                "Attempts": result.attempts,
                "Output structure": self.output_structure,
                "Error": result.error,
            },
        )
        return result

    def _invoke(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[RawSpecification, str, dict[str, Any], str]:
        if self.output_structure == "structured":
            try:
                return self._invoke_structured(system_prompt, user_prompt)
            except Exception as exc:
                logger.warning(
                    "specification: structured output failed; "
                    "falling back to raw manual JSON parsing "
                    f"({type(exc).__name__}: {exc})"
                )

        return self._invoke_raw(system_prompt, user_prompt)

    def _invoke_structured(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[RawSpecification, str, dict[str, Any], str]:
        messages = [
            ("system", system_prompt),
            ("human", user_prompt),
        ]

        structured_llm = self.llm.llm.with_structured_output(RawSpecification)
        metric = LLMRunMetrics()
        response, metric = invoke_with_metrics(structured_llm, messages, metric)

        if isinstance(response, RawSpecification):
            raw_spec = response
            raw_text = response.model_dump_json()
        else:
            data = self._coerce_specification_json(dict(response))
            raw_spec = RawSpecification.model_validate(data)
            raw_text = json.dumps(response, ensure_ascii=False, default=str)

        return raw_spec, raw_text, metric.as_dict(), "langchain_structured"

    def _invoke_raw(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[RawSpecification, str, dict[str, Any], str]:
        messages = [
            ("system", system_prompt),
            ("human", user_prompt),
        ]

        metric = LLMRunMetrics()
        response, metric = invoke_with_metrics(self.llm.llm, messages, metric)
        raw_text = self._response_to_text(response)

        try:
            data = extract_json_object(raw_text)
            data = self._coerce_specification_json(data)
            raw_spec = RawSpecification.model_validate(data)
            return raw_spec, raw_text, metric.as_dict(), "manual_json"
        except Exception as json_exc:
            logger.warning(
                "specification: raw JSON extraction/validation failed; "
                "falling back to legacy schema parser "
                f"({type(json_exc).__name__}: {json_exc})"
            )

        raw_spec = parse_raw_specification(raw_text)
        return raw_spec, raw_text, metric.as_dict(), "legacy_parser"

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
    def _coerce_specification_json(data: dict[str, Any]) -> dict[str, Any]:
        """Normalize common local-model JSON mistakes before Pydantic validation."""
        if not isinstance(data, dict):
            raise ValueError("Specification JSON must be an object.")

        if "contracts" not in data:
            for key in (
                "requirements",
                "nl_contracts",
                "natural_language_contracts",
                "behavioral_contracts",
                "specification",
            ):
                value = data.get(key)
                if isinstance(value, list):
                    data["contracts"] = value
                    break
                if isinstance(value, dict) and isinstance(value.get("contracts"), list):
                    data["contracts"] = value["contracts"]
                    if not data.get("summary") and value.get("summary"):
                        data["summary"] = value["summary"]
                    break

        contracts = data.get("contracts")
        if not isinstance(contracts, list):
            raise ValueError("Specification JSON must contain a contracts list.")

        if not str(data.get("summary") or "").strip():
            data["summary"] = (
                "The task requires producing output that satisfies the stated "
                "input, output, and resource constraints."
            )

        counters: dict[str, int] = {
            "precondition": 0,
            "postcondition": 0,
            "invariant": 0,
            "resource_bound": 0,
        }
        normalized_contracts: list[dict[str, str]] = []

        for item in contracts:
            if not isinstance(item, dict):
                continue

            contract_type = SpecificationStage._normalize_contract_type(
                str(item.get("type") or item.get("kind") or "")
            )
            if contract_type not in counters:
                contract_type = SpecificationStage._infer_contract_type(item)

            text = SpecificationStage._first_nonempty(
                item,
                "text",
                "contract",
                "description",
                "requirement",
                "condition",
                "obligation",
                "statement",
                "predicate",
            )
            evidence = SpecificationStage._first_nonempty(
                item,
                "evidence",
                "source",
                "grounding",
                "rationale",
                "support",
                "quote",
            )

            if not text:
                continue
            if not evidence:
                evidence = text

            counters[contract_type] += 1
            normalized_contracts.append(
                {
                    "type": contract_type,
                    "text": str(text).strip(),
                    "evidence": str(evidence).strip(),
                }
            )

        if not normalized_contracts:
            raise ValueError("Specification contains zero usable contracts.")

        return {
            "summary": str(data["summary"]).strip(),
            "contracts": normalized_contracts,
        }

    @staticmethod
    def _first_nonempty(item: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = item.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    @staticmethod
    def _normalize_contract_type(contract_type: str) -> str:
        value = contract_type.lower().strip().replace("-", "_").replace(" ", "_")
        aliases = {
            "pre": "precondition",
            "requires": "precondition",
            "require": "precondition",
            "input": "precondition",
            "input_constraint": "precondition",
            "input_constraints": "precondition",
            "assumption": "precondition",
            "caller_obligation": "precondition",
            "post": "postcondition",
            "ensures": "postcondition",
            "ensure": "postcondition",
            "output": "postcondition",
            "output_constraint": "postcondition",
            "output_constraints": "postcondition",
            "guarantee": "postcondition",
            "supplier_guarantee": "postcondition",
            "definition": "invariant",
            "semantic_rule": "invariant",
            "rule": "invariant",
            "resource": "resource_bound",
            "resource_bounds": "resource_bound",
            "time": "resource_bound",
            "memory": "resource_bound",
            "nfr": "resource_bound",
            "non_functional": "resource_bound",
            "nonfunctional": "resource_bound",
            "non_functional_requirement": "resource_bound",
        }
        return aliases.get(value, value)

    @staticmethod
    def _infer_contract_type(item: dict[str, Any]) -> str:
        text = " ".join(str(value) for value in item.values()).lower()
        if any(marker in text for marker in ("time", "memory", "second", "megabyte", "mb")):
            return "resource_bound"
        if any(marker in text for marker in ("input", "contains", "given", "satisfies", "constraint")):
            return "precondition"
        return "postcondition"

    @staticmethod
    def _compact_schema_json() -> str:
        return json.dumps(
            {
                "summary": "one concise sentence describing the task",
                "contracts": [
                    {
                        "type": "precondition | postcondition | invariant | resource_bound",
                        "text": "one atomic natural-language contract",
                        "evidence": "task text that grounds this contract",
                    }
                ],
            },
            indent=2,
            ensure_ascii=False,
        )

    @staticmethod
    def _normalize_output_structure(value: str) -> OutputStructure:
        normalized = str(value or "raw").strip().lower().replace("-", "_")
        if normalized in {"raw", "json", "manual"}:
            return "raw"
        if normalized in {"structured", "structure", "tool", "langchain"}:
            return "structured"
        raise ValueError("output_structure must be either 'raw' or 'structured'.")
