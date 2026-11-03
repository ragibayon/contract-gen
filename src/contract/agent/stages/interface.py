from __future__ import annotations

import ast
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from contract.agent.schema.interface import ProgramInterface, RawProgramInterface
from contract.agent.schema.specification import Specification
from contract.llm.callbacks.metrics import (
    LLMRunMetrics,
    invoke_with_metrics,
    merge_llm_metrics,
)
from contract.llm.client import LLMClient
from contract.utils.logger import logger
from contract.utils.prompt_renderer import render_prompt

OutputStructure = Literal["raw", "structured"]


@dataclass
class InterfaceStageResult:
    name: str
    output: Any = None
    raw_output: str = ""
    error: str | None = None
    attempts: int = 0
    elapsed_sec: float = 0.0
    llm_metrics: dict[str, Any] = field(default_factory=dict)
    refinements: list[dict[str, Any]] = field(default_factory=list)


class InterfaceStage:
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
        task_prompt: str,
        specification: Specification,
    ) -> InterfaceStageResult:
        started = time.time()
        last_error: str | None = None
        last_raw = ""
        repair_feedback = ""
        metrics_items: list[dict[str, Any]] = []
        refinements: list[dict[str, Any]] = []

        contracts_json = json.dumps(
            [c.model_dump() for c in specification.contracts],
            indent=2,
            ensure_ascii=False,
        )

        logger.info(
            f"interface: preparing interface from {len(specification.contracts)} contracts "
            f"(output_structure={self.output_structure}, prompt_variant={self.prompt_variant})"
        )

        system_prompt = render_prompt(
            "interface/system.j2",
            output_structure=self.output_structure,
            prompt_variant=self.prompt_variant,
        )
        base_user_prompt = render_prompt(
            "interface/user.j2",
            task_prompt=task_prompt,
            contracts_json=contracts_json,
            output_structure=self.output_structure,
            prompt_variant=self.prompt_variant,
        )

        for attempt in range(1, self.max_attempts + 1):
            user_prompt = base_user_prompt
            if repair_feedback:
                user_prompt += "\n\n" + render_prompt(
                    "interface/repair.j2",
                    feedback=repair_feedback,
                    output_structure=self.output_structure,
                    prompt_variant=self.prompt_variant,
                )

            logger.stage_attempt(
                "interface",
                attempt,
                self.max_attempts,
                repairing=bool(repair_feedback),
            )

            try:
                logger.info("interface: invoking LLM")
                raw, raw_text, metric, parser_name = self._invoke_llm(
                    system_prompt,
                    user_prompt,
                )
                metrics_items.append(metric)
                last_raw = raw_text

                interface = self._normalize(raw)
                self._validate_skeleton(interface.skeleton)

                result = InterfaceStageResult(
                    name="interface",
                    output=interface,
                    raw_output=last_raw,
                    attempts=attempt,
                    elapsed_sec=round(time.time() - started, 3),
                    llm_metrics=merge_llm_metrics(metrics_items),
                    refinements=refinements,
                )
                logger.stage_finished(
                    "interface",
                    True,
                    result.elapsed_sec,
                    {
                        "Attempts": attempt,
                        "Parser": parser_name,
                        "Skeleton lines": len(interface.skeleton.splitlines()),
                        "Static validation": "passed",
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
                    }
                )
                repair_feedback = last_error
                logger.warning(f"interface: attempt {attempt} failed: {last_error}")

        result = InterfaceStageResult(
            name="interface",
            output=None,
            raw_output=last_raw,
            error=last_error or "Unknown error.",
            attempts=self.max_attempts,
            elapsed_sec=round(time.time() - started, 3),
            llm_metrics=merge_llm_metrics(metrics_items),
            refinements=refinements,
        )
        logger.stage_finished(
            "interface",
            False,
            result.elapsed_sec,
            {"Attempts": result.attempts, "Error": result.error},
        )
        return result

    def _invoke_llm(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[RawProgramInterface, str, dict[str, Any], str]:
        messages = [("system", system_prompt), ("human", user_prompt)]

        if self.output_structure == "structured":
            try:
                structured_llm = self.llm.llm.with_structured_output(
                    RawProgramInterface
                )
                metric = LLMRunMetrics()
                response, metric = invoke_with_metrics(structured_llm, messages, metric)

                if isinstance(response, RawProgramInterface):
                    return (
                        response,
                        response.model_dump_json(),
                        metric.as_dict(),
                        "langchain_structured",
                    )

                raw = RawProgramInterface.model_validate(response)
                return (
                    raw,
                    json.dumps(response, ensure_ascii=False, default=str),
                    metric.as_dict(),
                    "langchain_structured",
                )

            except Exception as exc:
                logger.warning(
                    "interface: structured output failed; falling back to raw parsing "
                    f"({type(exc).__name__}: {exc})"
                )

        metric = LLMRunMetrics()
        response, metric = invoke_with_metrics(self.llm.llm, messages, metric)
        raw_text = self._response_to_text(response)

        if not raw_text:
            raise ValueError("Model output is empty; no interface skeleton found.")


        try:
            data = self._extract_json_object(raw_text)
            raw = RawProgramInterface.model_validate(data)
            return raw, raw_text, metric.as_dict(), "manual_json"
        except Exception:
            pass


        cleaned = self._strip_code_fence(raw_text)
        raw = RawProgramInterface(skeleton=cleaned)
        return raw, raw_text, metric.as_dict(), "raw_skeleton"

    @staticmethod
    def _normalize(raw: RawProgramInterface) -> ProgramInterface:
        return ProgramInterface(skeleton=raw.skeleton.strip() + "\n")

    @staticmethod
    def _validate_skeleton(skeleton: str) -> None:
        tree = ast.parse(skeleton)
        compile(tree, "<interface_skeleton>", "exec")

        solve_defs = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "solve"
        ]
        main_defs = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        ]

        if not solve_defs:
            raise ValueError("Skeleton must define solve(...).")
        if not main_defs:
            raise ValueError("Skeleton must define main().")

        solve_body = solve_defs[0].body
        executable_body = [
            node
            for node in solve_body
            if not isinstance(node, ast.Expr)
            or not isinstance(getattr(node, "value", None), ast.Constant)
            or not isinstance(node.value.value, str)
        ]

        if len(executable_body) != 1 or not isinstance(executable_body[0], ast.Pass):
            raise ValueError(
                "solve(...) body must contain only pass after the docstring."
            )

        has_if_main = False
        for node in tree.body:
            if not isinstance(node, ast.If):
                continue
            test = node.test
            if (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "__name__"
                and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Eq)
                and len(test.comparators) == 1
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value == "__main__"
            ):
                has_if_main = True
                break

        if not has_if_main:
            raise ValueError('Skeleton must include if __name__ == "__main__": guard.')

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
    def _extract_json_object(text: str) -> dict[str, Any]:
        cleaned = InterfaceStage._strip_code_fence(text)

        decoder = json.JSONDecoder()


        try:
            value = json.loads(cleaned)
            if isinstance(value, dict):
                return value
        except Exception:
            pass


        for match in re.finditer(r"\{", cleaned):
            start = match.start()
            try:
                value, _ = decoder.raw_decode(cleaned[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value

        raise ValueError("No JSON object found in model output.")

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        stripped = text.strip()

        fence_match = re.fullmatch(
            r"```(?:json|python|py)?\s*(.*?)\s*```",
            stripped,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if fence_match:
            return fence_match.group(1).strip()

        return stripped

    @staticmethod
    def _normalize_output_structure(value: str) -> OutputStructure:
        normalized = (value or "raw").strip().lower().replace("-", "_")

        if normalized in {"raw", "text", "manual"}:
            return "raw"
        if normalized in {"structured", "json", "schema"}:
            return "structured"

        raise ValueError(
            "output_structure must be either 'raw' or 'structured'; " f"got {value!r}."
        )
