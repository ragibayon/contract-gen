from __future__ import annotations

import ast
import json
import time
from dataclasses import dataclass, field
from typing import Any

from contract.agent.schema.code import CodeArtifact, RawCodeArtifact
from contract.agent.schema.interface import ProgramInterface
from contract.agent.schema.specification import Specification
from contract.eval.execution import ExecutionWitnessReport, evaluate_execution_witnesses
from contract.llm.callbacks.metrics import LLMRunMetrics, invoke_with_metrics, merge_llm_metrics
from contract.llm.client import LLMClient
from contract.utils.code_extract import extract_python_code, response_to_text
from contract.utils.logger import logger
from contract.utils.prompt_renderer import render_prompt


@dataclass
class CodeStageResult:
    name: str
    output: Any = None
    raw_output: str = ""
    error: str | None = None
    attempts: int = 0
    elapsed_sec: float = 0.0
    static_validation_passed: bool = False
    execution_witness_passed: bool | None = None
    witness_report: ExecutionWitnessReport | None = None
    llm_metrics: dict[str, Any] = field(default_factory=dict)
    refinements: list[dict[str, Any]] = field(default_factory=list)


class CodeGenerationStage:
    def __init__(
        self,
        llm: LLMClient,
        max_attempts: int = 3,
        output_structure: str = "raw",
        prompt_variant: str = "contract",
        max_refinement_attempts: int = 3,
        execution_timeout_sec: float = 5.0,
    ):
        self.llm = llm
        self.max_attempts = max(1, max_attempts)
        self.max_refinement_attempts = max(0, max_refinement_attempts)
        self.execution_timeout_sec = max(0.1, float(execution_timeout_sec))
        self.output_structure = self._normalize_output_structure(output_structure)
        self.prompt_variant = prompt_variant

    def run(
        self,
        task_prompt: str,
        specification: Specification,
        interface: ProgramInterface,
        public_witnesses: dict[str, list[Any]] | None = None,
    ) -> CodeStageResult:
        started = time.time()
        last_error: str | None = None
        last_raw = ""
        last_valid_code: str | None = None
        last_witness_report: ExecutionWitnessReport | None = None
        repair_feedback = ""
        metrics_items: list[dict[str, Any]] = []
        refinements: list[dict[str, Any]] = []

        contracts_json = json.dumps(
            [contract.model_dump() for contract in specification.contracts],
            indent=2,
            ensure_ascii=False,
        )

        has_public_witnesses = bool(
            public_witnesses
            and public_witnesses.get("inputs")
            and public_witnesses.get("outputs")
        )
        witness_count = min(
            len((public_witnesses or {}).get("inputs", [])),
            len((public_witnesses or {}).get("outputs", [])),
        )
        logger.info(
            "code_generation: preparing code generation "
            f"({len(specification.contracts)} contracts, "
            f"{witness_count} public witnesses, "
            f"output_structure={self.output_structure}, "
            f"prompt_variant={self.prompt_variant})"
        )

        max_total_attempts = self.max_attempts * (self.max_refinement_attempts + 1)
        attempt = 0
        static_attempt = 0
        refinement_round = 0
        witness_feedback = ""
        refinement_length_errors = 0

        while static_attempt < self.max_attempts:
            attempt += 1
            static_attempt += 1
            logger.stage_attempt(
                "code_generation",
                attempt,
                max_total_attempts,
                repairing=bool(repair_feedback),
            )
            logger.info(
                "code_generation: "
                f"refinement {refinement_round}/{self.max_refinement_attempts}, "
                f"static attempt {static_attempt}/{self.max_attempts}"
            )
            system_prompt = render_prompt(
                "code_generation/system.j2",
                output_structure=self.output_structure,
                prompt_variant=self.prompt_variant,
            )
            user_prompt = render_prompt(
                "code_generation/user.j2",
                task_prompt=task_prompt,
                contracts_json=contracts_json,
                skeleton=interface.skeleton,
                output_structure=self.output_structure,
                prompt_variant=self.prompt_variant,
            )
            if repair_feedback:
                user_prompt += "\n\n" + render_prompt(
                    "code_generation/repair.j2",
                    feedback=repair_feedback,
                    output_structure=self.output_structure,
                    prompt_variant=self.prompt_variant,
                )

            try:
                logger.info("code_generation: invoking LLM")
                raw, raw_text, metric = self._invoke_code_llm(system_prompt, user_prompt)
                metrics_items.append(metric)
                last_raw = raw_text
                code = raw.code.strip() + "\n"

                self._validate_code(code)
                self._validate_required_program_shape(code)
                last_valid_code = code
                logger.success("code_generation: static validation passed")

                witness_report = None
                if has_public_witnesses:
                    logger.info(f"code_generation: evaluating {witness_count} public witnesses")
                    witness_report = evaluate_execution_witnesses(
                        code=code,
                        witnesses=public_witnesses or {},
                        timeout_sec=self.execution_timeout_sec,
                    )
                    last_witness_report = witness_report
                    if witness_report.pass_ratio < 1.0:
                        last_error = "Public execution witness check failed."
                        logger.warning(
                            "code_generation: public witnesses failed "
                            f"({witness_report.passed}/{witness_report.total} passed)"
                        )
                        witness_feedback = self._format_witness_feedback(witness_report)
                        repair_feedback = witness_feedback
                        refinements.append(
                            {
                                "attempt": attempt,
                                "refinement_round": refinement_round,
                                "static_attempt": static_attempt,
                                "error": last_error,
                                "raw_output": raw_text,
                                "parsed_code": code,
                                "witness_report": witness_report,
                                "repair_feedback": repair_feedback,
                            }
                        )
                        if refinement_round >= self.max_refinement_attempts:
                            break
                        refinement_round += 1
                        static_attempt = 0
                        continue
                    logger.success(
                        "code_generation: public witnesses passed "
                        f"({witness_report.passed}/{witness_report.total})"
                    )

                result = CodeStageResult(
                    name="code_generation",
                    output=CodeArtifact(code=code),
                    raw_output=last_raw,
                    error=None,
                    attempts=attempt,
                    elapsed_sec=round(time.time() - started, 3),
                    static_validation_passed=True,
                    execution_witness_passed=(
                        witness_report.pass_ratio == 1.0
                        if witness_report is not None
                        else None
                    ),
                    witness_report=witness_report,
                    llm_metrics=merge_llm_metrics(metrics_items),
                    refinements=refinements,
                )
                logger.stage_finished(
                    "code_generation",
                    True,
                    result.elapsed_sec,
                    {
                        "Attempts": attempt,
                        "Parser": self.output_structure,
                        "Static validation": "passed",
                        "Execution witnesses": (
                            "not provided"
                            if witness_report is None
                            else f"{witness_report.passed}/{witness_report.total} passed"
                        ),
                    },
                )
                return result

            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                static_feedback = self._format_static_feedback(last_error)
                is_refinement_length_error = (
                    refinement_round > 0 and type(exc).__name__ == "LengthError"
                )
                if is_refinement_length_error:
                    refinement_length_errors += 1
                    if refinement_length_errors == 1:
                        static_feedback += (
                            "\nBe concise during your private reasoning so there is enough "
                            "output space for the complete code."
                        )
                repair_feedback = (
                    witness_feedback + "\n\n" + static_feedback
                    if witness_feedback
                    else static_feedback
                )
                refinements.append(
                    {
                        "attempt": attempt,
                        "refinement_round": refinement_round,
                        "static_attempt": static_attempt,
                        "error": last_error,
                        "raw_output": last_raw,
                        "repair_feedback": repair_feedback,
                    }
                )
                logger.warning(f"code_generation: attempt {attempt} failed: {last_error}")
                if is_refinement_length_error and refinement_length_errors >= 2:
                    logger.warning(
                        "code_generation: stopping refinement after a repeated LengthError"
                    )
                    break

        result = CodeStageResult(
            name="code_generation",
            output=CodeArtifact(code=last_valid_code) if last_valid_code else None,
            raw_output=last_raw,
            error=last_error or "Unknown error.",
            attempts=attempt,
            elapsed_sec=round(time.time() - started, 3),
            static_validation_passed=last_valid_code is not None,
            execution_witness_passed=(
                last_witness_report.pass_ratio == 1.0
                if last_witness_report is not None
                else None
            ),
            witness_report=last_witness_report,
            llm_metrics=merge_llm_metrics(metrics_items),
            refinements=refinements,
        )
        logger.stage_finished(
            "code_generation",
            False,
            result.elapsed_sec,
            {
                "Attempts": result.attempts,
                "Parser": self.output_structure,
                "Static validation": result.static_validation_passed,
                "Execution witnesses": result.execution_witness_passed,
                "Error": result.error,
            },
        )
        return result

    def _invoke_code_llm(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[RawCodeArtifact, str, dict[str, Any]]:
        messages = [("system", system_prompt), ("human", user_prompt)]

        if self.output_structure == "structured":
            try:
                structured_llm = self.llm.llm.with_structured_output(RawCodeArtifact)
                metric = LLMRunMetrics()
                raw_response, metric = invoke_with_metrics(structured_llm, messages, metric)
                if isinstance(raw_response, RawCodeArtifact):
                    return raw_response, raw_response.model_dump_json(), metric.as_dict()
                raw = RawCodeArtifact.model_validate(raw_response)
                return raw, json.dumps(raw_response, ensure_ascii=False, default=str), metric.as_dict()
            except Exception as exc:
                logger.warning(
                    "code_generation: structured output failed; "
                    f"falling back to raw code parsing ({type(exc).__name__}: {exc})"
                )

        metric = LLMRunMetrics()
        response, metric = invoke_with_metrics(self.llm.llm, messages, metric)
        raw_text = response_to_text(response)
        code = extract_python_code(raw_text)
        return RawCodeArtifact(code=code), raw_text, metric.as_dict()

    @staticmethod
    def _validate_code(code: str) -> None:
        tree = ast.parse(code)
        compiled = compile(tree, "<generated_code>", "exec")
        exec(compiled, {"__name__": "__contract_validation__"})

    @staticmethod
    def _validate_required_program_shape(code: str) -> None:
        tree = ast.parse(code)
        has_solve = any(
            isinstance(node, ast.FunctionDef) and node.name == "solve"
            for node in tree.body
        )
        if not has_solve:
            raise ValueError("Generated program does not define solve(...).")

        has_main = any(
            isinstance(node, ast.FunctionDef) and node.name == "main"
            for node in tree.body
        )
        if not has_main:
            raise ValueError("Generated program does not define main().")

    @staticmethod
    def _normalize_output_structure(value: str) -> str:
        normalized = (value or "raw").strip().lower().replace("-", "_")
        if normalized in {"raw", "text", "code"}:
            return "raw"
        if normalized in {"structured", "json", "schema"}:
            return "structured"
        raise ValueError(
            "output_structure must be either 'raw' or 'structured'; "
            f"got {value!r}."
        )

    @staticmethod
    def _format_static_feedback(error: str) -> str:
        return "\n".join(
            [
                "The previous code failed parsing or static validation.",
                "",
                "Validation feedback:",
                error,
                "",
                "Repair guidance:",
                "- Return a complete syntactically valid Python program.",
                "- Return Python code only; do not return JSON.",
                "- Do not include markdown fences.",
                "- Preserve solve(...) and main().",
                "- solve(...) must return the computed answer.",
                "- main() must parse stdin, call solve(...), and print the result.",
                "- Do not explain the solution.",
            ]
        )

    @staticmethod
    def _format_witness_feedback(report: ExecutionWitnessReport) -> str:
        parts = [
            "The previous code violated public execution witnesses used as contract-satisfaction feedback.",
            f"Passed {report.passed}/{report.total} public execution witnesses.",
        ]
        diagnoses: list[str] = []
        for failure in report.failures:
            stderr = failure.stderr or ""
            if "IndexError: list index out of range" in stderr:
                diagnoses.append("The program indexed outside a sequence. Re-check computed indexes or use a mapping/dictionary when the key space is not guaranteed to match sequence length.")
            if "KeyError" in stderr:
                diagnoses.append("The program accessed a missing dictionary key. Re-check key construction and default handling.")
            if "ValueError" in stderr:
                diagnoses.append("The program raised ValueError. Re-check input parsing and assumptions.")
            if "TypeError" in stderr:
                diagnoses.append("The program raised TypeError. Re-check value types, parsed inputs, and operations.")
            if "ZeroDivisionError" in stderr:
                diagnoses.append("The program divided by zero. Re-check boundary cases and denominator assumptions.")
            if "RecursionError" in stderr:
                diagnoses.append("The program exceeded recursion depth. Re-check recursion strategy or use an iterative approach.")
            if "IndentationError" in stderr:
                diagnoses.append("The program has invalid indentation. Return syntactically valid Python.")
            if "SyntaxError" in stderr:
                diagnoses.append("The program has invalid syntax. Return syntactically valid Python.")
            if "TimeoutExpired" in stderr or "timed out" in stderr.lower():
                diagnoses.append("The program timed out. Improve the algorithmic complexity and avoid brute force over large bounds.")

        if diagnoses:
            parts.append("Diagnostic guidance:\n" + "\n".join(f"- {item}" for item in sorted(set(diagnoses))))

        for failure in report.failures[:3]:
            parts.append(
                "\n".join(
                    [
                        f"Witness {failure.index}:",
                        "Input:",
                        str(failure.input),
                        "Expected:",
                        str(failure.expected),
                        "Actual:",
                        str(failure.actual),
                        "Stderr:",
                        str(failure.stderr),
                    ]
                )
            )
        parts.append(
            "\n".join(
                [
                    "Repair requirements:",
                    "- Preserve the existing solve(...) signature.",
                    "- Preserve main() and the if-main guard.",
                    "- Do not read stdin inside solve(...).",
                    "- Do not print inside solve(...).",
                    "- Return complete corrected Python code only.",
                ]
            )
        )
        return "\n\n".join(parts)
