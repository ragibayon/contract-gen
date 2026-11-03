from __future__ import annotations

import ast
import time
from dataclasses import dataclass, field
from typing import Any

from contract.agent.schema.code import CodeArtifact
from contract.agent.schema.executable_contract import ExecutableContractArtifact
from contract.agent.schema.merged_contract import MergedContractArtifact
from contract.eval.execution import ExecutionWitnessReport, evaluate_execution_witnesses
from contract.utils.logger import logger


@dataclass
class MergeContractStageResult:
    name: str
    output: Any = None
    error: str | None = None
    static_validation_passed: bool = False
    execution_witness_passed: bool | None = None
    witness_report: ExecutionWitnessReport | None = None
    attempts: int = 1
    elapsed_sec: float = 0.0
    refinements: list[dict[str, Any]] = field(default_factory=list)


class MergeContractStage:
    def __init__(self, execution_timeout_sec: float = 5.0):
        self.execution_timeout_sec = max(0.1, float(execution_timeout_sec))

    def run(
        self,
        code: CodeArtifact,
        executable_contract: ExecutableContractArtifact,
        public_witnesses: dict[str, list[Any]] | None = None,
    ) -> MergeContractStageResult:
        started = time.time()
        witness_count = min(
            len((public_witnesses or {}).get("inputs", [])),
            len((public_witnesses or {}).get("outputs", [])),
        )
        logger.info(f"merge_contract: merging artifacts ({witness_count} public witnesses)")
        try:
            merged = self._merge(
                code=code.code,
                imports=executable_contract.imports,
                decorators=executable_contract.decorators,
            )
            self._validate_static(merged)
            logger.success("merge_contract: static validation passed")

            witness_report = None
            has_public_witnesses = bool(
                public_witnesses and public_witnesses.get("inputs") and public_witnesses.get("outputs")
            )
            if has_public_witnesses:
                logger.info(f"merge_contract: evaluating {witness_count} public witnesses")
                witness_report = evaluate_execution_witnesses(
                    code=merged,
                    witnesses=public_witnesses or {},
                    timeout_sec=self.execution_timeout_sec,
                )
                if witness_report.pass_ratio < 1.0:
                    result = MergeContractStageResult(
                        name="merge_contract",
                        output=MergedContractArtifact(code=merged, coverage=executable_contract.coverage),
                        error="Merged contract code violated public execution witnesses.",
                        static_validation_passed=True,
                        execution_witness_passed=False,
                        witness_report=witness_report,
                        elapsed_sec=round(time.time() - started, 3),
                    )
                    logger.stage_finished(
                        "merge_contract",
                        False,
                        result.elapsed_sec,
                        {
                            "Static validation": "passed",
                            "Execution witnesses": f"{witness_report.passed}/{witness_report.total} passed",
                            "Error": result.error,
                        },
                    )
                    return result
                logger.success(
                    "merge_contract: public witnesses passed "
                    f"({witness_report.passed}/{witness_report.total})"
                )

            result = MergeContractStageResult(
                name="merge_contract",
                output=MergedContractArtifact(code=merged, coverage=executable_contract.coverage),
                static_validation_passed=True,
                execution_witness_passed=(witness_report.pass_ratio == 1.0 if witness_report else None),
                witness_report=witness_report,
                elapsed_sec=round(time.time() - started, 3),
            )
            logger.stage_finished(
                "merge_contract",
                True,
                result.elapsed_sec,
                {
                    "Static validation": "passed",
                    "Execution witnesses": (
                        "not provided" if witness_report is None else f"{witness_report.passed}/{witness_report.total} passed"
                    ),
                },
            )
            return result

        except Exception as exc:
            result = MergeContractStageResult(
                name="merge_contract",
                output=None,
                error=f"{type(exc).__name__}: {exc}",
                static_validation_passed=False,
                elapsed_sec=round(time.time() - started, 3),
            )
            logger.stage_finished(
                "merge_contract",
                False,
                result.elapsed_sec,
                {"Static validation": "failed", "Error": result.error},
            )
            return result

    @staticmethod
    def _merge(code: str, imports: str, decorators: str) -> str:
        code = code.strip() + "\n"
        imports = imports.strip()
        decorators = decorators.strip()
        if not decorators:
            raise ValueError("No decorators to merge.")

        lines = code.splitlines()
        solve_idx = None
        for i, line in enumerate(lines):
            if line.startswith("def solve("):
                solve_idx = i
                break
        if solve_idx is None:
            raise ValueError("No top-level solve(...) function found.")

        existing_imports: list[str] = []
        body_lines: list[str] = []
        for line in lines:
            if line.startswith("import ") or line.startswith("from "):
                existing_imports.append(line)
            else:
                body_lines.append(line)

        merged_imports: list[str] = []
        seen: set[str] = set()
        for line in imports.splitlines() + existing_imports:
            clean = line.strip()
            if clean and clean not in seen:
                merged_imports.append(clean)
                seen.add(clean)

        body_lines = [line for line in body_lines if line.strip()]
        for i, line in enumerate(body_lines):
            if line.startswith("def solve("):
                solve_idx = i
                break
        else:
            raise ValueError("No top-level solve(...) function found after import split.")

        decorator_lines = decorators.splitlines()
        merged_body = body_lines[:solve_idx] + decorator_lines + body_lines[solve_idx:]
        return "\n".join(merged_imports + [""] + merged_body).rstrip() + "\n"

    @staticmethod
    def _validate_static(code: str) -> None:
        tree = ast.parse(code)
        compiled = compile(tree, "<merged_contract_code>", "exec")
        exec(compiled, {"__name__": "__contract_validation__"})
