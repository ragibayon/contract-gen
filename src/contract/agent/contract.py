from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from contract.agent.schema.code import CodeArtifact
from contract.agent.schema.executable_contract import ExecutableContractArtifact
from contract.agent.schema.interface import ProgramInterface
from contract.agent.schema.specification import Specification
from contract.agent.stages.code_generation import CodeGenerationStage, CodeStageResult
from contract.agent.stages.contract_probe import (
    ContractProbeStage,
    ContractProbeStageResult,
)
from contract.agent.stages.executable_contract import (
    ExecutableContractStage,
    ExecutableContractStageResult,
)
from contract.agent.stages.interface import InterfaceStage, InterfaceStageResult
from contract.agent.stages.merge_contract import (
    MergeContractStage,
    MergeContractStageResult,
)
from contract.agent.stages.specification import AgentStageResult, SpecificationStage
from contract.llm.client import LLMClient
from contract.utils.logger import logger
from contract.utils.prompt_renderer import normalize_prompt_variant


@dataclass
class ContractAgentResult:
    task_id: str
    specification: AgentStageResult
    interface: InterfaceStageResult | None = None
    code: CodeStageResult | None = None
    executable_contract: ExecutableContractStageResult | None = None
    merged_contract: MergeContractStageResult | None = None
    contract_probe: ContractProbeStageResult | None = None


class ContractAgent:
    def __init__(
        self,
        llm: LLMClient | None = None,
        max_attempts: int = 3,
        output_structure: str = "raw",
        prompt_variant: str = "contract",
        code_refinement_attempts: int = 3,
        execution_timeout_sec: float = 5.0,
    ):
        self.llm = llm or LLMClient()
        self.max_attempts = max(1, max_attempts)
        self.code_refinement_attempts = max(0, code_refinement_attempts)
        self.execution_timeout_sec = max(0.1, float(execution_timeout_sec))
        self.output_structure = self._normalize_output_structure(output_structure)
        self.prompt_variant = normalize_prompt_variant(prompt_variant)

        try:
            self.specification_stage = SpecificationStage(
                self.llm,
                self.max_attempts,
                output_structure=self.output_structure,
                prompt_variant=self.prompt_variant,
            )
        except TypeError:
            self.specification_stage = SpecificationStage(
                self.llm,
                self.max_attempts,
            )

        try:
            self.interface_stage = InterfaceStage(
                self.llm,
                self.max_attempts,
                output_structure=self.output_structure,
                prompt_variant=self.prompt_variant,
            )
        except TypeError:
            self.interface_stage = InterfaceStage(
                self.llm,
                self.max_attempts,
                prompt_variant=self.prompt_variant,
            )

        self.code_stage = CodeGenerationStage(
            self.llm,
            self.max_attempts,
            output_structure=self.output_structure,
            prompt_variant=self.prompt_variant,
            max_refinement_attempts=self.code_refinement_attempts,
            execution_timeout_sec=self.execution_timeout_sec,
        )

        try:
            self.executable_contract_stage = ExecutableContractStage(
                self.llm,
                self.max_attempts,
                output_structure=self.output_structure,
                prompt_variant=self.prompt_variant,
            )
        except TypeError:
            self.executable_contract_stage = ExecutableContractStage(
                self.llm,
                self.max_attempts,
                prompt_variant=self.prompt_variant,
            )

        self.merge_contract_stage = MergeContractStage(
            execution_timeout_sec=self.execution_timeout_sec
        )

        try:
            self.contract_probe_stage = ContractProbeStage(
                self.llm,
                self.max_attempts,
                output_structure=self.output_structure,
                prompt_variant=self.prompt_variant,
            )
        except TypeError:
            self.contract_probe_stage = ContractProbeStage(
                self.llm,
                self.max_attempts,
                prompt_variant=self.prompt_variant,
            )

    def run(
        self,
        task_id: str,
        prompt: str,
        entry_point: str | None = None,
        public_witnesses: dict[str, list[Any]] | None = None,
        cached_specification: Specification | None = None,
        cached_interface: ProgramInterface | None = None,
        cached_code: CodeArtifact | None = None,
        cached_executable_contract: ExecutableContractArtifact | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ContractAgentResult:
        cached_stages = [
            name
            for name, value in (
                ("specification", cached_specification),
                ("interface", cached_interface),
                ("code_generation", cached_code),
                ("executable_contract", cached_executable_contract),
            )
            if value is not None
        ]

        run_mode = "resume" if cached_stages else "fresh"

        witness_count = min(
            len((public_witnesses or {}).get("inputs", [])),
            len((public_witnesses or {}).get("outputs", [])),
        )

        llm_config = getattr(self.llm, "config", None)

        run_metadata: dict[str, Any] = dict(metadata or {})
        run_metadata.update(
            {
                "Mode": run_mode,
                "Provider": getattr(llm_config, "provider", "unknown"),
                "Model": getattr(llm_config, "model", "unknown"),
                "Max attempts": self.max_attempts,
                "Code refinement attempts": self.code_refinement_attempts,
                "Public test timeout": self.execution_timeout_sec,
                "Output structure": self.output_structure,
                "Prompt variant": self.prompt_variant,
                "Public witnesses": witness_count,
                "Cached stages": ", ".join(cached_stages) or "none",
            }
        )

        logger.rule("Contract agent run")
        logger.task_header(
            task_id=task_id,
            entry_point=entry_point,
            task=prompt,
            metadata=run_metadata,
        )

        specification = self._run_specification(
            task_id,
            prompt,
            entry_point,
            cached_specification,
        )
        if specification.output is None:
            return ContractAgentResult(
                task_id=task_id,
                specification=specification,
            )

        interface = self._run_interface(
            prompt,
            specification.output,
            cached_interface,
        )
        if interface.output is None:
            return ContractAgentResult(
                task_id=task_id,
                specification=specification,
                interface=interface,
            )

        code = self._run_code(
            prompt,
            specification.output,
            interface.output,
            public_witnesses,
            cached_code,
        )
        if code.output is None or not code.static_validation_passed:
            return ContractAgentResult(
                task_id=task_id,
                specification=specification,
                interface=interface,
                code=code,
            )

        executable_contract = self._run_executable_contract(
            prompt,
            specification.output,
            interface.output,
            cached_executable_contract,
        )
        if (
            executable_contract.output is None
            or not executable_contract.static_validation_passed
        ):
            return ContractAgentResult(
                task_id=task_id,
                specification=specification,
                interface=interface,
                code=code,
                executable_contract=executable_contract,
            )

        logger.stage_started("merge_contract", mode="generated")
        merged_contract = self.merge_contract_stage.run(
            code=code.output,
            executable_contract=executable_contract.output,
            public_witnesses=public_witnesses,
        )
        if (
            merged_contract.output is None
            or not merged_contract.static_validation_passed
        ):
            return ContractAgentResult(
                task_id=task_id,
                specification=specification,
                interface=interface,
                code=code,
                executable_contract=executable_contract,
                merged_contract=merged_contract,
            )





        if merged_contract.execution_witness_passed is False:
            logger.warning(
                "contract_probe: merged contract failed public witnesses; "
                "continuing self-probes for code-only and contracted-code evidence."
            )

        logger.stage_started("contract_probe", mode="generated")
        contract_probe = self.contract_probe_stage.run(
            task_prompt=prompt,
            specification=specification.output,
            interface=interface.output,
            generated_code=code.output,
            merged_contract=merged_contract.output,
            public_witnesses=public_witnesses,
        )

        if (
            contract_probe.probe_report is not None
            and contract_probe.probe_report.total > 0
        ):
            logger.success(f"Contract agent run completed for task {task_id}")

        return ContractAgentResult(
            task_id=task_id,
            specification=specification,
            interface=interface,
            code=code,
            executable_contract=executable_contract,
            merged_contract=merged_contract,
            contract_probe=contract_probe,
        )

    def _run_specification(
        self,
        task_id: str,
        prompt: str,
        entry_point: str | None,
        cached_specification: Specification | None,
    ) -> AgentStageResult:
        mode = "cached" if cached_specification is not None else "generated"
        logger.stage_started("specification", mode=mode)

        if cached_specification is not None:
            result = AgentStageResult(
                name="specification",
                output=cached_specification,
                llm_metrics={"source": "cache"},
            )
            logger.stage_finished(
                "specification",
                True,
                0.0,
                {"Source": "cache"},
            )
            return result

        return self.specification_stage.run(
            task_id=task_id,
            prompt=prompt,
            entry_point=entry_point,
        )

    def _run_interface(
        self,
        prompt: str,
        specification: Specification,
        cached_interface: ProgramInterface | None,
    ) -> InterfaceStageResult:
        mode = "cached" if cached_interface is not None else "generated"
        logger.stage_started("interface", mode=mode)

        if cached_interface is not None:
            result = InterfaceStageResult(
                name="interface",
                output=cached_interface,
                llm_metrics={"source": "cache"},
            )
            logger.stage_finished(
                "interface",
                True,
                0.0,
                {"Source": "cache"},
            )
            return result

        return self.interface_stage.run(
            task_prompt=prompt,
            specification=specification,
        )

    def _run_code(
        self,
        prompt: str,
        specification: Specification,
        interface: ProgramInterface,
        public_witnesses: dict[str, list[Any]] | None,
        cached_code: CodeArtifact | None,
    ) -> CodeStageResult:
        mode = "cached" if cached_code is not None else "generated"
        logger.stage_started("code_generation", mode=mode)

        if cached_code is not None:
            result = CodeStageResult(
                name="code_generation",
                output=cached_code,
                attempts=0,
                static_validation_passed=True,
                llm_metrics={"source": "cache"},
            )
            logger.stage_finished(
                "code_generation",
                True,
                0.0,
                {
                    "Source": "cache",
                    "Static validation": "assumed passed",
                },
            )
            return result

        return self.code_stage.run(
            task_prompt=prompt,
            specification=specification,
            interface=interface,
            public_witnesses=public_witnesses,
        )

    def _run_executable_contract(
        self,
        prompt: str,
        specification: Specification,
        interface: ProgramInterface,
        cached_executable_contract: ExecutableContractArtifact | None,
    ) -> ExecutableContractStageResult:
        mode = "cached" if cached_executable_contract is not None else "generated"
        logger.stage_started("executable_contract", mode=mode)

        if cached_executable_contract is not None:
            result = ExecutableContractStageResult(
                name="executable_contract",
                output=cached_executable_contract,
                attempts=0,
                static_validation_passed=True,
                llm_metrics={"source": "cache"},
            )
            logger.stage_finished(
                "executable_contract",
                True,
                0.0,
                {
                    "Source": "cache",
                    "Static validation": "assumed passed",
                },
            )
            return result

        return self.executable_contract_stage.run(
            task_prompt=prompt,
            specification=specification,
            interface=interface,
        )

    @staticmethod
    def _normalize_output_structure(value: str) -> str:
        normalized = (value or "raw").strip().lower().replace("-", "_")

        if normalized in {"raw", "text", "manual"}:
            return "raw"

        if normalized in {"structured", "json", "schema"}:
            return "structured"

        raise ValueError(
            "output_structure must be either 'raw' or 'structured'; " f"got {value!r}."
        )
