from __future__ import annotations

from pydantic import BaseModel, Field

from contract.agent.schema.executable_contract import ContractLoweringItem


class MergedContractArtifact(BaseModel):
    """Generated code with executable contracts merged onto solve(...)."""

    code: str = Field(description="Complete Python program with contract decorators.")
    coverage: list[ContractLoweringItem] = Field(
        default_factory=list,
        description=(
            "Trace showing which NLContract requirement IDs are covered by the "
            "merged executable decorators. Multiple NLContracts may be covered by "
            "one merged decorator expression."
        ),
    )
