from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


LoweringKind = Literal["require", "ensure", "omitted"]


class ContractLoweringItem(BaseModel):
    """Trace from one NLContract to the executable artifact."""

    model_config = ConfigDict(extra="ignore")

    contract_id: str = Field(description="Exact NLContract ID.")
    lowered_to: LoweringKind = Field(
        description="One of: require, ensure, omitted."
    )
    expression: str = Field(
        default="",
        description="Decorator string when lowered; empty string when omitted.",
    )
    reason: str = Field(
        default="",
        description="Short reason for the lowering decision.",
    )


class RawExecutableContractArtifact(BaseModel):
    """LLM-generated executable contract decorators."""

    model_config = ConfigDict(extra="ignore")

    imports: str = Field(
        description="Python import lines for decorators; must include icontract."
    )
    decorators: str = Field(
        description="Only icontract decorator lines for solve(...); no function body."
    )
    coverage: list[ContractLoweringItem] = Field(
        default_factory=list,
        description="One lowering item for each NLContract ID.",
    )


class ExecutableContractArtifact(BaseModel):
    """Validated executable contract decorators."""

    imports: str
    decorators: str
    coverage: list[ContractLoweringItem] = Field(default_factory=list)
