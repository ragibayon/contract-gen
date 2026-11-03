from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ContractType = Literal[
    "precondition",
    "postcondition",
    "invariant",
    "resource_bound",
]


class RawNLContract(BaseModel):
    """Natural-language contract extracted directly by the LLM."""

    model_config = ConfigDict(extra="ignore")

    type: ContractType = Field(description="Design-by-contract obligation type.")

    text: str = Field(description="Natural-language contract statement.")

    evidence: str = Field(description="Task evidence that grounds this contract.")


class NLContract(BaseModel):
    """Normalized contract enriched by the runtime."""

    id: str = Field(description="Deterministic contract identifier.")

    type: ContractType = Field(description="Normalized contract type.")

    text: str = Field(description="Normalized contract statement.")

    evidence: str = Field(description="Grounding evidence from the task.")


class RawSpecification(BaseModel):
    """Specification produced directly by the LLM."""

    model_config = ConfigDict(extra="ignore")

    summary: str = Field(description="One-sentence task summary.")

    contracts: list[RawNLContract] = Field(
        default_factory=list, description="Extracted NLContract inventory."
    )


class Specification(BaseModel):
    """Normalized specification produced by the runtime."""

    task_id: str = Field(description="Unique task identifier.")

    entry_point: str | None = Field(
        default=None, description="Target function name if available."
    )

    summary: str = Field(description="Normalized task summary.")

    contracts: list[NLContract] = Field(
        default_factory=list, description="Normalized NLContract inventory."
    )
