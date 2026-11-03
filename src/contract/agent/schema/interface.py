from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RawProgramInterface(BaseModel):
    """LLM-generated canonical Python skeleton."""

    model_config = ConfigDict(extra="ignore")

    skeleton: str = Field(
        description=(
            "Complete Python skeleton string containing solve(...), main(), "
            "and the if-main guard."
        )
    )


class ProgramInterface(BaseModel):
    """Validated canonical Python skeleton."""

    skeleton: str = Field(description="Validated Python skeleton string.")
