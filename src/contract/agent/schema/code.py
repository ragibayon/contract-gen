from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RawCodeArtifact(BaseModel):
    """LLM-generated code artifact."""

    model_config = ConfigDict(extra="ignore")

    code: str = Field(description="Complete Python program string.")


class CodeArtifact(BaseModel):
    """Validated generated Python code."""

    code: str = Field(description="Complete Python program after syntax validation.")
