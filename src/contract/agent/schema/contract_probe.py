from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ProbeType = Literal["oracle", "violation", "metamorphic", "predicate"]
ProbePriority = Literal["high", "low"]


class ContractProbe(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: ProbeType = Field(
        description=(
            "Probe category. Use: "
            "'oracle' when a concrete expected stdout is known; "
            "'violation' when the input should trigger a generated precondition/postcondition violation; "
            "'metamorphic' when two inputs should have a known relation between their outputs; "
            "'predicate' when one output should satisfy a simple property."
        )
    )

    input: str = Field(
        description=(
            "Primary stdin input for the program. Write exactly what should be passed to stdin, "
            "including newlines if needed. Keep it compact and executable."
        )
    )

    expected: str | list[str] | None = Field(
        default=None,
        description=(
            "Allowed stdout for oracle probes only. Use a string for one expected output, "
            "or a list of strings when multiple stdout formats are acceptable. "
            "Use null for violation, metamorphic, and predicate probes."
        ),
    )

    paired_input: str | None = Field(
        default=None,
        description=(
            "Second stdin input for metamorphic probes only. This is compared with input "
            "using relation. Use null for oracle, violation, and predicate probes."
        ),
    )

    relation: str = Field(
        default="",
        description=(
            "Simple relation or predicate to check. For violation probes, use "
            "'contract_violation'. For metamorphic probes, use simple names such as "
            "'same_output', 'different_output', or 'output_increases'. For predicate probes, "
            "use simple names such as 'nonempty_stdout', 'stdout_length_2', 'contains_value', "
            "or another short check name already supported by the evaluator. Leave empty only "
            "for oracle probes."
        ),
    )

    requirement_ids: list[str] = Field(
        default_factory=list,
        description=(
            "NLContract IDs exercised by this probe. Use exact IDs from the provided contracts, "
            "for example ['NLC-PRE-001'] or ['NLC-POST-002']. Do not invent IDs."
        ),
    )

    reason: str = Field(
        default="",
        description=(
            "One short sentence explaining why this probe is useful. Keep it brief. "
            "Example: 'Checks the lower boundary of the input range.'"
        ),
    )

    priority: ProbePriority = Field(
        default="low",
        description=(
            "Use 'high' for core public-like examples, boundary cases, direct precondition checks, "
            "or direct postcondition checks. Use 'low' for supplemental or weaker evidence."
        ),
    )


class RawContractProbePlan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    probes: list[ContractProbe] = Field(
        ...,
        min_length=1,
        description=(
            "Non-empty list of generated contract probes. Generate at least one executable probe. "
            "Prefer 4-6 compact probes when possible. Keep probes simple so small/open-source models "
            "can produce valid JSON."
        ),
    )


class ContractProbePlan(BaseModel):
    probes: list[ContractProbe] = Field(
        default_factory=list,
        description="Validated contract probes using the same schema as RawContractProbePlan.",
    )


class ContractProbeReport(BaseModel):
    passed: int
    total: int
    pass_ratio: float
    failures: list[dict] = Field(default_factory=list)
    cases: list[dict] = Field(default_factory=list)
    contract_violations: int = 0
    invalid_violation_probes: int = 0
    runtime_errors: int = 0
    timeouts: int = 0
    wrong_outputs: int = 0
    predicate_failures: int = 0
    relation_failures: int = 0
    nonzero_exits: int = 0
    supported_violation_probes: int = 0
    unsupported_violation_probes: int = 0
    uncovered_violation_requirement_ids: list[str] = Field(default_factory=list)
