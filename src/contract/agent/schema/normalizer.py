from __future__ import annotations

from contract.agent.schema.specification import (
    NLContract,
    RawSpecification,
    Specification,
)

PREFIX = {
    "precondition": "PRE",
    "postcondition": "POST",
    "invariant": "INV",
    "resource_bound": "NFR",
}


def _clean(text: str | None, limit: int = 300) -> str:
    return " ".join((text or "").split()).strip()[:limit]


def normalize_spec(
    raw: RawSpecification,
    task_id: str,
    entry_point: str | None,
) -> Specification:
    counters: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()

    contracts: list[NLContract] = []

    for contract in raw.contracts:
        text = _clean(contract.text)
        if not text:
            continue

        key = (contract.type, text.lower())
        if key in seen:
            continue
        seen.add(key)

        prefix = PREFIX[contract.type]
        counters[prefix] = counters.get(prefix, 0) + 1

        contracts.append(
            NLContract(
                id=f"NLC-{prefix}-{counters[prefix]:03d}",
                type=contract.type,
                text=text,
                evidence=_clean(contract.evidence),
            )
        )

    return Specification(
        task_id=task_id,
        entry_point=entry_point,
        summary=_clean(raw.summary, 500),
        contracts=contracts,
    )
