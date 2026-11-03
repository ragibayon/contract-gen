from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

PROMPT_ROOT = Path(__file__).resolve().parents[1] / "agent" / "prompts"


_env = Environment(
    loader=FileSystemLoader(str(PROMPT_ROOT)),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


_PROMPT_VARIANT_ALIASES = {
    "v1": "stripped",
    "stripped": "stripped",
    "minimal": "stripped",
    "v2": "fewshot",
    "fewshot": "fewshot",
    "few-shot": "fewshot",
    "examples": "fewshot",
    "example": "fewshot",
    "v3": "cot",
    "cot": "cot",
    "chain_of_thought": "cot",
    "chain-of-thought": "cot",
    "v4": "contract",
    "contract": "contract",
    "contract_full": "contract",
    "full": "contract",
    "current": "contract",
    "selfcheck": "contract",
    "self-check": "contract",
}


def normalize_prompt_variant(variant: str | None) -> str:
    """Normalize prompt-ablation variant names used by the batch runner."""
    key = (variant or "contract").strip().lower().replace(" ", "_")
    try:
        return _PROMPT_VARIANT_ALIASES[key]
    except KeyError as exc:
        valid = ", ".join(sorted(_PROMPT_VARIANT_ALIASES))
        raise ValueError(f"Unknown prompt variant {variant!r}. Valid aliases: {valid}") from exc


def prompt_variant_flags(variant: str | None) -> dict[str, bool]:
    """Return Jinja switches for prompting-technique ablations.

    Variants:
    - stripped: rules only; no examples, no private reasoning, no self-check.
    - fewshot: rules + examples only.
    - cot: rules + examples + private reasoning.
    - contract: rules + examples + private reasoning + self-check/current behavior.
    """
    normalized = normalize_prompt_variant(variant)
    return {
        "include_examples": normalized in {"fewshot", "cot", "contract"},
        "include_cot": normalized in {"cot", "contract"},
        "include_selfcheck": normalized == "contract",
    }


def render_prompt(template_name: str, **context: Any) -> str:
    variant = normalize_prompt_variant(context.get("prompt_variant"))
    defaults: dict[str, Any] = {
        "prompt_variant": variant,
        **prompt_variant_flags(variant),
    }
    defaults.update(context)
    template = _env.get_template(template_name)
    return template.render(**defaults).strip()
