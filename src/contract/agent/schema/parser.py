from __future__ import annotations

import json
import re
from typing import Any

from contract.agent.schema.specification import RawSpecification


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")

    return json.loads(cleaned[start : end + 1])


def parse_raw_specification(text: str) -> RawSpecification:
    obj = extract_json_object(text)
    return RawSpecification.model_validate(obj)
