from __future__ import annotations

import json
import re
from typing import Any


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = _strip_common_wrappers(text)

    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    candidates = _balanced_json_candidates(cleaned)
    last_error: Exception | None = None

    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError as exc:
            last_error = exc

    if last_error is not None:
        raise ValueError(f"No valid JSON object found. Last error: {last_error}")

    raise ValueError("No JSON object found in model output.")


def _strip_common_wrappers(text: str) -> str:
    text = text.strip()

    fence_match = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fence_match:
        text = fence_match.group(1).strip()


    prefixes = [
        "json",
        "JSON:",
        "Here is the JSON:",
        "Here is the requested JSON:",
    ]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()

    return text


def _balanced_json_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    starts = [i for i, ch in enumerate(text) if ch == "{"]

    for start in starts:
        depth = 0
        in_string = False
        escaped = False

        for index in range(start, len(text)):
            ch = text[index]

            if escaped:
                escaped = False
                continue

            if ch == "\\":
                escaped = True
                continue

            if ch == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1

                if depth == 0:
                    candidates.append(text[start : index + 1])
                    break


    candidates.sort(key=len, reverse=True)
    return candidates
