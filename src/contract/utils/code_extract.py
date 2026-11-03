from __future__ import annotations

import json
import re
from typing import Any

from contract.utils.json_extract import extract_json_object


_CODE_FENCE_RE = re.compile(
    r"```(?:python|py)?\s*(.*?)\s*```",
    flags=re.IGNORECASE | re.DOTALL,
)
_ANY_FENCE_RE = re.compile(
    r"```\s*(.*?)\s*```",
    flags=re.DOTALL,
)


def extract_python_code(text: str) -> str:
    """Extract Python code from common LLM response formats.

    Accepted formats:
    - pure Python code
    - fenced Python code
    - generic fenced code
    - JSON object containing a `code`, `program`, or `solution` string
    - explanatory text followed by code
    """

    if text is None:
        raise ValueError("No model output to parse as Python code.")

    original = str(text).strip()
    if not original:
        raise ValueError("Model output is empty; no Python code found.")


    try:
        data = extract_json_object(original)
        if isinstance(data, dict):
            for key in ("code", "program", "solution", "python", "source"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return _cleanup_code(value)
    except Exception:
        pass


    python_blocks = _CODE_FENCE_RE.findall(original)
    if python_blocks:
        return _cleanup_code(max(python_blocks, key=len))


    any_blocks = _ANY_FENCE_RE.findall(original)
    if any_blocks:
        return _cleanup_code(max(any_blocks, key=len))


    return _cleanup_code(_strip_to_likely_python(original))


def response_to_text(response: Any) -> str:
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(str(item["content"]))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False, default=str))
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()

    return str(content).strip()


def _cleanup_code(code: str) -> str:
    code = str(code).strip()


    if code.lower().startswith("python\n"):
        code = code.split("\n", 1)[1].strip()
    if code.lower().startswith("py\n"):
        code = code.split("\n", 1)[1].strip()

    code = _strip_to_likely_python(code)

    if not code.endswith("\n"):
        code += "\n"

    return code


def _strip_to_likely_python(text: str) -> str:
    lines = text.strip().splitlines()
    if not lines:
        return ""


    first_index = 0
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if (
            stripped.startswith("import ")
            or stripped.startswith("from ")
            or stripped.startswith("def ")
            or stripped.startswith("class ")
            or stripped.startswith("@")
        ):
            first_index = index
            break

    lines = lines[first_index:]


    stop_markers = (
        "Explanation:",
        "Notes:",
        "Why this works:",
        "Complexity:",
        "Time complexity:",
        "Space complexity:",
        "The code above",
    )
    trimmed: list[str] = []
    for line in lines:
        if line.strip().startswith(stop_markers):
            break
        trimmed.append(line)

    return "\n".join(trimmed).strip()
