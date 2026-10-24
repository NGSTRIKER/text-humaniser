"""Parse and validate structured LLM output (Stage 7, plan §11).

Small local models often wrap JSON in markdown fences, emit ``<think>``
blocks, or add chatter.  :func:`parse_response` tolerates that, then
validates the object against the schema and the request's hard limits.
Model explanations are recorded as *claims*; they are not trusted.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from humanizer.llm_request import LLMRewriteRequest

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_UNCLOSED = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class OutputFormatError(ValueError):
    """The model reply could not be turned into a valid structured result."""


@dataclass
class ParsedLLMOutput:
    rewritten_text: str
    applied: list[str] = field(default_factory=list)
    rejected: list[dict[str, str]] = field(default_factory=list)
    uncertain: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _extract_json(raw: str) -> Any:
    text = _THINK.sub("", raw)
    text = _THINK_UNCLOSED.sub("", text).strip()
    text = _FENCE.sub("", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start == -1:
        raise OutputFormatError("reply contains no JSON object")
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
        return obj
    except json.JSONDecodeError as exc:
        raise OutputFormatError(f"invalid JSON: {exc.msg}") from exc


def _str_list(v: Any, name: str) -> list[str]:
    if v is None:
        return []
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise OutputFormatError(f"`{name}` must be a list of strings")
    return v


def parse_response(raw: str, request: LLMRewriteRequest) -> ParsedLLMOutput:
    """Validate ``raw`` against the schema and the request's limits.

    Raises :class:`OutputFormatError` for anything that must be retried:
    bad JSON, wrong types, empty text, length over the limit, or a
    protected string that went missing.  Softer problems become warnings.
    """
    obj = _extract_json(raw)
    if not isinstance(obj, dict):
        raise OutputFormatError("reply must be a JSON object")
    text = obj.get("rewritten_text")
    if not isinstance(text, str) or not text.strip():
        raise OutputFormatError("`rewritten_text` missing or empty")
    text = text.strip()

    rejected_raw = obj.get("rejected") or []
    if not isinstance(rejected_raw, list):
        raise OutputFormatError("`rejected` must be a list")
    rejected = []
    for r in rejected_raw:
        if isinstance(r, dict) and isinstance(r.get("id"), str):
            rejected.append({"id": r["id"], "reason": str(r.get("reason", ""))})
        elif isinstance(r, str):
            rejected.append({"id": r, "reason": ""})
        else:
            raise OutputFormatError("`rejected` items must be {id, reason}")

    out = ParsedLLMOutput(
        rewritten_text=text,
        applied=_str_list(obj.get("applied"), "applied"),
        rejected=rejected,
        uncertain=_str_list(obj.get("uncertain"), "uncertain"),
    )

    if request.max_output_chars and len(text) > request.max_output_chars:
        raise OutputFormatError(f"output {len(text)} chars exceeds limit {request.max_output_chars}")
    missing = [p for p in request.protected if p not in text and p.lower() not in text.lower()]
    if missing:
        raise OutputFormatError(f"protected text missing: {missing}")

    known = {s["id"] for s in request.suggestions}
    unknown = [i for i in out.applied + [r["id"] for r in out.rejected] if i not in known]
    if unknown:
        out.warnings.append(f"model referenced unknown suggestion ids {unknown}")
    if len(text) < 0.5 * len(request.text):
        out.warnings.append("output is less than half the input length; details may be lost")
    return out


REPAIR_INSTRUCTION = (
    "Your previous reply was rejected: {error}. Reply again with ONLY one JSON object "
    "matching output_schema. Keep all protected strings exactly and stay within max_output_chars."
)
