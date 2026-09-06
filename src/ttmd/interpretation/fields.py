"""Field-semantics phase: LLM proposes what each data field means.

Given field NAMES + observed stats + USER-DECLARED protocol context (J1939 /
NMEA / AIS / custom ...), the LLM proposes for each field:
  - description, unit, aggregation (avg / integral-for-rates / etc.), confidence,
  - is_standard (does it map to a named standard signal, or is it custom/derived).

Discipline: these are PROPOSALS, not facts. Name-based meaning is the LOWEST-
confidence source (below documents, below expert). Stored in the knowledge base
as a distinct 'field_semantics' tier, expert-reviewable (promote/reject). Protocol
context (from user metadata) raises confidence for standard fields; custom/derived
fields stay low. Nothing is authoritative until an expert confirms it.
"""
from __future__ import annotations

import json
import re

from .provider import LLMProvider

_SYSTEM = """You identify the meaning of sensor data field names for a {asset_type}.
{protocol_line}

For EACH field name given (with its observed value range), propose:
- "description": one plain sentence of what it measures,
- "unit": the most likely unit (e.g. rpm, degC, L/h, kPa, knots, deg); "" if unknown,
- "aggregation": how a total/summary is computed — "integral" for RATES (per-hour
  flows), else "avg"/"max"/"min"/"sum" as appropriate,
- "is_standard": true if it maps to a named signal in the stated protocol, false
  if it looks custom/derived/proprietary,
- "confidence": "high" | "medium" | "low",
- "role": the SEMANTIC ROLE this field plays, for spatial/identity queries. One of:
    "latitude"    — a geographic latitude in degrees,
    "longitude"   — a geographic longitude in degrees,
    "identifier"  — a unique id for the entity/vessel/device this row is about
                    (e.g. an AIS MMSI, a device or asset id),
    "entity_name" — a human-readable name/label for that entity (e.g. a ship name
                    or callsign),
    "timestamp"   — the time of the observation,
    "none"        — anything else (most sensor measurements are "none").
  Assign a role ONLY when you're reasonably sure from the name/protocol/range;
  otherwise use "none". At most one field should be "latitude", one "longitude".

Rules:
- If a protocol is stated, use it: map names to that standard's signals and their
  DEFINED units, with higher confidence. But you may be misremembering a spec —
  never claim certainty; this is a PROPOSAL for expert review.
- Custom / derived / cryptic names -> is_standard false, low confidence.
- Do NOT invent fields not in the list.
Return ONLY a JSON array, one object per field, key "field" = the exact name."""

_USER = """Fields to identify:
{fields}
Return the JSON array."""


def describe_fields(source: str, columns_with_stats: list[dict],
                    asset_type: str, provider: LLMProvider,
                    protocol: str | None = None,
                    source_description: str | None = None) -> list[dict]:
    """Propose semantics for a source's fields. columns_with_stats: [{name,min,max}].

    Returns [{field, description, unit, aggregation, is_standard, confidence}].
    """
    if protocol:
        protocol_line = (f'These fields come from a "{protocol}" source'
                         + (f" ({source_description})" if source_description else "")
                         + ". Interpret names against that protocol's standard signals.")
    else:
        protocol_line = ("No protocol was declared for this source, so rely on the "
                         "field names alone and keep confidence low.")

    field_lines = "\n".join(
        f"  - {c['name']} (observed {c.get('min','?')}..{c.get('max','?')})"
        for c in columns_with_stats)
    system = _SYSTEM.format(asset_type=asset_type, protocol_line=protocol_line)

    # Scale the output budget to the field count: each field is a JSON object of
    # ~150 tokens. Without this, many-field sources (e.g. AIS with 20 columns)
    # truncate mid-array and parse to nothing.
    budget = max(1024, 200 * len(columns_with_stats) + 256)
    raw = provider.complete(system, _USER.format(fields=field_lines),
                            max_tokens=budget).strip()
    proposed = _parse_json_array(raw)

    # keep only fields we actually asked about; normalize
    valid_names = {c["name"] for c in columns_with_stats}
    out = []
    for item in proposed:
        name = item.get("field", "").strip()
        if name not in valid_names:
            continue
        role = (item.get("role") or "none").strip().lower()
        if role not in _ROLES:
            role = "none"
        out.append({
            "field": name,
            "description": (item.get("description") or "").strip(),
            "unit": (item.get("unit") or "").strip(),
            "aggregation": (item.get("aggregation") or "avg").strip(),
            "is_standard": bool(item.get("is_standard", False)),
            "confidence": (item.get("confidence") or "low").strip().lower(),
            "role": role,
            "protocol": protocol or "none",
        })
    return out


_ROLES = {"latitude", "longitude", "identifier", "entity_name", "timestamp", "none"}


def _parse_json_array(raw: str) -> list:
    # Normal case: a well-formed [...] array (possibly inside a ```json fence).
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, list):
                return data
        except Exception:
            pass
    # Salvage a TRUNCATED array (model hit the token limit before the closing
    # ']'): parse each complete top-level {...} object individually.
    start = raw.find("[")
    if start == -1:
        return []
    body = raw[start + 1:]
    out, depth, obj_start = [], 0, None
    for i, ch in enumerate(body):
        if ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start is not None:
                try:
                    out.append(json.loads(body[obj_start:i + 1]))
                except Exception:
                    pass
                obj_start = None
    return out
