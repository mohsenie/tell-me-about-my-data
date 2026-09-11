"""What can the user query? — describe available signals per source.

Reads the actual data schema (honest: only lists what exists) + basic ranges, so
the user knows the queryable fields. No LLM needed.
"""
from __future__ import annotations

import glob

import duckdb


_NUMERIC = ("DOUBLE", "FLOAT", "BIGINT", "INTEGER", "HUGEINT", "DECIMAL",
            "SMALLINT", "TINYINT")


def list_capabilities(source_globs: dict[str, str]) -> dict:
    """For each source with data: its numeric signals + observed min/max.

    source_globs: {source_name: glob}. Returns
      {source: {"signals": [{name, min, max, unit_hint}], "dates": [...]}}
    """
    con = duckdb.connect()
    out: dict = {}
    for src, pattern in source_globs.items():
        files = glob.glob(pattern)
        if not files:
            continue
        rel = f"read_parquet('{pattern}', union_by_name=true)"
        schema = con.execute(f"DESCRIBE SELECT * FROM {rel}").fetchall()
        cols = [r[0] for r in schema
                if any(r[1].upper().startswith(t) for t in _NUMERIC)
                and r[0].lower() != "timestamp"]
        signals, excluded = [], []
        for c in cols:
            try:
                mn, mx = con.execute(
                    f"SELECT min(\"{c}\"), max(\"{c}\") FROM {rel} WHERE \"{c}\" IS NOT NULL"
                ).fetchone()
            except Exception:
                mn = mx = None
            if mn is None:
                excluded.append({"name": c, "reason": "no data (all null)"})
            elif mn == mx:
                excluded.append({"name": c, "reason": f"constant ({mn:g})"})
            else:
                signals.append({"name": c, "min": float(mn), "max": float(mx)})
        out[src] = {
            "signals": signals, "signal_count": len(signals),
            "excluded": excluded, "total_fields": len(cols),
        }
    con.close()
    return out


def describe_capabilities(caps: dict) -> str:
    """Plain-language description of what can be queried."""
    if not caps:
        return "No queryable data found."
    lines = ["You can query these signals (values computed on request):", ""]
    for src, info in caps.items():
        ex = info.get("excluded", [])
        header = (f"**{src}** — {info['total_fields']} fields "
                  f"({info['signal_count']} queryable"
                  + (f", {len(ex)} not queryable" if ex else "") + "):")
        lines.append(header)
        for s in info["signals"]:
            lines.append(f"  - {s['name']}  (observed {s['min']:.1f} … {s['max']:.1f})")
        for e in ex:
            lines.append(f"  - {e['name']}  [not queryable: {e['reason']}]")
        lines.append("")
    examples = _example_requests(caps)
    if examples:
        lines.append("Example requests:")
        lines += [f'  - "{e}"' for e in examples]
    return "\n".join(lines)


def _example_requests(caps: dict) -> list[str]:
    """Build example queries from the ACTUAL queryable signals (data-agnostic —
    no hardcoded fuel/engine names). Picks real signals from the described sources
    and phrases avg/max/plot queries over a window."""
    picks = []
    for src, info in caps.items():
        for s in info.get("signals", []):
            picks.append(s["name"])
    if not picks:
        return []
    templates = ["average {sig} over the last 2 days",
                 "max {sig} yesterday",
                 "plot {sig} over the last 3 days"]
    return [t.format(sig=picks[i]) for i, t in enumerate(templates) if i < len(picks)]
