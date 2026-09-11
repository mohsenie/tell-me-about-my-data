"""Parse natural time expressions to an epoch instant, relative to the DATA.

"now"/"currently" -> latest data timestamp (NOT wall clock — data may be historic).
"yesterday" / "2 days ago" / "at 13:00 yesterday" / "13:00" -> resolved against
the data's own latest date. Returns an epoch (float) or None if not parseable.

Deterministic string/date math; no LLM. The LLM extracts the phrase, this maps it.
"""
from __future__ import annotations

import calendar
import datetime as dt
import re


def _to_epoch(d: dt.datetime) -> float:
    """Naive datetime treated as UTC -> epoch (avoids local-timezone shift)."""
    return float(calendar.timegm(d.timetuple())) + d.microsecond / 1e6


def parse_instant(text: str, data_min: float, data_max: float) -> float | None:
    """Resolve a time phrase in `text` to an epoch, within [data_min, data_max].

    Reference "now" = data_max (the latest reading), so "yesterday" is relative to
    the data, not today's wall clock. All datetimes are treated as UTC.
    """
    low = (text or "").lower()
    ref = dt.datetime.utcfromtimestamp(data_max)

    # base day offset
    day = ref
    abs_date = False
    # absolute ISO date "2026-09-02" (optionally "on 2026-09-02") sets the day.
    m = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", low)
    if m:
        try:
            day = dt.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            abs_date = True
        except ValueError:
            pass
    if not abs_date:
        if "yesterday" in low:
            day = ref - dt.timedelta(days=1)
        elif "today" in low or "now" in low or "current" in low:
            day = ref
        m = re.search(r"(\d+)\s*days?\s*ago", low)
        if m:
            day = ref - dt.timedelta(days=int(m.group(1)))
    m = re.search(r"(\d+)\s*hours?\s*ago", low)
    if m:
        inst = ref - dt.timedelta(hours=int(m.group(1)))
        return _clamp(_to_epoch(inst), data_min, data_max)

    # explicit clock time "13:00" / "1pm" / "at 9"
    hh = mm = None
    m = re.search(r"\b(\d{1,2}):(\d{2})\b", low)
    if m:
        hh, mm = int(m.group(1)), int(m.group(2))
    else:
        m = re.search(r"\b(\d{1,2})\s*(am|pm)\b", low)
        if m:
            hh = int(m.group(1)) % 12 + (12 if m.group(2) == "pm" else 0)
            mm = 0
        elif re.search(r"\bat\s+(\d{1,2})\b", low):
            hh = int(re.search(r"\bat\s+(\d{1,2})\b", low).group(1)); mm = 0

    if hh is not None:
        inst = day.replace(hour=hh, minute=mm or 0, second=0, microsecond=0)
        return _clamp(_to_epoch(inst), data_min, data_max)

    # a day mentioned but no clock time -> noon of that day (a reasonable instant)
    if abs_date or any(k in low for k in ("yesterday", "today", "days ago",
                                          "now", "current")):
        inst = day.replace(hour=12, minute=0, second=0, microsecond=0)
        return _clamp(_to_epoch(inst), data_min, data_max)

    return None


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))
