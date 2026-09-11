"""Parse natural time expressions to an epoch instant, with a HYBRID anchor.

Relative phrases ("2 days ago", "yesterday", "now") anchor to the WALL CLOCK when
the data is live (latest reading within ~48h of now), and to the DATA's latest
reading when the data is historic (so the phrase still lands inside the data
instead of a range with no rows). parse_instant_explained also returns a short note
saying which anchor was used, so a historic resolution is never a silent surprise.
Absolute dates ("2026-09-02") and clock times ("13:00") are taken as-is.

Deterministic string/date math; no LLM. The LLM extracts the phrase, this maps it.
"""
from __future__ import annotations

import calendar
import datetime as dt
import re


def _to_epoch(d: dt.datetime) -> float:
    """Naive datetime treated as UTC -> epoch (avoids local-timezone shift)."""
    return float(calendar.timegm(d.timetuple())) + d.microsecond / 1e6


# if the latest reading is within this many hours of the wall clock, the data is
# "live" and relative phrases ("2 days ago") anchor to NOW; otherwise the data is
# historic and they anchor to the latest reading (so they still land in-range).
_FRESH_HOURS = 48


def _reference(data_max: float, now: float | None) -> tuple[dt.datetime, bool]:
    """Pick the anchor for relative phrases: wall-clock NOW when the data is fresh
    (latest reading within _FRESH_HOURS of now), else the data's latest reading.
    Returns (reference_datetime_utc, anchored_to_now)."""
    now = now if now is not None else _to_epoch(dt.datetime.utcnow())
    fresh = (now - data_max) <= _FRESH_HOURS * 3600
    if fresh:
        return dt.datetime.utcfromtimestamp(now), True
    return dt.datetime.utcfromtimestamp(data_max), False


def parse_instant(text: str, data_min: float, data_max: float,
                  now: float | None = None) -> float | None:
    """Resolve a time phrase in `text` to an epoch, within [data_min, data_max].

    HYBRID anchor: if the data is LIVE (latest reading within ~48h of the wall
    clock) relative phrases like "2 days ago" are relative to NOW; if the data is
    HISTORIC they're relative to the latest reading (so they still land in-range).
    All datetimes are treated as UTC. See parse_instant_explained for the anchor
    note."""
    epoch, _note = parse_instant_explained(text, data_min, data_max, now=now)
    return epoch


def parse_instant_explained(text: str, data_min: float, data_max: float,
                            now: float | None = None) -> tuple[float | None, str]:
    """Same as parse_instant but also returns a short human note explaining which
    anchor was used (empty string when nothing was parsed). Lets callers tell the
    user 'X days ago = <date> (relative to the latest reading, as the data isn't
    live)' so the resolution is never a surprise."""
    low = (text or "").lower()
    ref, anchored_now = _reference(data_max, now)

    # base day offset
    day = ref
    abs_date = False
    relative = False           # used a relative phrase (days/hours ago, yesterday)
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
            day = ref - dt.timedelta(days=1); relative = True
        elif "today" in low or "now" in low or "current" in low:
            day = ref; relative = True
        m = re.search(r"(\d+)\s*days?\s*ago", low)
        if m:
            day = ref - dt.timedelta(days=int(m.group(1))); relative = True

    def _final(inst):
        epoch = _clamp(_to_epoch(inst), data_min, data_max)
        note = ""
        # explain the anchor only for a RELATIVE phrase on HISTORIC data — that's
        # the surprising case (the number isn't relative to today's wall clock).
        if relative and not anchored_now and not abs_date:
            d = dt.datetime.utcfromtimestamp(epoch).strftime("%Y-%m-%d")
            latest = ref.strftime("%Y-%m-%d")
            note = (f"(interpreted relative to the latest reading on {latest} — the "
                    f"data isn't live — so that resolves to {d}.)")
        return epoch, note

    m = re.search(r"(\d+)\s*hours?\s*ago", low)
    if m:
        relative = True
        inst = ref - dt.timedelta(hours=int(m.group(1)))
        return _final(inst)

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
        return _final(inst)

    # a day mentioned but no clock time -> noon of that day (a reasonable instant)
    if abs_date or any(k in low for k in ("yesterday", "today", "days ago",
                                          "now", "current")):
        inst = day.replace(hour=12, minute=0, second=0, microsecond=0)
        return _final(inst)

    return None, ""


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))
