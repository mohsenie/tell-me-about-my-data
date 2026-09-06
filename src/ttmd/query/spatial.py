"""Positional / spatial queries (deterministic).

Answers "where is the ship now?" and "what ships were nearby at time T?" from
lat/lon data. Column names (lat/lon/id/timestamp) are PARAMETERS resolved by the
caller from field semantics — nothing is hardcoded to specific field names.
"""
from __future__ import annotations

import math

import duckdb


def _src(globs: list[str]) -> str:
    arr = ", ".join("'" + g + "'" for g in globs)
    return f"read_parquet([{arr}], union_by_name=true)"


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * r * math.asin(math.sqrt(a))


def current_position(globs: list[str], lat_col: str, lon_col: str) -> dict:
    """Most-recent lat/lon as a position. Returns {lat, lon, timestamp} or None."""
    con = duckdb.connect()
    row = con.execute(
        f'SELECT "{lat_col}", "{lon_col}", timestamp FROM {_src(globs)} '
        f'WHERE "{lat_col}" IS NOT NULL AND "{lon_col}" IS NOT NULL '
        f'ORDER BY timestamp DESC LIMIT 1'
    ).fetchone()
    con.close()
    if not row:
        return {}
    return {"lat": float(row[0]), "lon": float(row[1]), "timestamp": float(row[2])}


def ships_nearby(globs: list[str], at_epoch: float, own_lat: float, own_lon: float,
                 lat_col: str, lon_col: str, id_col: str,
                 radius_km: float = 50.0, tolerance_s: float = 300.0,
                 name_col: str | None = None) -> list[dict]:
    """Distinct vessels near (own_lat, own_lon) around time at_epoch.

    Takes each vessel's report CLOSEST to at_epoch (within tolerance), computes
    distance to own-ship, returns those within radius_km. Column names are
    parameters (id_col = the vessel key, e.g. mmsi).
    """
    con = duckdb.connect()
    src = _src(globs)
    # Names (AIS static data, msg type 5) are broadcast far less often than
    # positions, so the position record CLOSEST in time usually has a blank name.
    # Resolve each vessel's name from ANY record where it reported one (the most
    # recent up to at_epoch, else the earliest after) and LEFT JOIN it in.
    if name_col:
        names_cte = f"""
        , names AS (
          SELECT vid, arg_min(nm, ord) AS name FROM (
            SELECT "{id_col}" AS vid, "{name_col}" AS nm,
                   -- prefer names known AT/BEFORE the instant, then nearest after
                   CASE WHEN timestamp <= {at_epoch}
                        THEN {at_epoch} - timestamp
                        ELSE (timestamp - {at_epoch}) + 1e12 END AS ord
            FROM {src}
            WHERE "{name_col}" IS NOT NULL AND length(trim("{name_col}")) > 0
          ) GROUP BY vid
        )"""
        name_join = "LEFT JOIN names USING (vid)"
        name_sel = ", names.name AS name"
    else:
        names_cte, name_join, name_sel = "", "", ""

    rows = con.execute(f"""
        WITH near AS (
          SELECT "{id_col}" AS vid, "{lat_col}" AS lat, "{lon_col}" AS lon,
                 timestamp AS t,
                 row_number() OVER (PARTITION BY "{id_col}"
                     ORDER BY abs(timestamp - {at_epoch})) AS rn
          FROM {src}
          WHERE "{lat_col}" IS NOT NULL AND "{lon_col}" IS NOT NULL
            AND abs(timestamp - {at_epoch}) <= {tolerance_s}
        ){names_cte}
        SELECT near.vid AS vid, near.lat AS lat, near.lon AS lon{name_sel}
        FROM near {name_join}
        WHERE near.rn = 1
    """).fetchall()
    cols = [d[0] for d in con.description]
    con.close()

    out = []
    for r in rows:
        rec = dict(zip(cols, r))
        d = _haversine_km(own_lat, own_lon, rec["lat"], rec["lon"])
        if d <= radius_km:
            nm = rec.get("name")
            out.append({"id": rec["vid"], "distance_km": round(d, 2),
                        "lat": rec["lat"], "lon": rec["lon"],
                        "name": nm.strip() if isinstance(nm, str) else nm})
    out.sort(key=lambda x: x["distance_km"])
    return out


def _closest_approach(con, src, lat_col, lon_col, lat, lon, after_epoch=None):
    """Timestamp (and distance) of the track's CLOSEST approach to (lat, lon).

    Pure SQL haversine so the whole track is scanned in the engine. If
    after_epoch is given, only points at/after it are considered (used to force
    arrival to come after departure). Returns {timestamp, distance_km} or None.
    """
    after = f"AND timestamp >= {after_epoch}" if after_epoch is not None else ""
    row = con.execute(f"""
        SELECT timestamp,
               2 * 6371.0 * asin(sqrt(
                 pow(sin(radians("{lat_col}" - {lat}) / 2), 2)
                 + cos(radians({lat})) * cos(radians("{lat_col}"))
                 * pow(sin(radians("{lon_col}" - {lon}) / 2), 2)
               )) AS dist_km
        FROM {src}
        WHERE "{lat_col}" IS NOT NULL AND "{lon_col}" IS NOT NULL {after}
        ORDER BY dist_km ASC
        LIMIT 1
    """).fetchone()
    if not row or row[0] is None:
        return None
    return {"timestamp": float(row[0]), "distance_km": round(float(row[1]), 2)}


def voyage_window(globs: list[str], lat_col: str, lon_col: str,
                  start_lat: float, start_lon: float,
                  end_lat: float, end_lon: float) -> dict:
    """Time window of the voyage from a start point to an end point, derived from
    the position track.

    Departure = the track's closest approach to the START point. Arrival = the
    closest approach to the END point that occurs AT/AFTER departure (so the
    window is directional and always positive). Column names are parameters
    (role-resolved by the caller) — nothing hardcoded.

    Returns {t_start, t_end, start_dist_km, end_dist_km, track_km} or {} if the
    track can't be matched (e.g. it never goes near one of the points).
    """
    con = duckdb.connect()
    src = _src(globs)
    dep = _closest_approach(con, src, lat_col, lon_col, start_lat, start_lon)
    if not dep:
        con.close()
        return {}
    arr = _closest_approach(con, src, lat_col, lon_col, end_lat, end_lon,
                            after_epoch=dep["timestamp"])
    if not arr:
        # end point never reached AFTER departure — try the global closest as a
        # fallback and let the caller see the (possibly reversed) window.
        arr = _closest_approach(con, src, lat_col, lon_col, end_lat, end_lon)
        if not arr:
            con.close()
            return {}
    # cumulative great-circle track length between the two times (sum of
    # consecutive-fix hops) — informational "distance travelled".
    lo, hi = sorted((dep["timestamp"], arr["timestamp"]))
    track = con.execute(f"""
        WITH pts AS (
          SELECT timestamp AS t, "{lat_col}" AS la, "{lon_col}" AS lo
          FROM {src}
          WHERE "{lat_col}" IS NOT NULL AND "{lon_col}" IS NOT NULL
            AND timestamp BETWEEN {lo} AND {hi}
          ORDER BY timestamp
        ), hops AS (
          SELECT 2 * 6371.0 * asin(sqrt(
                   pow(sin(radians(la - lag(la) OVER (ORDER BY t)) / 2), 2)
                   + cos(radians(lag(la) OVER (ORDER BY t))) * cos(radians(la))
                   * pow(sin(radians(lo - lag(lo) OVER (ORDER BY t)) / 2), 2)
                 )) AS hop_km
          FROM pts
        )
        SELECT coalesce(sum(hop_km), 0) FROM hops
    """).fetchone()[0]
    con.close()
    return {
        "t_start": dep["timestamp"], "t_end": arr["timestamp"],
        "start_dist_km": dep["distance_km"], "end_dist_km": arr["distance_km"],
        "track_km": round(float(track), 1),
    }


def track_distance_km(globs: list[str], lat_col: str, lon_col: str,
                      t_range: tuple[float, float] | None = None) -> float:
    """Great-circle length of the position track (sum of consecutive-fix hops),
    optionally restricted to a time window. Used for distance-normalized metrics
    like fuel-per-km. Column names are parameters (role-resolved by the caller)."""
    where = f"WHERE \"{lat_col}\" IS NOT NULL AND \"{lon_col}\" IS NOT NULL"
    if t_range:
        lo, hi = sorted(t_range)
        where += f" AND timestamp BETWEEN {lo} AND {hi}"
    con = duckdb.connect()
    row = con.execute(f"""
        WITH pts AS (
          SELECT timestamp AS t, "{lat_col}" AS la, "{lon_col}" AS lo
          FROM {_src(globs)} {where}
          ORDER BY timestamp
        ), hops AS (
          SELECT 2 * 6371.0 * asin(sqrt(
                   pow(sin(radians(la - lag(la) OVER (ORDER BY t)) / 2), 2)
                   + cos(radians(lag(la) OVER (ORDER BY t))) * cos(radians(la))
                   * pow(sin(radians(lo - lag(lo) OVER (ORDER BY t)) / 2), 2)
                 )) AS hop_km
          FROM pts
        )
        SELECT coalesce(sum(hop_km), 0) FROM hops
    """).fetchone()
    con.close()
    return round(float(row[0]), 2)


def distance_segments(globs: list[str], lat_col: str, lon_col: str,
                      seg_km: float, t_range: tuple[float, float] | None = None
                      ) -> list[dict]:
    """Split the position track into consecutive segments of ~`seg_km` each,
    walking it in time order and accumulating great-circle distance. Each segment
    carries its distance span and its TIME window, so a per-distance metric (e.g.
    fuel per 20 km) can integrate the rate over each segment's time window.

    Returns [{km_start, km_end, t_start, t_end}], distance in km. The last
    segment may be shorter than seg_km (the remaining tail). Column names are
    parameters (role-resolved by the caller) — nothing hardcoded.
    """
    if seg_km <= 0:
        return []
    where = f'WHERE "{lat_col}" IS NOT NULL AND "{lon_col}" IS NOT NULL'
    if t_range:
        lo, hi = sorted(t_range)
        where += f" AND timestamp BETWEEN {lo} AND {hi}"
    con = duckdb.connect()
    rows = con.execute(
        f'SELECT timestamp, "{lat_col}", "{lon_col}" FROM {_src(globs)} {where} '
        f'ORDER BY timestamp'
    ).fetchall()
    con.close()
    if len(rows) < 2:
        return []

    segs = []
    cum = 0.0                      # cumulative distance from track start
    seg_start_km = 0.0
    seg_start_t = float(rows[0][0])
    prev = rows[0]
    for cur in rows[1:]:
        hop = _haversine_km(float(prev[1]), float(prev[2]),
                            float(cur[1]), float(cur[2]))
        cum += hop
        prev = cur
        # close as many segment boundaries as this hop crossed
        while cum - seg_start_km >= seg_km:
            seg_start_km += seg_km
            segs.append({
                "km_start": round(seg_start_km - seg_km, 3),
                "km_end": round(seg_start_km, 3),
                "t_start": seg_start_t,
                "t_end": float(cur[0]),
            })
            seg_start_t = float(cur[0])
    # trailing partial segment (remaining distance since the last boundary)
    if cum - seg_start_km > 1e-6:
        segs.append({
            "km_start": round(seg_start_km, 3),
            "km_end": round(cum, 3),
            "t_start": seg_start_t,
            "t_end": float(prev[0]),
        })
    return segs


def entity_position_at(globs, lat_col, lon_col, id_col, name_col, term,
                       at_epoch, tolerance_s=3600):
    """Position of a named/identified entity closest to `at_epoch`.

    Names (AIS static data) are broadcast SEPARATELY from positions, so position
    rows usually have a blank name — filtering positions by name finds nothing.
    So we first resolve the term to an IDENTIFIER (from any record that carries
    the name, or the term used directly as an id), then find that id's position
    nearest the instant. Column names are role-resolved parameters."""
    con = duckdb.connect()
    src = _src(globs)
    t = str(term).strip().replace("'", "''")

    # 1. resolve term -> id. Try name match (any record that reported a name);
    # else treat the term itself as an id.
    vid = None
    if name_col:
        r = con.execute(f'''
            SELECT "{id_col}" FROM {src}
            WHERE "{name_col}" IS NOT NULL AND lower("{name_col}") LIKE lower('%{t}%')
            GROUP BY "{id_col}" ORDER BY count(*) DESC LIMIT 1
        ''').fetchone()
        if r:
            vid = r[0]
    if vid is None:
        r = con.execute(f'''SELECT "{id_col}" FROM {src}
                            WHERE lower(CAST("{id_col}" AS VARCHAR)) = lower('{t}')
                            LIMIT 1''').fetchone()
        vid = r[0] if r else None
    if vid is None:
        con.close()
        return None

    # 2. that id's position nearest the instant
    row = con.execute(f'''
        SELECT "{lat_col}" AS la, "{lon_col}" AS lo, timestamp AS ts,
               abs(timestamp - {at_epoch}) AS dt
        FROM {src}
        WHERE "{id_col}" = ? AND "{lat_col}" IS NOT NULL AND "{lon_col}" IS NOT NULL
          AND abs(timestamp - {at_epoch}) <= {tolerance_s}
        ORDER BY dt ASC LIMIT 1
    ''', [vid]).fetchone()
    # resolved name (best-known) for display
    nm = None
    if name_col:
        rn = con.execute(f'''SELECT max("{name_col}") FROM {src}
                             WHERE "{id_col}" = ? AND "{name_col}" IS NOT NULL
                               AND length(trim("{name_col}")) > 0''', [vid]).fetchone()
        nm = rn[0] if rn else None
    con.close()
    if not row:
        return None
    return {"id": vid, "name": nm.strip() if isinstance(nm, str) else nm,
            "lat": float(row[0]), "lon": float(row[1]), "timestamp": float(row[2]),
            "distance_s": float(row[3])}


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Public great-circle distance (km) between two points."""
    return round(_haversine_km(lat1, lon1, lat2, lon2), 3)


def signal_by_location(pos_globs, lat_col, lon_col, value_globs, value_exprs,
                       cell_deg=0.1, t_range=None, max_tolerance_s=600):
    """Aggregate a signal (or a combined magnitude of several) over GEOGRAPHIC
    cells of the position track. Cross-source: each position fix is matched to the
    nearest-in-time value reading (ASOF join within max_tolerance_s), a per-fix
    value is computed from `value_exprs`, then averaged per lat/lon cell.

    - value_exprs: list of column names. One -> that signal; several -> Euclidean
      MAGNITUDE sqrt(sum(col^2)) (e.g. acceleration_x/y/z -> vibration magnitude).
    - cell_deg: grid cell size in degrees (~11 km at 0.1 deg).
    Returns [{lat, lon, value, n}] per non-empty cell, plus is_magnitude flag via
    the tuple (cells, is_magnitude). Column names are role-resolved parameters.
    """
    is_mag = len(value_exprs) > 1
    # per-fix value expression on the VALUE side
    if is_mag:
        val_expr = "sqrt(" + " + ".join(f'pow(v."{c}",2)' for c in value_exprs) + ")"
    else:
        val_expr = f'v."{value_exprs[0]}"'
    notnull = " AND ".join(f'v."{c}" IS NOT NULL' for c in value_exprs)

    pos_where = f'p."{lat_col}" IS NOT NULL AND p."{lon_col}" IS NOT NULL'
    if t_range:
        lo, hi = sorted(t_range)
        pos_where += f" AND p.timestamp BETWEEN {lo} AND {hi}"

    con = duckdb.connect()
    # ASOF join: match each position fix to the most recent value reading at/before
    # it (fast, O(n log n)), keep only matches within the tolerance, then bin by
    # geographic cell and average.
    rows = con.execute(f"""
        WITH pos AS (
          SELECT timestamp AS t, "{lat_col}" AS la, "{lon_col}" AS lo
          FROM {_src(pos_globs)} p WHERE {pos_where.replace('p.','')}
        ), val AS (
          SELECT timestamp AS t, {val_expr.replace('v.','')} AS val
          FROM {_src(value_globs)} v WHERE {notnull.replace('v.','')}
        ), joined AS (
          SELECT pos.la AS la, pos.lo AS lo, val.val AS v, val.t AS vt, pos.t AS pt
          FROM pos ASOF LEFT JOIN val ON pos.t >= val.t
        )
        SELECT round(la/{cell_deg})*{cell_deg} AS clat,
               round(lo/{cell_deg})*{cell_deg} AS clon,
               avg(v) AS mean_v, count(v) AS n
        FROM joined
        WHERE v IS NOT NULL AND abs(pt - vt) <= {max_tolerance_s}
        GROUP BY clat, clon HAVING count(v) > 0
        ORDER BY mean_v DESC
    """).fetchall()
    con.close()
    cells = [{"lat": float(r[0]), "lon": float(r[1]),
              "value": float(r[2]), "n": int(r[3])} for r in rows]
    return cells, is_mag
