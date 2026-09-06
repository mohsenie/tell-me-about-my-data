"""Project paths. Single source of truth for locations used by the CLI.

Tuning constants for the discovery/profiling logic live in
src/ttmd/constants.py (package-internal), not here.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Partitioned ship data:
#   ship-data/<vessel>/<source>/logs/date=YYYY-MM-DD/<source>-<vessel>-<ts>.parquet
SHIP_DATA_DIR = ROOT / "ship-data"
DEFAULT_VESSEL = "vessel-001"
# Fallback only; real asset type comes from sources.yaml (asset_type key).
FALLBACK_ASSET_TYPE = "asset"

# Customer-provided documentation (manuals, datasheets) for root-cause grounding.
USER_DOC_DIR = ROOT / "user-documentation"

ARTIFACTS_DIR = ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def source_glob(source: str, vessel: str = DEFAULT_VESSEL) -> str:
    """Glob matching all parquet files for a source across all date partitions."""
    return str(SHIP_DATA_DIR / vessel / source / "logs" / "date=*" / "*.parquet")


def source_globs_for_dates(source: str, dates: list[str],
                           vessel: str = DEFAULT_VESSEL) -> list[str]:
    """Globs for specific date partitions (YYYY-MM-DD), for time-window queries."""
    base = SHIP_DATA_DIR / vessel / source / "logs"
    return [str(base / f"date={d}" / "*.parquet") for d in dates]


def available_dates(source: str, vessel: str = DEFAULT_VESSEL) -> list[str]:
    """Dates present for a source, sorted ascending (from date=YYYY-MM-DD dirs)."""
    base = SHIP_DATA_DIR / vessel / source / "logs"
    if not base.is_dir():
        return []
    dates = [p.name.split("=", 1)[1] for p in base.iterdir()
             if p.is_dir() and p.name.startswith("date=")]
    return sorted(dates)


def source_metadata(vessel: str = DEFAULT_VESSEL) -> dict:
    """User-provided per-source metadata (protocol, description) for context.

    Read from ship-data/<vessel>/sources.yaml if present. This is USER input
    (declared at onboarding), not hardcoded: the customer knows their equipment.
    Missing file / missing source -> empty dict (system degrades gracefully to
    name-only guessing). Returns {source: {"protocol": str, "description": str}}.
    """
    import yaml
    path = SHIP_DATA_DIR / vessel / "sources.yaml"
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    return data.get("sources", {})


def known_places(vessel: str = DEFAULT_VESSEL) -> list[dict]:
    """User-provided place/port list from sources.yaml (top-level 'places'), for
    turning coordinates into human place names WITHOUT any network geocoder.

    Each entry: {name, lat, lon, [radius_km]}. USER input (the customer knows the
    ports on their route); missing -> empty list (system degrades to raw coords).
    NOT hardcoded and NOT fetched from the internet (offline by design)."""
    import yaml
    path = SHIP_DATA_DIR / vessel / "sources.yaml"
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text()) or {}
    out = []
    for p in data.get("places", []) or []:
        if p.get("name") is not None and p.get("lat") is not None and p.get("lon") is not None:
            out.append({"name": str(p["name"]), "lat": float(p["lat"]),
                        "lon": float(p["lon"]),
                        "radius_km": float(p["radius_km"]) if p.get("radius_km") else None})
    return out


def asset_type(vessel: str = DEFAULT_VESSEL) -> str:
    """Asset type declared by the user in sources.yaml (top-level 'asset_type'),
    else a neutral fallback. NOT hardcoded to 'ship'."""
    import yaml
    path = SHIP_DATA_DIR / vessel / "sources.yaml"
    if path.exists():
        data = yaml.safe_load(path.read_text()) or {}
        if data.get("asset_type"):
            return data["asset_type"]
    return FALLBACK_ASSET_TYPE


def discover_sources(vessel: str = DEFAULT_VESSEL) -> list[str]:
    """Auto-detect available sources for a vessel from the filesystem.

    A directory under ship-data/<vessel>/ counts as a source if it contains at
    least one parquet file (any depth). Enables zero-config onboarding of new
    data types: drop a new source folder in and it is picked up automatically.
    """
    base = SHIP_DATA_DIR / vessel
    if not base.is_dir():
        return []
    found = []
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        if any(d.rglob("*.parquet")):
            found.append(d.name)
    return found
