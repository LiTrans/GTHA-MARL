import csv
import hashlib
import json
import math
from pathlib import Path

PROFILE_TRANSFER = "hold_registered_2022_2025_profile"


def load_projection(path, end_year):
    path = Path(path)
    meta_path = path.with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if meta.get("schedule_sha256") != digest:
        raise ValueError("net projection schedule hash mismatch")
    if meta.get("schema") != "net_population_projection_v1":
        raise ValueError("unknown net projection schema")
    if meta.get("entrant_profile_transfer") != PROFILE_TRANSFER:
        raise ValueError("The net population projection needs its profile settings")
    if meta.get("timing") != "ending_year_january_december" or meta.get("start_year") != 2026:
        raise ValueError("net projection timing/start-year mismatch")
    if meta.get("end_year") != end_year or end_year < 2026:
        raise ValueError("net projection end-year mismatch")
    with path.open(encoding="utf-8-sig", newline="") as f:
        data = list(csv.DictReader(f))
    if [str(r.get("year")) for r in data] != [str(y) for y in range(2026, end_year + 2)]:
        raise ValueError(
            "net projection years must be unique, contiguous and include terminal stub"
        )
    rows = {}
    for r in data:
        year = int(r["year"])
        value = float(r["net_people"])
        if not math.isfinite(value) or r.get("units") != "persons_age_15plus":
            raise ValueError("net projection values/units invalid")
        is_stub = year == end_year + 1
        if r.get("terminal_stub") != str(is_stub) or (is_stub and value != 0):
            raise ValueError("net projection terminal stub invalid")
        rows[year] = value
    return (
        rows,
        {
            **meta,
            "schedule_sha256": digest,
            "metadata_sha256": hashlib.sha256(meta_path.read_bytes()).hexdigest(),
        },
    )
