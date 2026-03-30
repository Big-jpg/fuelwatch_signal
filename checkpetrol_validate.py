
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

BASE_URL = "https://checkpetrol.com.au"
API_BASE = f"{BASE_URL}/api/v1"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/146.0.0.0 Safari/537.36"
)

SUPPORTED_CHECKPETROL_FUELS = ["U91", "DSL", "LPG", "E10", "P95", "P98"]

CHECKPETROL_TO_CANONICAL = {
    "U91": "ULP",
    "DSL": "DSL",
    "LPG": "LPG",
    "E10": "E10",
    "P95": "PUP",
    "P98": "98R",
}

CANONICAL_TO_CHECKPETROL = {
    "ULP": "U91",
    "DSL": "DSL",
    "LPG": "LPG",
    "E10": "E10",
    "PUP": "P95",
    "98R": "P98",
}

AU_BOUNDS = {
    "lat_min": -44.0,
    "lat_max": -10.0,
    "lon_min": 112.0,
    "lon_max": 154.0,
}

PLAUSIBLE_PRICE_RANGES = {
    "U91": (80.0, 400.0),
    "DSL": (80.0, 450.0),
    "P95": (80.0, 450.0),
    "P98": (80.0, 450.0),
    "E10": (80.0, 400.0),
    "LPG": (20.0, 250.0),
}

STATE_CODES = {"WA", "NSW", "VIC", "QLD", "SA", "TAS", "NT", "ACT"}


@dataclass
class DatasetManifestRecord:
    dataset: str
    endpoint: str
    params: dict[str, Any]
    ok: bool
    status_code: int | None
    content_type: str | None
    output_path: str
    row_count: int | None = None
    warning_count: int = 0
    warning_sample: list[str] | None = None
    notes: dict[str, Any] | None = None


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None

#
# Price normalisation helper
#
# CheckPetrol sometimes reports price values in tenths of a cent (e.g.
# ``3029`` representing ``302.9 c/L``).  To ensure prices are
# comparable with other data sources (which are typically in cents per
# litre), this helper scales down any unusually large values.  Values
# above 1000 are divided by 10 and rounded to three decimal places.
# Otherwise the value is returned unchanged.  Non‑numeric inputs
# produce ``None``.
def normalize_price_cpl(value: Any) -> float | None:
    """Normalise a price value to cents per litre.

    Values above 1000 are assumed to be scaled by 10 (tenths of a
    cent) and are divided accordingly.  Missing or non‑numeric inputs
    return ``None``.
    """
    v = safe_float(value)
    if v is None:
        return None
    if v > 1000:
        return round(v / 10.0, 3)
    return v


def parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().upper().split())


def normalize_brand(value: Any) -> str:
    v = normalize_text(value)
    replacements = {
        "BP CONNECT": "BP",
        "SHELL COLES EXPRESS": "SHELL",
        "COLES EXPRESS": "SHELL",
        "PUMA ENERGY": "PUMA",
        "7-ELEVEN": "7 ELEVEN",
    }
    return replacements.get(v, v)


def normalize_suburb(value: Any) -> str:
    return normalize_text(value)


class CheckPetrolClient:
    def __init__(self, base_url: str = API_BASE) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json, text/plain, */*",
                "User-Agent": USER_AGENT,
                "Referer": BASE_URL + "/",
                "Origin": BASE_URL,
            }
        )

    def _url(self, path: str) -> str:
        if path.startswith(("http://", "https://")):
            return path
        if not path.startswith("/"):
            path = "/" + path
        return self.base_url + path

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> tuple[requests.Response, Any]:
        response = self.session.get(self._url(path), params=params, timeout=120)
        response.raise_for_status()
        ctype = response.headers.get("Content-Type", "")
        if "json" not in ctype.lower():
            raise RuntimeError(f"Expected JSON from {path}, got {ctype!r}")
        return response, response.json()


def issue_row(
    severity: str,
    issue_type: str,
    dataset: str,
    entity_id: Any,
    details: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "severity": severity,
        "issue_type": issue_type,
        "dataset": dataset,
        "entity_id": entity_id,
        "details": details,
    }
    if extra:
        row.update(extra)
    return row


def flatten_station_features(payload: dict[str, Any], query_fuel: str, collected_at: datetime) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    features = payload.get("features") or []
    station_rows: list[dict[str, Any]] = []
    price_rows: list[dict[str, Any]] = []

    for feature in features:
        props = feature.get("properties") or {}
        geom = feature.get("geometry") or {}
        coords = geom.get("coordinates") or [None, None]
        lon = safe_float(coords[0]) if len(coords) > 0 else None
        lat = safe_float(coords[1]) if len(coords) > 1 else None
        updated_at = parse_timestamp(props.get("updated_at"))
        station_id = props.get("id")
        collected_at_iso = collected_at.isoformat()

        station_rows.append(
            {
                "query_fuel": query_fuel,
                "station_id": station_id,
                "name": props.get("name"),
                "brand": props.get("brand"),
                "brand_norm": normalize_brand(props.get("brand")),
                "source": props.get("source"),
                "state": props.get("state"),
                "suburb": props.get("suburb"),
                "suburb_norm": normalize_suburb(props.get("suburb")),
                "postcode": props.get("postcode"),
                "address": props.get("address"),
                "fuel_type": props.get("fuel_type"),
                # Use normalised price to ensure tenths-of-a-cent values are
                # scaled correctly.  Keep the column name ``price_cents`` for
                # backward compatibility even though the unit is cents per litre.
                "price_cents": normalize_price_cpl(props.get("price_cents")),
                "has_outage": bool(props.get("has_outage")),
                "is_fallback": bool(props.get("is_fallback")),
                "is_stale": bool(props.get("is_stale")),
                "no_price": bool(props.get("no_price")),
                "outage_fuels_count": len(props.get("outage_fuels") or []),
                "updated_at": updated_at.isoformat() if updated_at else None,
                "latitude": lat,
                "longitude": lon,
                "all_prices_json": json.dumps(props.get("all_prices") or {}, ensure_ascii=False, sort_keys=True),
                "collected_at": collected_at_iso,
            }
        )

        all_prices = props.get("all_prices") or {}
        if isinstance(all_prices, dict):
            for fuel_code, price in all_prices.items():
                price_rows.append(
                    {
                        "query_fuel": query_fuel,
                        "station_id": station_id,
                        "fuel_code": fuel_code,
                        "canonical_fuel_code": CHECKPETROL_TO_CANONICAL.get(fuel_code),
                        # Apply the same normalisation to the per‑fuel prices.  Use
                        # the original key ``price_cents`` for compatibility.
                        "price_cents": normalize_price_cpl(price),
                        "is_primary_for_query": fuel_code == query_fuel,
                        "collected_at": collected_at_iso,
                        "updated_at": updated_at.isoformat() if updated_at else None,
                    }
                )

    return station_rows, price_rows


def collect_raw_datasets(client: CheckPetrolClient, root: Path) -> tuple[list[DatasetManifestRecord], dict[str, Any]]:
    raw_root = ensure_dir(root / "raw")
    collected_at = now_utc()

    manifest: list[DatasetManifestRecord] = []
    payloads: dict[str, Any] = {}

    jobs: list[tuple[str, str, dict[str, Any] | None]] = [
        ("health", "/health", None),
        ("availability", "/stats/availability", None),
    ]
    jobs.extend((f"stations_{fuel}", "/stations", {"fuel": fuel}) for fuel in SUPPORTED_CHECKPETROL_FUELS)

    for dataset, endpoint, params in jobs:
        dest = raw_root / f"{dataset}.json"
        warnings: list[str] = []
        notes: dict[str, Any] = {"collected_at": collected_at.isoformat()}

        try:
            response, payload = client.get_json(endpoint, params=params)
            json_dump(dest, payload)

            row_count = None
            if isinstance(payload, dict) and isinstance(payload.get("features"), list):
                row_count = len(payload["features"])
            elif isinstance(payload, list):
                row_count = len(payload)

            payloads[dataset] = payload
            manifest.append(
                DatasetManifestRecord(
                    dataset=dataset,
                    endpoint=endpoint,
                    params=params or {},
                    ok=True,
                    status_code=response.status_code,
                    content_type=response.headers.get("Content-Type"),
                    output_path=str(dest),
                    row_count=row_count,
                    warning_count=len(warnings),
                    warning_sample=warnings[:10],
                    notes=notes,
                )
            )
        except Exception as exc:
            manifest.append(
                DatasetManifestRecord(
                    dataset=dataset,
                    endpoint=endpoint,
                    params=params or {},
                    ok=False,
                    status_code=None,
                    content_type=None,
                    output_path=str(dest),
                    row_count=None,
                    warning_count=1,
                    warning_sample=[str(exc)],
                    notes=notes,
                )
            )

    json_dump(root / "manifest.json", [asdict(x) for x in manifest])
    return manifest, payloads


def build_normalized_tables(payloads: dict[str, Any], root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    flat_root = ensure_dir(root / "flat")
    collected_at = now_utc()

    station_frames: list[pd.DataFrame] = []
    price_frames: list[pd.DataFrame] = []
    query_context_frames: list[pd.DataFrame] = []

    for fuel in SUPPORTED_CHECKPETROL_FUELS:
        dataset = f"stations_{fuel}"
        payload = payloads.get(dataset) or {}
        station_rows, price_rows = flatten_station_features(payload, fuel, collected_at)
        station_df = pd.DataFrame(station_rows)
        price_df = pd.DataFrame(price_rows)

        if station_df.empty:
            continue

        station_frames.append(station_df)
        if not price_df.empty:
            price_frames.append(price_df)

        ctx = station_df[
            [
                "query_fuel",
                "station_id",
                "fuel_type",
                "price_cents",
                "has_outage",
                "is_fallback",
                "is_stale",
                "no_price",
                "updated_at",
                "collected_at",
                "source",
                "state",
                "suburb",
                "brand",
            ]
        ].copy()
        query_context_frames.append(ctx)

    stations_snapshot = (
        pd.concat(station_frames, ignore_index=True, sort=False)
        if station_frames
        else pd.DataFrame()
    )
    station_prices_snapshot = (
        pd.concat(price_frames, ignore_index=True, sort=False)
        if price_frames
        else pd.DataFrame()
    )
    station_query_context = (
        pd.concat(query_context_frames, ignore_index=True, sort=False)
        if query_context_frames
        else pd.DataFrame()
    )

    if not stations_snapshot.empty:
        stations_snapshot["__rank"] = stations_snapshot["query_fuel"].map({f: i for i, f in enumerate(SUPPORTED_CHECKPETROL_FUELS)})
        stations_snapshot = (
            stations_snapshot.sort_values(["station_id", "__rank"])
            .drop_duplicates(subset=["station_id"], keep="first")
            .drop(columns=["__rank"])
        )

    if not station_prices_snapshot.empty:
        station_prices_snapshot = (
            station_prices_snapshot.sort_values(["station_id", "fuel_code", "query_fuel"])
            .drop_duplicates(subset=["station_id", "fuel_code"], keep="first")
        )

    stations_snapshot.to_csv(flat_root / "stations_snapshot.csv", index=False)
    station_prices_snapshot.to_csv(flat_root / "station_prices_snapshot.csv", index=False)
    station_query_context.to_csv(flat_root / "station_query_context.csv", index=False)

    return stations_snapshot, station_prices_snapshot, station_query_context


def validate_availability_payload(payload: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}

    required = ["availability", "outage_count", "outage_percent", "outages_by_state", "total_stations"]
    if not isinstance(payload, dict):
        issues.append(issue_row("error", "availability_not_object", "availability", None, "Payload is not an object"))
        return issues, summary

    for key in required:
        if key not in payload:
            issues.append(issue_row("error", "availability_missing_key", "availability", key, f"Missing key: {key}"))

    summary["availability_total_stations"] = payload.get("total_stations")
    summary["availability_outage_count"] = payload.get("outage_count")
    summary["availability_outage_percent"] = payload.get("outage_percent")
    summary["availability_outages_by_state_count"] = len(payload.get("outages_by_state") or []) if isinstance(payload.get("outages_by_state"), list) else None
    summary["availability_entries"] = len(payload.get("availability") or []) if isinstance(payload.get("availability"), list) else None

    if isinstance(payload.get("outages_by_state"), list):
        for row in payload["outages_by_state"]:
            state = row.get("state") if isinstance(row, dict) else None
            if state and normalize_text(state) not in STATE_CODES:
                issues.append(issue_row("warning", "availability_unknown_state", "availability", state, f"Unexpected state code: {state}"))

    return issues, summary


def compute_completeness(stations_snapshot: pd.DataFrame, station_prices_snapshot: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    if not stations_snapshot.empty:
        for col in [
            "station_id",
            "name",
            "brand",
            "source",
            "state",
            "suburb",
            "postcode",
            "updated_at",
            "latitude",
            "longitude",
            "fuel_type",
            "price_cents",
            "all_prices_json",
        ]:
            if col in stations_snapshot.columns:
                null_pct = float(stations_snapshot[col].isna().mean() * 100.0)
                rows.append({"table_name": "stations_snapshot", "field_name": col, "null_pct": round(null_pct, 4)})

    if not station_prices_snapshot.empty:
        for col in ["station_id", "fuel_code", "price_cents", "updated_at"]:
            if col in station_prices_snapshot.columns:
                null_pct = float(station_prices_snapshot[col].isna().mean() * 100.0)
                rows.append({"table_name": "station_prices_snapshot", "field_name": col, "null_pct": round(null_pct, 4)})

    return pd.DataFrame(rows)


def validate_stations(
    stations_snapshot: pd.DataFrame,
    station_prices_snapshot: pd.DataFrame,
    station_query_context: pd.DataFrame,
    availability_payload: Any,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    issues: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    now = now_utc()

    if stations_snapshot.empty:
        issues.append(issue_row("error", "stations_empty", "stations_snapshot", None, "No station rows were normalized"))
        return (
            pd.DataFrame(issues),
            summary,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )

    stations = stations_snapshot.copy()
    prices = station_prices_snapshot.copy()
    queries = station_query_context.copy()

    stations["updated_at_dt"] = pd.to_datetime(stations["updated_at"], errors="coerce", utc=True)
    stations["age_minutes"] = (now - stations["updated_at_dt"]).dt.total_seconds() / 60.0
    stations["state_norm"] = stations["state"].astype(str).str.upper().str.strip()

    # Required fields
    required_station_fields = ["station_id", "name", "brand", "source", "state", "suburb", "postcode", "updated_at", "all_prices_json"]
    for field in required_station_fields:
        missing = int(stations[field].isna().sum()) if field in stations.columns else len(stations)
        if missing:
            issues.append(issue_row("warning", "missing_required_field", "stations_snapshot", field, f"{missing} rows missing required field", {"missing_rows": missing}))

    # Station key uniqueness
    duplicate_station_ids = stations["station_id"].duplicated(keep=False)
    dup_count = int(duplicate_station_ids.sum())
    if dup_count:
        issues.append(issue_row("error", "duplicate_station_id", "stations_snapshot", None, f"{dup_count} duplicate station_id rows found", {"duplicate_rows": dup_count}))

    # Coordinates and Australia bounds
    invalid_coords = stations["latitude"].isna() | stations["longitude"].isna()
    if int(invalid_coords.sum()):
        issues.append(issue_row("warning", "missing_coordinates", "stations_snapshot", None, f"{int(invalid_coords.sum())} rows missing coordinates"))

    out_of_bounds = (
        stations["latitude"].notna()
        & stations["longitude"].notna()
        & (
            (stations["latitude"] < AU_BOUNDS["lat_min"])
            | (stations["latitude"] > AU_BOUNDS["lat_max"])
            | (stations["longitude"] < AU_BOUNDS["lon_min"])
            | (stations["longitude"] > AU_BOUNDS["lon_max"])
        )
    )
    if int(out_of_bounds.sum()):
        issues.append(issue_row("error", "coordinates_out_of_au_bounds", "stations_snapshot", None, f"{int(out_of_bounds.sum())} rows outside Australia bounds"))

    # Timestamp validity and freshness
    bad_ts = stations["updated_at_dt"].isna()
    if int(bad_ts.sum()):
        issues.append(issue_row("error", "invalid_updated_at", "stations_snapshot", None, f"{int(bad_ts.sum())} rows have invalid updated_at timestamps"))

    future_ts = stations["updated_at_dt"].notna() & (stations["updated_at_dt"] > now + pd.Timedelta(minutes=5))
    if int(future_ts.sum()):
        issues.append(issue_row("error", "future_timestamp", "stations_snapshot", None, f"{int(future_ts.sum())} rows have future timestamps"))

    very_old = stations["age_minutes"].notna() & (stations["age_minutes"] > 60 * 24 * 7)
    if int(very_old.sum()):
        issues.append(issue_row("warning", "very_old_timestamp", "stations_snapshot", None, f"{int(very_old.sum())} rows are older than 7 days"))

    # Stale flag coherence
    stale_flag_mismatch = stations["is_stale"] & stations["age_minutes"].notna() & (stations["age_minutes"] < 60 * 24)
    if int(stale_flag_mismatch.sum()):
        issues.append(issue_row("warning", "stale_flag_recent_timestamp", "stations_snapshot", None, f"{int(stale_flag_mismatch.sum())} rows marked stale but updated within 24 hours"))

    non_stale_but_old = (~stations["is_stale"]) & stations["age_minutes"].notna() & (stations["age_minutes"] > 60 * 24 * 3)
    if int(non_stale_but_old.sum()):
        issues.append(issue_row("warning", "non_stale_but_old", "stations_snapshot", None, f"{int(non_stale_but_old.sum())} rows not marked stale but older than 3 days"))

    # Flag coherence
    no_price_with_value = stations["no_price"] & stations["price_cents"].notna()
    if int(no_price_with_value.sum()):
        issues.append(issue_row("warning", "no_price_with_value", "stations_snapshot", None, f"{int(no_price_with_value.sum())} rows flagged no_price but price_cents exists"))

    outage_without_outage_fuels = stations["has_outage"] & (stations["outage_fuels_count"].fillna(0) == 0)
    if int(outage_without_outage_fuels.sum()):
        issues.append(issue_row("warning", "outage_without_outage_fuels", "stations_snapshot", None, f"{int(outage_without_outage_fuels.sum())} rows flagged outage with no outage_fuels entries"))

    # Price sanity by fuel
    if not prices.empty:
        for fuel_code, (low, high) in PLAUSIBLE_PRICE_RANGES.items():
            scoped = prices[prices["fuel_code"] == fuel_code]
            if scoped.empty:
                continue
            bad_range = scoped["price_cents"].notna() & ((scoped["price_cents"] < low) | (scoped["price_cents"] > high))
            count = int(bad_range.sum())
            if count:
                issues.append(issue_row("warning", "price_out_of_range", "station_prices_snapshot", fuel_code, f"{count} {fuel_code} rows outside plausible range {low}-{high} c/L", {"fuel_code": fuel_code, "bad_rows": count}))

    # Query consistency: query fuel price_cents should match exploded price for query_fuel
    if not queries.empty and not prices.empty:
        query_match = queries.merge(
            prices[["station_id", "fuel_code", "price_cents"]].rename(columns={"fuel_code": "query_fuel", "price_cents": "exploded_price_cents"}),
            on=["station_id", "query_fuel"],
            how="left",
        )
        has_exploded = query_match["exploded_price_cents"].notna()
        mismatch = has_exploded & (query_match["price_cents"].round(3) != query_match["exploded_price_cents"].round(3))
        missing_for_query = ~has_exploded
        if int(mismatch.sum()):
            issues.append(issue_row("error", "query_price_mismatch", "station_query_context", None, f"{int(mismatch.sum())} query rows where price_cents != all_prices[query_fuel]"))
        if int(missing_for_query.sum()):
            issues.append(issue_row("warning", "query_fuel_missing_in_all_prices", "station_query_context", None, f"{int(missing_for_query.sum())} query rows where query_fuel was absent from all_prices"))

    # Source / state / fuel summaries
    source_quality_summary = (
        stations.groupby("source", dropna=False)
        .agg(
            stations=("station_id", "nunique"),
            outage_rate=("has_outage", "mean"),
            stale_rate=("is_stale", "mean"),
            fallback_rate=("is_fallback", "mean"),
            no_price_rate=("no_price", "mean"),
            median_age_minutes=("age_minutes", "median"),
        )
        .reset_index()
    )
    for col in ["outage_rate", "stale_rate", "fallback_rate", "no_price_rate"]:
        source_quality_summary[col] = (source_quality_summary[col].fillna(0.0) * 100.0).round(2)

    state_station_summary = (
        stations.groupby("state_norm", dropna=False)
        .agg(
            stations=("station_id", "nunique"),
            outage_rate=("has_outage", "mean"),
            stale_rate=("is_stale", "mean"),
            median_age_minutes=("age_minutes", "median"),
        )
        .reset_index()
        .rename(columns={"state_norm": "state"})
    )
    for col in ["outage_rate", "stale_rate"]:
        state_station_summary[col] = (state_station_summary[col].fillna(0.0) * 100.0).round(2)

    fuel_coverage_summary = pd.DataFrame()
    if not prices.empty:
        total_stations = stations["station_id"].nunique()
        fuel_coverage_summary = (
            prices.groupby("fuel_code", dropna=False)
            .agg(
                stations_with_fuel=("station_id", "nunique"),
                avg_price_cents=("price_cents", "mean"),
                median_price_cents=("price_cents", "median"),
            )
            .reset_index()
        )
        fuel_coverage_summary["coverage_pct"] = (fuel_coverage_summary["stations_with_fuel"] / total_stations * 100.0).round(2)
        fuel_coverage_summary["canonical_fuel_code"] = fuel_coverage_summary["fuel_code"].map(CHECKPETROL_TO_CANONICAL)
        fuel_coverage_summary = fuel_coverage_summary.sort_values("stations_with_fuel", ascending=False)

    state_fuel_coverage_summary = pd.DataFrame()
    if not prices.empty:
        state_fuel_coverage_summary = (
            prices.merge(stations[["station_id", "state_norm"]], on="station_id", how="left")
            .groupby(["state_norm", "fuel_code"], dropna=False)
            .agg(stations_with_fuel=("station_id", "nunique"))
            .reset_index()
            .rename(columns={"state_norm": "state"})
        )
        total_by_state = (
            stations.groupby("state_norm", dropna=False)["station_id"].nunique().rename("total_stations").reset_index().rename(columns={"state_norm": "state"})
        )
        state_fuel_coverage_summary = state_fuel_coverage_summary.merge(total_by_state, on="state", how="left")
        state_fuel_coverage_summary["coverage_pct"] = (
            state_fuel_coverage_summary["stations_with_fuel"] / state_fuel_coverage_summary["total_stations"] * 100.0
        ).round(2)

    freshness_summary = pd.DataFrame()
    freshness_buckets = [
        (-1, 60, "0-60 min"),
        (60, 360, "1-6 hrs"),
        (360, 1440, "6-24 hrs"),
        (1440, 4320, "1-3 days"),
        (4320, 1000000000, ">3 days"),
    ]
    freshness_rows: list[dict[str, Any]] = []
    valid_age = stations["age_minutes"].dropna()
    for low, high, label in freshness_buckets:
        count = int(((valid_age > low) & (valid_age <= high)).sum())
        freshness_rows.append({"scope": "overall", "bucket": label, "rows": count})
    freshness_summary = pd.DataFrame(freshness_rows)

    availability_issues, availability_summary = validate_availability_payload(availability_payload)
    issues.extend(availability_issues)
    summary.update(availability_summary)

    # Summary numbers
    summary.update(
        {
            "validated_at": now.isoformat(),
            "station_count": int(stations["station_id"].nunique()),
            "station_price_rows": int(len(prices)),
            "query_context_rows": int(len(queries)),
            "source_count": int(stations["source"].nunique(dropna=True)),
            "state_count": int(stations["state_norm"].nunique(dropna=True)),
            "fuel_codes_observed": sorted(prices["fuel_code"].dropna().astype(str).unique().tolist()) if not prices.empty else [],
            "median_age_minutes": round(float(valid_age.median()), 2) if not valid_age.empty else None,
            "p90_age_minutes": round(float(valid_age.quantile(0.9)), 2) if not valid_age.empty else None,
            "outage_rate_pct": round(float(stations["has_outage"].mean() * 100.0), 2),
            "stale_rate_pct": round(float(stations["is_stale"].mean() * 100.0), 2),
            "fallback_rate_pct": round(float(stations["is_fallback"].mean() * 100.0), 2),
            "no_price_rate_pct": round(float(stations["no_price"].mean() * 100.0), 2),
        }
    )

    return (
        pd.DataFrame(issues),
        summary,
        source_quality_summary,
        state_station_summary,
        fuel_coverage_summary,
        state_fuel_coverage_summary,
        freshness_summary,
    )


def load_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def find_fuelwatch_current_prices(fuelwatch_root: Path) -> Path | None:
    candidates = [
        fuelwatch_root / "flat" / "current_prices_effective.csv",
        fuelwatch_root / "flat" / "current_prices.csv",
        fuelwatch_root / "current_prices_effective.csv",
        fuelwatch_root / "current_prices.csv",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def validate_wa_concordance(
    stations_snapshot: pd.DataFrame,
    station_prices_snapshot: pd.DataFrame,
    fuelwatch_root: Path | None,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    if fuelwatch_root is None:
        return pd.DataFrame(), {"wa_concordance_status": "skipped_no_fuelwatch_root"}, pd.DataFrame()

    fuelwatch_path = find_fuelwatch_current_prices(fuelwatch_root)
    if fuelwatch_path is None:
        return pd.DataFrame(), {"wa_concordance_status": "skipped_no_current_prices_file"}, pd.DataFrame()

    fuelwatch_df = load_csv_if_exists(fuelwatch_path)
    if fuelwatch_df.empty:
        return pd.DataFrame(), {"wa_concordance_status": "skipped_empty_current_prices"}, pd.DataFrame()

    fw_required = {"site_name", "brand_name", "suburb", "product_short_name", "price_today"}
    if not fw_required.issubset(set(fuelwatch_df.columns)):
        return pd.DataFrame(), {"wa_concordance_status": "skipped_missing_required_columns", "fuelwatch_columns_present": sorted(fuelwatch_df.columns.tolist())}, pd.DataFrame()

    cp_wa = stations_snapshot[stations_snapshot["state"].astype(str).str.upper().eq("WA")].copy()
    if cp_wa.empty:
        return pd.DataFrame(), {"wa_concordance_status": "skipped_no_wa_rows"}, pd.DataFrame()

    cp_prices = station_prices_snapshot.copy()
    if cp_prices.empty:
        return pd.DataFrame(), {"wa_concordance_status": "skipped_no_station_prices"}, pd.DataFrame()

    cp_wa_prices = cp_wa[["station_id", "brand_norm", "suburb_norm"]].merge(
        cp_prices[["station_id", "fuel_code", "canonical_fuel_code", "price_cents"]],
        on="station_id",
        how="inner",
    )
    cp_wa_prices = cp_wa_prices[cp_wa_prices["canonical_fuel_code"].notna()].copy()

    fuelwatch_df = fuelwatch_df.copy()
    fuelwatch_df["brand_norm"] = fuelwatch_df["brand_name"].map(normalize_brand)
    fuelwatch_df["suburb_norm"] = fuelwatch_df["suburb"].map(normalize_suburb)
    fuelwatch_df["canonical_fuel_code"] = fuelwatch_df["product_short_name"].astype(str)
    fuelwatch_df["price_today"] = pd.to_numeric(fuelwatch_df["price_today"], errors="coerce")

    fw_wa = fuelwatch_df[
        fuelwatch_df["canonical_fuel_code"].isin(set(CANONICAL_TO_CHECKPETROL.keys()))
    ].copy()

    cp_suburb = (
        cp_wa_prices.groupby(["suburb_norm", "canonical_fuel_code"], dropna=False)
        .agg(
            cp_station_count=("station_id", "nunique"),
            cp_median_price=("price_cents", "median"),
            cp_mean_price=("price_cents", "mean"),
        )
        .reset_index()
    )
    fw_suburb = (
        fw_wa.groupby(["suburb_norm", "canonical_fuel_code"], dropna=False)
        .agg(
            fw_station_count=("site_name", "nunique"),
            fw_median_price=("price_today", "median"),
            fw_mean_price=("price_today", "mean"),
        )
        .reset_index()
    )

    concordance = cp_suburb.merge(fw_suburb, on=["suburb_norm", "canonical_fuel_code"], how="inner")
    if concordance.empty:
        return pd.DataFrame(), {"wa_concordance_status": "no_joinable_suburb_fuel_rows"}, pd.DataFrame()

    concordance["median_price_delta"] = concordance["cp_median_price"] - concordance["fw_median_price"]
    concordance["mean_price_delta"] = concordance["cp_mean_price"] - concordance["fw_mean_price"]
    concordance["abs_median_price_delta"] = concordance["median_price_delta"].abs()
    concordance["abs_mean_price_delta"] = concordance["mean_price_delta"].abs()

    summary = {
        "wa_concordance_status": "ok",
        "wa_concordance_rows": int(len(concordance)),
        "wa_concordance_suburbs": int(concordance["suburb_norm"].nunique()),
        "wa_concordance_fuels": sorted(concordance["canonical_fuel_code"].unique().tolist()),
        "wa_median_abs_median_price_delta": round(float(concordance["abs_median_price_delta"].median()), 3),
        "wa_p90_abs_median_price_delta": round(float(concordance["abs_median_price_delta"].quantile(0.9)), 3),
        "wa_median_abs_mean_price_delta": round(float(concordance["abs_mean_price_delta"].median()), 3),
    }

    by_fuel = (
        concordance.groupby("canonical_fuel_code", dropna=False)
        .agg(
            rows=("suburb_norm", "count"),
            suburbs=("suburb_norm", "nunique"),
            median_abs_median_price_delta=("abs_median_price_delta", "median"),
            p90_abs_median_price_delta=("abs_median_price_delta", lambda s: s.quantile(0.9)),
        )
        .reset_index()
        .sort_values("rows", ascending=False)
    )

    return concordance, summary, by_fuel


def markdown_report(
    summary: dict[str, Any],
    issues_df: pd.DataFrame,
    source_quality_summary: pd.DataFrame,
    fuel_coverage_summary: pd.DataFrame,
    wa_summary: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# CheckPetrol validation summary")
    lines.append("")
    lines.append("## Core metrics")
    for key in sorted(summary.keys()):
        lines.append(f"- {key}: {summary[key]}")
    lines.append("")
    lines.append("## WA concordance")
    for key in sorted(wa_summary.keys()):
        lines.append(f"- {key}: {wa_summary[key]}")
    lines.append("")
    lines.append("## Issue counts")
    if issues_df.empty:
        lines.append("- No issues logged.")
    else:
        issue_counts = issues_df.groupby(["severity", "issue_type"]).size().reset_index(name="rows")
        for _, row in issue_counts.iterrows():
            lines.append(f"- {row['severity']} / {row['issue_type']}: {int(row['rows'])}")
    lines.append("")
    lines.append("## Top sources")
    if source_quality_summary.empty:
        lines.append("- No source summary available.")
    else:
        top = source_quality_summary.sort_values("stations", ascending=False).head(10)
        for _, row in top.iterrows():
            lines.append(
                f"- {row['source']}: stations={int(row['stations'])}, "
                f"outage_rate={row['outage_rate']}%, stale_rate={row['stale_rate']}%, "
                f"fallback_rate={row['fallback_rate']}%, no_price_rate={row['no_price_rate']}%"
            )
    lines.append("")
    lines.append("## Fuel coverage")
    if fuel_coverage_summary.empty:
        lines.append("- No fuel coverage summary available.")
    else:
        for _, row in fuel_coverage_summary.sort_values("stations_with_fuel", ascending=False).iterrows():
            lines.append(
                f"- {row['fuel_code']} ({row.get('canonical_fuel_code') or '-'}): "
                f"stations={int(row['stations_with_fuel'])}, coverage={row['coverage_pct']}%, "
                f"median_price_cents={round(float(row['median_price_cents']), 3) if pd.notna(row['median_price_cents']) else 'NA'}"
            )
    lines.append("")
    return "\n".join(lines)


def write_outputs(
    root: Path,
    issues_df: pd.DataFrame,
    completeness_df: pd.DataFrame,
    summary: dict[str, Any],
    source_quality_summary: pd.DataFrame,
    state_station_summary: pd.DataFrame,
    fuel_coverage_summary: pd.DataFrame,
    state_fuel_coverage_summary: pd.DataFrame,
    freshness_summary: pd.DataFrame,
    wa_concordance_df: pd.DataFrame,
    wa_summary: dict[str, Any],
    wa_by_fuel_df: pd.DataFrame,
) -> None:
    reports_root = ensure_dir(root / "reports")
    diagnostics_root = ensure_dir(root / "diagnostics")

    issues_df.to_csv(diagnostics_root / "validation_issues.csv", index=False)
    completeness_df.to_csv(diagnostics_root / "field_completeness.csv", index=False)
    source_quality_summary.to_csv(diagnostics_root / "source_quality_summary.csv", index=False)
    state_station_summary.to_csv(diagnostics_root / "state_station_summary.csv", index=False)
    fuel_coverage_summary.to_csv(diagnostics_root / "fuel_coverage_summary.csv", index=False)
    state_fuel_coverage_summary.to_csv(diagnostics_root / "state_fuel_coverage_summary.csv", index=False)
    freshness_summary.to_csv(diagnostics_root / "freshness_summary.csv", index=False)
    wa_concordance_df.to_csv(diagnostics_root / "wa_concordance_suburb_fuel.csv", index=False)
    wa_by_fuel_df.to_csv(diagnostics_root / "wa_concordance_by_fuel.csv", index=False)

    validation_summary = {
        "summary": summary,
        "wa_concordance": wa_summary,
        "issue_count": int(len(issues_df)),
        "error_count": int((issues_df["severity"] == "error").sum()) if not issues_df.empty else 0,
        "warning_count": int((issues_df["severity"] == "warning").sum()) if not issues_df.empty else 0,
        "generated_at": now_utc().isoformat(),
    }
    json_dump(reports_root / "validation_summary.json", validation_summary)

    md = markdown_report(summary, issues_df, source_quality_summary, fuel_coverage_summary, wa_summary)
    (reports_root / "validation_summary.md").write_text(md, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the CheckPetrol dataset end-to-end.")
    parser.add_argument(
        "--output-root",
        default=f"checkpetrol_validation/{iso_utc_now()}",
        help="Output root folder for raw, flat, reports, and diagnostics outputs.",
    )
    parser.add_argument(
        "--fuelwatch-root",
        default=None,
        help="Optional FuelWatch run root. If supplied, the validator will attempt WA concordance against current_prices.csv/current_prices_effective.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = ensure_dir(args.output_root)

    client = CheckPetrolClient()
    manifest, payloads = collect_raw_datasets(client, root)
    stations_snapshot, station_prices_snapshot, station_query_context = build_normalized_tables(payloads, root)
    completeness_df = compute_completeness(stations_snapshot, station_prices_snapshot)

    availability_payload = payloads.get("availability")
    (
        issues_df,
        summary,
        source_quality_summary,
        state_station_summary,
        fuel_coverage_summary,
        state_fuel_coverage_summary,
        freshness_summary,
    ) = validate_stations(
        stations_snapshot,
        station_prices_snapshot,
        station_query_context,
        availability_payload,
    )

    fuelwatch_root = Path(args.fuelwatch_root) if args.fuelwatch_root else None
    wa_concordance_df, wa_summary, wa_by_fuel_df = validate_wa_concordance(
        stations_snapshot,
        station_prices_snapshot,
        fuelwatch_root,
    )

    write_outputs(
        root,
        issues_df,
        completeness_df,
        summary,
        source_quality_summary,
        state_station_summary,
        fuel_coverage_summary,
        state_fuel_coverage_summary,
        freshness_summary,
        wa_concordance_df,
        wa_summary,
        wa_by_fuel_df,
    )

    run_summary = {
        "output_root": str(root),
        "manifest_records": len(manifest),
        "station_count": summary.get("station_count"),
        "station_price_rows": summary.get("station_price_rows"),
        "issue_count": int(len(issues_df)),
        "error_count": int((issues_df["severity"] == "error").sum()) if not issues_df.empty else 0,
        "warning_count": int((issues_df["severity"] == "warning").sum()) if not issues_df.empty else 0,
        "wa_concordance_status": wa_summary.get("wa_concordance_status"),
    }
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
