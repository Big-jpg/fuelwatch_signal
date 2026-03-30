from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from fuelwatch_client import (
    FuelWatchClient,
    default_monthly_window,
    ensure_dir,
    iso_utc_now,
    perth_date_range_to_gmt_strings,
    save_blob_listing_and_files,
)

DEFAULT_SITE_FUELS = ["ULP", "DSL", "98R"]
DEFAULT_HISTORY_FUELS = ["ULP"]
DEFAULT_REGIONS = ["Metro"]
VALIDATED_DAILY_HISTORY_FUELS = {"ULP"}
VALIDATED_MONTHLY_HISTORY_FUELS = {"ULP"}
ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass
class ManifestRecord:
    dataset: str
    endpoint: str
    params: dict[str, Any]
    status: str
    rows: int | None
    output_path: str
    warning: str | None = None
    non_null_price_today: int | None = None
    non_null_price_tomorrow: int | None = None
    non_null_delta_abs: int | None = None


class Collector:
    def __init__(self, client: FuelWatchClient, root: str | Path) -> None:
        self.client = client
        self.root = ensure_dir(root)
        self.raw_root = ensure_dir(self.root / "raw")
        self.flat_root = ensure_dir(self.root / "flat")
        self.manifest: list[ManifestRecord] = []

    def record(
        self,
        dataset: str,
        endpoint: str,
        params: dict[str, Any],
        status: str,
        output_path: Path,
        rows: int | None = None,
        warning: str | None = None,
        non_null_price_today: int | None = None,
        non_null_price_tomorrow: int | None = None,
        non_null_delta_abs: int | None = None,
    ) -> None:
        self.manifest.append(
            ManifestRecord(
                dataset,
                endpoint,
                params,
                status,
                rows,
                str(output_path),
                warning,
                non_null_price_today,
                non_null_price_tomorrow,
                non_null_delta_abs,
            )
        )

    def write_json(self, dataset: str, endpoint: str, payload: Any, path: Path, params: dict[str, Any]) -> None:
        self.client.write_json(payload, path)
        rows = len(payload) if isinstance(payload, list) else None
        status = "ok"
        warning = None
        if isinstance(payload, list) and len(payload) == 0:
            status = "empty"
            warning = "Empty payload returned"
        self.record(dataset, endpoint, params, status, path, rows, warning)

    def write_csv(self, name: str, rows: list[dict[str, Any]]) -> Path:
        path = self.flat_root / name
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def save_manifest(self) -> tuple[Path, Path]:
        manifest_json = self.root / "manifest.json"
        manifest_csv = self.root / "manifest.csv"
        records = [asdict(x) for x in self.manifest]
        manifest_json.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame(records).to_csv(manifest_csv, index=False)
        return manifest_json, manifest_csv


def emit_progress(callback: ProgressCallback | None, pile: str, current: int, total: int, detail: str = "", status: str = "running") -> None:
    if callback:
        callback({"pile": pile, "current": current, "total": total, "detail": detail, "status": status})


def load_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _nested_get(obj: Any, *path: str) -> Any:
    current = obj
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None

#
# Price normalization helper
#
# The upstream FuelWatch API and other external sources sometimes provide
# prices in an unexpected unit – e.g. tenths of a cent.  In most cases
# prices are reported in cents per litre, so values larger than 1000 are
# likely scaled by a factor of 10.  This helper normalises any numeric
# price to a sensible range by downscaling unusually large values.  If a
# value is greater than 1000 it is divided by 10 and rounded to three
# decimal places.  Otherwise the value is returned unchanged.  When
# provided with ``None`` or a non‐numeric value the result is also
# ``None``.
def normalize_price_cpl(value: Any) -> float | None:
    """Normalise a price value to cents per litre.

    Values above 1000 are assumed to be scaled by 10 (tenths of a cent)
    and are divided accordingly.  Missing or non‑numeric inputs are
    returned as ``None``.
    """
    v = _coerce_float(value)
    if v is None:
        return None
    # If the value is implausibly large assume it is in tenths of a cent
    # and convert to cents by dividing by 10.  For example ``3029``
    # becomes ``302.9``.
    if v > 1000:
        return round(v / 10.0, 3)
    return v


def _first_float(*values: Any) -> float | None:
    for value in values:
        coerced = _coerce_float(value)
        if coerced is not None:
            return coerced
    return None


def previous_run_roots(current_root: Path) -> list[Path]:
    parent = current_root.parent
    if not parent.exists():
        return []
    runs = [p for p in parent.iterdir() if p.is_dir() and p != current_root]
    return sorted(runs, key=lambda p: p.name, reverse=True)


def _latest_non_null_by_key(df: pd.DataFrame, key_cols: list[str], value_col: str, source_col: str = "source_run_id") -> pd.DataFrame:
    if df.empty or value_col not in df.columns:
        cols = key_cols + [value_col, source_col]
        return pd.DataFrame(columns=cols)
    work = df.copy()
    work = work[work[value_col].notna()]
    if work.empty:
        cols = key_cols + [value_col, source_col]
        return pd.DataFrame(columns=cols)
    work = work.sort_values(source_col, ascending=False)
    return work[key_cols + [value_col, source_col]].drop_duplicates(subset=key_cols, keep="first")


def build_effective_current_prices(current_rows: list[dict[str, Any]], current_root: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    current_df = pd.DataFrame(current_rows).copy()
    if current_df.empty:
        return current_rows, {"cached_rows_added": 0, "cached_today_fills": 0, "cached_tomorrow_fills": 0, "cached_delta_recomputed": 0}

    current_df["source_run_id_today"] = current_df.get("run_id", current_root.name)
    current_df["source_run_id_tomorrow"] = current_df.get("run_id", current_root.name)
    current_df["effective_source"] = "current"
    key_cols = ["site_id", "fuel_type_requested", "product_short_name"]

    prior_frames: list[pd.DataFrame] = []
    for prior_root in previous_run_roots(current_root):
        candidate_paths = [
            prior_root / "flat" / "current_prices_effective.csv",
            prior_root / "flat" / "current_prices.csv",
        ]
        prior_df = pd.DataFrame()
        for candidate in candidate_paths:
            prior_df = load_csv_if_exists(candidate)
            if not prior_df.empty:
                break
        if prior_df.empty:
            continue
        missing_keys = [c for c in key_cols if c not in prior_df.columns]
        if missing_keys:
            continue
        prior_df = prior_df.copy()
        prior_df["source_run_id"] = prior_root.name
        prior_frames.append(prior_df)

    if not prior_frames:
        return current_df.to_dict(orient="records"), {"cached_rows_added": 0, "cached_today_fills": 0, "cached_tomorrow_fills": 0, "cached_delta_recomputed": 0}

    prior_all = pd.concat(prior_frames, ignore_index=True, sort=False)

    today_lookup = _latest_non_null_by_key(prior_all, key_cols, "price_today").rename(columns={"price_today": "cached_price_today", "source_run_id": "cached_today_run_id"})
    tomorrow_lookup = _latest_non_null_by_key(prior_all, key_cols, "price_tomorrow").rename(columns={"price_tomorrow": "cached_price_tomorrow", "source_run_id": "cached_tomorrow_run_id"})
    full_row_lookup = prior_all.sort_values("source_run_id", ascending=False).drop_duplicates(subset=key_cols, keep="first")

    merged = current_df.merge(today_lookup, on=key_cols, how="left")
    merged = merged.merge(tomorrow_lookup, on=key_cols, how="left")

    fill_today_mask = merged["price_today"].isna() & merged["cached_price_today"].notna()
    fill_tomorrow_mask = merged["price_tomorrow"].isna() & merged["cached_price_tomorrow"].notna()
    merged.loc[fill_today_mask, "price_today"] = merged.loc[fill_today_mask, "cached_price_today"]
    merged.loc[fill_today_mask, "source_run_id_today"] = merged.loc[fill_today_mask, "cached_today_run_id"]
    merged.loc[fill_tomorrow_mask, "price_tomorrow"] = merged.loc[fill_tomorrow_mask, "cached_price_tomorrow"]
    merged.loc[fill_tomorrow_mask, "source_run_id_tomorrow"] = merged.loc[fill_tomorrow_mask, "cached_tomorrow_run_id"]
    merged.loc[fill_today_mask | fill_tomorrow_mask, "effective_source"] = "mixed_cache"

    current_keys = set(map(tuple, merged[key_cols].astype(str).itertuples(index=False, name=None)))
    row_candidates = full_row_lookup.copy()
    row_candidates["__key"] = list(map(tuple, row_candidates[key_cols].astype(str).itertuples(index=False, name=None)))
    add_rows = row_candidates[~row_candidates["__key"].isin(current_keys)].copy()
    added_count = len(add_rows)
    if added_count:
        add_rows["run_id"] = current_root.name
        add_rows["source_run_id_today"] = add_rows.get("source_run_id", add_rows.get("run_id", current_root.name))
        add_rows["source_run_id_tomorrow"] = add_rows.get("source_run_id", add_rows.get("run_id", current_root.name))
        add_rows["effective_source"] = "cache_only"
        keep_cols = list(dict.fromkeys(list(merged.columns) + [c for c in add_rows.columns if c in merged.columns]))
        add_rows = add_rows.reindex(columns=keep_cols)
        merged = pd.concat([merged, add_rows], ignore_index=True, sort=False)

    recompute_mask = merged["price_today"].notna() & merged["price_tomorrow"].notna()
    delta_before = merged.get("delta_abs")
    merged.loc[recompute_mask, "delta_abs"] = (pd.to_numeric(merged.loc[recompute_mask, "price_tomorrow"], errors="coerce") - pd.to_numeric(merged.loc[recompute_mask, "price_today"], errors="coerce")).round(3)
    merged.loc[~recompute_mask, "delta_abs"] = pd.NA
    if delta_before is None:
        delta_recomputed = int(recompute_mask.sum())
    else:
        delta_before_num = pd.to_numeric(delta_before, errors="coerce")
        delta_after_num = pd.to_numeric(merged["delta_abs"], errors="coerce")
        delta_recomputed = int((delta_after_num.notna() & delta_before_num.isna()).sum())

    drop_cols = [c for c in ["cached_price_today", "cached_today_run_id", "cached_price_tomorrow", "cached_tomorrow_run_id", "__key", "source_run_id"] if c in merged.columns]
    merged = merged.drop(columns=drop_cols, errors="ignore")
    merged = merged.sort_values(["fuel_type_requested", "product_short_name", "site_name"], na_position="last")
    return merged.to_dict(orient="records"), {
        "cached_rows_added": int(added_count),
        "cached_today_fills": int(fill_today_mask.sum()),
        "cached_tomorrow_fills": int(fill_tomorrow_mask.sum()),
        "cached_delta_recomputed": int(delta_recomputed),
    }


def build_effective_terminal_gate(rows: list[dict[str, Any]], current_root: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    current_df = pd.DataFrame(rows).copy()
    if current_df.empty:
        return rows, {"cached_next_fills": 0, "cached_previous_fills": 0, "cached_delta_recomputed": 0}

    current_df["source_run_id_previous"] = current_df.get("run_id", current_root.name)
    current_df["source_run_id_next"] = current_df.get("run_id", current_root.name)
    current_df["effective_source"] = "current"
    key_cols = ["centre_id", "fuel_type"]

    prior_frames: list[pd.DataFrame] = []
    for prior_root in previous_run_roots(current_root):
        candidate_paths = [
            prior_root / "flat" / "terminal_gate_prices_effective.csv",
            prior_root / "flat" / "terminal_gate_prices.csv",
        ]
        prior_df = pd.DataFrame()
        for candidate in candidate_paths:
            prior_df = load_csv_if_exists(candidate)
            if not prior_df.empty:
                break
        if prior_df.empty:
            continue
        missing_keys = [c for c in key_cols if c not in prior_df.columns]
        if missing_keys:
            continue
        prior_df = prior_df.copy()
        prior_df["source_run_id"] = prior_root.name
        prior_frames.append(prior_df)

    if not prior_frames:
        return current_df.to_dict(orient="records"), {"cached_next_fills": 0, "cached_previous_fills": 0, "cached_delta_recomputed": 0}

    prior_all = pd.concat(prior_frames, ignore_index=True, sort=False)
    prev_lookup = _latest_non_null_by_key(prior_all, key_cols, "price_previous").rename(columns={"price_previous": "cached_price_previous", "source_run_id": "cached_previous_run_id"})
    next_lookup = _latest_non_null_by_key(prior_all, key_cols, "price_next").rename(columns={"price_next": "cached_price_next", "source_run_id": "cached_next_run_id"})
    merged = current_df.merge(prev_lookup, on=key_cols, how="left")
    merged = merged.merge(next_lookup, on=key_cols, how="left")

    fill_prev_mask = merged["price_previous"].isna() & merged["cached_price_previous"].notna()
    fill_next_mask = merged["price_next"].isna() & merged["cached_price_next"].notna()
    merged.loc[fill_prev_mask, "price_previous"] = merged.loc[fill_prev_mask, "cached_price_previous"]
    merged.loc[fill_prev_mask, "source_run_id_previous"] = merged.loc[fill_prev_mask, "cached_previous_run_id"]
    merged.loc[fill_next_mask, "price_next"] = merged.loc[fill_next_mask, "cached_price_next"]
    merged.loc[fill_next_mask, "source_run_id_next"] = merged.loc[fill_next_mask, "cached_next_run_id"]
    merged.loc[fill_prev_mask | fill_next_mask, "effective_source"] = "mixed_cache"

    recompute_mask = pd.to_numeric(merged["price_current"], errors="coerce").notna() & pd.to_numeric(merged["price_next"], errors="coerce").notna()
    delta_before_num = pd.to_numeric(merged.get("delta_next_current"), errors="coerce")
    merged.loc[recompute_mask, "delta_next_current"] = (pd.to_numeric(merged.loc[recompute_mask, "price_next"], errors="coerce") - pd.to_numeric(merged.loc[recompute_mask, "price_current"], errors="coerce")).round(3)
    merged.loc[~recompute_mask, "delta_next_current"] = pd.NA
    delta_after_num = pd.to_numeric(merged["delta_next_current"], errors="coerce")
    delta_recomputed = int((delta_after_num.notna() & delta_before_num.isna()).sum())

    drop_cols = [c for c in ["cached_price_previous", "cached_previous_run_id", "cached_price_next", "cached_next_run_id"] if c in merged.columns]
    merged = merged.drop(columns=drop_cols, errors="ignore")
    return merged.to_dict(orient="records"), {
        "cached_next_fills": int(fill_next_mask.sum()),
        "cached_previous_fills": int(fill_prev_mask.sum()),
        "cached_delta_recomputed": int(delta_recomputed),
    }


def flatten_sites(payload: list[dict[str, Any]], fuel_type: str, run_id: str) -> list[dict[str, Any]]:
    rows = []
    for site in payload:
        address = site.get("address") or {}
        product = site.get("product") or {}
        # Normalise price values to cents per litre.  Upstream values
        # occasionally arrive in tenths of a cent (e.g. 3029 -> 302.9).  Use
        # the helper to ensure consistent units across all data sources.
        price_today_raw = product.get("priceToday")
        price_tomorrow_raw = product.get("priceTomorrow")
        price_today = normalize_price_cpl(price_today_raw)
        price_tomorrow = normalize_price_cpl(price_tomorrow_raw)
        delta_abs = None
        if price_today is not None and price_tomorrow is not None:
            delta_abs = round(price_tomorrow - price_today, 3)

        latitude = _first_float(
            site.get("latitude"),
            site.get("lat"),
            _nested_get(site, "location", "latitude"),
            _nested_get(site, "location", "lat"),
            _nested_get(site, "coordinates", "latitude"),
            _nested_get(site, "coordinates", "lat"),
            address.get("latitude"),
            address.get("lat"),
            _nested_get(address, "coordinates", "latitude"),
            _nested_get(address, "coordinates", "lat"),
        )
        longitude = _first_float(
            site.get("longitude"),
            site.get("lon"),
            site.get("lng"),
            _nested_get(site, "location", "longitude"),
            _nested_get(site, "location", "lon"),
            _nested_get(site, "location", "lng"),
            _nested_get(site, "coordinates", "longitude"),
            _nested_get(site, "coordinates", "lon"),
            _nested_get(site, "coordinates", "lng"),
            address.get("longitude"),
            address.get("lon"),
            address.get("lng"),
            _nested_get(address, "coordinates", "longitude"),
            _nested_get(address, "coordinates", "lon"),
            _nested_get(address, "coordinates", "lng"),
        )

        rows.append(
            {
                "run_id": run_id,
                "fuel_type_requested": fuel_type,
                "site_id": site.get("id"),
                "site_name": site.get("siteName"),
                "brand_name": site.get("brandName"),
                "address_line1": address.get("line1"),
                "suburb": address.get("location"),
                "postcode": address.get("postCode"),
                "state": address.get("state"),
                "latitude": latitude,
                "longitude": longitude,
                "driveway_service": site.get("drivewayService"),
                "is_closed_now": site.get("isClosedNow"),
                "is_closed_all_day_tomorrow": site.get("isClosedAllDayTomorrow"),
                "manned": site.get("manned"),
                "membership_required": site.get("membershipRequired"),
                "operates_24_7": site.get("operates247"),
                "product_short_name": product.get("shortName"),
                # Normalised prices
                "price_today": price_today,
                "price_tomorrow": price_tomorrow,
                "delta_abs": delta_abs,
            }
        )
    return rows


def flatten_daily_history(payload: list[dict[str, Any]], region: str, fuel_type: str, run_id: str) -> list[dict[str, Any]]:
    return [{"run_id": run_id, "region": region, "fuel_type": fuel_type, "publish_date": row.get("publishDate"), "average_price": row.get("averagePrice")} for row in payload]


def flatten_monthly_history(payload: list[dict[str, Any]], region: str, fuel_type: str, run_id: str) -> list[dict[str, Any]]:
    return [{"run_id": run_id, "region": row.get("region", region), "fuel_type": row.get("product", fuel_type), "month": row.get("month"), "average_price": row.get("average")} for row in payload]


def flatten_terminal_gate_prices(
    payload: list[dict[str, Any]],
    run_id: str,
    centre_lookup: dict[Any, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    centre_lookup = centre_lookup or {}
    for centre in payload:
        centre_id = centre.get("id")
        meta = centre_lookup.get(centre_id, {})
        latitude = _first_float(
            centre.get("latitude"),
            centre.get("lat"),
            _nested_get(centre, "coordinates", "latitude"),
            _nested_get(centre, "coordinates", "lat"),
            meta.get("latitude"),
            meta.get("lat"),
            _nested_get(meta, "coordinates", "latitude"),
            _nested_get(meta, "coordinates", "lat"),
        )
        longitude = _first_float(
            centre.get("longitude"),
            centre.get("lon"),
            centre.get("lng"),
            _nested_get(centre, "coordinates", "longitude"),
            _nested_get(centre, "coordinates", "lon"),
            _nested_get(centre, "coordinates", "lng"),
            meta.get("longitude"),
            meta.get("lon"),
            meta.get("lng"),
            _nested_get(meta, "coordinates", "longitude"),
            _nested_get(meta, "coordinates", "lon"),
            _nested_get(meta, "coordinates", "lng"),
        )
        for model in centre.get("terminalGateProductModels") or []:
            current = model.get("priceCurrent")
            nxt = model.get("priceNext")
            previous = model.get("pricePrevious")
            delta = None
            if current is not None and nxt is not None:
                delta = round(float(nxt) - float(current), 3)
            rows.append(
                {
                    "run_id": run_id,
                    "centre_id": centre_id,
                    "centre_name": centre.get("description") or meta.get("description"),
                    "centre_latitude": latitude,
                    "centre_longitude": longitude,
                    "default_terminal_gate_price_change_time": centre.get("defaultTerminalGatePriceChangeTime") or meta.get("defaultTerminalGatePriceChangeTime"),
                    "terminal_gate_price_change_time": centre.get("terminalGatePriceChangeTime") or meta.get("terminalGatePriceChangeTime"),
                    "fuel_type": model.get("fuelType"),
                    "price_previous": previous,
                    "price_current": current,
                    "price_next": nxt,
                    "delta_next_current": delta,
                }
            )
    return rows


def collect_reference(collector: Collector, progress_callback: ProgressCallback | None = None) -> None:
    ref_dir = ensure_dir(collector.raw_root / "reference")
    endpoints = {
        "brands": ("/brands", collector.client.brands),
        "products": ("/products", collector.client.products),
        "suburbs": ("/sites/suburbs", collector.client.suburbs),
        "parameters": ("/configuration/parameter", collector.client.parameters),
        "toggles": ("/configuration/toggle", collector.client.toggles),
        "alerts": ("/alerts", collector.client.alerts),
        "groups": ("/groups", collector.client.groups),
        "regions": ("/region", collector.client.regions),
    }
    total = len(endpoints)
    emit_progress(progress_callback, "Reference datasets", 0, total, "Starting")
    for i, (dataset, (endpoint, fn)) in enumerate(endpoints.items(), start=1):
        path = ref_dir / f"{dataset}.json"
        try:
            collector.write_json(dataset, endpoint, fn(), path, {})
        except Exception as exc:
            collector.record(dataset, endpoint, {}, "error", path, warning=str(exc))
        emit_progress(progress_callback, "Reference datasets", i, total, dataset)
    emit_progress(progress_callback, "Reference datasets", total, total, "Complete", "complete")


def collect_current_prices(collector: Collector, fuels: list[str], run_id: str, progress_callback: ProgressCallback | None = None) -> None:
    out_dir = ensure_dir(collector.raw_root / "sites")
    rows = []
    total = len(fuels)
    emit_progress(progress_callback, "Current site prices", 0, total, "Starting")
    for i, fuel in enumerate(fuels, start=1):
        path = out_dir / f"sites_{fuel}.json"
        params = {"fuelType": fuel}
        try:
            payload = collector.client.current_site_prices(fuel)
            collector.write_json(f"sites_{fuel}", "/sites", payload, path, params)
            rows.extend(flatten_sites(payload, fuel, run_id))
        except Exception as exc:
            collector.record(f"sites_{fuel}", "/sites", params, "error", path, warning=str(exc))
        emit_progress(progress_callback, "Current site prices", i, total, fuel)

    raw_path = collector.write_csv("current_prices.csv", rows)
    flat_df = pd.DataFrame(rows)
    non_null_price_today = int(flat_df["price_today"].notna().sum()) if not flat_df.empty and "price_today" in flat_df.columns else 0
    non_null_price_tomorrow = int(flat_df["price_tomorrow"].notna().sum()) if not flat_df.empty and "price_tomorrow" in flat_df.columns else 0
    non_null_delta_abs = int(flat_df["delta_abs"].notna().sum()) if not flat_df.empty and "delta_abs" in flat_df.columns else 0
    status = "ok"
    warning = None
    if rows and non_null_price_tomorrow == 0:
        status = "warning"
        warning = "Rows returned but all tomorrow prices are null in the flattened current prices dataset"
    elif rows and non_null_price_tomorrow < len(rows):
        status = "warning"
        warning = f"Rows returned but tomorrow prices are only populated for {non_null_price_tomorrow} of {len(rows)} rows"
    collector.record(
        "current_prices_flat",
        "derived",
        {"fuels": fuels},
        status,
        raw_path,
        rows=len(rows),
        warning=warning,
        non_null_price_today=non_null_price_today,
        non_null_price_tomorrow=non_null_price_tomorrow,
        non_null_delta_abs=non_null_delta_abs,
    )

    effective_rows, cache_stats = build_effective_current_prices(rows, collector.root)
    effective_path = collector.write_csv("current_prices_effective.csv", effective_rows)
    effective_df = pd.DataFrame(effective_rows)
    eff_non_null_price_today = int(effective_df["price_today"].notna().sum()) if not effective_df.empty and "price_today" in effective_df.columns else 0
    eff_non_null_price_tomorrow = int(effective_df["price_tomorrow"].notna().sum()) if not effective_df.empty and "price_tomorrow" in effective_df.columns else 0
    eff_non_null_delta_abs = int(effective_df["delta_abs"].notna().sum()) if not effective_df.empty and "delta_abs" in effective_df.columns else 0
    effective_warning = None
    if cache_stats["cached_tomorrow_fills"] > 0 or cache_stats["cached_rows_added"] > 0:
        effective_warning = (
            f"Applied cache backfill: today fills={cache_stats['cached_today_fills']}, "
            f"tomorrow fills={cache_stats['cached_tomorrow_fills']}, "
            f"rows added={cache_stats['cached_rows_added']}, "
            f"delta recomputed={cache_stats['cached_delta_recomputed']}"
        )
    collector.record(
        "current_prices_effective_flat",
        "derived-cache",
        {"fuels": fuels, **cache_stats},
        "ok",
        effective_path,
        rows=len(effective_rows),
        warning=effective_warning,
        non_null_price_today=eff_non_null_price_today,
        non_null_price_tomorrow=eff_non_null_price_tomorrow,
        non_null_delta_abs=eff_non_null_delta_abs,
    )
    emit_progress(progress_callback, "Current site prices", total, total, "Complete", "complete")


def collect_historical_series(collector: Collector, regions: list[str], fuels: list[str], start_date: date, end_date: date, run_id: str, progress_callback: ProgressCallback | None = None) -> None:
    out_dir = ensure_dir(collector.raw_root / "reports")
    date_from, date_to = perth_date_range_to_gmt_strings(start_date, end_date)
    daily_tasks = [(r, f) for r in regions for f in fuels if f in VALIDATED_DAILY_HISTORY_FUELS]
    monthly_tasks = [(r, f) for r in regions for f in fuels if f in VALIDATED_MONTHLY_HISTORY_FUELS]
    daily_rows = []
    monthly_rows = []

    emit_progress(progress_callback, "Historical daily", 0, max(len(daily_tasks), 1), "Starting")
    for i, (region, fuel) in enumerate(daily_tasks, start=1):
        params = {"region": region, "fuelType": fuel}
        path = out_dir / f"price_trends_{region}_{fuel}.json"
        try:
            payload = collector.client.historical_daily_prices(region, fuel)
            collector.write_json(f"price_trends_{region}_{fuel}", "/report/price-trends", payload, path, params)
            daily_rows.extend(flatten_daily_history(payload, region, fuel, run_id))
        except Exception as exc:
            collector.record(f"price_trends_{region}_{fuel}", "/report/price-trends", params, "error", path, warning=str(exc))
        emit_progress(progress_callback, "Historical daily", i, max(len(daily_tasks), 1), f"{region} {fuel}")
    emit_progress(progress_callback, "Historical daily", max(len(daily_tasks),1), max(len(daily_tasks),1), "Complete", "complete")

    emit_progress(progress_callback, "Historical monthly", 0, max(len(monthly_tasks), 1), "Starting")
    for i, (region, fuel) in enumerate(monthly_tasks, start=1):
        params = {"region": region, "fuelType": fuel, "dateFrom": date_from, "dateTo": date_to}
        path = out_dir / f"monthly_average_prices_{region}_{fuel}.json"
        try:
            payload = collector.client.historical_monthly_prices(region, fuel, date_from, date_to)
            collector.write_json(f"monthly_average_prices_{region}_{fuel}", "/report/monthly-average-prices", payload, path, params)
            monthly_rows.extend(flatten_monthly_history(payload, region, fuel, run_id))
        except Exception as exc:
            collector.record(f"monthly_average_prices_{region}_{fuel}", "/report/monthly-average-prices", params, "error", path, warning=str(exc))
        emit_progress(progress_callback, "Historical monthly", i, max(len(monthly_tasks), 1), f"{region} {fuel}")
    emit_progress(progress_callback, "Historical monthly", max(len(monthly_tasks),1), max(len(monthly_tasks),1), "Complete", "complete")

    daily_path = collector.write_csv("historical_daily_prices.csv", daily_rows)
    collector.record("historical_daily_prices_flat", "derived", {"regions": regions, "fuels": fuels}, "ok", daily_path, rows=len(daily_rows))
    monthly_path = collector.write_csv("historical_monthly_prices.csv", monthly_rows)
    collector.record("historical_monthly_prices_flat", "derived", {"regions": regions, "fuels": fuels, "dateFrom": date_from, "dateTo": date_to}, "ok", monthly_path, rows=len(monthly_rows))


def collect_archive_indexes(collector: Collector, progress_callback: ProgressCallback | None = None) -> None:
    out_dir = ensure_dir(collector.raw_root / "reports")
    endpoints = {
        "weekly_retail_prices_index": ("/report/weekly-retail-prices", collector.client.weekly_retail_prices),
        "monthly_retail_prices_index": ("/report/monthly-retail-prices", collector.client.monthly_retail_prices),
        "terminal_gate_prices_index": ("/report/terminal-gate-prices", collector.client.terminal_gate_report),
    }
    total = len(endpoints)
    emit_progress(progress_callback, "Archive indexes", 0, total, "Starting")
    for i, (dataset, (endpoint, fn)) in enumerate(endpoints.items(), start=1):
        path = out_dir / f"{dataset}.json"
        try:
            collector.write_json(dataset, endpoint, fn(), path, {})
        except Exception as exc:
            collector.record(dataset, endpoint, {}, "error", path, warning=str(exc))
        emit_progress(progress_callback, "Archive indexes", i, total, dataset)
    emit_progress(progress_callback, "Archive indexes", total, total, "Complete", "complete")


def collect_terminal_gate(collector: Collector, run_id: str, progress_callback: ProgressCallback | None = None) -> None:
    out_dir = ensure_dir(collector.raw_root / "terminal_gate")
    centres_path = out_dir / "centres.json"
    emit_progress(progress_callback, "Terminal gate", 0, 1, "Loading centres")
    try:
        centres = collector.client.terminal_gate_centres()
        collector.write_json("terminal_gate_centres", "/terminalgate/centres", centres, centres_path, {})
    except Exception as exc:
        collector.record("terminal_gate_centres", "/terminalgate/centres", {}, "error", centres_path, warning=str(exc))
        emit_progress(progress_callback, "Terminal gate", 1, 1, str(exc), "error")
        return
    rows = []
    centre_lookup = {centre.get("id"): centre for centre in centres if centre.get("id") is not None}
    total = len(centres) + 1
    emit_progress(progress_callback, "Terminal gate", 1, total, f"Loaded {len(centres)} centres")
    current = 1
    for centre in centres:
        cid = centre.get("id")
        if cid is None:
            continue
        path = out_dir / f"prices_{cid}.json"
        params = {"centreId": cid}
        try:
            payload = collector.client.terminal_gate_prices(cid)
            collector.write_json(f"terminal_gate_prices_{cid}", f"/terminalgate/prices/{cid}", payload, path, params)
            rows.extend(flatten_terminal_gate_prices(payload, run_id, centre_lookup=centre_lookup))
        except Exception as exc:
            collector.record(f"terminal_gate_prices_{cid}", f"/terminalgate/prices/{cid}", params, "error", path, warning=str(exc))
        current += 1
        emit_progress(progress_callback, "Terminal gate", current, total, centre.get("description", str(cid)))

    out = collector.write_csv("terminal_gate_prices.csv", rows)
    collector.record("terminal_gate_prices_flat", "derived", {}, "ok", out, rows=len(rows))

    effective_rows, cache_stats = build_effective_terminal_gate(rows, collector.root)
    effective_out = collector.write_csv("terminal_gate_prices_effective.csv", effective_rows)
    effective_warning = None
    if cache_stats["cached_next_fills"] > 0:
        effective_warning = (
            f"Applied cache backfill: previous fills={cache_stats['cached_previous_fills']}, "
            f"next fills={cache_stats['cached_next_fills']}, "
            f"delta recomputed={cache_stats['cached_delta_recomputed']}"
        )
    collector.record("terminal_gate_prices_effective_flat", "derived-cache", cache_stats, "ok", effective_out, rows=len(effective_rows), warning=effective_warning)
    emit_progress(progress_callback, "Terminal gate", total, total, "Complete", "complete")


def collect_blobs(collector: Collector, download_weekly: bool, download_monthly: bool, download_wholesale: bool, progress_callback: ProgressCallback | None = None) -> None:
    out_dir = ensure_dir(collector.raw_root / "blobs")
    jobs = []
    if download_weekly:
        jobs.append(("Weekly retail files", "weekly_retail_blobs", collector.client.weekly_retail_prices, out_dir / "weekly_retail_prices"))
    if download_monthly:
        jobs.append(("Monthly retail files", "monthly_retail_blobs", collector.client.monthly_retail_prices, out_dir / "monthly_retail_prices"))
    if download_wholesale:
        jobs.append(("Wholesale CSV files", "wholesale_blobs", collector.client.terminal_gate_report, out_dir / "terminal_gate_prices"))
    for pile_name, dataset, fn, target in jobs:
        try:
            listing = fn()
            total = len(listing)
            emit_progress(progress_callback, pile_name, 0, max(total,1), f"{total} files listed")
            def _blob_progress(current: int, total_files: int, item: dict[str, Any] | None) -> None:
                detail = ''
                if item:
                    detail = f"{item.get('action','')}: {item.get('file_name','')}"
                emit_progress(progress_callback, pile_name, current, max(total_files,1), detail)
            written = save_blob_listing_and_files(collector.client, listing, target, skip_existing=True, progress_callback=_blob_progress)
            collector.record(dataset, "blob-download", {"count": len(listing)}, "ok", target, rows=len(written))
            emit_progress(progress_callback, pile_name, max(total,1), max(total,1), "Complete", "complete")
        except Exception as exc:
            collector.record(dataset, "blob-download", {}, "error", target, warning=str(exc))
            emit_progress(progress_callback, pile_name, 1, 1, str(exc), "error")


def run_collection(output_root: str | Path, site_fuels: list[str], history_regions: list[str], history_fuels: list[str], start_date: date, end_date: date, collect_blob_indexes: bool = True, collect_terminal: bool = True, download_weekly: bool = False, download_monthly: bool = False, download_wholesale: bool = False, progress_callback: ProgressCallback | None = None) -> dict[str, Path]:
    client = FuelWatchClient()
    collector = Collector(client, output_root)
    run_id = Path(output_root).name
    collect_reference(collector, progress_callback)
    collect_current_prices(collector, site_fuels, run_id, progress_callback)
    collect_historical_series(collector, history_regions, history_fuels, start_date, end_date, run_id, progress_callback)
    if collect_blob_indexes:
        collect_archive_indexes(collector, progress_callback)
    if collect_terminal:
        collect_terminal_gate(collector, run_id, progress_callback)
    collect_blobs(collector, download_weekly, download_monthly, download_wholesale, progress_callback)
    manifest_json, manifest_csv = collector.save_manifest()
    return {"root": collector.root, "manifest_json": manifest_json, "manifest_csv": manifest_csv, "flat_dir": collector.flat_root, "raw_dir": collector.raw_root}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect validated FuelWatch reporting datasets")
    start_date, end_date = default_monthly_window()
    parser.add_argument("--output-root", default=f"fuelwatch_runs/{iso_utc_now()}")
    parser.add_argument("--site-fuels", nargs="*", default=DEFAULT_SITE_FUELS)
    parser.add_argument("--history-fuels", nargs="*", default=DEFAULT_HISTORY_FUELS)
    parser.add_argument("--regions", nargs="*", default=DEFAULT_REGIONS)
    parser.add_argument("--start-date", default=start_date.isoformat())
    parser.add_argument("--end-date", default=end_date.isoformat())
    parser.add_argument("--no-archive-indexes", action="store_true")
    parser.add_argument("--no-terminal", action="store_true")
    parser.add_argument("--download-weekly", action="store_true")
    parser.add_argument("--download-monthly", action="store_true")
    parser.add_argument("--download-wholesale", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_collection(
        output_root=args.output_root,
        site_fuels=args.site_fuels,
        history_regions=args.regions,
        history_fuels=args.history_fuels,
        start_date=date.fromisoformat(args.start_date),
        end_date=date.fromisoformat(args.end_date),
        collect_blob_indexes=not args.no_archive_indexes,
        collect_terminal=not args.no_terminal,
        download_weekly=args.download_weekly,
        download_monthly=args.download_monthly,
        download_wholesale=args.download_wholesale,
    )


if __name__ == "__main__":
    main()
