from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import requests


BASE_URL = "https://checkpetrol.com.au/api/v1"
HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/146.0.0.0 Safari/537.36"
    ),
    "Referer": "https://checkpetrol.com.au/",
    "Origin": "https://checkpetrol.com.au",
}


def get_json(path: str, params: dict[str, Any] | None = None) -> Any:
    r = requests.get(f"{BASE_URL}{path}", params=params, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.json()


def is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def station_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for feature in payload.get("features", []):
        props = feature.get("properties") or {}
        geom = feature.get("geometry") or {}
        coords = geom.get("coordinates") or [None, None]

        rows.append(
            {
                "id": props.get("id"),
                "name": props.get("name"),
                "brand": props.get("brand"),
                "source": props.get("source"),
                "state": props.get("state"),
                "suburb": props.get("suburb"),
                "postcode": props.get("postcode"),
                "fuel_type": props.get("fuel_type"),
                "price_cents": props.get("price_cents"),
                "all_prices": props.get("all_prices") or {},
                "has_outage": props.get("has_outage"),
                "is_fallback": props.get("is_fallback"),
                "is_stale": props.get("is_stale"),
                "no_price": props.get("no_price"),
                "updated_at": props.get("updated_at"),
                "longitude": coords[0] if len(coords) > 0 else None,
                "latitude": coords[1] if len(coords) > 1 else None,
            }
        )
    return rows


def summarise_requested_fuel(rows: list[dict[str, Any]], requested_fuel: str) -> dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {"requested_fuel": requested_fuel, "total_rows": 0}

    exact_fuel = 0
    has_requested_in_all_prices = 0
    query_equals_all_prices = 0
    missing_requested_in_all_prices = 0

    by_source_missing = Counter()
    by_state_missing = Counter()
    fuel_type_counter = Counter()
    available_fuel_keys = Counter()

    outage_count = 0
    stale_count = 0
    fallback_count = 0
    no_price_count = 0

    mismatch_examples = []

    for row in rows:
        fuel_type = row["fuel_type"]
        all_prices = row["all_prices"]

        fuel_type_counter[fuel_type] += 1

        for k in all_prices.keys():
            available_fuel_keys[k] += 1

        if fuel_type == requested_fuel:
            exact_fuel += 1

        requested_value = all_prices.get(requested_fuel)
        if is_number(requested_value):
            has_requested_in_all_prices += 1
            if is_number(row["price_cents"]) and float(row["price_cents"]) == float(requested_value):
                query_equals_all_prices += 1
        else:
            missing_requested_in_all_prices += 1
            by_source_missing[row["source"]] += 1
            by_state_missing[row["state"]] += 1
            if len(mismatch_examples) < 25:
                mismatch_examples.append(
                    {
                        "id": row["id"],
                        "name": row["name"],
                        "state": row["state"],
                        "source": row["source"],
                        "fuel_type": fuel_type,
                        "price_cents": row["price_cents"],
                        "all_prices": all_prices,
                    }
                )

        if row["has_outage"]:
            outage_count += 1
        if row["is_stale"]:
            stale_count += 1
        if row["is_fallback"]:
            fallback_count += 1
        if row["no_price"]:
            no_price_count += 1

    return {
        "requested_fuel": requested_fuel,
        "total_rows": total,
        "exact_fuel_rows": exact_fuel,
        "exact_fuel_pct": round(exact_fuel / total * 100, 2),
        "has_requested_in_all_prices_rows": has_requested_in_all_prices,
        "has_requested_in_all_prices_pct": round(has_requested_in_all_prices / total * 100, 2),
        "query_equals_all_prices_rows": query_equals_all_prices,
        "query_equals_all_prices_pct": round(query_equals_all_prices / total * 100, 2),
        "missing_requested_in_all_prices_rows": missing_requested_in_all_prices,
        "missing_requested_in_all_prices_pct": round(missing_requested_in_all_prices / total * 100, 2),
        "outage_count": outage_count,
        "stale_count": stale_count,
        "fallback_count": fallback_count,
        "no_price_count": no_price_count,
        "fuel_type_distribution": dict(fuel_type_counter.most_common()),
        "available_fuel_keys": dict(available_fuel_keys.most_common()),
        "missing_by_source": dict(by_source_missing.most_common()),
        "missing_by_state": dict(by_state_missing.most_common()),
        "mismatch_examples": mismatch_examples,
    }


def run():
    output = Path("checkpetrol_probe_phase2")
    output.mkdir(parents=True, exist_ok=True)

    fuels_to_test = ["U91", "DSL", "LPG", "E10", "E85", "U95", "U98", "98", "P95", "P98", "98RON"]
    summary = []

    for fuel in fuels_to_test:
        payload = get_json("/stations", params={"fuel": fuel})
        rows = station_rows(payload)
        result = summarise_requested_fuel(rows, fuel)

        (output / f"stations_{fuel}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        pd.DataFrame(rows).to_csv(output / f"stations_{fuel}.csv", index=False)
        (output / f"summary_{fuel}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

        summary.append({
            "fuel": fuel,
            "total_rows": result["total_rows"],
            "exact_fuel_pct": result.get("exact_fuel_pct"),
            "has_requested_in_all_prices_pct": result.get("has_requested_in_all_prices_pct"),
            "query_equals_all_prices_pct": result.get("query_equals_all_prices_pct"),
            "missing_requested_in_all_prices_rows": result.get("missing_requested_in_all_prices_rows"),
        })

    pd.DataFrame(summary).to_csv(output / "fuel_probe_summary.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()