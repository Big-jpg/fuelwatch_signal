from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


BASE_URL = "https://checkpetrol.com.au"
API_BASE = f"{BASE_URL}/api/v1"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/146.0.0.0 Safari/537.36"
)

FUEL_MAP = {
    "U91": "ULP",
    "U95": "PUP",
    "U98": "98R",
    "E10": "E10",
    "E85": "E85",
    "DSL": "DSL",
    "B20": "BDL",
    "LPG": "LPG",
}


@dataclass
class ProbeResult:
    name: str
    path: str
    ok: bool
    status_code: int | None
    content_type: str | None
    row_count: int | None = None
    warnings: list[str] | None = None
    sample_keys: list[str] | None = None


class CheckPetrolProbe:
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
        response = self.session.get(self._url(path), params=params, timeout=60)
        response.raise_for_status()
        ctype = response.headers.get("Content-Type", "")
        if "json" not in ctype.lower():
            raise ValueError(f"Expected JSON but got {ctype!r} from {path}")
        return response, response.json()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_get(d: dict[str, Any], *keys: str) -> Any:
    current: Any = d
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def parse_isoish(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    text = value.replace("Z", "+00:00")
    try:
        datetime.fromisoformat(text)
        return True
    except Exception:
        return False


def validate_health(payload: Any) -> list[str]:
    warnings: list[str] = []
    if not isinstance(payload, dict):
        return ["Health payload is not an object"]

    if payload.get("status") != "ok":
        warnings.append(f"Unexpected health status: {payload.get('status')!r}")

    if not isinstance(payload.get("uptime"), (int, float)):
        warnings.append("Missing or non-numeric uptime")

    fetched = payload.get("fetchers")
    if not isinstance(fetched, list) or not fetched:
        warnings.append("Missing fetchers array")

    station_counts = payload.get("station_counts")
    if not isinstance(station_counts, list) or not station_counts:
        warnings.append("Missing station_counts array")

    return warnings


def validate_availability(payload: Any) -> list[str]:
    warnings: list[str] = []

    if not isinstance(payload, dict):
        return ["Availability payload is not an object"]

    required = [
        "availability",
        "outage_count",
        "outage_percent",
        "outages_by_state",
        "total_stations",
    ]
    for key in required:
        if key not in payload:
            warnings.append(f"Missing {key}")

    if "availability" in payload and not isinstance(payload["availability"], list):
        warnings.append("availability is not a list")

    if "outages_by_state" in payload and not isinstance(payload["outages_by_state"], list):
        warnings.append("outages_by_state is not a list")

    if "outage_count" in payload and not isinstance(payload["outage_count"], int):
        warnings.append("outage_count is not an integer")

    if "outage_percent" in payload and not isinstance(payload["outage_percent"], (int, float)):
        warnings.append("outage_percent is not numeric")

    if "total_stations" in payload and not isinstance(payload["total_stations"], int):
        warnings.append("total_stations is not an integer")

    return warnings
    warnings: list[str] = []

    if not isinstance(payload, dict):
        return ["Availability payload is not an object"]

    features = payload.get("features")
    if not isinstance(features, list):
        return ["GeoJSON features missing or not a list"]

    if not features:
        warnings.append("No features returned")
        return warnings

    sample = features[0]
    if sample.get("type") != "Feature":
        warnings.append("Feature.type is not 'Feature'")

    geometry = sample.get("geometry")
    if not isinstance(geometry, dict):
        warnings.append("Missing geometry object")
    else:
        coords = geometry.get("coordinates")
        if not isinstance(coords, list) or len(coords) < 2:
            warnings.append("Geometry coordinates missing or malformed")

    props = sample.get("properties")
    if not isinstance(props, dict):
        warnings.append("Missing feature properties")
        return warnings

    required = ["id", "name", "brand", "fuel_type", "state", "source", "updated_at"]
    for key in required:
        if key not in props:
            warnings.append(f"Missing properties.{key}")

    if "all_prices" in props and not isinstance(props["all_prices"], dict):
        warnings.append("properties.all_prices is not an object")

    return warnings


def validate_stations(payload: Any, requested_fuel: str) -> list[str]:
    warnings: list[str] = []

    if not isinstance(payload, dict):
        return ["Stations payload is not an object"]

    features = payload.get("features")
    if not isinstance(features, list):
        return ["Stations.features missing or not a list"]

    if not features:
        warnings.append("No station features returned")
        return warnings

    bad_coords = 0
    bad_updated = 0
    missing_prices = 0
    wrong_fuel = 0
    stale_flags = 0
    outage_flags = 0

    for feature in features[:500]:
        props = feature.get("properties") or {}
        geom = feature.get("geometry") or {}
        coords = geom.get("coordinates") or []

        if len(coords) < 2 or not is_number(coords[0]) or not is_number(coords[1]):
            bad_coords += 1

        updated_at = props.get("updated_at")
        if not parse_isoish(updated_at):
            bad_updated += 1

        fuel_type = props.get("fuel_type")
        if fuel_type != requested_fuel:
            wrong_fuel += 1

        all_prices = props.get("all_prices") or {}
        if requested_fuel not in all_prices:
            missing_prices += 1

        if props.get("is_stale") is True:
            stale_flags += 1

        if props.get("has_outage") is True:
            outage_flags += 1

    if bad_coords:
        warnings.append(f"{bad_coords} sampled rows had invalid coordinates")
    if bad_updated:
        warnings.append(f"{bad_updated} sampled rows had invalid updated_at")
    if wrong_fuel:
        warnings.append(f"{wrong_fuel} sampled rows had mismatched fuel_type")
    if missing_prices:
        warnings.append(f"{missing_prices} sampled rows missing all_prices[{requested_fuel}]")

    if stale_flags:
        warnings.append(f"{stale_flags} sampled rows flagged is_stale=true")
    if outage_flags:
        warnings.append(f"{outage_flags} sampled rows flagged has_outage=true")

    return warnings


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_probe(output_root: str | Path = "checkpetrol_probe") -> dict[str, Any]:
    root = ensure_dir(output_root)
    raw_dir = ensure_dir(root / "raw")
    client = CheckPetrolProbe()

    manifest: list[ProbeResult] = []

    jobs = [
        ("health", "/health", None, validate_health),
        ("availability", "/stats/availability", None, validate_availability),
        ("stations_U91", "/stations", {"fuel": "U91"}, lambda p: validate_stations(p, "U91")),
        ("stations_DSL", "/stations", {"fuel": "DSL"}, lambda p: validate_stations(p, "DSL")),
        ("stations_U98", "/stations", {"fuel": "U98"}, lambda p: validate_stations(p, "U98")),
    ]

    for name, path, params, validator in jobs:
        file_name = f"{name}.json"
        target = raw_dir / file_name
        warnings: list[str] = []
        try:
            response, payload = client.get_json(path, params=params)
            write_json(target, payload)

            if isinstance(payload, dict) and isinstance(payload.get("features"), list):
                row_count = len(payload["features"])
                sample_keys = sorted(list((payload["features"][0].get("properties") or {}).keys()))[:30] if row_count else []
            elif isinstance(payload, dict):
                row_count = None
                sample_keys = sorted(list(payload.keys()))[:30]
            elif isinstance(payload, list):
                row_count = len(payload)
                sample_keys = sorted(list(payload[0].keys()))[:30] if row_count and isinstance(payload[0], dict) else []
            else:
                row_count = None
                sample_keys = []

            warnings.extend(validator(payload))

            manifest.append(
                ProbeResult(
                    name=name,
                    path=path if params is None else f"{path}?{ '&'.join(f'{k}={v}' for k, v in params.items()) }",
                    ok=True,
                    status_code=response.status_code,
                    content_type=response.headers.get("Content-Type"),
                    row_count=row_count,
                    warnings=warnings,
                    sample_keys=sample_keys,
                )
            )
        except Exception as exc:
            manifest.append(
                ProbeResult(
                    name=name,
                    path=path,
                    ok=False,
                    status_code=None,
                    content_type=None,
                    warnings=[str(exc)],
                    sample_keys=[],
                )
            )

    manifest_dicts = [asdict(x) for x in manifest]
    write_json(root / "manifest.json", manifest_dicts)

    return {
        "root": str(root),
        "ok": sum(1 for x in manifest if x.ok),
        "failed": sum(1 for x in manifest if not x.ok),
        "warnings": sum(len(x.warnings or []) for x in manifest),
        "manifest": manifest_dicts,
    }


if __name__ == "__main__":
    result = run_probe()
    print(json.dumps(result, indent=2))