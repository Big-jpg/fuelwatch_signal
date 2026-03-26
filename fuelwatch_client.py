from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Callable

import requests

BASE_URL = "https://www.fuelwatch.wa.gov.au"
API_BASE = f"{BASE_URL}/api"
PERTH_TZ = timezone(timedelta(hours=8))


class FuelWatchError(RuntimeError):
    pass


@dataclass
class DownloadResult:
    endpoint: str
    path: Path
    rows: int | None = None
    bytes_written: int | None = None


class FuelWatchClient:
    def __init__(self, base_url: str = API_BASE, user_agent: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": user_agent
                or (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/146.0.0.0 Safari/537.36"
                ),
                "Referer": BASE_URL + "/",
                "Origin": BASE_URL,
            }
        )
        self._token: str | None = None

    def _url(self, path: str) -> str:
        if path.startswith(("http://", "https://")):
            return path
        if not path.startswith("/"):
            path = "/" + path
        return self.base_url + path

    def get_token(self, force: bool = False) -> str | None:
        if self._token and not force:
            return self._token
        response = self.session.get(self._url("/token"), timeout=60)
        response.raise_for_status()
        token = response.text.strip().strip('"')
        self._token = token or None
        if self._token:
            self.session.headers["Authorization"] = f"Bearer {self._token}"
        return self._token

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        if "Authorization" not in self.session.headers:
            self.get_token()
        response = self.session.get(self._url(path), params=params, timeout=120)
        if response.status_code == 401:
            self.get_token(force=True)
            response = self.session.get(self._url(path), params=params, timeout=120)
        response.raise_for_status()
        ctype = response.headers.get("Content-Type", "")
        if "json" not in ctype:
            raise FuelWatchError(f"Expected JSON from {path}, got {ctype!r}")
        return response.json()

    def download_binary(self, url: str, dest: str | Path, skip_existing: bool = True) -> Path:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if skip_existing and dest.exists() and dest.stat().st_size > 0:
            return dest
        response = self.session.get(url, timeout=300)
        response.raise_for_status()
        dest.write_bytes(response.content)
        return dest

    def brands(self) -> list[dict[str, Any]]:
        return self.get_json("/brands")

    def products(self) -> list[dict[str, Any]]:
        return self.get_json("/products")

    def suburbs(self) -> list[dict[str, Any]]:
        return self.get_json("/sites/suburbs")

    def parameters(self) -> dict[str, Any]:
        return self.get_json("/configuration/parameter")

    def toggles(self) -> list[dict[str, Any]]:
        return self.get_json("/configuration/toggle")

    def alerts(self) -> list[dict[str, Any]]:
        return self.get_json("/alerts")

    def groups(self) -> list[dict[str, Any]]:
        return self.get_json("/groups")

    def regions(self) -> list[dict[str, Any]]:
        return self.get_json("/region")

    def current_site_prices(self, fuel_type: str) -> list[dict[str, Any]]:
        return self.get_json("/sites", params={"fuelType": fuel_type})

    def historical_daily_prices(self, region: str, fuel_type: str) -> list[dict[str, Any]]:
        return self.get_json("/report/price-trends", params={"region": region, "fuelType": fuel_type})

    def historical_monthly_prices(self, region: str, fuel_type: str, date_from: str, date_to: str) -> list[dict[str, Any]]:
        return self.get_json(
            "/report/monthly-average-prices",
            params={"region": region, "fuelType": fuel_type, "dateFrom": date_from, "dateTo": date_to},
        )

    def weekly_retail_prices(self) -> list[dict[str, Any]]:
        return self.get_json("/report/weekly-retail-prices")

    def monthly_retail_prices(self) -> list[dict[str, Any]]:
        return self.get_json("/report/monthly-retail-prices")

    def terminal_gate_report(self) -> list[dict[str, Any]]:
        return self.get_json("/report/terminal-gate-prices")

    def terminal_gate_centres(self) -> list[dict[str, Any]]:
        return self.get_json("/terminalgate/centres")

    def terminal_gate_prices(self, centre_id: int | str) -> list[dict[str, Any]]:
        return self.get_json(f"/terminalgate/prices/{centre_id}")

    def write_json(self, payload: Any, dest: str | Path) -> DownloadResult:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        rows = len(payload) if isinstance(payload, list) else None
        return DownloadResult(endpoint=str(dest), path=dest, rows=rows, bytes_written=dest.stat().st_size)


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def perth_date_range_to_gmt_strings(start_date: date, end_date: date) -> tuple[str, str]:
    start_local = datetime.combine(start_date, time(0, 0, 0), tzinfo=PERTH_TZ)
    end_local = datetime.combine(end_date, time(23, 59, 59), tzinfo=PERTH_TZ)
    start_utc = start_local.astimezone(timezone.utc)
    end_utc = end_local.astimezone(timezone.utc)
    return (
        start_utc.strftime("%a, %d %b %Y %H:%M:%S GMT"),
        end_utc.strftime("%a, %d %b %Y %H:%M:%S GMT"),
    )


def default_monthly_window(months_back: int = 15) -> tuple[date, date]:
    today = datetime.now(PERTH_TZ).date()
    end_date = today
    first_of_current = today.replace(day=1)
    year = first_of_current.year
    month = first_of_current.month - (months_back - 1)
    while month <= 0:
        month += 12
        year -= 1
    start_date = date(year, month, 1)
    return start_date, end_date


def save_blob_listing_and_files(
    client: FuelWatchClient,
    listing: Iterable[dict[str, Any]],
    output_dir: str | Path,
    limit: int | None = None,
    skip_existing: bool = True,
    progress_callback: Callable[[int, int, dict[str, Any] | None], None] | None = None,
) -> list[Path]:
    output_dir = ensure_dir(output_dir)
    items = list(listing)
    if limit is not None:
        items = items[:limit]
    total = len(items)
    written: list[Path] = []
    for i, item in enumerate(items, start=1):
        url = item.get("url")
        file_name = item.get("fileName") or (Path(url).name if url else None)
        if not url or not file_name:
            if progress_callback:
                progress_callback(i, total, {"action": "skipped", "file_name": file_name or "unknown"})
            continue
        dest = output_dir / file_name
        action = "downloaded"
        if skip_existing and dest.exists() and dest.stat().st_size > 0:
            action = "reused"
        client.download_binary(url, dest, skip_existing=skip_existing)
        written.append(dest)
        if progress_callback:
            progress_callback(i, total, {"action": action, "file_name": file_name})
    return written
