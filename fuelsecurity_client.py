"""Client for the FuelSecurity API.

This module wraps a handful of JSON endpoints from https://www.fuelsecurity.com.au/api
and provides simple functions to fetch structured data. Functions return
Python dictionaries parsed from JSON. If a request fails or the response
is invalid, the function will return an empty dict.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import requests

_BASE_URL = "https://www.fuelsecurity.com.au/api"


def _fetch_json(path: str) -> Dict[str, Any]:
    """Fetch JSON from the FuelSecurity API.

    Args:
        path: endpoint path relative to the API root.

    Returns:
        Parsed JSON as a dictionary, or an empty dict if the request fails.
    """
    url = f"{_BASE_URL}/{path.lstrip('/')}"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()  # type: ignore[no-any-return]
    except Exception as exc:  # catch network errors and JSON decode errors
        logging.warning("FuelSecurity API request failed for %s: %s", url, exc)
        return {}


def get_status() -> Dict[str, Any]:
    """Return the status metadata for the FuelSecurity API."""
    return _fetch_json("status")


def get_prices_summary() -> Dict[str, Any]:
    """Return price summaries across capitals and national averages."""
    return _fetch_json("prices/summary")


def get_reserves_current() -> Dict[str, Any]:
    """Return current fuel reserves information."""
    return _fetch_json("reserves/current")


def get_inbound_summary() -> Dict[str, Any]:
    """Return inbound supply summary of ships arriving in Australia."""
    return _fetch_json("supply/inbound-summary")


def get_outages() -> Dict[str, Any]:
    """Return summary and trend of fuel outages by state and fuel type."""
    return _fetch_json("outages")


def get_tankers_map() -> Dict[str, Any]:
    """Return information about tankers tracked by AIS."""
    return _fetch_json("tankers/map")