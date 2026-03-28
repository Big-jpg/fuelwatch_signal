from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pandas.errors import EmptyDataError, ParserError

from collector import (
    DEFAULT_HISTORY_FUELS,
    DEFAULT_REGIONS,
    DEFAULT_SITE_FUELS,
    VALIDATED_DAILY_HISTORY_FUELS,
    VALIDATED_MONTHLY_HISTORY_FUELS,
    run_collection,
)
from fuelwatch_client import default_monthly_window, iso_utc_now

st.set_page_config(page_title="FuelWatch Signals", layout="wide")


# ---------- file + formatting helpers ----------
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (EmptyDataError, ParserError):
        return pd.DataFrame()



def load_json(path: Path):
    if path.exists() and path.stat().st_size > 0:
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def choose_existing_path(*paths: Path) -> Path:
    for path in paths:
        if path.exists() and path.stat().st_size > 0:
            return path
    return paths[0]


def fmt_price(value) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.1f} c/L"


def fmt_delta(value) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):+.1f} c"


def fmt_pct(value) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.1f}%"


def safe_float(value):
    if value is None or pd.isna(value):
        return None
    return float(value)


def safe_corr(x: pd.Series, y: pd.Series) -> float | None:
    pairs = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    if len(pairs) < 3:
        return None
    if pairs.iloc[:, 0].nunique() < 2 or pairs.iloc[:, 1].nunique() < 2:
        return None
    return safe_float(pairs.iloc[:, 0].corr(pairs.iloc[:, 1]))


def robust_zscore(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    median = numeric.median()
    mad = (numeric - median).abs().median()
    if pd.isna(mad) or mad == 0:
        return pd.Series(0.0, index=series.index, dtype="float64")
    return (0.6745 * (numeric - median) / mad).fillna(0.0)


def movement_label(value, flat_band: float = 0.0) -> str:
    if value is None or pd.isna(value):
        return "No projection"
    if float(value) > flat_band:
        return "Rising"
    if float(value) < -flat_band:
        return "Falling"
    return "Flat"


def quadrant_label(row: pd.Series) -> str:
    if pd.isna(row.get("delta_abs")) or pd.isna(row.get("vs_suburb_today")):
        return "Unclassified"
    if row["vs_suburb_today"] >= 0 and row["delta_abs"] >= 0:
        return "Expensive + Rising"
    if row["vs_suburb_today"] >= 0 and row["delta_abs"] < 0:
        return "Expensive + Falling"
    if row["vs_suburb_today"] < 0 and row["delta_abs"] >= 0:
        return "Cheap + Rising"
    return "Cheap + Falling"


def soften_theme() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1600px;}
        .stMetric {
            background: rgba(255,255,255,0.025);
            border: 1px solid rgba(255,255,255,0.06);
            padding: .75rem 1rem;
            border-radius: 14px;
        }
        .small-note {
            font-size: 0.9rem;
            color: rgba(250,250,250,0.72);
            margin-top: -0.25rem;
            margin-bottom: 0.5rem;
        }
        h1, h2, h3 {letter-spacing: -0.02em;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def display_table(
    df: pd.DataFrame,
    columns: list[str],
    price_cols: list[str] | None = None,
    delta_cols: list[str] | None = None,
    pct_cols: list[str] | None = None,
    height: int = 360,
):
    if df.empty:
        st.info("No rows available.")
        return
    cols = [c for c in columns if c in df.columns]
    view = df[cols].copy()
    price_cols = [c for c in (price_cols or []) if c in view.columns]
    delta_cols = [c for c in (delta_cols or []) if c in view.columns]
    pct_cols = [c for c in (pct_cols or []) if c in view.columns]
    fmt = {c: "{:.1f}" for c in price_cols}
    fmt.update({c: "{:+.1f}" for c in delta_cols})
    fmt.update({c: "{:.1f}%" for c in pct_cols})
    st.dataframe(view.style.format(fmt), use_container_width=True, height=height)


def render_progress_cards(config: dict[str, str]):
    st.subheader("Collection progress")
    cols = st.columns(2)
    pile_ui = {}
    for idx, (pile, detail) in enumerate(config.items()):
        with cols[idx % 2]:
            card = st.container(border=True)
            card.markdown(f"**{pile}**")
            metric = card.caption("0 / 0")
            bar = card.progress(0.0)
            detail_placeholder = card.caption(detail)
            pile_ui[pile] = {"metric": metric, "bar": bar, "detail": detail_placeholder}
    return pile_ui


def update_progress_cards(pile_ui: dict, event: dict):
    pile = event["pile"]
    if pile not in pile_ui:
        return
    current = int(event.get("current", 0))
    total = int(event.get("total", 0))
    ratio = 0.0 if total <= 0 else min(max(current / total, 0.0), 1.0)
    pile_ui[pile]["metric"].caption(f"{current} / {total}")
    pile_ui[pile]["bar"].progress(ratio)
    suffix = ""
    if event.get("status") == "complete":
        suffix = " - done"
    elif event.get("status") == "error":
        suffix = " - issue"
    pile_ui[pile]["detail"].caption(f"{event.get('detail', '')}{suffix}")


# ---------- coordinate + distance helpers ----------
def standardise_coordinate_columns(
    df: pd.DataFrame,
    lat_candidates: list[str],
    lon_candidates: list[str],
    latitude_name: str = "latitude",
    longitude_name: str = "longitude",
) -> pd.DataFrame:
    out = df.copy()
    lat_source = next((c for c in lat_candidates if c in out.columns), None)
    lon_source = next((c for c in lon_candidates if c in out.columns), None)
    if latitude_name not in out.columns:
        out[latitude_name] = pd.NA
    if longitude_name not in out.columns:
        out[longitude_name] = pd.NA
    if lat_source:
        out[latitude_name] = pd.to_numeric(out[lat_source], errors="coerce")
    else:
        out[latitude_name] = pd.to_numeric(out[latitude_name], errors="coerce")
    if lon_source:
        out[longitude_name] = pd.to_numeric(out[lon_source], errors="coerce")
    else:
        out[longitude_name] = pd.to_numeric(out[longitude_name], errors="coerce")
    return out


def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(pd.to_numeric(lat1, errors="coerce"))
    lon1 = np.radians(pd.to_numeric(lon1, errors="coerce"))
    lat2 = np.radians(pd.to_numeric(lat2, errors="coerce"))
    lon2 = np.radians(pd.to_numeric(lon2, errors="coerce"))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0088 * c


def distance_bucket(value) -> str:
    if value is None or pd.isna(value):
        return "Unknown"
    value = float(value)
    if value < 10:
        return "0-10 km"
    if value < 25:
        return "10-25 km"
    if value < 50:
        return "25-50 km"
    if value < 100:
        return "50-100 km"
    return "100+ km"


def collapse_categories(series: pd.Series, top_n: int = 12, other_label: str = "Other") -> pd.Series:
    s = series.fillna("Unknown").astype(str)
    keep = s.value_counts().head(top_n).index
    return s.where(s.isin(keep), other_label)


@st.cache_data(show_spinner=False)
def load_distance_nodes(root_str: str, terminal_csv_path: str | None = None) -> pd.DataFrame:
    root = Path(root_str)
    candidates = [
        root / "distance_nodes.csv",
        root / "reference" / "distance_nodes.csv",
        root.parent / "distance_nodes.csv",
        Path("distance_nodes.csv"),
    ]
    frames: list[pd.DataFrame] = []
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            df = pd.read_csv(candidate)
            df["node_source"] = str(candidate)
            frames.append(df)
            break

    if terminal_csv_path:
        terminal_df = load_csv(Path(terminal_csv_path))
        if not terminal_df.empty:
            terminal_df = standardise_coordinate_columns(
                terminal_df,
                ["centre_latitude", "latitude", "lat"],
                ["centre_longitude", "longitude", "lon", "lng"],
                latitude_name="latitude",
                longitude_name="longitude",
            )
            if terminal_df[["latitude", "longitude"]].notna().all(axis=1).any():
                terminal_nodes = (
                    terminal_df[["centre_id", "centre_name", "latitude", "longitude"]]
                    .dropna(subset=["latitude", "longitude"])
                    .drop_duplicates()
                    .rename(columns={"centre_name": "node_name"})
                )
                terminal_nodes["node_type"] = "terminal"
                terminal_nodes["node_source"] = "terminal_gate_prices"
                frames.append(terminal_nodes)

    if not frames:
        return pd.DataFrame(columns=["node_name", "node_type", "latitude", "longitude", "node_source"])

    nodes = pd.concat(frames, ignore_index=True, sort=False)
    if "node_name" not in nodes.columns:
        return pd.DataFrame(columns=["node_name", "node_type", "latitude", "longitude", "node_source"])
    if "node_type" not in nodes.columns:
        nodes["node_type"] = "reference"
    nodes = standardise_coordinate_columns(nodes, ["latitude", "lat"], ["longitude", "lon", "lng"])
    nodes = nodes.dropna(subset=["latitude", "longitude"])
    nodes["node_name"] = nodes["node_name"].astype(str)
    return nodes.drop_duplicates(subset=["node_name", "latitude", "longitude"])


def attach_nearest_nodes(sites: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    if sites.empty:
        return sites.copy()
    out = sites.copy()
    if nodes.empty:
        out["nearest_node"] = pd.NA
        out["nearest_node_type"] = pd.NA
        out["distance_km"] = pd.NA
        out["distance_band"] = "Unknown"
        return out
    if not {"latitude", "longitude"}.issubset(out.columns):
        out["nearest_node"] = pd.NA
        out["nearest_node_type"] = pd.NA
        out["distance_km"] = pd.NA
        out["distance_band"] = "Unknown"
        return out

    out = out.copy()
    best_distance = pd.Series(np.nan, index=out.index, dtype="float64")
    best_node = pd.Series(pd.NA, index=out.index, dtype="object")
    best_type = pd.Series(pd.NA, index=out.index, dtype="object")
    valid_sites = out[["latitude", "longitude"]].notna().all(axis=1)
    for _, node in nodes.iterrows():
        dist = pd.Series(np.nan, index=out.index, dtype="float64")
        dist.loc[valid_sites] = haversine_km(
            out.loc[valid_sites, "latitude"],
            out.loc[valid_sites, "longitude"],
            node["latitude"],
            node["longitude"],
        )
        update_mask = best_distance.isna() | ((dist < best_distance) & dist.notna())
        best_distance.loc[update_mask] = dist.loc[update_mask]
        best_node.loc[update_mask] = node.get("node_name")
        best_type.loc[update_mask] = node.get("node_type")

    out["nearest_node"] = best_node
    out["nearest_node_type"] = best_type
    out["distance_km"] = best_distance
    out["distance_band"] = out["distance_km"].apply(distance_bucket)
    return out


# ---------- data preparation ----------
def projection_coverage(df: pd.DataFrame) -> dict[str, float | int | None]:
    if df.empty:
        return {
            "rows": 0,
            "today_rows": 0,
            "tomorrow_rows": 0,
            "delta_rows": 0,
            "today_pct": None,
            "tomorrow_pct": None,
            "delta_pct": None,
        }
    rows = len(df)
    today_rows = int(df["price_today"].notna().sum()) if "price_today" in df.columns else 0
    tomorrow_rows = int(df["price_tomorrow"].notna().sum()) if "price_tomorrow" in df.columns else 0
    delta_rows = int(df["delta_abs"].notna().sum()) if "delta_abs" in df.columns else 0
    return {
        "rows": rows,
        "today_rows": today_rows,
        "tomorrow_rows": tomorrow_rows,
        "delta_rows": delta_rows,
        "today_pct": (today_rows / rows * 100.0) if rows else None,
        "tomorrow_pct": (tomorrow_rows / rows * 100.0) if rows else None,
        "delta_pct": (delta_rows / rows * 100.0) if rows else None,
    }


def prepare_current(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out = standardise_coordinate_columns(
        out,
        ["latitude", "lat", "site_latitude", "siteLatitude"],
        ["longitude", "lon", "lng", "site_longitude", "siteLongitude"],
        latitude_name="latitude",
        longitude_name="longitude",
    )

    for col in ["price_today", "price_tomorrow", "delta_abs"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ["is_closed_now", "is_closed_all_day_tomorrow", "manned", "membership_required", "operates_24_7"]:
        if col in out.columns:
            out[col] = out[col].fillna(False).astype(bool)

    fuel_group = ["product_short_name"]
    suburb_group = ["product_short_name", "suburb"]
    brand_group = ["product_short_name", "brand_name"]

    out["market_median_today"] = out.groupby(fuel_group)["price_today"].transform("median")
    out["market_mean_today"] = out.groupby(fuel_group)["price_today"].transform("mean")
    out["market_median_delta"] = out.groupby(fuel_group)["delta_abs"].transform("median")
    out["suburb_median_today"] = out.groupby(suburb_group)["price_today"].transform("median")
    out["suburb_median_tomorrow"] = out.groupby(suburb_group)["price_tomorrow"].transform("median")
    out["suburb_median_delta"] = out.groupby(suburb_group)["delta_abs"].transform("median")
    out["brand_median_delta"] = out.groupby(brand_group)["delta_abs"].transform("median")
    out["brand_mean_delta"] = out.groupby(brand_group)["delta_abs"].transform("mean")

    out["vs_market_today"] = out["price_today"] - out["market_median_today"]
    out["vs_suburb_today"] = out["price_today"] - out["suburb_median_today"]
    out["vs_suburb_tomorrow"] = out["price_tomorrow"] - out["suburb_median_tomorrow"]
    out["delta_vs_market"] = out["delta_abs"] - out["market_median_delta"]
    out["delta_vs_suburb"] = out["delta_abs"] - out["suburb_median_delta"]
    out["delta_vs_brand"] = out["delta_abs"] - out["brand_median_delta"]

    out["price_robust_z"] = out.groupby("product_short_name")["price_today"].transform(robust_zscore)
    out["delta_robust_z"] = out.groupby("product_short_name")["delta_abs"].transform(robust_zscore)
    out["vs_suburb_robust_z"] = out.groupby("product_short_name")["vs_suburb_today"].transform(robust_zscore)
    out["outlier_price"] = out["price_robust_z"].abs() >= 2.5
    out["outlier_delta"] = out["delta_robust_z"].abs() >= 2.5

    out["movement"] = out["delta_abs"].apply(movement_label)
    out["quadrant"] = out.apply(quadrant_label, axis=1)
    out["signal_score"] = (
        out["delta_abs"].abs().fillna(0)
        + out["delta_vs_suburb"].abs().fillna(0)
        + out["delta_vs_brand"].abs().fillna(0)
    )
    return out


def prepare_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["average_price"] = pd.to_numeric(out["average_price"], errors="coerce")
    out["publish_date"] = pd.to_datetime(out["publish_date"], errors="coerce")
    out = out.sort_values(["region", "fuel_type", "publish_date"])
    grp = out.groupby(["region", "fuel_type"])["average_price"]
    out["daily_delta"] = grp.diff()
    out["rolling_7d"] = grp.transform(lambda s: s.rolling(7, min_periods=1).mean())
    out["volatility_14d"] = grp.transform(lambda s: s.diff().rolling(14, min_periods=3).std())
    out["change_7d"] = grp.transform(lambda s: s.diff(7))
    return out


def prepare_monthly(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["average_price"] = pd.to_numeric(out["average_price"], errors="coerce")
    out["month_date"] = pd.to_datetime(out["month"], format="%B %Y", errors="coerce")
    out = out.sort_values(["region", "fuel_type", "month_date"])
    grp = out.groupby(["region", "fuel_type"])["average_price"]
    out["mom_delta"] = grp.diff()
    out["mom_pct"] = grp.pct_change() * 100.0
    out["rolling_3m"] = grp.transform(lambda s: s.rolling(3, min_periods=1).mean())
    return out


def prepare_terminal(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out = standardise_coordinate_columns(
        out,
        ["centre_latitude", "latitude", "lat"],
        ["centre_longitude", "longitude", "lon", "lng"],
        latitude_name="centre_latitude",
        longitude_name="centre_longitude",
    )
    for col in ["price_previous", "price_current", "price_next", "delta_next_current"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["delta_current_previous"] = out["price_current"] - out["price_previous"]
    return out


# ---------- aggregations ----------
def signal_summary(df: pd.DataFrame) -> dict[str, float | int | None]:
    coverage = projection_coverage(df)
    if df.empty:
        return {
            "rising": 0,
            "flat": 0,
            "falling": 0,
            "median_delta": None,
            "p90_delta": None,
            "today_spread": None,
            "tomorrow_spread": None,
            "projection_state": "no_rows",
            "outlier_share": None,
            "price_delta_corr": None,
            "distance_delta_corr": None,
            **coverage,
        }
    delta = df["delta_abs"].dropna()
    rising = int((delta > 0).sum())
    flat = int((delta == 0).sum())
    falling = int((delta < 0).sum())
    projection_state = "ready" if not delta.empty else "missing_tomorrow_prices"
    outlier_share = ((df["outlier_delta"].fillna(False)).mean() * 100.0) if "outlier_delta" in df.columns and len(df) else None
    distance_delta_corr = safe_corr(df.get("distance_km", pd.Series(dtype=float)), df.get("delta_abs", pd.Series(dtype=float)))
    return {
        "rising": rising,
        "flat": flat,
        "falling": falling,
        "median_delta": float(delta.median()) if not delta.empty else None,
        "p90_delta": float(delta.quantile(0.9)) if not delta.empty else None,
        "today_spread": float(df["price_today"].max() - df["price_today"].min()) if df["price_today"].notna().any() else None,
        "tomorrow_spread": float(df["price_tomorrow"].max() - df["price_tomorrow"].min()) if df["price_tomorrow"].notna().any() else None,
        "projection_state": projection_state,
        "outlier_share": outlier_share,
        "price_delta_corr": safe_corr(df["price_today"], df["delta_abs"]),
        "distance_delta_corr": distance_delta_corr,
        **coverage,
    }


def suburb_stress(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    agg = (
        df.groupby(["product_short_name", "suburb"], dropna=False)
        .agg(
            sites=("site_id", "nunique"),
            avg_today=("price_today", "mean"),
            avg_tomorrow=("price_tomorrow", "mean"),
            avg_delta=("delta_abs", "mean"),
            median_delta=("delta_abs", "median"),
            max_delta=("delta_abs", "max"),
            delta_std=("delta_abs", "std"),
            pct_rising=("delta_abs", lambda s: (pd.to_numeric(s, errors="coerce").fillna(0) > 0).mean() * 100),
            pct_falling=("delta_abs", lambda s: (pd.to_numeric(s, errors="coerce").fillna(0) < 0).mean() * 100),
        )
        .reset_index()
    )
    agg["stress_index"] = agg["avg_delta"].fillna(0) + agg["delta_std"].fillna(0) + agg["pct_rising"].fillna(0) / 20.0
    return agg.sort_values(["stress_index", "avg_delta"], ascending=False)


def suburb_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    agg = (
        df.groupby("suburb", dropna=False)
        .agg(
            sites=("site_id", "nunique"),
            median_today=("price_today", "median"),
            mean_today=("price_today", "mean"),
            median_delta=("delta_abs", "median"),
            mean_delta=("delta_abs", "mean"),
            pct_rising=("delta_abs", lambda s: (pd.to_numeric(s, errors="coerce") > 0).mean() * 100),
            pct_falling=("delta_abs", lambda s: (pd.to_numeric(s, errors="coerce") < 0).mean() * 100),
            outlier_share=("outlier_delta", lambda s: pd.Series(s).fillna(False).mean() * 100),
            median_distance_km=("distance_km", "median"),
        )
        .reset_index()
    )
    agg["pressure_index"] = agg["mean_delta"].fillna(0) + agg["pct_rising"].fillna(0) / 25.0 + agg["outlier_share"].fillna(0) / 50.0
    return agg.sort_values(["pressure_index", "sites"], ascending=[False, False])


def brand_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    agg = (
        df.groupby("brand_name", dropna=False)
        .agg(
            sites=("site_id", "nunique"),
            median_today=("price_today", "median"),
            mean_today=("price_today", "mean"),
            median_delta=("delta_abs", "median"),
            mean_delta=("delta_abs", "mean"),
            pct_rising=("delta_abs", lambda s: (pd.to_numeric(s, errors="coerce") > 0).mean() * 100),
            pct_falling=("delta_abs", lambda s: (pd.to_numeric(s, errors="coerce") < 0).mean() * 100),
            outlier_share=("outlier_delta", lambda s: pd.Series(s).fillna(False).mean() * 100),
            median_distance_km=("distance_km", "median"),
        )
        .reset_index()
    )
    agg["pressure_index"] = agg["mean_delta"].fillna(0) + agg["pct_rising"].fillna(0) / 25.0 + agg["outlier_share"].fillna(0) / 50.0
    return agg.sort_values(["pressure_index", "sites"], ascending=[False, False])


def fuel_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    agg = (
        df.groupby("product_short_name", dropna=False)
        .agg(
            sites=("site_id", "nunique"),
            mean_today=("price_today", "mean"),
            median_today=("price_today", "median"),
            mean_delta=("delta_abs", "mean"),
            median_delta=("delta_abs", "median"),
            pct_rising=("delta_abs", lambda s: (pd.to_numeric(s, errors="coerce") > 0).mean() * 100),
            pct_falling=("delta_abs", lambda s: (pd.to_numeric(s, errors="coerce") < 0).mean() * 100),
            outlier_share=("outlier_delta", lambda s: pd.Series(s).fillna(False).mean() * 100),
        )
        .reset_index()
    )
    return agg.sort_values("mean_delta", ascending=False)


def distance_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "distance_km" not in df.columns:
        return pd.DataFrame()
    scoped = df[df["distance_km"].notna()].copy()
    if scoped.empty:
        return pd.DataFrame()
    agg = (
        scoped.groupby("distance_band", dropna=False)
        .agg(
            sites=("site_id", "nunique"),
            mean_today=("price_today", "mean"),
            mean_delta=("delta_abs", "mean"),
            median_delta=("delta_abs", "median"),
            pct_rising=("delta_abs", lambda s: (pd.to_numeric(s, errors="coerce") > 0).mean() * 100),
            pct_falling=("delta_abs", lambda s: (pd.to_numeric(s, errors="coerce") < 0).mean() * 100),
            max_distance_km=("distance_km", "max"),
        )
        .reset_index()
    )
    order = ["0-10 km", "10-25 km", "25-50 km", "50-100 km", "100+ km", "Unknown"]
    agg["distance_band"] = pd.Categorical(agg["distance_band"], categories=order, ordered=True)
    return agg.sort_values("distance_band")


def market_regime_text(summary: dict[str, float | int | None]) -> str:
    projection_state = summary.get("projection_state")
    tomorrow_pct = summary.get("tomorrow_pct")
    if projection_state == "no_rows":
        return "No current rows available for this fuel."
    if projection_state == "missing_tomorrow_prices":
        return "Tomorrow pricing has not been published for this fuel yet, or the upstream payload is missing tomorrow prices."
    if tomorrow_pct is not None and tomorrow_pct < 80:
        return f"Projection coverage is partial ({tomorrow_pct:.1f}% of rows have tomorrow prices). Interpret signal metrics cautiously."
    rising = summary["rising"]
    falling = summary["falling"]
    p90 = summary["p90_delta"] or 0
    if rising > falling * 4 and p90 >= 10:
        return "Broad upward pressure with strong outliers."
    if falling > rising * 1.5 and abs(summary.get("median_delta") or 0) >= 1:
        return "Broad softening pressure is visible rather than isolated discounting."
    if rising > falling and p90 >= 5:
        return "Mostly rising market with moderate volatility."
    if falling > rising:
        return "Mixed-to-falling market."
    return "Relatively stable market."


# ---------- snapshot history ----------
def parse_run_timestamp(run_name: str):
    return pd.to_datetime(run_name, format="%Y%m%dT%H%M%SZ", errors="coerce", utc=True)


@st.cache_data(show_spinner=False)
def load_snapshot_history(root_str: str, use_cached_effective: bool, max_runs: int = 24) -> pd.DataFrame:
    root = Path(root_str)
    parent = root.parent if root.parent.exists() else root
    if not parent.exists():
        return pd.DataFrame()
    run_roots = sorted([p for p in parent.iterdir() if p.is_dir()], key=lambda p: p.name)
    run_roots = run_roots[-max_runs:]
    frames: list[pd.DataFrame] = []
    for run_root in run_roots:
        candidates = []
        if use_cached_effective:
            candidates.append(run_root / "flat" / "current_prices_effective.csv")
        candidates.append(run_root / "flat" / "current_prices.csv")
        chosen = next((p for p in candidates if p.exists() and p.stat().st_size > 0), None)
        if chosen is None:
            continue
        df = load_csv(chosen)
        if df.empty:
            continue
        df["snapshot_run_id"] = run_root.name
        df["snapshot_ts"] = parse_run_timestamp(run_root.name)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = prepare_current(combined)
    return combined.dropna(subset=["snapshot_ts"])


def snapshot_trajectory(snapshot_df: pd.DataFrame, fuel: str, dimension: str) -> pd.DataFrame:
    if snapshot_df.empty:
        return pd.DataFrame()
    scoped = snapshot_df[snapshot_df["product_short_name"] == fuel].copy()
    if scoped.empty or dimension not in scoped.columns:
        return pd.DataFrame()
    agg = (
        scoped.groupby(["snapshot_ts", dimension], dropna=False)
        .agg(
            sites=("site_id", "nunique"),
            mean_today=("price_today", "mean"),
            median_today=("price_today", "median"),
            mean_delta=("delta_abs", "mean"),
            median_delta=("delta_abs", "median"),
            pct_rising=("delta_abs", lambda s: (pd.to_numeric(s, errors="coerce") > 0).mean() * 100),
            pct_falling=("delta_abs", lambda s: (pd.to_numeric(s, errors="coerce") < 0).mean() * 100),
        )
        .reset_index()
    )
    return agg.sort_values([dimension, "snapshot_ts"])


# ---------- app ----------
soften_theme()
start_default, end_default = default_monthly_window()
if "last_root" not in st.session_state:
    st.session_state["last_root"] = None

st.title("FuelWatch Signals")
st.caption("Collect latest data, then review signal-first fuel analytics for current, tomorrow, distance, and historical movement.")

with st.sidebar:
    st.header("Collect latest data")
    output_root = st.text_input("Output root", value=f"fuelwatch_runs/{iso_utc_now()}")
    site_fuels = st.multiselect("Current price fuels", ["ULP", "PUP", "DSL", "BDL", "LPG", "98R", "E85"], default=DEFAULT_SITE_FUELS)
    history_regions = st.multiselect("Historical regions", ["Metro", "Country", "Albany", "Bunbury", "Geraldton", "Kalgoorlie", "North", "South"], default=DEFAULT_REGIONS)
    history_fuels = st.multiselect("Historical JSON fuels", ["ULP", "PUP", "DSL", "BDL", "LPG", "98R", "E85"], default=DEFAULT_HISTORY_FUELS)
    start_date = st.date_input("Monthly history start", value=start_default)
    end_date = st.date_input("Monthly history end", value=end_default)
    st.divider()
    collect_archive_indexes = st.checkbox("Collect archive indexes", value=True)
    collect_terminal = st.checkbox("Collect terminal gate", value=True)
    download_monthly = st.checkbox("Download monthly retail CSV files")
    download_weekly = st.checkbox("Download weekly retail PDFs")
    download_wholesale = st.checkbox("Download wholesale CSV files")
    use_cached_effective = st.checkbox(
        "Use latest available cached values",
        value=True,
        help="When enabled, the app will prefer effective flat files that backfill missing current-cycle values from earlier runs.",
    )
    snapshot_runs_to_scan = st.slider("Run history depth", min_value=6, max_value=48, value=24, step=6)
    collect_btn = st.button("Collect latest data", type="primary", use_container_width=True)

progress_area = st.container()
if collect_btn:
    if isinstance(start_date, tuple) or isinstance(end_date, tuple):
        st.error("Select single dates for the monthly history window.")
    elif start_date > end_date:
        st.error("Start date must be on or before end date.")
    else:
        with progress_area:
            config = {
                "Reference datasets": "Waiting",
                "Current site prices": "Waiting",
                "Historical daily": "Waiting",
                "Historical monthly": "Waiting",
            }
            if collect_archive_indexes:
                config["Archive indexes"] = "Waiting"
            if collect_terminal:
                config["Terminal gate"] = "Waiting"
            if download_weekly:
                config["Weekly retail files"] = "Waiting"
            if download_monthly:
                config["Monthly retail files"] = "Waiting"
            if download_wholesale:
                config["Wholesale CSV files"] = "Waiting"
            pile_ui = render_progress_cards(config)
            log_box = st.container(border=True)
            log_placeholder = log_box.empty()
            log_lines: list[str] = []

            def _progress(event: dict):
                update_progress_cards(pile_ui, event)
                line = f"{event['pile']}: {event['current']}/{event['total']} - {event.get('detail', '')}"
                if event.get("status") != "running":
                    line += f" [{event.get('status')}]"
                log_lines.append(line)
                log_placeholder.code("\n".join(log_lines[-12:]))

            with st.status("Collecting latest data...", expanded=False) as status:
                result = run_collection(
                    output_root=output_root,
                    site_fuels=site_fuels,
                    history_regions=history_regions,
                    history_fuels=history_fuels,
                    start_date=start_date,
                    end_date=end_date,
                    collect_blob_indexes=collect_archive_indexes,
                    collect_terminal=collect_terminal,
                    download_weekly=download_weekly,
                    download_monthly=download_monthly,
                    download_wholesale=download_wholesale,
                    progress_callback=_progress,
                )
                st.session_state["last_root"] = str(result["root"])
                status.update(label=f"Done: {result['root']}", state="complete")
                st.success("Latest data collected. Review the signal tabs below.")

root = Path(st.session_state.get("last_root") or output_root)
manifest_df = load_csv(root / "manifest.csv")
current_prices_path = choose_existing_path(
    root / "flat" / "current_prices_effective.csv" if use_cached_effective else root / "flat" / "current_prices.csv",
    root / "flat" / "current_prices.csv",
)
terminal_prices_path = choose_existing_path(
    root / "flat" / "terminal_gate_prices_effective.csv" if use_cached_effective else root / "flat" / "terminal_gate_prices.csv",
    root / "flat" / "terminal_gate_prices.csv",
)
current_df = prepare_current(load_csv(current_prices_path))
daily_df = prepare_daily(load_csv(root / "flat" / "historical_daily_prices.csv"))
monthly_df = prepare_monthly(load_csv(root / "flat" / "historical_monthly_prices.csv"))
terminal_df = prepare_terminal(load_csv(terminal_prices_path))
parameters = load_json(root / "raw" / "reference" / "parameters.json") or {}
nodes_df = load_distance_nodes(str(root), str(terminal_prices_path))
current_df = attach_nearest_nodes(current_df, nodes_df)
snapshot_df = load_snapshot_history(str(root), use_cached_effective=use_cached_effective, max_runs=snapshot_runs_to_scan)
if not snapshot_df.empty:
    snapshot_df = attach_nearest_nodes(snapshot_df, nodes_df)

summary_tab, signals_tab, current_tab, historical_tab, terminal_tab, archive_tab = st.tabs([
    "Summary",
    "Signals",
    "Current prices",
    "Historical",
    "Terminal gate",
    "Archives",
])

with summary_tab:
    st.subheader("Run summary")
    if manifest_df.empty:
        st.info("Press 'Collect latest data' to begin.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Datasets", len(manifest_df))
        c2.metric("OK", int((manifest_df["status"] == "ok").sum()))
        c3.metric("Empty", int((manifest_df["status"] == "empty").sum()))
        c4.metric("Warnings / skipped / error", int((manifest_df["status"].isin(["warning", "skipped", "error"])).sum()))
        status_counts = manifest_df["status"].fillna("unknown").value_counts().rename_axis("status").reset_index(name="count")
        fig = px.bar(status_counts, x="status", y="count", title="Manifest status counts")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig, use_container_width=True)
        if parameters:
            p1, p2, p3 = st.columns(3)
            p1.metric("Retail price change", parameters.get("RetailPriceChangeTime", "-"))
            p2.metric("Terminal gate change", parameters.get("TerminalGatePriceChangeTime", "-"))
            p3.metric("Wholesale changeover", parameters.get("WholesaleChangeOverTime", "-"))
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Current sites", f"{current_df['site_id'].nunique():,}" if not current_df.empty else "0")
        s2.metric("Distance nodes", f"{len(nodes_df):,}")
        s3.metric("Geocoded sites", f"{int(current_df[['latitude', 'longitude']].notna().all(axis=1).sum()):,}" if not current_df.empty and {'latitude','longitude'}.issubset(current_df.columns) else "0")
        s4.metric("Snapshot runs loaded", f"{snapshot_df['snapshot_run_id'].nunique():,}" if not snapshot_df.empty else "0")
        with st.expander("Manifest detail"):
            st.dataframe(manifest_df, use_container_width=True)
        with st.expander("Run paths and validated history scope"):
            st.write(
                {
                    "root": str(root),
                    "current_prices_path": str(current_prices_path),
                    "terminal_gate_prices_path": str(terminal_prices_path),
                    "use_cached_effective": use_cached_effective,
                    "validated_daily_json_fuels": sorted(VALIDATED_DAILY_HISTORY_FUELS),
                    "validated_monthly_json_fuels": sorted(VALIDATED_MONTHLY_HISTORY_FUELS),
                    "distance_nodes_loaded": len(nodes_df),
                    "snapshot_runs_loaded": int(snapshot_df["snapshot_run_id"].nunique()) if not snapshot_df.empty else 0,
                }
            )

with signals_tab:
    st.subheader("Signal view")
    if current_df.empty:
        st.info("Collect latest data to build the signal board.")
    else:
        fuels = sorted(current_df["product_short_name"].dropna().unique().tolist())
        signal_fuel = st.selectbox("Fuel", fuels, index=0, key="signal_fuel")
        sdf = current_df[current_df["product_short_name"] == signal_fuel].copy()
        summary = signal_summary(sdf)

        overview_subtab, scatter_subtab, distance_subtab, history_subtab = st.tabs([
            "Pulse",
            "Scatter + distribution",
            "Distance",
            "Run history",
        ])

        with overview_subtab:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sites rising tomorrow", summary["rising"])
            c2.metric("Flat sites", summary["flat"])
            c3.metric("Falling sites", summary["falling"])
            c4.metric("Median delta", fmt_delta(summary["median_delta"]))
            c5, c6, c7, c8 = st.columns(4)
            c5.metric("90th percentile delta", fmt_delta(summary["p90_delta"]))
            c6.metric("Today spread", fmt_delta(summary["today_spread"]))
            c7.metric("Tomorrow spread", fmt_delta(summary["tomorrow_spread"]))
            c8.metric("Outlier share", fmt_pct(summary["outlier_share"]))
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Tomorrow coverage", f"{summary['tomorrow_rows']:,} / {summary['rows']:,}")
            d2.metric("Tomorrow coverage %", f"{summary['tomorrow_pct']:.1f}%" if summary["tomorrow_pct"] is not None else "-")
            d3.metric("Price -> delta corr", f"{summary['price_delta_corr']:.2f}" if summary["price_delta_corr"] is not None else "-")
            d4.metric("Projection state", summary["projection_state"].replace("_", " ").title())

            st.info(market_regime_text(summary))
            has_projection_data = summary["delta_rows"] > 0
            if has_projection_data:
                left, right = st.columns(2)
                shock_up = sdf.sort_values(["delta_abs", "signal_score"], ascending=[False, False], na_position="last").head(15)
                fig = px.bar(
                    shock_up,
                    x="site_name",
                    y="delta_abs",
                    hover_data=["suburb", "brand_name", "signal_score", "price_today", "price_tomorrow"],
                    title=f"Tomorrow shock board - biggest increases ({signal_fuel})",
                )
                fig.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
                left.plotly_chart(fig, use_container_width=True)

                shock_down = sdf.sort_values(["delta_abs", "signal_score"], ascending=[True, False], na_position="last").head(15)
                fig = px.bar(
                    shock_down,
                    x="site_name",
                    y="delta_abs",
                    hover_data=["suburb", "brand_name", "signal_score", "price_today", "price_tomorrow"],
                    title=f"Tomorrow shock board - biggest falls ({signal_fuel})",
                )
                fig.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
                right.plotly_chart(fig, use_container_width=True)

                left, right = st.columns(2)
                stress = suburb_stress(sdf)
                if not stress.empty:
                    top_stress = stress.head(15)
                    fig = px.bar(
                        top_stress,
                        x="suburb",
                        y="stress_index",
                        hover_data=["avg_delta", "pct_rising", "pct_falling", "delta_std", "sites"],
                        title=f"Suburb stress index ({signal_fuel})",
                    )
                    fig.update_layout(height=380, margin=dict(l=10, r=10, t=45, b=10))
                    left.plotly_chart(fig, use_container_width=True)

                divergence = sdf.sort_values("delta_vs_suburb", ascending=False, na_position="last").head(15)
                fig = px.bar(
                    divergence,
                    x="site_name",
                    y="delta_vs_suburb",
                    hover_data=["suburb", "brand_name", "delta_abs", "suburb_median_delta", "distance_km"],
                    title=f"Delta divergence vs suburb median ({signal_fuel})",
                )
                fig.update_layout(height=380, margin=dict(l=10, r=10, t=45, b=10))
                right.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No usable tomorrow-vs-today deltas are available for this fuel in the selected dataset. The signal charts are suppressed until tomorrow prices arrive or cached values are available.")

            st.markdown("**High-signal sites**")
            display_table(
                sdf.sort_values(["signal_score", "delta_abs"], ascending=[False, False], na_position="last"),
                [
                    "site_name",
                    "brand_name",
                    "suburb",
                    "price_today",
                    "price_tomorrow",
                    "delta_abs",
                    "delta_vs_suburb",
                    "delta_vs_brand",
                    "signal_score",
                    "distance_km",
                    "nearest_node",
                ],
                price_cols=["price_today", "price_tomorrow", "signal_score", "distance_km"],
                delta_cols=["delta_abs", "delta_vs_suburb", "delta_vs_brand"],
                height=360,
            )

        with scatter_subtab:
            if summary["delta_rows"] == 0:
                st.warning("Scatter and distribution views need delta values. Collect again once tomorrow prices are present or rely on cached effective values.")
            else:
                left, right = st.columns([1, 1])
                color_mode = left.selectbox("Site scatter colour", ["quadrant", "movement", "brand_name", "suburb"], key="site_scatter_color")
                point_size = right.selectbox("Site scatter size", ["signal_score", "delta_abs", "price_today"], key="site_scatter_size")
                site_scatter = sdf.dropna(subset=["price_today", "delta_abs"]).copy()
                if color_mode in {"brand_name", "suburb"}:
                    site_scatter["colour_group"] = collapse_categories(site_scatter[color_mode], top_n=12)
                    color_col = "colour_group"
                else:
                    color_col = color_mode
                fig = px.scatter(
                    site_scatter,
                    x="price_today",
                    y="delta_abs",
                    color=color_col,
                    size=point_size,
                    hover_data=[
                        "site_name",
                        "brand_name",
                        "suburb",
                        "price_tomorrow",
                        "delta_vs_suburb",
                        "delta_vs_brand",
                        "distance_km",
                        "nearest_node",
                    ],
                    title=f"Site distribution: price today vs tomorrow delta ({signal_fuel})",
                )
                fig.add_hline(y=0)
                fig.update_layout(height=500, margin=dict(l=10, r=10, t=45, b=10))
                st.plotly_chart(fig, use_container_width=True)

                suburb_agg = suburb_distribution(sdf)
                brand_agg = brand_distribution(sdf)
                fuel_agg = fuel_distribution(current_df)

                left, right = st.columns(2)
                if not suburb_agg.empty:
                    fig = px.scatter(
                        suburb_agg.head(40),
                        x="median_today",
                        y="median_delta",
                        size="sites",
                        color="pct_rising",
                        hover_data=["suburb", "mean_delta", "pct_falling", "outlier_share", "median_distance_km"],
                        title=f"Suburb distribution: median price vs median delta ({signal_fuel})",
                    )
                    fig.add_hline(y=0)
                    fig.update_layout(height=420, margin=dict(l=10, r=10, t=45, b=10))
                    left.plotly_chart(fig, use_container_width=True)
                if not brand_agg.empty:
                    fig = px.scatter(
                        brand_agg,
                        x="median_today",
                        y="median_delta",
                        size="sites",
                        color="pct_rising",
                        hover_data=["brand_name", "mean_delta", "pct_falling", "outlier_share", "median_distance_km"],
                        title=f"Brand distribution: median price vs median delta ({signal_fuel})",
                    )
                    fig.add_hline(y=0)
                    fig.update_layout(height=420, margin=dict(l=10, r=10, t=45, b=10))
                    right.plotly_chart(fig, use_container_width=True)

                left, right = st.columns(2)
                fig = px.histogram(sdf, x="delta_abs", nbins=30, title=f"Delta distribution ({signal_fuel})")
                fig.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
                left.plotly_chart(fig, use_container_width=True)
                if not fuel_agg.empty:
                    fig = px.scatter(
                        fuel_agg,
                        x="mean_today",
                        y="mean_delta",
                        size="sites",
                        color="pct_rising",
                        hover_data=["product_short_name", "median_delta", "pct_falling", "outlier_share"],
                        title="Cross-fuel distribution: mean price vs mean delta",
                    )
                    fig.add_hline(y=0)
                    fig.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
                    right.plotly_chart(fig, use_container_width=True)

                st.markdown("**Distribution tables**")
                left, right = st.columns(2)
                with left:
                    st.markdown("Suburb pressure")
                    display_table(
                        suburb_agg,
                        ["suburb", "sites", "median_today", "median_delta", "pct_rising", "pct_falling", "outlier_share", "median_distance_km", "pressure_index"],
                        price_cols=["median_today", "median_distance_km", "pressure_index"],
                        delta_cols=["median_delta"],
                        pct_cols=["pct_rising", "pct_falling", "outlier_share"],
                        height=320,
                    )
                with right:
                    st.markdown("Brand pressure")
                    display_table(
                        brand_agg,
                        ["brand_name", "sites", "median_today", "median_delta", "pct_rising", "pct_falling", "outlier_share", "median_distance_km", "pressure_index"],
                        price_cols=["median_today", "median_distance_km", "pressure_index"],
                        delta_cols=["median_delta"],
                        pct_cols=["pct_rising", "pct_falling", "outlier_share"],
                        height=320,
                    )

        with distance_subtab:
            geocoded_sites = sdf[["latitude", "longitude"]].notna().all(axis=1).sum() if {"latitude", "longitude"}.issubset(sdf.columns) else 0
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Distance nodes", f"{len(nodes_df):,}")
            d2.metric("Geocoded sites", f"{int(geocoded_sites):,}")
            d3.metric("Median site distance", f"{sdf["distance_km"].median():.1f} km" if "distance_km" in sdf.columns and sdf["distance_km"].notna().any() else "-")
            d4.metric("Distance -> delta corr", f"{summary['distance_delta_corr']:.2f}" if summary["distance_delta_corr"] is not None else "-")

            distance_ready = len(nodes_df) > 0 and {"latitude", "longitude"}.issubset(sdf.columns) and geocoded_sites > 0
            if not distance_ready:
                st.info("Distance analysis is ready, but the current dataset still lacks enough geocoded site rows or reference nodes. The collector now preserves coordinates when the upstream payload exposes them. You can also provide a node catalog manually.")
                st.code(
                    "node_name,node_type,latitude,longitude\nKwinana Terminal,terminal,-32.235,115.770\nKewdale Depot,distribution,-31.978,115.951",
                    language="csv",
                )
                st.caption("Save this as distance_nodes.csv either beside app.py, beside the run folder, or inside the run folder.")
            else:
                node_types = sorted(nodes_df["node_type"].fillna("reference").unique().tolist())
                selected_node_types = st.multiselect("Node types", node_types, default=node_types, key="distance_node_types")
                distance_scope = sdf.copy()
                available_nodes = nodes_df[nodes_df["node_type"].isin(selected_node_types)].copy()
                distance_scope = attach_nearest_nodes(distance_scope, available_nodes)
                distance_scope = distance_scope[distance_scope["distance_km"].notna()].copy()

                if distance_scope.empty:
                    st.warning("No sites could be matched to the selected node set.")
                else:
                    left, right = st.columns(2)
                    fig = px.scatter(
                        distance_scope,
                        x="distance_km",
                        y="delta_abs",
                        color=collapse_categories(distance_scope["nearest_node"], top_n=10),
                        size="signal_score",
                        hover_data=["site_name", "suburb", "brand_name", "price_today", "price_tomorrow", "nearest_node"],
                        title=f"Distance vs tomorrow delta ({signal_fuel})",
                    )
                    fig.add_hline(y=0)
                    fig.update_layout(height=430, margin=dict(l=10, r=10, t=45, b=10))
                    left.plotly_chart(fig, use_container_width=True)

                    dist_agg = distance_distribution(distance_scope)
                    fig = px.bar(
                        dist_agg,
                        x="distance_band",
                        y="mean_delta",
                        hover_data=["sites", "pct_rising", "pct_falling", "max_distance_km"],
                        title=f"Average delta by distance band ({signal_fuel})",
                    )
                    fig.update_layout(height=430, margin=dict(l=10, r=10, t=45, b=10))
                    right.plotly_chart(fig, use_container_width=True)

                    left, right = st.columns(2)
                    suburb_dist = suburb_distribution(distance_scope)
                    if not suburb_dist.empty:
                        fig = px.scatter(
                            suburb_dist.head(40),
                            x="median_distance_km",
                            y="median_delta",
                            size="sites",
                            color="pct_rising",
                            hover_data=["suburb", "median_today", "pct_falling", "pressure_index"],
                            title=f"Suburb pressure vs distance ({signal_fuel})",
                        )
                        fig.add_hline(y=0)
                        fig.update_layout(height=390, margin=dict(l=10, r=10, t=45, b=10))
                        left.plotly_chart(fig, use_container_width=True)

                    nearest_summary = (
                        distance_scope.groupby(["nearest_node", "distance_band"], dropna=False)
                        .agg(
                            sites=("site_id", "nunique"),
                            mean_delta=("delta_abs", "mean"),
                            median_delta=("delta_abs", "median"),
                            mean_today=("price_today", "mean"),
                        )
                        .reset_index()
                        .sort_values(["mean_delta", "sites"], ascending=[False, False])
                    )
                    fig = px.bar(
                        nearest_summary.head(30),
                        x="nearest_node",
                        y="mean_delta",
                        color="distance_band",
                        hover_data=["sites", "median_delta", "mean_today"],
                        title=f"Node pressure composition ({signal_fuel})",
                    )
                    fig.update_layout(height=390, margin=dict(l=10, r=10, t=45, b=10))
                    right.plotly_chart(fig, use_container_width=True)

                    st.markdown("**Distance-aware site table**")
                    display_table(
                        distance_scope.sort_values(["distance_km", "delta_abs"], ascending=[False, False]),
                        ["site_name", "brand_name", "suburb", "price_today", "price_tomorrow", "delta_abs", "distance_km", "distance_band", "nearest_node"],
                        price_cols=["price_today", "price_tomorrow", "distance_km"],
                        delta_cols=["delta_abs"],
                        height=340,
                    )

        with history_subtab:
            if snapshot_df.empty or snapshot_df["snapshot_run_id"].nunique() < 2:
                st.info("Run history needs at least two saved collection runs in the same parent folder. The app will automatically scan previous run folders.")
            else:
                dim_label = st.selectbox("Trajectory dimension", ["suburb", "brand_name", "product_short_name", "distance_band", "nearest_node"], key="trajectory_dim")
                metric = st.selectbox("Trajectory metric", ["median_delta", "mean_delta", "median_today", "mean_today", "pct_rising", "pct_falling"], key="trajectory_metric")
                traj = snapshot_trajectory(snapshot_df, signal_fuel, dim_label if dim_label != "product_short_name" else "product_short_name")
                if traj.empty:
                    st.warning("No snapshot history is available for this selection yet.")
                else:
                    entity_options = traj[dim_label].fillna("Unknown").astype(str).value_counts().head(12).index.tolist()
                    selected_entities = st.multiselect("Entities", entity_options, default=entity_options[:4], key="trajectory_entities")
                    if selected_entities:
                        view = traj[traj[dim_label].fillna("Unknown").astype(str).isin(selected_entities)].copy()
                    else:
                        view = traj.copy()
                    fig = px.line(
                        view,
                        x="snapshot_ts",
                        y=metric,
                        color=dim_label,
                        markers=True,
                        hover_data=["sites", "median_today", "median_delta", "pct_rising", "pct_falling"],
                        title=f"Run-to-run {metric.replace('_', ' ')} trajectory ({signal_fuel})",
                    )
                    fig.update_layout(height=460, margin=dict(l=10, r=10, t=45, b=10))
                    st.plotly_chart(fig, use_container_width=True)

                    latest_ts = view["snapshot_ts"].max()
                    latest_slice = view[view["snapshot_ts"] == latest_ts].copy().sort_values(metric, ascending=False)
                    display_table(
                        latest_slice,
                        [dim_label, "sites", "median_today", "mean_today", "median_delta", "mean_delta", "pct_rising", "pct_falling"],
                        price_cols=["median_today", "mean_today"],
                        delta_cols=["median_delta", "mean_delta"],
                        pct_cols=["pct_rising", "pct_falling"],
                        height=320,
                    )

with current_tab:
    st.subheader("Current prices")
    if current_df.empty:
        st.info("No current price dataset found.")
    else:
        fuels = sorted(current_df["product_short_name"].dropna().unique().tolist())
        selected_fuel = st.selectbox("Fuel", fuels, index=0, key="current_fuel")
        df = current_df[current_df["product_short_name"] == selected_fuel].copy()
        col1, col2 = st.columns(2)
        suburbs = sorted(df["suburb"].dropna().unique().tolist())
        brands = sorted(df["brand_name"].dropna().unique().tolist())
        selected_suburbs = col1.multiselect("Suburbs (optional)", suburbs)
        selected_brands = col2.multiselect("Brands (optional)", brands)
        if selected_suburbs:
            df = df[df["suburb"].isin(selected_suburbs)]
        if selected_brands:
            df = df[df["brand_name"].isin(selected_brands)]

        coverage = projection_coverage(df)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Sites", f"{df['site_id'].nunique():,}")
        c3.metric("Avg today", fmt_price(df["price_today"].mean()))
        c4.metric("Avg tomorrow", fmt_price(df["price_tomorrow"].mean()))
        c5.metric("Avg delta", fmt_delta(df["delta_abs"].mean()))
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Tomorrow coverage", f"{coverage['tomorrow_rows']:,} / {coverage['rows']:,}")
        d2.metric("Tomorrow coverage %", f"{coverage['tomorrow_pct']:.1f}%" if coverage["tomorrow_pct"] is not None else "-")
        d3.metric("Delta coverage", f"{coverage['delta_rows']:,} / {coverage['rows']:,}")
        d4.metric("Median distance", f"{df['distance_km'].median():.1f} km" if "distance_km" in df.columns and df["distance_km"].notna().any() else "-")
        if coverage["tomorrow_rows"] == 0:
            st.warning("Tomorrow prices are absent for the current selection. This is usually an upstream publication-timing issue rather than a local calculation issue.")
        elif coverage["tomorrow_rows"] < coverage["rows"]:
            st.warning(f"Tomorrow prices are only available for {coverage['tomorrow_rows']:,} of {coverage['rows']:,} rows in the current selection.")

        suburb_summary = (
            df.groupby("suburb", dropna=False)
            .agg(sites=("site_id", "nunique"), avg_today=("price_today", "mean"), avg_tomorrow=("price_tomorrow", "mean"), avg_delta=("delta_abs", "mean"))
            .reset_index()
            .sort_values(["avg_today", "suburb"], na_position="last")
        )
        left, right = st.columns(2)
        fig = px.bar(suburb_summary.head(15), x="suburb", y="avg_today", title=f"Cheapest suburbs today - {selected_fuel}")
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=45, b=10))
        left.plotly_chart(fig, use_container_width=True)
        fig = px.histogram(df, x="delta_abs", nbins=25, title=f"Tomorrow - today delta distribution - {selected_fuel}")
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=45, b=10))
        right.plotly_chart(fig, use_container_width=True)
        display_table(
            df.sort_values(["price_today", "site_name"], na_position="last"),
            [
                "site_name",
                "brand_name",
                "suburb",
                "address_line1",
                "price_today",
                "price_tomorrow",
                "delta_abs",
                "distance_km",
                "nearest_node",
                "membership_required",
                "is_closed_now",
                "manned",
            ],
            price_cols=["price_today", "price_tomorrow", "distance_km"],
            delta_cols=["delta_abs"],
            height=360,
        )

with historical_tab:
    st.subheader("Historical")
    if daily_df.empty and monthly_df.empty:
        st.info("No historical datasets found.")
    else:
        region_opts = sorted(set(daily_df.get("region", pd.Series(dtype=str)).dropna().tolist()) | set(monthly_df.get("region", pd.Series(dtype=str)).dropna().tolist()))
        fuel_opts = sorted(set(daily_df.get("fuel_type", pd.Series(dtype=str)).dropna().tolist()) | set(monthly_df.get("fuel_type", pd.Series(dtype=str)).dropna().tolist()))
        region = st.selectbox("Region", region_opts, key="hist_region")
        fuel = st.selectbox("Fuel", fuel_opts, key="hist_fuel")
        dd = daily_df[(daily_df["region"] == region) & (daily_df["fuel_type"] == fuel)].copy() if not daily_df.empty else pd.DataFrame()
        md = monthly_df[(monthly_df["region"] == region) & (monthly_df["fuel_type"] == fuel)].copy() if not monthly_df.empty else pd.DataFrame()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Daily rows", len(dd))
        c2.metric("Monthly rows", len(md))
        c3.metric("Latest daily", fmt_price(dd["average_price"].iloc[-1] if not dd.empty else None))
        c4.metric("Latest monthly", fmt_price(md["average_price"].iloc[-1] if not md.empty else None))
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Latest daily change", fmt_delta(dd["daily_delta"].iloc[-1] if not dd.empty else None))
        c6.metric("7-day change", fmt_delta(dd["change_7d"].iloc[-1] if not dd.empty else None))
        c7.metric("14-day volatility", fmt_delta(dd["volatility_14d"].iloc[-1] if not dd.empty else None))
        c8.metric("MoM change", fmt_delta(md["mom_delta"].iloc[-1] if not md.empty else None))
        left, right = st.columns(2)
        if not dd.empty:
            long = pd.melt(dd, id_vars=["publish_date"], value_vars=["average_price", "rolling_7d"], var_name="series", value_name="value")
            fig = px.line(long, x="publish_date", y="value", color="series", title=f"Daily price trend - {region} {fuel}")
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
            left.plotly_chart(fig, use_container_width=True)
        if not md.empty:
            fig = px.line(md, x="month_date", y="average_price", markers=True, title=f"Monthly average trend - {region} {fuel}")
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
            right.plotly_chart(fig, use_container_width=True)
            fig = px.bar(md, x="month", y="mom_delta", title=f"Month-over-month delta - {region} {fuel}")
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(fig, use_container_width=True)
        left, right = st.columns(2)
        with left:
            st.markdown("**Daily historical prices**")
            display_table(
                dd.sort_values("publish_date", ascending=False),
                ["publish_date", "average_price", "daily_delta", "change_7d", "volatility_14d", "rolling_7d"],
                price_cols=["average_price", "rolling_7d"],
                delta_cols=["daily_delta", "change_7d", "volatility_14d"],
                height=320,
            )
        with right:
            st.markdown("**Monthly historical prices**")
            display_table(
                md.sort_values("month_date", ascending=False),
                ["month", "average_price", "mom_delta", "mom_pct", "rolling_3m"],
                price_cols=["average_price", "rolling_3m"],
                delta_cols=["mom_delta"],
                pct_cols=["mom_pct"],
                height=320,
            )

with terminal_tab:
    st.subheader("Terminal gate")
    if terminal_df.empty:
        st.info("No terminal gate dataset found.")
    else:
        fuel_opts = sorted(terminal_df["fuel_type"].dropna().unique().tolist())
        fuel = st.selectbox("Terminal fuel", fuel_opts, key="terminal_fuel")
        td = terminal_df[terminal_df["fuel_type"] == fuel].copy()
        centres = sorted(td["centre_name"].dropna().unique().tolist())
        selected_centres = st.multiselect("Terminal centres (optional)", centres)
        if selected_centres:
            td = td[td["centre_name"].isin(selected_centres)]
        retail_same = current_df[current_df["product_short_name"] == fuel] if not current_df.empty else pd.DataFrame()
        retail_spread = retail_same["price_today"].mean() - td["price_current"].mean() if not retail_same.empty and not td.empty else None
        pass_through_ratio = None
        if not retail_same.empty and td["delta_next_current"].notna().any() and retail_same["delta_abs"].notna().any():
            terminal_move = td["delta_next_current"].mean()
            retail_move = retail_same["delta_abs"].mean()
            if terminal_move not in [0, None] and not pd.isna(terminal_move):
                pass_through_ratio = retail_move / terminal_move

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Centres", td["centre_id"].nunique())
        c2.metric("Avg current", fmt_price(td["price_current"].mean()))
        c3.metric("Avg next", fmt_price(td["price_next"].mean()))
        c4.metric("Retail - terminal spread", fmt_delta(retail_spread))
        c5.metric("Retail pass-through", f"{pass_through_ratio:.2f}x" if pass_through_ratio is not None else "-")

        melt = td.melt(id_vars=["centre_name"], value_vars=["price_previous", "price_current", "price_next"], var_name="series", value_name="price")
        left, right = st.columns(2)
        fig = px.bar(melt, x="centre_name", y="price", color="series", barmode="group", title=f"Terminal previous/current/next - {fuel}")
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=45, b=10))
        left.plotly_chart(fig, use_container_width=True)
        fig = px.bar(td.sort_values("delta_next_current", ascending=False), x="centre_name", y="delta_next_current", title=f"Terminal next-current delta - {fuel}")
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=45, b=10))
        right.plotly_chart(fig, use_container_width=True)
        if not retail_same.empty and td["price_current"].notna().any():
            fig = px.scatter(
                td,
                x="price_current",
                y="delta_next_current",
                size="price_next",
                color="centre_name",
                hover_data=["price_previous", "price_next"],
                title=f"Terminal price vs terminal delta ({fuel})",
            )
            fig.add_hline(y=0)
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(fig, use_container_width=True)
        display_table(
            td.sort_values(["price_current", "centre_name"]),
            ["centre_name", "price_previous", "price_current", "price_next", "delta_current_previous", "delta_next_current", "terminal_gate_price_change_time"],
            price_cols=["price_previous", "price_current", "price_next"],
            delta_cols=["delta_current_previous", "delta_next_current"],
            height=340,
        )

with archive_tab:
    st.subheader("Archives")
    report_dir = root / "raw" / "reports"
    blob_dir = root / "raw" / "blobs"
    for name in ["weekly_retail_prices_index.json", "monthly_retail_prices_index.json", "terminal_gate_prices_index.json"]:
        path = report_dir / name
        data = load_json(path)
        if data is not None:
            st.markdown(f"**{name}**")
            st.write(f"Entries: {len(data) if isinstance(data, list) else 1}")
            if isinstance(data, list) and data:
                st.dataframe(pd.json_normalize(data[:100]), use_container_width=True, height=320)
    if blob_dir.exists():
        files = sorted([p for p in blob_dir.rglob("*") if p.is_file()])
        if files:
            st.markdown("**Downloaded archives**")
            st.dataframe(pd.DataFrame({"file": [str(p.relative_to(root)) for p in files]}), use_container_width=True, height=260)
