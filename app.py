from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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


# ---------- helpers ----------
def load_csv(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)
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


def soften_theme() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1500px;}
        .stMetric {background: rgba(255,255,255,0.02); padding: .75rem 1rem; border-radius: 12px;}
        h1, h2, h3 {letter-spacing: -0.02em;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def prepare_current(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in ["price_today", "price_tomorrow", "delta_abs"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ["is_closed_now", "is_closed_all_day_tomorrow", "manned", "membership_required", "operates_24_7"]:
        if col in out.columns:
            out[col] = out[col].fillna(False).astype(bool)

    out["suburb_median_today"] = out.groupby(["product_short_name", "suburb"])["price_today"].transform("median")
    out["suburb_median_tomorrow"] = out.groupby(["product_short_name", "suburb"])["price_tomorrow"].transform("median")
    out["suburb_median_delta"] = out.groupby(["product_short_name", "suburb"])["delta_abs"].transform("median")
    out["brand_median_delta"] = out.groupby(["product_short_name", "brand_name"])["delta_abs"].transform("median")
    out["vs_suburb_today"] = out["price_today"] - out["suburb_median_today"]
    out["vs_suburb_tomorrow"] = out["price_tomorrow"] - out["suburb_median_tomorrow"]
    out["delta_vs_suburb"] = out["delta_abs"] - out["suburb_median_delta"]
    out["delta_vs_brand"] = out["delta_abs"] - out["brand_median_delta"]
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
    out["rolling_7d"] = out.groupby(["region", "fuel_type"])["average_price"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    return out


def prepare_monthly(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["average_price"] = pd.to_numeric(out["average_price"], errors="coerce")
    out["month_date"] = pd.to_datetime(out["month"], format="%B %Y", errors="coerce")
    out = out.sort_values(["region", "fuel_type", "month_date"])
    out["mom_delta"] = out.groupby(["region", "fuel_type"])["average_price"].diff()
    return out


def prepare_terminal(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in ["price_previous", "price_current", "price_next", "delta_next_current"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def display_table(df: pd.DataFrame, columns: list[str], price_cols: list[str] | None = None, delta_cols: list[str] | None = None, height: int = 360):
    if df.empty:
        st.info("No rows available.")
        return
    cols = [c for c in columns if c in df.columns]
    view = df[cols].copy()
    price_cols = [c for c in (price_cols or []) if c in view.columns]
    delta_cols = [c for c in (delta_cols or []) if c in view.columns]
    fmt = {c: "{:.1f}" for c in price_cols}
    fmt.update({c: "{:+.1f}" for c in delta_cols})
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
            **coverage,
        }
    delta = df["delta_abs"].dropna()
    rising = int((delta > 0).sum())
    flat = int((delta == 0).sum())
    falling = int((delta < 0).sum())
    projection_state = "ready" if not delta.empty else "missing_tomorrow_prices"
    return {
        "rising": rising,
        "flat": flat,
        "falling": falling,
        "median_delta": float(delta.median()) if not delta.empty else None,
        "p90_delta": float(delta.quantile(0.9)) if not delta.empty else None,
        "today_spread": float(df["price_today"].max() - df["price_today"].min()) if df["price_today"].notna().any() else None,
        "tomorrow_spread": float(df["price_tomorrow"].max() - df["price_tomorrow"].min()) if df["price_tomorrow"].notna().any() else None,
        "projection_state": projection_state,
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
            pct_rising=("delta_abs", lambda s: (s.fillna(0) > 0).mean() * 100),
        )
        .reset_index()
    )
    agg["stress_index"] = agg["avg_delta"].fillna(0) + agg["delta_std"].fillna(0) + (agg["pct_rising"].fillna(0) / 20)
    return agg.sort_values(["stress_index", "avg_delta"], ascending=False)


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
    if rising > falling and p90 >= 5:
        return "Mostly rising market with moderate volatility."
    if falling > rising:
        return "Mixed-to-falling market."
    return "Relatively stable market."


soften_theme()
start_default, end_default = default_monthly_window()
if "last_root" not in st.session_state:
    st.session_state["last_root"] = None

st.title("FuelWatch Signals")
st.caption("Collect latest data, then review signal-first fuel analytics for current, tomorrow, and historical movement.")

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
    use_cached_effective = st.checkbox("Use latest available cached values", value=True, help="When enabled, the app will prefer effective flat files that backfill missing current-cycle values from earlier runs.")
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
                line = f"{event['pile']}: {event['current']}/{event['total']} - {event.get('detail','')}"
                if event.get('status') != 'running':
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
        c4.metric("Warnings / skipped / error", int((manifest_df["status"].isin(["warning", "skipped", "error"])) .sum()))
        status_counts = manifest_df["status"].fillna("unknown").value_counts().rename_axis("status").reset_index(name="count")
        fig = px.bar(status_counts, x="status", y="count", title="Manifest status counts")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig, use_container_width=True)
        if parameters:
            p1, p2, p3 = st.columns(3)
            p1.metric("Retail price change", parameters.get("RetailPriceChangeTime", "-"))
            p2.metric("Terminal gate change", parameters.get("TerminalGatePriceChangeTime", "-"))
            p3.metric("Wholesale changeover", parameters.get("WholesaleChangeOverTime", "-"))
        with st.expander("Manifest detail"):
            st.dataframe(manifest_df, use_container_width=True)
        with st.expander("Run paths and validated history scope"):
            st.write({
                "root": str(root),
                "current_prices_path": str(current_prices_path),
                "terminal_gate_prices_path": str(terminal_prices_path),
                "use_cached_effective": use_cached_effective,
                "validated_daily_json_fuels": sorted(VALIDATED_DAILY_HISTORY_FUELS),
                "validated_monthly_json_fuels": sorted(VALIDATED_MONTHLY_HISTORY_FUELS),
            })

with signals_tab:
    st.subheader("Signal view")
    if current_df.empty:
        st.info("Collect latest data to build the signal board.")
    else:
        fuels = sorted(current_df["product_short_name"].dropna().unique().tolist())
        signal_fuel = st.selectbox("Fuel", fuels, index=0, key="signal_fuel")
        sdf = current_df[current_df["product_short_name"] == signal_fuel].copy()
        summary = signal_summary(sdf)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sites rising tomorrow", summary["rising"])
        c2.metric("Flat sites", summary["flat"])
        c3.metric("Falling sites", summary["falling"])
        c4.metric("Median delta", fmt_delta(summary["median_delta"]))
        c5, c6, c7 = st.columns(3)
        c5.metric("90th percentile delta", fmt_delta(summary["p90_delta"]))
        c6.metric("Today spread", fmt_delta(summary["today_spread"]))
        c7.metric("Tomorrow spread", fmt_delta(summary["tomorrow_spread"]))

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Tomorrow coverage", f"{summary['tomorrow_rows']:,} / {summary['rows']:,}")
        d2.metric("Tomorrow coverage %", f"{summary['tomorrow_pct']:.1f}%" if summary["tomorrow_pct"] is not None else "-")
        d3.metric("Delta coverage", f"{summary['delta_rows']:,} / {summary['rows']:,}")
        d4.metric("Projection state", summary["projection_state"].replace("_", " ").title())

        st.info(market_regime_text(summary))

        has_projection_data = summary["delta_rows"] > 0
        if has_projection_data:
            left, right = st.columns(2)
            shock_up = sdf.sort_values(["delta_abs", "signal_score"], ascending=[False, False], na_position="last").head(15)
            fig = px.bar(shock_up, x="site_name", y="delta_abs", hover_data=["suburb", "brand_name", "signal_score"], title=f"Tomorrow shock board - biggest increases ({signal_fuel})")
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
            left.plotly_chart(fig, use_container_width=True)
            shock_down = sdf.sort_values(["delta_abs", "signal_score"], ascending=[True, False], na_position="last").head(15)
            fig = px.bar(shock_down, x="site_name", y="delta_abs", hover_data=["suburb", "brand_name", "signal_score"], title=f"Tomorrow shock board - biggest falls ({signal_fuel})")
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
            right.plotly_chart(fig, use_container_width=True)

            left, right = st.columns(2)
            stress = suburb_stress(sdf)
            if not stress.empty:
                top_stress = stress.head(15)
                fig = px.bar(top_stress, x="suburb", y="stress_index", hover_data=["avg_delta", "pct_rising", "delta_std", "sites"], title=f"Suburb stress index ({signal_fuel})")
                fig.update_layout(height=380, margin=dict(l=10, r=10, t=45, b=10))
                left.plotly_chart(fig, use_container_width=True)
            divergence = sdf.sort_values("delta_vs_suburb", ascending=False, na_position="last").head(15)
            fig = px.bar(divergence, x="site_name", y="delta_vs_suburb", hover_data=["suburb", "brand_name", "delta_abs", "suburb_median_delta"], title=f"Delta divergence vs suburb median ({signal_fuel})")
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=45, b=10))
            right.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No usable tomorrow-vs-today deltas are available for this fuel in the selected dataset. The signal charts are suppressed until tomorrow prices arrive or cached values are available.")

        st.markdown("**High-signal sites**")
        display_table(
            sdf.sort_values(["signal_score", "delta_abs"], ascending=[False, False], na_position="last"),
            ["site_name", "brand_name", "suburb", "price_today", "price_tomorrow", "delta_abs", "delta_vs_suburb", "delta_vs_brand", "signal_score"],
            price_cols=["price_today", "price_tomorrow", "signal_score"],
            delta_cols=["delta_abs", "delta_vs_suburb", "delta_vs_brand"],
            height=360,
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
        d4.metric("Today coverage %", f"{coverage['today_pct']:.1f}%" if coverage["today_pct"] is not None else "-")
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
        display_table(df.sort_values(["price_today", "site_name"], na_position="last"), ["site_name", "brand_name", "suburb", "address_line1", "price_today", "price_tomorrow", "delta_abs", "membership_required", "is_closed_now", "manned"], price_cols=["price_today", "price_tomorrow"], delta_cols=["delta_abs"], height=360)

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
            display_table(dd.sort_values("publish_date", ascending=False), ["publish_date", "average_price", "rolling_7d"], price_cols=["average_price", "rolling_7d"], height=320)
        with right:
            st.markdown("**Monthly historical prices**")
            display_table(md.sort_values("month_date", ascending=False), ["month", "average_price", "mom_delta"], price_cols=["average_price"], delta_cols=["mom_delta"], height=320)

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
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Centres", td["centre_id"].nunique())
        c2.metric("Avg current", fmt_price(td["price_current"].mean()))
        c3.metric("Avg next", fmt_price(td["price_next"].mean()))
        c4.metric("Retail - terminal spread", fmt_delta(retail_spread))
        melt = td.melt(id_vars=["centre_name"], value_vars=["price_previous", "price_current", "price_next"], var_name="series", value_name="price")
        left, right = st.columns(2)
        fig = px.bar(melt, x="centre_name", y="price", color="series", barmode="group", title=f"Terminal previous/current/next - {fuel}")
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=45, b=10))
        left.plotly_chart(fig, use_container_width=True)
        fig = px.bar(td.sort_values("delta_next_current", ascending=False), x="centre_name", y="delta_next_current", title=f"Terminal next-current delta - {fuel}")
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=45, b=10))
        right.plotly_chart(fig, use_container_width=True)
        display_table(td.sort_values(["price_current", "centre_name"]), ["centre_name", "price_previous", "price_current", "price_next", "delta_next_current", "terminal_gate_price_change_time"], price_cols=["price_previous", "price_current", "price_next"], delta_cols=["delta_next_current"], height=340)

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
