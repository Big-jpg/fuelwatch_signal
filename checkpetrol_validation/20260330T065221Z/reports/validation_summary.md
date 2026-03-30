# CheckPetrol validation summary

## Core metrics
- availability_entries: 7
- availability_outage_count: 592
- availability_outage_percent: 7
- availability_outages_by_state_count: 8
- availability_total_stations: 8353
- fallback_rate_pct: 10.39
- fuel_codes_observed: ['DSL', 'E10', 'E85', 'LPG', 'P95', 'P98', 'PDSL', 'U91']
- median_age_minutes: 2147.51
- no_price_rate_pct: 0.88
- outage_rate_pct: 7.06
- p90_age_minutes: 6847.21
- query_context_rows: 50316
- source_count: 5
- stale_rate_pct: 0.0
- state_count: 8
- station_count: 8386
- station_price_rows: 30689
- validated_at: 2026-03-30T06:52:34.492661+00:00

## WA concordance
- wa_concordance_fuels: ['98R', 'DSL', 'ULP']
- wa_concordance_rows: 939
- wa_concordance_status: ok
- wa_concordance_suburbs: 361
- wa_median_abs_mean_price_delta: 2534.1
- wa_median_abs_median_price_delta: 2537.1
- wa_p90_abs_median_price_delta: 2902.68

## Issue counts
- error / future_timestamp: 1
- error / invalid_updated_at: 1
- error / query_price_mismatch: 1
- warning / non_stale_but_old: 1
- warning / price_out_of_range: 6
- warning / query_fuel_missing_in_all_prices: 1
- warning / very_old_timestamp: 1

## Top sources
- nsw_fuelcheck: stations=2688, outage_rate=6.29%, stale_rate=0.0%, fallback_rate=12.72%, no_price_rate=0.0%
- qld_live: stations=1758, outage_rate=10.13%, stale_rate=0.0%, fallback_rate=11.21%, no_price_rate=1.08%
- vic_servosaver: stations=1701, outage_rate=13.87%, stale_rate=0.0%, fallback_rate=5.41%, no_price_rate=2.41%
- petrolmate: stations=1409, outage_rate=0.35%, stale_rate=0.0%, fallback_rate=14.76%, no_price_rate=0.0%
- wa_fuelwatch: stations=830, outage_rate=0.48%, stale_rate=0.0%, fallback_rate=3.86%, no_price_rate=1.69%

## Fuel coverage
- U91 (ULP): stations=7441, coverage=88.73%, median_price_cents=2599.0
- P98 (98R): stations=5461, coverage=65.12%, median_price_cents=2839.0
- DSL (DSL): stations=5140, coverage=61.29%, median_price_cents=3229.0
- P95 (PUP): stations=3966, coverage=47.29%, median_price_cents=2749.0
- PDSL (nan): stations=3897, coverage=46.47%, median_price_cents=3239.0
- E10 (E10): stations=3670, coverage=43.76%, median_price_cents=2579.0
- LPG (LPG): stations=898, coverage=10.71%, median_price_cents=1099.0
- E85 (nan): stations=216, coverage=2.58%, median_price_cents=2499.0
