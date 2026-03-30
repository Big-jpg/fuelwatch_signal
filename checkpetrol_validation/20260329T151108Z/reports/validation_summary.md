# CheckPetrol validation summary

## Core metrics
- availability_entries: 7
- availability_outage_count: 565
- availability_outage_percent: 9
- availability_outages_by_state_count: 8
- availability_total_stations: 6542
- fallback_rate_pct: 10.27
- fuel_codes_observed: ['DSL', 'E10', 'E85', 'LPG', 'P95', 'P98', 'PDSL', 'U91']
- median_age_minutes: 2649.34
- no_price_rate_pct: 0.64
- outage_rate_pct: 8.64
- p90_age_minutes: 7205.29
- query_context_rows: 39252
- source_count: 4
- stale_rate_pct: 0.0
- state_count: 8
- station_count: 6542
- station_price_rows: 23686
- validated_at: 2026-03-29T15:11:22.549333+00:00

## WA concordance
- wa_concordance_status: skipped_no_fuelwatch_root

## Issue counts
- error / invalid_updated_at: 1
- error / query_price_mismatch: 1
- warning / non_stale_but_old: 1
- warning / price_out_of_range: 6
- warning / query_fuel_missing_in_all_prices: 1
- warning / very_old_timestamp: 1

## Top sources
- nsw_fuelcheck: stations=2683, outage_rate=9.5%, stale_rate=0.0%, fallback_rate=12.9%, no_price_rate=0.0%
- vic_servosaver: stations=1701, outage_rate=16.4%, stale_rate=0.0%, fallback_rate=5.35%, no_price_rate=2.47%
- petrolmate: stations=1396, outage_rate=1.43%, stale_rate=0.0%, fallback_rate=14.54%, no_price_rate=0.0%
- wa_fuelwatch: stations=762, outage_rate=1.44%, stale_rate=0.0%, fallback_rate=4.2%, no_price_rate=0.0%

## Fuel coverage
- U91 (ULP): stations=5828, coverage=89.09%, median_price_cents=2599.0
- DSL (DSL): stations=4132, coverage=63.16%, median_price_cents=3215.0
- P98 (98R): stations=3541, coverage=54.13%, median_price_cents=2839.0
- E10 (E10): stations=3083, coverage=47.13%, median_price_cents=2579.0
- P95 (PUP): stations=3010, coverage=46.01%, median_price_cents=2749.0
- PDSL (nan): stations=2529, coverage=38.66%, median_price_cents=3219.0
- LPG (LPG): stations=822, coverage=12.56%, median_price_cents=1089.0
- E85 (nan): stations=741, coverage=11.33%, median_price_cents=2829.0
