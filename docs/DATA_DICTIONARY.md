# Data Dictionary

## File: `outputs/powerbi/powerbi_forecast_dashboard_15_companies.csv`

> Two files share this schema:
> - `..._15_companies.csv` — the showcase 15 firms (default run).
> - `..._all_companies.csv` — every eligible firm (`--all` run); the showcase
>   15 are flagged `is_showcase = 1`.

One row per **company × target quarter × forecast scenario**. With 15 companies,
4 quarters of 2025, and 3 scenarios, a full file has 180 rows. The "long /
scenario" shape lets Power BI plot Actual, YOY Baseline, and YOY + Random Forest
Adjusted on a single line-chart axis using `forecast_scenario` as the legend and
`scenario_value_musd` as the value.

All monetary fields are in **millions of USD** (suffix `_musd`).

### Identity and grouping

| Column | Type | Description |
|---|---|---|
| `cik` | string | SEC Central Index Key; unique firm identifier. |
| `company_name` | string | Raw company name as filed with the SEC. |
| `company_label` | string | Cleaned, chart-friendly company name (suffixes removed). |
| `sic` | string | SEC Standard Industrial Classification code. |
| `firm_size_group` | string | `Big`, `Mid`, or `Small` for showcase firms. In the `--all` file, non-showcase firms are bucketed by revenue percentile and may also be `Other`. |
| `selection_reason` | string | Showcase firms: `Top 5 by 2025 actual revenue`, `Closest to median revenue percentile`, or `Lower revenue percentile representative`. In the `--all` file, non-showcase firms show `Eligible firm (not in showcase 15)`. |
| `revenue_rank_2025` | int | Dense rank by total 2025 actual revenue across eligible firms (1 = largest). |
| `revenue_percentile_2025` | float | Percentile (0–1) of total 2025 actual revenue (higher = larger). |
| `actual_2025_revenue_musd` | float | Firm's total actual 2025 revenue (sum of 4 quarters), millions USD. |
| `model_name_technical` | string | Technical model id: `yoy_plus_random_forest_macro_controls_residual`. |
| `is_showcase` | int | 1 if the firm is one of the showcase 15; 0 otherwise. Always 1 in the 15-company file. |

### Time fields

| Column | Type | Description |
|---|---|---|
| `calendar_period` | string | Feature quarter (the quarter whose data is used to forecast). |
| `target_calendar_period` | string | Target quarter being forecast (e.g. `2025Q2`). |
| `target_quarter_sort` | int | Ordinal sort key = `year*4 + quarter`. Use as the Power BI sort-by column. |
| `target_quarter_start_date` | date | First day of the target quarter. Use for a continuous date axis. |

### Scenario fields (the long-form plotting columns)

| Column | Type | Description |
|---|---|---|
| `forecast_scenario` | string | `Actual`, `YOY Baseline Forecast`, or `YOY + Random Forest Adjusted Forecast`. Legend field. |
| `scenario_value_musd` | float | The value for this scenario: actual revenue, baseline forecast, or adjusted forecast. Primary line-chart measure. |
| `scenario_variance_musd` | float | Actual − scenario forecast. `NaN` for the Actual scenario. |
| `scenario_absolute_error_musd` | float | Absolute value of the scenario variance. `NaN` for Actual. |
| `scenario_accuracy_pct` | float | `max(0, 1 − |actual − forecast| / |actual|) × 100`. `NaN` for Actual. |

### Wide forecast fields (repeated on every scenario row for the same company-quarter)

| Column | Type | Description |
|---|---|---|
| `actual_revenue_musd` | float | Actual next-quarter revenue. |
| `yoy_baseline_forecast_musd` | float | YOY adjusted seasonal baseline: `revenue_lag_3 + (revenue − revenue_lag_4)`. |
| `random_forest_adjustment_musd` | float | Predicted residual added by the Random Forest model. |
| `yoy_random_forest_adjusted_forecast_musd` | float | Final forecast: baseline + RF adjustment. |
| `current_revenue_musd` | float | Revenue in the feature quarter. |

### Variance / error / accuracy (wide, per scenario type)

| Column | Type | Description |
|---|---|---|
| `yoy_baseline_variance_musd` | float | Actual − YOY baseline forecast. |
| `yoy_random_forest_adjusted_variance_musd` | float | Actual − adjusted forecast. |
| `yoy_baseline_abs_error_musd` | float | `|baseline variance|`. |
| `yoy_random_forest_adjusted_abs_error_musd` | float | `|adjusted variance|`. |
| `yoy_baseline_accuracy_pct` | float | Accuracy of the baseline forecast. |
| `yoy_random_forest_adjusted_accuracy_pct` | float | Accuracy of the adjusted forecast. |

> Note: in the long file these wide columns are present, but the canonical
> per-scenario figures are `scenario_*`. Use the `scenario_*` columns for charts
> driven by `forecast_scenario`, and the wide columns for direct baseline-vs-adjusted
> comparison cards.

### Risk and index

| Column | Type | Description |
|---|---|---|
| `forecast_abs_error_ratio` | float | Adjusted absolute error ÷ |actual|. |
| `forecast_risk_flag` | string | `High risk` if ratio ≥ 0.15, else `Low risk`; `Unknown` if actual is 0. |
| `actual_revenue_index_base100` | float | Actual revenue indexed to 100 at the firm's first 2025 quarter. |

### Macroeconomic variables (feature quarter)

| Column | Type | Description |
|---|---|---|
| `cpi` | float | CPI (CPIAUCSL), quarterly mean. |
| `fed_funds_rate` | float | Effective federal funds rate (FEDFUNDS), quarterly mean. |
| `unemployment_rate` | float | Unemployment rate (UNRATE), quarterly mean. |
| `cpi_qoq_change` | float | CPI quarter-over-quarter change. |
| `cpi_qoq_pct_change` | float | CPI quarter-over-quarter percent change. |
| `fed_funds_rate_qoq_change` | float | Fed funds rate QoQ change. |
| `unemployment_rate_qoq_change` | float | Unemployment rate QoQ change. |

### Revenue history and quality

| Column | Type | Description |
|---|---|---|
| `revenue_lag_1_musd` … `revenue_lag_4_musd` | float | Revenue 1–4 quarters before the feature quarter, millions USD. |
| `is_derived_q4` | int | 1 if the feature row's Q4 revenue was derived (annual − Q1–Q3), else 0. |
| `firm_obs_total` | int | Number of quarterly observations available for the firm. |

---

## Final-model selection rule (for reproducibility)

1. Start from `outputs/report_tables/yoy_macro_residual_predictions.csv`, filtered to
   `model == yoy_plus_random_forest_macro_controls_residual` and `split == test_2025`.
2. Keep firms with all 4 target quarters in 2025, positive total actual 2025
   revenue, and no missing actual/forecast values.
3. Compute `actual_2025_revenue_musd`, `revenue_rank_2025`, `revenue_percentile_2025`.
4. Select **Big** = top 5 by total actual 2025 revenue; **Mid** = 5 firms closest
   to the 50th percentile; **Small** = 5 firms spread across the 10th–25th
   percentile band (representative, not the extreme bottom).
