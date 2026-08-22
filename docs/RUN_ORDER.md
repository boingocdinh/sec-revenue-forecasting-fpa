# Final Run Order

Activate the environment first (Windows example):

    .\capstone_env\Scripts\python.exe

Run from the project root. Every script also accepts the root as the first arg
and auto-detects whether it lives in the repo root or in `scripts/`.

## Required pipeline (final model = Script 19)

| # | Command | Produces |
|---|---|---|
| 1 | `python scripts/06b_extract_sec_revenue_panel_with_2026q1_source.py .` | `data/processed/sec_revenue_panel_2021_2025.csv` |
| 2 | `python scripts/07b_diagnose_q4_coverage_with_2026q1_source.py .` | `outputs/diagnostics/q4_derivation_diagnostic_by_cik_fy.csv` |
| 3 | `python scripts/08_build_revenue_panel_with_derived_q4.py .` | `data/processed/sec_revenue_panel_2021_2025_q4_enhanced.csv` |
| 4 | `python scripts/09_validate_enhanced_panel_for_modeling.py .` | validation reports in `outputs/diagnostics/` |
| 5 | `python scripts/10_build_forecasting_dataset.py .` | `data/processed/forecasting_dataset_2021_2025.csv` |
| 6 | `python scripts/12_build_macro_quarterly.py .` | `data/processed/macro_quarterly.csv` |
| 7 | `python scripts/13_merge_sec_macro_dataset.py .` | `data/processed/model_ready_sec_macro_2021_2025.csv` |
| 8 | `python scripts/19_test_yoy_macro_residual_correction.py .` | `outputs/report_tables/yoy_macro_residual_predictions.csv`, `models/yoy_macro_residual_best_model.pkl` |
| 9 | `python scripts/21_prepare_powerbi_dashboard.py .` | `outputs/powerbi/powerbi_forecast_dashboard_15_companies.csv` |

Final public deliverable:
`outputs/powerbi/powerbi_forecast_dashboard_15_companies.csv`

### Optional: export every eligible firm

    python scripts/21_prepare_powerbi_dashboard.py . --all

Produces `outputs/powerbi/powerbi_forecast_dashboard_all_companies.csv` with
all eligible firms. The showcase 15 are tagged `is_showcase = 1`. In Power BI,
slice on `company_label` to drill into any company, and filter `is_showcase = 1`
to default the page to the defended 15.

## Optional (in `scripts/optional/`, report use only)

- `optional/11_train_sec_only_models_REPORT.py` — SEC-only benchmark models.
- `optional/18_macro_revenue_relationship_regression.py` — exploratory macro-revenue
  association analysis; not used for statistical inference or final model selection
  (needs `statsmodels`).
- `optional/20_compare_actual_original_new_forecast.py` — actual vs YOY vs
  adjusted comparison table/chart. Superseded for delivery by Script 21.

Run from the project root, e.g. `python scripts/optional/18_...py .`

## Dropped from the final submission

- `15_final_holdout_evaluation.py` and `16_shap_sec_only_xgboost.py` require a
  SEC-only `_v2` model artifact that no included script produces. They are excluded.

## FRED files (before step 6)

Place in `data/raw/fred/`:
- `CPIAUCSL.csv`
- `FEDFUNDS.csv`
- `UNRATE.csv`

Script 12 reads `data/raw/fred` and falls back to legacy `data_raw/fred` with a
warning if the new folder is absent.
