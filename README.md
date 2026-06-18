# Revenue Forecasting & Variance Analysis

Forecasting next-quarter revenue for 3,500+ U.S. public companies using SEC financial
filings and macroeconomic data — then framing the results for FP&A-style variance review.

An end-to-end project spanning **Python, SQL, Power BI, and machine learning**, built to
mirror how a finance team actually forecasts revenue and reviews the variance.

---

## Dashboard

![Revenue Forecast Dashboard](images/dashboard.png)

*Power BI dashboard: forecast accuracy, variance, and error-by-industry across the 2025
test year.*

---

## Headline results (held-out 2025 test set)

| Metric | Result |
| --- | --- |
| Mean absolute error (MAE) | **$147M** (vs $184M seasonal baseline — ~20% better) |
| Typical forecast error (median) | **~8%** |
| Forecasts within ±10% of actual | **~61%** |
| Impossible negative forecasts | **0** (vs 64 in the baseline) |
| Companies analyzed | **3,500+** |

> Honest note: dollar error (MAE) is concentrated in a small number of very large,
> volatile firms (e.g. Dell, NVIDIA). The *typical* company is forecast within ~8%;
> the average is pulled up by a few mega-cap outliers (the ~18% mean error reflects these outliers). Both views are reported.

---

## What the project does

1. **Extracts** quarterly revenue for thousands of public companies from raw SEC
   Financial Statement Data Sets (2021–2025), using a priority-based revenue-tag rule
   and deduplication to one clean value per company-quarter.
2. **Derives** missing Q4 revenue (annual − Q1–Q3) where companies don't report it
   directly, with each derived row flagged.
3. **Engineers** firm-level features (revenue lags, year-over-year and quarter-over-
   quarter growth, seasonality, volatility, sector) and joins FRED macroeconomic
   indicators (CPI, fed funds rate, unemployment).
4. **Forecasts** next-quarter revenue with a **log-growth model** — predicting
   `log(next_revenue / current_revenue)` and reconstructing the dollar forecast as
   `current_revenue × exp(prediction)`, which is mathematically non-negative.
5. **Reports** the results as FP&A artifacts: an Excel variance workbook, a Power BI
   forecast-accuracy dashboard, and a SQL data layer.

---

## Why log-growth (the key modeling decision)

The first machine-learning model corrected a seasonal baseline on the raw-dollar scale.
It lowered dollar error but produced **1,836 impossible negative revenue forecasts** and
over-focused on the largest firms.

Reformulating into **log-growth space** fixed this: forecasts became non-negative by
construction, accuracy spread proportionally across firm sizes, and the model improved on
the seasonal baseline across MAE, sMAPE, and percentage error — while eliminating every
impossible forecast. A naive "same as last quarter" forecast remains hard to beat on
percentage error alone, which reflects how persistent quarterly revenue is; the model's
clear edge is dollar-weighted accuracy and economically sensible output.

---

## Tools

| Tool | Use |
| --- | --- |
| **Python** (pandas, scikit-learn) | data pipeline, feature engineering, modeling |
| **SQL** (MySQL) | cleaning, deduplication, aggregation, YoY growth, macro join |
| **Power BI** | forecast-accuracy & variance dashboard |
| **Excel** | budget-vs-actual variance workbook |

---

## Repository structure

```
.
├── README.md
├── scripts/
│   └── improved_revenue_forecast.py        # final log-growth model (runs on full data)
├── sql/
│   └── revenue_forecasting_queries.sql     # MySQL data layer (load, clean, aggregate, YoY)
├── data/
│   └── sample/
│       └── sample_revenue_panel.csv        # 24-company demo sample (see note below)
├── outputs/
│   ├── Revenue_Forecast_Variance_Analysis.xlsx
│   ├── powerbi_forecast_data.csv
│   ├── powerbi_accuracy_bands.csv
│   └── improved_predictions_test_2025.csv
├── reports/
│   ├── Capstone_Academic_Report.docx       # full academic report
│   └── FPA_Business_Report.docx            # FP&A leadership briefing
└── images/
    └── dashboard.png
```

---

## How to run

**The model** (`scripts/improved_revenue_forecast.py`):
```bash
python scripts/improved_revenue_forecast.py --data data/processed/model_ready_sec_macro_2021_2025.csv
```
Trains on 2021–2023, selects hyperparameters on 2024, reports final results on 2025, and
writes `improved_predictions_test_2025.csv`.

**The SQL** (`sql/revenue_forecasting_queries.sql`): open in MySQL Workbench, run Section 1
to create the table, import the data, then run the queries (filtering, deduplication,
aggregation, YoY growth, macro join).

> **Data note:** the full ~51,000-row modeling panel is not committed to this repo (large
> file). A 24-company **demonstration sample** (`data/sample/sample_revenue_panel.csv`) is
> included so the SQL runs against real-shaped data. The sample is for demonstration only —
> it does **not** reproduce the headline results, which come from the full panel. SEC
> Financial Statement Data Sets and FRED series are publicly available for the full run.

---

## Limitations

- A quarterly **revenue** forecast is not a 13-week **cash-flow** forecast.
- ~1,600 firm-quarters with zero/negative reported revenue use a seasonal fallback (they
  can't be modeled in log space) — they are flagged, not deleted.
- Macroeconomic variables are weak standalone predictors; firm history dominates.
- Derived Q4 revenue can introduce noise.
- Dollar error is dominated by a few mega-cap firms.
- The model is **more production-ready, not perfect**, and would require monitoring in use.

---

## Reports

Two write-ups accompany the code: a full **academic capstone report** (methodology,
results, limitations) and an **FP&A business briefing** (what leadership should trust and
do). Both are in `reports/`.
