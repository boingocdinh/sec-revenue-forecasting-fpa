# Next-Quarter Revenue Forecasting for U.S. Public Companies

An end-to-end forecasting and FP&A analytics project that uses SEC financial filings,
macroeconomic indicators, and a Random Forest model to estimate next-quarter revenue for
U.S. public companies.

The project covers data extraction, fiscal-Q4 derivation, timing validation, feature
engineering, model evaluation, sensitivity testing, Excel variance analysis, and Power BI
company-level review.

---

## Business question

Can historical SEC revenue, company information, and a small group of economic indicators
forecast next-quarter revenue more accurately than simple rules such as repeating revenue
from the same quarter last year?

The final model is designed as a consistent starting point for FP&A review, budgeting,
resource planning, and variance analysis. It is not intended to replace business judgment.

---

## Dashboard

![Revenue Forecast Dashboard](images/dashboard.png)

*Power BI supports company and quarter selection, forecast-versus-actual comparison, and
absolute percentage error review. Company examples are illustrative; the headline metrics
come from the complete held-out 2025 test set.*

---

## Final results: held-out 2025 test set

The final test contains **12,924 firm-quarter observations from 3,586 firms**. The model
was trained on earlier years, tuned on 2024, and evaluated on 2025 only after model
selection was complete.

| Metric | Seasonal baseline | Last-quarter baseline | Final Random Forest |
| --- | ---: | ---: | ---: |
| MAE | $191.94M | $171.85M | **$153.36M** |
| RMSE | $1,182.73M | Not reported | **$1,018.51M** |
| Median absolute percentage error | 10.63% | Not reported | **7.31%** |
| sMAPE | 25.1% | Not reported | **18.2%** |
| Forecasts within ±10% of actual | 47.80% | Not reported | **58.78%** |
| Forecasts within ±20% of actual | 70.62% | Not reported | **76.70%** |
| Negative forecasts | 64 | 54 | **0** |

### Headline finding

The final log-growth Random Forest reduced 2025 MAE by **20.10%** compared with the
seasonal baseline and produced no negative forecasts.

MAE and RMSE are influenced by a small number of very large, volatile firms. Median APE,
sMAPE, and within-range measures provide a complementary view across companies of
different sizes. Percentage-based metrics use only observations with positive actual
revenue.

---

## Data pipeline and timing control

The pipeline converts raw SEC filings into a timing-valid forecasting panel:

| Stage | Observations | Description |
| --- | ---: | --- |
| Direct SEC revenue panel | 67,094 | Firm-quarter revenue reported directly in SEC filings |
| Panel after fiscal-Q4 derivation | 84,108 | Adds Q4 when annual revenue and Q1-Q3 are available |
| Candidate modeling rows | 51,168 | Creates next-quarter targets and model features |
| Final timing-valid sample | **48,802** | Excludes 2,366 rows that failed the timing rule |

The final sample covers **4,528 firms** and contains no duplicate company-period rows or
missing targets.

The timing rule requires:

```text
filed_date < target_period_end_date
```

This prevents the model from using a filing that became available only after the target
quarter had already ended.

### Chronological split

| Split | Target period | Observations | Firms |
| --- | --- | ---: | ---: |
| Training | 2022Q2-2023Q4 | 22,559 | 4,068 |
| Validation | 2024Q1-2024Q4 | 13,319 | 3,714 |
| Test | 2025Q1-2025Q4 | 12,924 | 3,586 |

---

## Modeling approach

The final model predicts proportional revenue movement rather than revenue dollars
directly:

```text
log_growth = log(next_quarter_revenue / current_quarter_revenue)
forecast = current_quarter_revenue × exp(predicted_log_growth)
```

This scale allows one model to learn from both small and large companies.

### Final model configuration

- Model: `RandomForestRegressor`
- Target: next-quarter log revenue growth
- Features: 22 total
  - 7 revenue-history features
  - 6 macroeconomic features
  - 9 sector indicators
- Trees: 500
- Minimum samples per leaf: 20
- Random state: 0
- Fallback: seasonal forecast when current revenue is zero or negative
- Final safeguard: clip any remaining negative forecast to zero

The macroeconomic features are based on inflation, the federal funds rate,
unemployment, and their quarterly changes.

---

## Derived fiscal-Q4 sensitivity

Some annual filings report full-year revenue without a separate fourth-quarter value. In
those cases, fiscal Q4 is derived as:

```text
fiscal_Q4 = annual_revenue - Q1 - Q2 - Q3
```

Derived-Q4 observations account for **2,709 of 12,924 test rows, or 20.96%** of the 2025
test set.

| Current-quarter revenue source | Test rows | Final-model MAE | Improvement vs seasonal |
| --- | ---: | ---: | ---: |
| Directly reported SEC revenue | 10,215 | **$140.90M** | **27.38% better** |
| Derived fiscal Q4 | 2,709 | **$200.36M** | **8.83% worse** |

The overall improvement is therefore not uniform. Derived fiscal Q4 remains the clearest
weakness and should be flagged for review. A production workflow should consider a
Q4-specific model, seasonal override, or ensemble.

---

## Macro-feature ablation

A controlled ablation compared two models using the same training rows, test rows,
Random Forest settings, and fallback rule. The only difference was whether the six macro
features were included.

| Model version | MAE | Median APE | sMAPE | RMSE |
| --- | ---: | ---: | ---: | ---: |
| With macro features | **$153.36M** | **7.31%** | **18.2%** | $1,018.51M |
| Without macro features | $154.88M | 7.56% | 18.4% | **$1,007.50M** |

Macro features improved MAE by only **0.98%** and slightly improved typical percentage
errors, but they did not reduce the largest misses. Company-specific revenue history
remained the main forecasting signal.

---

## FP&A deliverables

- **Power BI dashboard:** full-test summary and company-level forecast review
- **Excel variance workbook:** forecast-versus-actual analysis and official metrics
- **Academic report:** data, methodology, safeguards, findings, and limitations
- **Presentation:** business question, pipeline, model results, sensitivity tests, and
  Power BI example
- **Model outputs:** final 2025 predictions, evaluation metrics, Q4 sensitivity results,
  and macro-ablation results

---

## Tools

| Tool | Use |
| --- | --- |
| Python: pandas, NumPy, scikit-learn | Data preparation, feature engineering, modeling, and evaluation |
| SQL: MySQL | Cleaning, deduplication, aggregation, growth calculations, and macro joins |
| Power BI | Interactive company-level forecast and variance review |
| Excel | FP&A-style variance analysis and metric reporting |
| SEC Financial Statement Data Sets | Company financial-statement data |
| FRED | CPI, federal funds rate, and unemployment data |

---

## Repository structure

```text
.
├── README.md
├── scripts/
│   └── improved_revenue_forecast.py
├── sql/
│   └── revenue_forecasting_queries.sql
├── data/
│   └── sample/
│       └── sample_revenue_panel.csv
├── outputs/
│   ├── Revenue_Forecast_Variance_Analysis.xlsx
│   ├── log_growth_random_forest_metrics.csv
│   ├── log_growth_random_forest_predictions_test_2025.csv
│   ├── derived_q4_counts_by_split.csv
│   ├── derived_q4_sensitivity_test_2025.csv
│   └── powerbi/
│       ├── powerbi_forecast_data.csv
│       └── powerbi_model_metrics.csv
├── reports/
│   └── NgocDinh_MSDS_Capstone_Report_Final_REVISED.docx
├── presentations/
│   └── NgocDinh_MSDS_Capstone_Presentation_FINAL_REVISED.pptx
└── images/
    └── dashboard.png
```

The tree highlights the principal deliverables rather than every intermediate audit or
development file.

---

## How to run

### Python model

```bash
python scripts/improved_revenue_forecast.py \
  --data data/processed/model_ready_sec_macro_2021_2025.csv
```

The script trains on the historical training period, uses 2024 for model selection,
evaluates the final model on 2025, and writes the prediction and metric outputs.

### SQL workflow

Open `sql/revenue_forecasting_queries.sql` in MySQL Workbench. Create the table, import
the source data, and run the cleaning, deduplication, aggregation, growth, and macro-join
queries in order.

### Data note

The full **48,802-row timing-valid modeling panel** is not committed to this repository.
The demonstration sample in `data/sample/` is included so the repository structure and
SQL workflow can be reviewed without publishing the full processed dataset. The sample
does not reproduce the official results.

---

## Limitations

- This is a timing-screened historical evaluation, not a complete real-time backtest
  rebuilt from one fixed forecasting date.
- Filing availability was screened, but forecast lead times vary across observations.
- FRED series may contain later revisions rather than the exact historical data vintage
  available at each forecast date.
- Derived fiscal Q4 introduces additional measurement and period-alignment risk.
- Rows with zero or negative current revenue require a seasonal fallback because log
  growth cannot be calculated normally.
- Dollar-error measures are influenced by the largest firms.
- Quarterly revenue forecasting is not the same as a 13-week cash-flow forecast.
- Power BI company examples illustrate the review process; they do not replace the full
  test-set evaluation.

---

## Conclusion and future work

The project answers the research question with a **qualified yes**. The final Random
Forest improved overall 2025 accuracy, produced economically sensible non-negative
forecasts, and supported company-level review through Power BI. However, performance was
not consistent across every reporting situation.

Future work should:

1. Test a Q4-specific model, seasonal override, or ensemble.
2. Rebuild the data from fixed forecast-origin dates using point-in-time data vintages.
3. Add prediction intervals around each forecast.
4. Evaluate industry-specific models.
5. Explore richer company information such as management guidance and filing text.

The final model should be viewed as a useful starting point for review rather than a
replacement for business judgment.
