"""
Improved revenue forecasting model
===================================
Reformulates the original dollar-scale residual model into LOG-GROWTH space.

Why this is better (and the honest tradeoff):
  * Predicting log(next_revenue / current_revenue) instead of raw dollars means
    the reconstructed forecast = current_revenue * exp(pred) is ALWAYS positive.
    This eliminates the impossible negative forecasts the original model produced.
  * Modeling growth (not level) stops the model from simply learning "big firms
    are big," so accuracy is spread proportionally across firm sizes rather than
    concentrated on a handful of mega-caps.
  * Tradeoff to be aware of: because it no longer over-optimizes for the giants,
    raw-dollar MAE (which is dominated by mega-caps) may be flat or slightly worse,
    while proportional metrics (sMAPE, median % error) improve. Report BOTH and
    let the business objective decide which matters.

Splits (uses the model_split column already in the file):
    train_2021_2023  -> fit
    validation_2024  -> hyperparameter selection
    test_2025        -> final, out-of-sample reporting

Run:
    python improved_revenue_forecast.py --data data/processed/model_ready_sec_macro_2021_2025.csv
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# ----------------------------- helpers ------------------------------------- #
def smape(actual, forecast):
    """Symmetric MAPE in [0,2]. Robust to scale; 0 is perfect."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    denom = np.abs(actual) + np.abs(forecast)
    with np.errstate(divide="ignore", invalid="ignore"):
        per = np.where(denom == 0, 0.0, 2.0 * np.abs(forecast - actual) / denom)
    return float(np.nanmean(per))


def metrics_block(name, actual, forecast):
    """Return a dict of the metrics we care about for one model."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    abs_err = np.abs(forecast - actual)
    # median absolute % error, computed only where actual > 0 (else undefined)
    pos = actual > 0
    med_pct = float(np.median(np.abs(forecast[pos] - actual[pos]) / actual[pos]))
    return {
        "model": name,
        "MAE_$mm": abs_err.mean() / 1e6,
        "median_abs_pct": med_pct,
        "sMAPE": smape(actual, forecast),
        "neg_forecasts": int((forecast < 0).sum()),
        "n": len(actual),
    }


def sector_from_sic(sic):
    try:
        s = int(sic)
    except (ValueError, TypeError):
        return "other"
    buckets = [
        (100, 1000, "agriculture"), (1000, 1500, "mining"), (1500, 1800, "construction"),
        (2000, 4000, "manufacturing"), (4000, 5000, "utilities_transport"),
        (5000, 5200, "wholesale"), (5200, 6000, "retail"),
        (6000, 6800, "finance"), (7000, 9000, "services"),
    ]
    for lo, hi, label in buckets:
        if lo <= s < hi:
            return label
    return "other"


# ----------------------------- feature build ------------------------------- #
def build_features(df):
    """Engineer the firm-level feature matrix. Returns (X, df_aligned)."""
    df = df.copy()
    df["sector"] = df["sic"].apply(sector_from_sic)

    # Numeric features. These all already exist in the file; we lean on the
    # log/growth lags because they carry firm-level signal that macro vars don't.
    numeric = [
        "signed_log_revenue",            # current revenue, log scale
        "signed_log_revenue_lag_1",
        "signed_log_revenue_lag_4",      # same quarter last year (seasonality)
        "signed_log_revenue_change_lag_1",
        "signed_log_revenue_change_lag_4",
        "revenue_growth_lag_1",
        "revenue_growth_lag_4",
        "cpi", "fed_funds_rate", "unemployment_rate",
        "cpi_qoq_pct_change", "fed_funds_rate_qoq_change", "unemployment_rate_qoq_change",
    ]
    numeric = [c for c in numeric if c in df.columns]
    X = pd.get_dummies(df[numeric + ["sector"]], columns=["sector"], dummy_na=False)
    # Median-impute any remaining gaps so the model never sees NaN
    X = X.fillna(X.median(numeric_only=True))
    return X, df


# ----------------------------- main ---------------------------------------- #
def main(path, mega_cap_threshold_mm=5000):
    df = pd.read_csv(path)

    # Target = log growth. Only defined where both current and next revenue are positive.
    df["pos_pair"] = (df["revenue_usd"] > 0) & (df["target_revenue_next_qtr"] > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = df["target_revenue_next_qtr"] / df["revenue_usd"]
        df["log_growth"] = np.where(df["pos_pair"], np.log(ratio.where(ratio > 0)), np.nan)

    X, df = build_features(df)

    tr = df["model_split"] == "train_2021_2023"
    va = df["model_split"] == "validation_2024"
    te = df["model_split"] == "test_2025"

    # Train only on rows with a defined log-growth target.
    fit_mask = tr & df["pos_pair"]

    # ---- hyperparameter selection on validation_2024 ----
    candidates = [
        {"n_estimators": 300, "min_samples_leaf": 5},
        {"n_estimators": 400, "min_samples_leaf": 10},
        {"n_estimators": 500, "min_samples_leaf": 20},
    ]
    best, best_smape = None, np.inf
    for params in candidates:
        m = RandomForestRegressor(random_state=0, n_jobs=-1, **params)
        m.fit(X[fit_mask], df.loc[fit_mask, "log_growth"])
        va_eval = va & df["pos_pair"]
        pred_lg = m.predict(X[va_eval])
        fcst = df.loc[va_eval, "revenue_usd"].to_numpy() * np.exp(pred_lg)
        s = smape(df.loc[va_eval, "target_revenue_next_qtr"], fcst)
        print(f"  validation sMAPE {s:.4f}  params={params}")
        if s < best_smape:
            best, best_smape = params, s
    print(f"Selected params: {best}  (validation sMAPE {best_smape:.4f})\n")

    # ---- refit on train, predict on test ----
    model = RandomForestRegressor(random_state=0, n_jobs=-1, **best)
    model.fit(X[fit_mask], df.loc[fit_mask, "log_growth"])

    test = df[te].copy()
    Xte = X[te]

    # Improved forecast: reconstruct dollars from predicted log-growth, floor at 0.
    # Where the log-growth target was undefined (non-positive revenue), fall back
    # to the seasonal naive forecast so every row gets a prediction.
    pred_lg_test = model.predict(Xte)
    improved = test["revenue_usd"].to_numpy() * np.exp(pred_lg_test)
    fallback = test["seasonal_naive_forecast_next_qtr"].to_numpy()
    improved = np.where(test["pos_pair"].to_numpy(), improved, fallback)
    improved = np.clip(improved, 0, None)            # non-negativity guarantee
    test["pred_improved"] = improved

    # ---- OPTIONAL mega-cap segmentation -----------------------------------
    # Giants drive dollar error. Reporting a version where, for very large firms,
    # we defer to the seasonal naive (which tracks their scale well) recovers
    # dollar MAE without reintroducing negatives. Shown as a separate row.
    big = test["revenue_usd"] >= mega_cap_threshold_mm * 1e6
    segmented = improved.copy()
    segmented[big.to_numpy()] = np.clip(
        test.loc[big, "seasonal_naive_forecast_next_qtr"].to_numpy(), 0, None
    )
    test["pred_segmented"] = segmented

    actual = test["target_revenue_next_qtr"]
    results = pd.DataFrame([
        metrics_block("Seasonal baseline", actual, test["seasonal_naive_forecast_next_qtr"]),
        metrics_block("Naive (last qtr)", actual, test["naive_forecast_next_qtr"])
        if "naive_forecast_next_qtr" in test.columns else
        metrics_block("Seasonal baseline (dup)", actual, test["seasonal_naive_forecast_next_qtr"]),
        metrics_block("Improved (log-growth)", actual, test["pred_improved"]),
        metrics_block("Improved + mega-cap seg...", actual, test["pred_segmented"]),
    ])

    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
    print("FINAL RESULTS — test_2025\n")
    print(results.to_string(index=False))

    base_mae = results.loc[0, "MAE_$mm"]
    imp_mae = results.loc[2, "MAE_$mm"]
    seg_mae = results.loc[3, "MAE_$mm"]
    print(f"\nImproved vs seasonal baseline (MAE):       {(base_mae-imp_mae)/base_mae*100:+.1f}%")
    print(f"Segmented  vs seasonal baseline (MAE):     {(base_mae-seg_mae)/base_mae*100:+.1f}%")
    print(f"Negative forecasts eliminated:  improved={results.loc[2,'neg_forecasts']}, "
          f"segmented={results.loc[3,'neg_forecasts']}")

    # Save test-set predictions for the variance workbook / Power BI
    out = test[["cik", "company_name", "sic", "target_calendar_period",
                "target_revenue_next_qtr", "seasonal_naive_forecast_next_qtr",
                "pred_improved", "pred_segmented"]].copy()
    out.to_csv("improved_predictions_test_2025.csv", index=False)
    print("\nWrote improved_predictions_test_2025.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/model_ready_sec_macro_2021_2025.csv")
    ap.add_argument("--mega_cap_threshold_mm", type=float, default=5000)
    args = ap.parse_args()
    main(args.data, args.mega_cap_threshold_mm)
