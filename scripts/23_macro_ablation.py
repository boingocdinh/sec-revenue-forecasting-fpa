"""
Macro-feature ablation for the locked final revenue forecasting model.

Compares:
1) the saved final model with macro features
2) the same Random Forest specification with the six macro features removed

Everything else is held fixed:
- same timing-valid dataset
- same train/test rows
- same preprocessing from improved_revenue_forecast.build_features()
- same 500 trees
- same min_samples_leaf=20
- same random_state=0
- same log-growth target
- same seasonal fallback routing
- same official metric function

Run from project root:
    python scripts/23_macro_ablation.py

Optional explicit data path:
    python scripts/23_macro_ablation.py data/processed/<final_dataset>.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import improved_revenue_forecast as finalmod  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "log_growth_random_forest_final.pkl"
LOCKED_METRICS_PATH = PROJECT_ROOT / "outputs" / "report_tables" / "log_growth_random_forest_metrics.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "report_tables"
DETAIL_OUTPUT = OUTPUT_DIR / "macro_ablation_test_2025.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "macro_ablation_summary.csv"

MACRO_FEATURES = [
    "cpi",
    "fed_funds_rate",
    "unemployment_rate",
    "cpi_qoq_pct_change",
    "fed_funds_rate_qoq_change",
    "unemployment_rate_qoq_change",
]

EXPECTED_TOTAL_ROWS = 48_802
EXPECTED_TRAIN_CANDIDATES = 22_559
EXPECTED_TRAIN_USED = 21_781
EXPECTED_TEST_ROWS = 12_924
EXPECTED_TEST_FIRMS = 3_586


def find_final_dataset() -> Path:
    processed = PROJECT_ROOT / "data" / "processed"
    required = {
        "cik",
        "sic",
        "model_split",
        "revenue_usd",
        "target_revenue_next_qtr",
        "seasonal_naive_forecast_next_qtr",
        *MACRO_FEATURES,
    }

    candidates = []
    for path in sorted(processed.glob("*.csv")):
        if "timing_audit" in path.name.lower():
            continue
        try:
            header = pd.read_csv(path, nrows=0)
        except Exception:
            continue
        if not required.issubset(set(header.columns)):
            continue
        try:
            rows = len(pd.read_csv(path, usecols=["model_split"]))
        except Exception:
            continue
        if rows == EXPECTED_TOTAL_ROWS:
            candidates.append(path)

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise FileNotFoundError(
            "Could not automatically find a 48,802-row final modeling CSV in data/processed. "
            "Re-run with the dataset path explicitly."
        )

    names = "\n".join(f"  - {p}" for p in candidates)
    raise RuntimeError(
        "More than one 48,802-row candidate dataset was found. "
        "Pass the correct one explicitly:\n" + names
    )


def extract_feature_matrix(build_result):
    if isinstance(build_result, pd.DataFrame):
        return build_result

    if isinstance(build_result, tuple):
        for item in build_result:
            if isinstance(item, pd.DataFrame):
                return item

    raise TypeError(
        "Could not identify the feature matrix returned by build_features()."
    )


def routed_forecast(model, X, rows):
    predicted_log_growth = model.predict(X)

    current_revenue = rows["revenue_usd"].to_numpy(dtype=float)
    model_forecast = current_revenue * np.exp(predicted_log_growth)

    fallback_forecast = rows[
        "seasonal_naive_forecast_next_qtr"
    ].to_numpy(dtype=float)

    current_positive = current_revenue > 0
    fallback_used = ~current_positive

    forecast = np.where(
        current_positive,
        model_forecast,
        fallback_forecast,
    )

    forecast = np.clip(forecast, 0.0, None)
    return forecast, fallback_used


def check_close(name, observed, expected, atol=1e-6, rtol=1e-9):
    if not np.isclose(observed, expected, atol=atol, rtol=rtol):
        raise RuntimeError(
            f"Locked-model reproduction failed for {name}: "
            f"reproduced={observed}, locked={expected}."
        )


def main(data_path: Path) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("MACRO FEATURE ABLATION — FINAL TIMING-VALID LOG-GROWTH RANDOM FOREST")
    print("=" * 88)
    print(f"Dataset: {data_path}")
    print(f"Saved model: {MODEL_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Saved model not found: {MODEL_PATH}")

    bundle = joblib.load(MODEL_PATH)
    saved_model = bundle["model"]

    checks = {
        "model_version": "final_v2_timing_valid",
        "random_seed": 0,
        "training_rows_used": EXPECTED_TRAIN_USED,
        "test_rows": EXPECTED_TEST_ROWS,
        "test_firms": EXPECTED_TEST_FIRMS,
    }
    for key, expected in checks.items():
        if bundle.get(key) != expected:
            raise RuntimeError(
                f"Unexpected saved-model metadata for {key}: "
                f"{bundle.get(key)!r} != {expected!r}"
            )

    params = saved_model.get_params()
    if params.get("n_estimators") != 500:
        raise RuntimeError("Saved model does not have n_estimators=500.")
    if params.get("min_samples_leaf") != 20:
        raise RuntimeError("Saved model does not have min_samples_leaf=20.")
    if params.get("random_state") != 0:
        raise RuntimeError("Saved model does not have random_state=0.")

    full_feature_columns = list(bundle["feature_columns"])
    missing_macro = [c for c in MACRO_FEATURES if c not in full_feature_columns]
    if missing_macro:
        raise RuntimeError(
            f"Saved model is missing expected macro features: {missing_macro}"
        )

    no_macro_columns = [
        c for c in full_feature_columns if c not in MACRO_FEATURES
    ]

    print(f"Full-model features: {len(full_feature_columns)}")
    print(f"No-macro features:   {len(no_macro_columns)}")
    print("Removed macro features:")
    for feature in MACRO_FEATURES:
        print(f"  - {feature}")

    df = pd.read_csv(
        data_path,
        dtype={"cik": "string", "sic": "string"},
    )
    df = finalmod.validate_input_dataset(df)

    if len(df) != EXPECTED_TOTAL_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_ROWS:,} final rows; found {len(df):,}."
        )

    pos_pair = (
        (df["revenue_usd"] > 0)
        & (df["target_revenue_next_qtr"] > 0)
    )
    revenue_ratio = df["target_revenue_next_qtr"] / df["revenue_usd"]
    df["log_growth"] = np.where(
        pos_pair,
        np.log(revenue_ratio.where(revenue_ratio > 0)),
        np.nan,
    )

    feature_matrix = extract_feature_matrix(finalmod.build_features(df))

    missing_features = [
        c for c in full_feature_columns if c not in feature_matrix.columns
    ]
    if missing_features:
        raise RuntimeError(
            "Feature matrix is missing saved-model features: "
            + ", ".join(missing_features)
        )

    X_full = feature_matrix.loc[:, full_feature_columns].copy()
    X_no_macro = feature_matrix.loc[:, no_macro_columns].copy()

    train_mask = df["model_split"].eq("train_2021_2023")
    test_mask = df["model_split"].eq("test_2025")
    fit_mask = train_mask & pos_pair

    if int(train_mask.sum()) != EXPECTED_TRAIN_CANDIDATES:
        raise RuntimeError(
            f"Expected {EXPECTED_TRAIN_CANDIDATES:,} training candidates; "
            f"found {int(train_mask.sum()):,}."
        )
    if int(fit_mask.sum()) != EXPECTED_TRAIN_USED:
        raise RuntimeError(
            f"Expected {EXPECTED_TRAIN_USED:,} training rows used; "
            f"found {int(fit_mask.sum()):,}."
        )
    if int(test_mask.sum()) != EXPECTED_TEST_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST_ROWS:,} test rows; "
            f"found {int(test_mask.sum()):,}."
        )

    test = df.loc[test_mask].copy()
    if int(test["cik"].nunique()) != EXPECTED_TEST_FIRMS:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST_FIRMS:,} test firms; "
            f"found {int(test['cik'].nunique()):,}."
        )

    with_macro_forecast, with_macro_fallback = routed_forecast(
        saved_model,
        X_full.loc[test_mask],
        test,
    )

    actual = test["target_revenue_next_qtr"].to_numpy(dtype=float)
    firms = test["cik"]

    with_macro_metrics = finalmod.metrics_block(
        "with_macro_saved_final",
        actual,
        with_macro_forecast,
        firms,
        fallback_observations=int(with_macro_fallback.sum()),
    )

    locked = pd.read_csv(LOCKED_METRICS_PATH)
    locked_final = locked.loc[
        locked["model"].eq("log_growth_random_forest")
    ].iloc[0]

    check_close(
        "MAE",
        with_macro_metrics["mae_usd"],
        float(locked_final["mae_usd"]),
        atol=1e-3,
        rtol=1e-12,
    )
    check_close(
        "RMSE",
        with_macro_metrics["rmse_usd"],
        float(locked_final["rmse_usd"]),
        atol=1e-3,
        rtol=1e-12,
    )
    check_close(
        "sMAPE",
        with_macro_metrics["smape_positive_actual"],
        float(locked_final["smape_positive_actual"]),
        atol=1e-12,
        rtol=1e-12,
    )

    print("\nLocked WITH-MACRO model reproduced successfully.")

    no_macro_model = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=20,
        random_state=0,
        n_jobs=-1,
    )

    no_macro_model.fit(
        X_no_macro.loc[fit_mask],
        df.loc[fit_mask, "log_growth"],
    )

    no_macro_forecast, no_macro_fallback = routed_forecast(
        no_macro_model,
        X_no_macro.loc[test_mask],
        test,
    )

    no_macro_metrics = finalmod.metrics_block(
        "without_macro_same_rf",
        actual,
        no_macro_forecast,
        firms,
        fallback_observations=int(no_macro_fallback.sum()),
    )

    comparison = pd.DataFrame(
        [with_macro_metrics, no_macro_metrics]
    )

    metric_cols = [
        "model",
        "test_observations",
        "test_firms",
        "mae_usd_millions",
        "rmse_usd_millions",
        "median_absolute_percentage_error_pct",
        "smape_positive_actual",
        "within_5_pct",
        "within_10_pct",
        "within_20_pct",
        "negative_forecasts",
        "fallback_observations",
    ]
    comparison.loc[:, metric_cols].to_csv(
        DETAIL_OUTPUT,
        index=False,
    )

    with_mae = float(with_macro_metrics["mae_usd"])
    no_mae = float(no_macro_metrics["mae_usd"])
    macro_mae_improvement_pct = (
        (no_mae - with_mae) / no_mae * 100.0
    )

    summary = pd.DataFrame(
        [
            {
                "comparison": "with_macro_vs_without_macro_same_rf",
                "with_macro_mae_usd_millions":
                    with_macro_metrics["mae_usd_millions"],
                "without_macro_mae_usd_millions":
                    no_macro_metrics["mae_usd_millions"],
                "macro_mae_improvement_pct":
                    macro_mae_improvement_pct,
                "with_macro_rmse_usd_millions":
                    with_macro_metrics["rmse_usd_millions"],
                "without_macro_rmse_usd_millions":
                    no_macro_metrics["rmse_usd_millions"],
                "with_macro_median_ape_pct":
                    with_macro_metrics[
                        "median_absolute_percentage_error_pct"
                    ],
                "without_macro_median_ape_pct":
                    no_macro_metrics[
                        "median_absolute_percentage_error_pct"
                    ],
                "with_macro_smape":
                    with_macro_metrics["smape_positive_actual"],
                "without_macro_smape":
                    no_macro_metrics["smape_positive_actual"],
                "with_macro_within_10_pct":
                    with_macro_metrics["within_10_pct"],
                "without_macro_within_10_pct":
                    no_macro_metrics["within_10_pct"],
                "with_macro_features": len(full_feature_columns),
                "without_macro_features": len(no_macro_columns),
                "training_rows_used": int(fit_mask.sum()),
                "test_rows": int(test_mask.sum()),
                "test_firms": int(test["cik"].nunique()),
                "random_seed": 0,
                "n_estimators": 500,
                "min_samples_leaf": 20,
            }
        ]
    )
    summary.to_csv(SUMMARY_OUTPUT, index=False)

    print("\n" + "=" * 88)
    print("TEST-SET MACRO ABLATION RESULTS")
    print("=" * 88)
    print(
        comparison.loc[:, metric_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:,.6f}",
        )
    )

    print("\nMacro contribution to MAE:")
    print(
        "  Positive value = WITH macro has lower MAE than WITHOUT macro."
    )
    print(
        f"  MAE improvement from macro features: "
        f"{macro_mae_improvement_pct:.6f}%"
    )

    print("\nOutputs saved:")
    print(f"  Detailed metrics: {DETAIL_OUTPUT}")
    print(f"  Summary:          {SUMMARY_OUTPUT}")
    print("=" * 88)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        supplied = Path(sys.argv[1])
        if not supplied.is_absolute():
            supplied = PROJECT_ROOT / supplied
        data_path = supplied.resolve()
    else:
        data_path = find_final_dataset()

    main(data_path)
