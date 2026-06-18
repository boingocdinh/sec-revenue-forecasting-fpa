from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# Script 11: Train SEC-Only Benchmark Models
# Project: Panel Forecasting of Firm Revenue Using SEC + Macro Data
#
# Important:
# - Train on train_2021_2023
# - Compare models on validation_2024
# - Do NOT use test_2025 yet
# ============================================================

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent

if SCRIPT_DIR.name == "optional" and SCRIPT_DIR.parent.name == "scripts":
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
elif SCRIPT_DIR.name == "scripts":
    PROJECT_ROOT = SCRIPT_DIR.parent
else:
    PROJECT_ROOT = SCRIPT_DIR

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecasting_dataset_2021_2025.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

METRICS_PATH = RESULTS_DIR / "sec_only_model_metrics.csv"
PREDICTIONS_PATH = RESULTS_DIR / "sec_only_validation_predictions.csv"
BEST_MODEL_PATH = MODELS_DIR / "sec_only_best_model.pkl"


def inverse_signed_log(series):
    """
    Inverse of signed log transform:
        signed_log = sign(x) * log1p(abs(x))

    Inverse:
        x = sign(y) * expm1(abs(y))
    """
    series = np.asarray(series)
    return np.sign(series) * np.expm1(np.abs(series))


def smape(y_true, y_pred):
    """
    Symmetric mean absolute percentage error.

    This is more stable than MAPE when actual revenue is near zero.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2

    mask = denominator != 0

    if mask.sum() == 0:
        return np.nan

    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask])


def make_metrics(y_true, y_pred, model_name, split_name):
    """
    Return one metrics row.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return {
        "model": model_name,
        "split": split_name,
        "rows": len(y_true),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse,
        "r2": r2_score(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "median_absolute_error": np.median(np.abs(y_true - y_pred)),
    }


def main():
    print("=" * 80)
    print("SCRIPT 11: TRAIN SEC-ONLY BENCHMARK MODELS")
    print("=" * 80)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file:\n{INPUT_PATH}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading forecasting dataset:")
    print(INPUT_PATH)

    df = pd.read_csv(INPUT_PATH, low_memory=False)

    print(f"\nRows loaded: {len(df):,}")
    print(f"Columns loaded: {len(df.columns):,}")

    # ------------------------------------------------------------
    # Basic validation
    # ------------------------------------------------------------
    required_cols = [
        "cik",
        "period_date",
        "calendar_period",
        "model_split",
        "target_revenue_next_qtr",
        "target_signed_log_revenue_next_qtr",
        "naive_forecast_next_qtr",
        "seasonal_naive_forecast_next_qtr",
    ]

    missing_required = [col for col in required_cols if col not in df.columns]

    if missing_required:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_required)
        )

    train = df[df["model_split"] == "train_2021_2023"].copy()
    val = df[df["model_split"] == "validation_2024"].copy()
    test = df[df["model_split"] == "test_2025"].copy()

    print("\nSplit counts:")
    print(f"  Train rows:      {len(train):,}")
    print(f"  Validation rows: {len(val):,}")
    print(f"  Test rows locked, not used: {len(test):,}")

    if train.empty or val.empty:
        raise ValueError("Train or validation set is empty. Stop and inspect model_split.")

    # ------------------------------------------------------------
    # Feature selection
    # SEC-only features. No macro variables yet.
    # ------------------------------------------------------------
    candidate_features = [
        "revenue_usd",
        "revenue_millions_usd",
        "signed_log_revenue",
        "revenue_lag_1",
        "revenue_lag_2",
        "revenue_lag_3",
        "revenue_lag_4",
        "revenue_growth_lag_1",
        "revenue_growth_lag_4",
        "revenue_growth_lag_1_undefined",
        "revenue_growth_lag_4_undefined",
        "signed_log_revenue_lag_1",
        "signed_log_revenue_lag_2",
        "signed_log_revenue_lag_3",
        "signed_log_revenue_lag_4",
        "signed_log_revenue_change_lag_1",
        "signed_log_revenue_change_lag_4",
        "calendar_year",
        "calendar_quarter",
        "quarter_index",
        "is_derived_q4",
        "firm_obs_total",
        "fy",
        "qtrs",
        "tag_priority",
    ]

    features = [col for col in candidate_features if col in df.columns]

    print("\nSEC-only features used:")
    for col in features:
        print(f"  - {col}")

    if not features:
        raise ValueError("No usable features found.")

    X_train = train[features]
    X_val = val[features]

    # Train models on signed-log target for stability.
    # Then convert predictions back to raw revenue dollars for evaluation.
    y_train_log = train["target_signed_log_revenue_next_qtr"].astype(float)
    y_val_raw = val["target_revenue_next_qtr"].astype(float)

    # ------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------
    numeric_preprocess = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor_scaled = ColumnTransformer(
        transformers=[
            ("num", numeric_preprocess, features),
        ],
        remainder="drop",
    )

    preprocessor_unscaled = ColumnTransformer(
        transformers=[
            (
                "num",
                SimpleImputer(strategy="median"),
                features,
            ),
        ],
        remainder="drop",
    )

    # ------------------------------------------------------------
    # Models
    # ------------------------------------------------------------
    models = {
        "linear_regression_signed_log": Pipeline(
            steps=[
                ("preprocess", preprocessor_scaled),
                ("model", LinearRegression()),
            ]
        ),
        "ridge_signed_log": Pipeline(
            steps=[
                ("preprocess", preprocessor_scaled),
                ("model", Ridge(alpha=1.0, random_state=42)),
            ]
        ),
        "hist_gradient_boosting_signed_log": Pipeline(
            steps=[
                ("preprocess", preprocessor_unscaled),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=300,
                        learning_rate=0.05,
                        max_leaf_nodes=31,
                        l2_regularization=0.1,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest_signed_log": Pipeline(
            steps=[
                ("preprocess", preprocessor_unscaled),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=200,
                        max_depth=12,
                        min_samples_leaf=5,
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    # Optional XGBoost if installed
    try:
        from xgboost import XGBRegressor

        models["xgboost_signed_log"] = Pipeline(
            steps=[
                ("preprocess", preprocessor_unscaled),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=500,
                        learning_rate=0.03,
                        max_depth=5,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="reg:squarederror",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        print("\nXGBoost is installed. Added xgboost_signed_log model.")

    except Exception:
        print("\nXGBoost not installed. Skipping XGBoost for Script 11.")

    # ------------------------------------------------------------
    # Baseline metrics on validation
    # ------------------------------------------------------------
    metrics_rows = []
    prediction_output = val[
        [
            "cik",
            "company_name",
            "sic",
            "period_date",
            "calendar_period",
            "target_period_date",
            "target_calendar_period",
            "target_revenue_next_qtr",
            "model_split",
        ]
    ].copy()

    baseline_cols = [
        ("naive_current_revenue", "naive_forecast_next_qtr"),
        ("seasonal_naive_same_quarter_last_year", "seasonal_naive_forecast_next_qtr"),
    ]

    for baseline_name, pred_col in baseline_cols:
        y_pred = val[pred_col].astype(float)

        metrics_rows.append(
            make_metrics(
                y_true=y_val_raw.values,
                y_pred=y_pred.values,
                model_name=baseline_name,
                split_name="validation_2024",
            )
        )

        prediction_output[f"pred_{baseline_name}"] = y_pred.values

    # ------------------------------------------------------------
    # Train ML models and evaluate on validation
    # ------------------------------------------------------------
    fitted_models = {}

    for model_name, pipeline in models.items():
        print("\n" + "-" * 80)
        print(f"Training model: {model_name}")
        print("-" * 80)

        pipeline.fit(X_train, y_train_log)

        pred_log = pipeline.predict(X_val)
        pred_raw = inverse_signed_log(pred_log)

        fitted_models[model_name] = pipeline

        metrics = make_metrics(
            y_true=y_val_raw.values,
            y_pred=pred_raw,
            model_name=model_name,
            split_name="validation_2024",
        )

        metrics_rows.append(metrics)

        prediction_output[f"pred_{model_name}"] = pred_raw

        print(f"Validation MAE:  {metrics['mae']:,.2f}")
        print(f"Validation RMSE: {metrics['rmse']:,.2f}")
        print(f"Validation R2:   {metrics['r2']:,.4f}")
        print(f"Validation sMAPE:{metrics['smape']:,.4f}")

    # ------------------------------------------------------------
    # Save metrics and choose best model by validation MAE
    # ------------------------------------------------------------
    metrics_df = pd.DataFrame(metrics_rows)

    metrics_df = metrics_df.sort_values(
        ["mae", "rmse"],
        ascending=[True, True],
    ).reset_index(drop=True)

    metrics_df.to_csv(METRICS_PATH, index=False)
    prediction_output.to_csv(PREDICTIONS_PATH, index=False)

    print("\n" + "=" * 80)
    print("VALIDATION MODEL COMPARISON")
    print("=" * 80)
    print(metrics_df.to_string(index=False))

    # Only choose among ML models, not naive baselines
    ml_metrics = metrics_df[
        ~metrics_df["model"].isin(
            [
                "naive_current_revenue",
                "seasonal_naive_same_quarter_last_year",
            ]
        )
    ].copy()

    best_model_name = ml_metrics.iloc[0]["model"]

    best_bundle = {
        "model_name": best_model_name,
        "model": fitted_models[best_model_name],
        "features": features,
        "target_used_for_training": "target_signed_log_revenue_next_qtr",
        "prediction_inverse_transform": "sign(y) * expm1(abs(y))",
        "selected_by": "lowest validation MAE among ML models",
        "validation_metrics": metrics_df.to_dict(orient="records"),
    }

    joblib.dump(best_bundle, BEST_MODEL_PATH)

    print("\n" + "=" * 80)
    print("OUTPUTS SAVED")
    print("=" * 80)
    print(f"Metrics:      {METRICS_PATH}")
    print(f"Predictions:  {PREDICTIONS_PATH}")
    print(f"Best model:   {BEST_MODEL_PATH}")

    print("\nBest SEC-only ML model selected by validation MAE:")
    print(f"  {best_model_name}")

    best_row = metrics_df[metrics_df["model"] == best_model_name].iloc[0]

    print("\nBest model validation metrics:")
    print(f"  MAE:   {best_row['mae']:,.2f}")
    print(f"  RMSE:  {best_row['rmse']:,.2f}")
    print(f"  R2:    {best_row['r2']:,.4f}")
    print(f"  sMAPE: {best_row['smape']:,.4f}")

    print("\nReminder:")
    print("  The 2025 test set was not used in this script.")
    print("  Use validation results only for model comparison right now.")

    print("\nScript 11 completed successfully.")


if __name__ == "__main__":
    main()