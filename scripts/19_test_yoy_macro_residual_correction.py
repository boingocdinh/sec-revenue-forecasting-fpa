from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# Script 19: Test YOY Seasonal + Macro Residual Correction
#
# Question:
# Can macro variables improve the strong YOY adjusted seasonal forecast?
#
# Method:
# 1. Base forecast:
#       yoy_pred = revenue_lag_3 + (revenue_usd - revenue_lag_4)
#
# 2. Residual target:
#       residual = actual_next_revenue - yoy_pred
#
# 3. Train models to predict residual using macro variables
#    and optional firm-history controls.
#
# 4. Final forecast:
#       corrected_pred = yoy_pred + predicted_residual
#
# Important:
# - Model selection uses validation_2024.
# - test_2025 is reported as a supplemental/final check.
# ============================================================

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent

if SCRIPT_DIR.name == "scripts":
    PROJECT_ROOT = SCRIPT_DIR.parent
else:
    PROJECT_ROOT = SCRIPT_DIR

if len(sys.argv) > 1:
    PROJECT_ROOT = Path(sys.argv[1]).expanduser().resolve()

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_ready_sec_macro_2021_2025.csv"
)

RESULTS_DIR = PROJECT_ROOT / "outputs" / "report_tables"
CHARTS_DIR = PROJECT_ROOT / "outputs" / "charts"
MODELS_DIR = PROJECT_ROOT / "models"

METRICS_PATH = RESULTS_DIR / "yoy_macro_residual_metrics.csv"
PREDICTIONS_PATH = RESULTS_DIR / "yoy_macro_residual_predictions.csv"
PLOT_PATH = CHARTS_DIR / "yoy_macro_residual_model_comparison.png"
BEST_MODEL_PATH = MODELS_DIR / "yoy_macro_residual_best_model.pkl"


def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0

    if mask.sum() == 0:
        return np.nan

    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask])


def make_metrics(y_true, y_pred, model_name, model_group, split_name):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    smape_value = smape(y_true, y_pred)

    return {
        "model": model_name,
        "model_group": model_group,
        "split": split_name,
        "rows": len(y_true),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "smape": smape_value,
        "approx_accuracy_from_smape": 1 - smape_value,
        "median_absolute_error": np.median(np.abs(y_true - y_pred)),
    }


def add_features(df):
    df = df.copy()

    if "sic" in df.columns:
        df["sic_numeric"] = pd.to_numeric(df["sic"], errors="coerce")
        df["sic_2digit"] = np.floor(df["sic_numeric"] / 100)
    else:
        df["sic_numeric"] = np.nan
        df["sic_2digit"] = np.nan

    current_and_past_4 = [
        "revenue_usd",
        "revenue_lag_1",
        "revenue_lag_2",
        "revenue_lag_3",
    ]

    lag_1_to_4 = [
        "revenue_lag_1",
        "revenue_lag_2",
        "revenue_lag_3",
        "revenue_lag_4",
    ]

    df["revenue_roll4_mean_current"] = df[current_and_past_4].mean(axis=1)
    df["revenue_roll4_median_current"] = df[current_and_past_4].median(axis=1)
    df["revenue_lag_mean_1_4"] = df[lag_1_to_4].mean(axis=1)
    df["revenue_lag_median_1_4"] = df[lag_1_to_4].median(axis=1)

    df["revenue_qoq_change"] = df["revenue_usd"] - df["revenue_lag_1"]
    df["revenue_yoy_change"] = df["revenue_usd"] - df["revenue_lag_4"]

    df["signed_log_qoq_change"] = (
        df["signed_log_revenue"] - df["signed_log_revenue_lag_1"]
    )

    df["signed_log_yoy_change"] = (
        df["signed_log_revenue"] - df["signed_log_revenue_lag_4"]
    )

    df["current_to_lag_mean_ratio"] = np.where(
        df["revenue_lag_mean_1_4"].abs() > 0,
        df["revenue_usd"] / df["revenue_lag_mean_1_4"].abs(),
        0,
    )

    # YOY adjusted seasonal baseline
    df["pred_yoy_adjusted_seasonal"] = (
        df["revenue_lag_3"].astype(float)
        + (
            df["revenue_usd"].astype(float)
            - df["revenue_lag_4"].astype(float)
        )
    )

    # Residual we want macro/control model to explain
    df["yoy_residual"] = (
        df["target_revenue_next_qtr"].astype(float)
        - df["pred_yoy_adjusted_seasonal"].astype(float)
    )

    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def make_preprocessor(features, scaled=True):
    if scaled:
        steps = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        steps = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

    return ColumnTransformer(
        transformers=[
            ("num", steps, features),
        ],
        remainder="drop",
    )


def evaluate_model_on_split(model, split_df, features, model_name):
    X = split_df[features]
    residual_pred = model.predict(X)

    corrected_pred = (
        split_df["pred_yoy_adjusted_seasonal"].astype(float).values
        + residual_pred
    )

    return corrected_pred, residual_pred


def main():
    print("=" * 80)
    print("SCRIPT 19: YOY SEASONAL + MACRO RESIDUAL CORRECTION")
    print("=" * 80)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file:\n{INPUT_PATH}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading data:")
    print(INPUT_PATH)

    df = pd.read_csv(INPUT_PATH, low_memory=False)
    df = add_features(df)

    print(f"\nRows loaded: {len(df):,}")
    print(f"Columns after feature engineering: {len(df.columns):,}")

    train = df[df["model_split"] == "train_2021_2023"].copy()
    val = df[df["model_split"] == "validation_2024"].copy()
    test = df[df["model_split"] == "test_2025"].copy()

    print("\nSplit counts:")
    print(f"  Train:      {len(train):,}")
    print(f"  Validation: {len(val):,}")
    print(f"  Test:       {len(test):,}")

    required_cols = [
        "target_revenue_next_qtr",
        "pred_yoy_adjusted_seasonal",
        "yoy_residual",
        "cpi",
        "fed_funds_rate",
        "unemployment_rate",
        "cpi_qoq_pct_change",
        "fed_funds_rate_qoq_change",
        "unemployment_rate_qoq_change",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError("Missing required columns: " + ", ".join(missing_cols))

    # ------------------------------------------------------------
    # Feature sets
    # ------------------------------------------------------------
    macro_features = [
        "cpi",
        "fed_funds_rate",
        "unemployment_rate",
        "cpi_qoq_change",
        "cpi_qoq_pct_change",
        "fed_funds_rate_qoq_change",
        "unemployment_rate_qoq_change",
    ]

    macro_plus_controls_features = [
        # macro
        "cpi",
        "fed_funds_rate",
        "unemployment_rate",
        "cpi_qoq_change",
        "cpi_qoq_pct_change",
        "fed_funds_rate_qoq_change",
        "unemployment_rate_qoq_change",

        # current firm revenue context
        "revenue_usd",
        "signed_log_revenue",
        "revenue_lag_1",
        "revenue_lag_2",
        "revenue_lag_3",
        "revenue_lag_4",
        "revenue_qoq_change",
        "revenue_yoy_change",
        "revenue_growth_lag_1",
        "revenue_growth_lag_4",
        "signed_log_qoq_change",
        "signed_log_yoy_change",
        "current_to_lag_mean_ratio",
        "revenue_lag_mean_1_4",
        "revenue_lag_median_1_4",
        "revenue_roll4_mean_current",
        "revenue_roll4_median_current",

        # calendar / firm descriptors
        "calendar_quarter",
        "quarter_index",
        "is_derived_q4",
        "firm_obs_total",
        "sic_numeric",
        "sic_2digit",
    ]

    macro_features = [col for col in macro_features if col in df.columns]
    macro_plus_controls_features = [
        col for col in macro_plus_controls_features if col in df.columns
    ]

    print("\nFeature sets:")
    print(f"  Macro-only features: {len(macro_features)}")
    print(f"  Macro + controls features: {len(macro_plus_controls_features)}")

    # ------------------------------------------------------------
    # Training data
    # ------------------------------------------------------------
    y_train_residual = train["yoy_residual"].astype(float)

    # ------------------------------------------------------------
    # Candidate residual models
    # ------------------------------------------------------------
    models = {}

    models["yoy_plus_linear_macro_residual"] = {
        "features": macro_features,
        "pipeline": Pipeline(
            steps=[
                ("preprocess", make_preprocessor(macro_features, scaled=True)),
                ("model", LinearRegression()),
            ]
        ),
        "model_group": "yoy_macro_residual",
    }

    models["yoy_plus_ridge_macro_residual"] = {
        "features": macro_features,
        "pipeline": Pipeline(
            steps=[
                ("preprocess", make_preprocessor(macro_features, scaled=True)),
                ("model", Ridge(alpha=10.0, random_state=42)),
            ]
        ),
        "model_group": "yoy_macro_residual",
    }

    models["yoy_plus_ridge_macro_controls_residual"] = {
        "features": macro_plus_controls_features,
        "pipeline": Pipeline(
            steps=[
                ("preprocess", make_preprocessor(macro_plus_controls_features, scaled=True)),
                ("model", Ridge(alpha=10.0, random_state=42)),
            ]
        ),
        "model_group": "yoy_macro_controls_residual",
    }

    models["yoy_plus_histgb_macro_controls_residual"] = {
        "features": macro_plus_controls_features,
        "pipeline": Pipeline(
            steps=[
                ("preprocess", make_preprocessor(macro_plus_controls_features, scaled=False)),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=300,
                        learning_rate=0.03,
                        max_leaf_nodes=31,
                        l2_regularization=0.1,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "model_group": "yoy_macro_controls_residual",
    }

    models["yoy_plus_random_forest_macro_controls_residual"] = {
        "features": macro_plus_controls_features,
        "pipeline": Pipeline(
            steps=[
                ("preprocess", make_preprocessor(macro_plus_controls_features, scaled=False)),
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
        "model_group": "yoy_macro_controls_residual",
    }

    try:
        from xgboost import XGBRegressor

        models["yoy_plus_xgboost_macro_controls_residual"] = {
            "features": macro_plus_controls_features,
            "pipeline": Pipeline(
                steps=[
                    ("preprocess", make_preprocessor(macro_plus_controls_features, scaled=False)),
                    (
                        "model",
                        XGBRegressor(
                            n_estimators=400,
                            learning_rate=0.03,
                            max_depth=4,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            objective="reg:squarederror",
                            random_state=42,
                            n_jobs=-1,
                            verbosity=0,
                        ),
                    ),
                ]
            ),
            "model_group": "yoy_macro_controls_residual",
        }

        print("\nXGBoost available: added residual XGBoost model.")

    except Exception:
        print("\nXGBoost not available: skipping residual XGBoost model.")

    # ------------------------------------------------------------
    # Fit and evaluate
    # ------------------------------------------------------------
    metrics_rows = []
    prediction_rows = []

    # Base YOY baseline on validation and test
    for split_name, split_df in [
        ("validation_2024", val),
        ("test_2025", test),
    ]:
        y_true = split_df["target_revenue_next_qtr"].astype(float).values
        y_pred = split_df["pred_yoy_adjusted_seasonal"].astype(float).values

        metrics_rows.append(
            make_metrics(
                y_true=y_true,
                y_pred=y_pred,
                model_name="yoy_adjusted_seasonal",
                model_group="baseline",
                split_name=split_name,
            )
        )

    fitted_models = {}

    for model_name, config in models.items():
        print("\n" + "-" * 80)
        print(f"Training residual model: {model_name}")
        print("-" * 80)

        features = config["features"]
        pipeline = config["pipeline"]

        X_train = train[features]

        pipeline.fit(X_train, y_train_residual)

        fitted_models[model_name] = {
            "pipeline": pipeline,
            "features": features,
            "model_group": config["model_group"],
        }

        for split_name, split_df in [
            ("validation_2024", val),
            ("test_2025", test),
        ]:
            corrected_pred, residual_pred = evaluate_model_on_split(
                model=pipeline,
                split_df=split_df,
                features=features,
                model_name=model_name,
            )

            y_true = split_df["target_revenue_next_qtr"].astype(float).values

            metrics = make_metrics(
                y_true=y_true,
                y_pred=corrected_pred,
                model_name=model_name,
                model_group=config["model_group"],
                split_name=split_name,
            )

            metrics_rows.append(metrics)

            print(
                f"{split_name}: MAE={metrics['mae']:,.2f}, "
                f"RMSE={metrics['rmse']:,.2f}, "
                f"sMAPE={metrics['smape']:.4f}"
            )

            temp_pred = split_df[
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
                    "revenue_usd",
                    "pred_yoy_adjusted_seasonal",
                    "yoy_residual",
                    "cpi",
                    "fed_funds_rate",
                    "unemployment_rate",
                ]
            ].copy()

            temp_pred["model"] = model_name
            temp_pred["split"] = split_name
            temp_pred["predicted_residual"] = residual_pred
            temp_pred["corrected_prediction"] = corrected_pred
            temp_pred["absolute_error"] = (
                temp_pred["target_revenue_next_qtr"]
                - temp_pred["corrected_prediction"]
            ).abs()

            prediction_rows.append(temp_pred)

    metrics_df = pd.DataFrame(metrics_rows)

    metrics_df = metrics_df.sort_values(
        ["split", "mae", "rmse"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    predictions_df = pd.concat(prediction_rows, ignore_index=True)

    # ------------------------------------------------------------
    # Select best residual model by validation MAE
    # ------------------------------------------------------------
    validation_metrics = metrics_df[
        metrics_df["split"] == "validation_2024"
    ].copy()

    yoy_val = validation_metrics[
        validation_metrics["model"] == "yoy_adjusted_seasonal"
    ].iloc[0]

    residual_validation_metrics = validation_metrics[
        validation_metrics["model"] != "yoy_adjusted_seasonal"
    ].copy()

    best_residual_row = residual_validation_metrics.iloc[0]
    best_residual_model_name = best_residual_row["model"]

    yoy_val_mae = yoy_val["mae"]
    best_residual_val_mae = best_residual_row["mae"]

    val_improvement = (yoy_val_mae - best_residual_val_mae) / yoy_val_mae

    # Test comparison for selected model
    test_metrics = metrics_df[metrics_df["split"] == "test_2025"].copy()

    yoy_test = test_metrics[
        test_metrics["model"] == "yoy_adjusted_seasonal"
    ].iloc[0]

    selected_test = test_metrics[
        test_metrics["model"] == best_residual_model_name
    ].iloc[0]

    test_improvement = (
        yoy_test["mae"] - selected_test["mae"]
    ) / yoy_test["mae"]

    # Save best residual model bundle
    best_bundle = {
        "model_name": best_residual_model_name,
        "model": fitted_models[best_residual_model_name]["pipeline"],
        "features": fitted_models[best_residual_model_name]["features"],
        "model_group": fitted_models[best_residual_model_name]["model_group"],
        "base_forecast": "yoy_adjusted_seasonal",
        "residual_target": "actual_next_revenue - yoy_adjusted_seasonal_prediction",
        "final_prediction": "yoy_adjusted_seasonal_prediction + predicted_residual",
        "selected_by": "lowest validation MAE among residual correction models",
        "validation_improvement_vs_yoy": val_improvement,
        "test_improvement_vs_yoy": test_improvement,
    }

    joblib.dump(best_bundle, BEST_MODEL_PATH)

    # ------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------
    metrics_df.to_csv(METRICS_PATH, index=False)
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)

    # ------------------------------------------------------------
    # Plot validation and test MAE
    # ------------------------------------------------------------
    plot_df = metrics_df.copy()

    plot_df["model_short"] = plot_df["model"].replace(
        {
            "yoy_adjusted_seasonal": "YOY seasonal",
            "yoy_plus_linear_macro_residual": "YOY + linear macro",
            "yoy_plus_ridge_macro_residual": "YOY + ridge macro",
            "yoy_plus_ridge_macro_controls_residual": "YOY + ridge macro+controls",
            "yoy_plus_histgb_macro_controls_residual": "YOY + HistGB residual",
            "yoy_plus_random_forest_macro_controls_residual": "YOY + RF residual",
            "yoy_plus_xgboost_macro_controls_residual": "YOY + XGB residual",
        }
    )

    pivot = plot_df.pivot(index="model_short", columns="split", values="mae")
    pivot = pivot.sort_values("validation_2024")

    ax = pivot.plot(kind="bar", figsize=(14, 6))
    ax.set_ylabel("MAE")
    ax.set_title("YOY Seasonal vs YOY + Macro Residual Correction")
    ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=300)
    plt.close()

    # ------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("YOY + MACRO RESIDUAL CORRECTION METRICS")
    print("=" * 80)

    print(metrics_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("DIRECT ANSWER")
    print("=" * 80)

    print("\nValidation comparison:")
    print(f"  YOY seasonal MAE:             {yoy_val_mae:,.2f}")
    print(f"  Best residual correction:     {best_residual_model_name}")
    print(f"  Best residual validation MAE: {best_residual_val_mae:,.2f}")
    print(f"  Improvement vs YOY:           {val_improvement:.2%}")

    print("\nTest comparison for selected residual model:")
    print(f"  YOY seasonal test MAE:        {yoy_test['mae']:,.2f}")
    print(f"  Selected residual test MAE:   {selected_test['mae']:,.2f}")
    print(f"  Test improvement vs YOY:      {test_improvement:.2%}")

    if val_improvement > 0:
        print("\nAnswer:")
        print("  On validation, macro/residual correction improves the YOY seasonal forecast.")
    else:
        print("\nAnswer:")
        print("  On validation, macro/residual correction does NOT improve the YOY seasonal forecast.")

    if test_improvement > 0:
        print("  On 2025 test, the selected residual correction also improves the YOY forecast.")
    else:
        print("  On 2025 test, the selected residual correction does NOT improve the YOY forecast.")

    print("\nImportant interpretation:")
    print("  If improvement is small or negative, it means the YOY seasonal forecast already")
    print("  captures most predictable structure, and macro variables do not explain much")
    print("  of the remaining forecast error.")

    print("\n" + "=" * 80)
    print("OUTPUTS SAVED")
    print("=" * 80)
    print(f"Metrics:      {METRICS_PATH}")
    print(f"Predictions:  {PREDICTIONS_PATH}")
    print(f"Plot:         {PLOT_PATH}")
    print(f"Best model:   {BEST_MODEL_PATH}")

    print("\nScript 19 completed successfully.")


if __name__ == "__main__":
    main()