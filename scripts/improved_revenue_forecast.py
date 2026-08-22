"""
Final timing-valid log-growth Random Forest revenue forecast
============================================================

This is the official final-model script for the project.

Methodology
-----------
* Input sample: 48,802 timing-valid firm-quarter observations.
* Train: target quarters through 2023.
* Validation: 2024 target quarters for hyperparameter selection.
* Test: 2025 target quarters for final out-of-sample reporting.
* Target: log(next-quarter revenue / current-quarter revenue).
* Forecast routing uses current revenue only. It never uses the future target.
* Rows with non-positive current revenue use the seasonal-naive fallback.
* Final forecasts are floored at zero.
* Percentage metrics use only observations where actual revenue is positive.

Outputs
-------
outputs/report_tables/log_growth_random_forest_metrics.csv
outputs/report_tables/log_growth_random_forest_predictions_test_2025.csv
models/log_growth_random_forest_final.pkl

Run from the project root:
    python scripts/improved_revenue_forecast.py

An optional project-root argument is also supported:
    python scripts/improved_revenue_forecast.py .
"""

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


RANDOM_SEED = 0
MODEL_NAME = "log_growth_random_forest"
MODEL_VERSION = "final_v2_timing_valid"

EXPECTED_TOTAL_ROWS = 48_802
EXPECTED_SPLIT_COUNTS = {
    "train_2021_2023": 22_559,
    "validation_2024": 13_319,
    "test_2025": 12_924,
}

NUMERIC_FEATURES = [
    "signed_log_revenue",
    "signed_log_revenue_lag_1",
    "signed_log_revenue_lag_4",
    "signed_log_revenue_change_lag_1",
    "signed_log_revenue_change_lag_4",
    "revenue_growth_lag_1",
    "revenue_growth_lag_4",
    "cpi",
    "fed_funds_rate",
    "unemployment_rate",
    "cpi_qoq_pct_change",
    "fed_funds_rate_qoq_change",
    "unemployment_rate_qoq_change",
]

SECTOR_CATEGORIES = [
    "agriculture",
    "construction",
    "finance",
    "manufacturing",
    "mining",
    "retail",
    "services",
    "utilities_transport",
    "wholesale",
]

CANDIDATE_PARAMETERS = [
    {
        "n_estimators": 300,
        "min_samples_leaf": 5,
    },
    {
        "n_estimators": 400,
        "min_samples_leaf": 10,
    },
    {
        "n_estimators": 500,
        "min_samples_leaf": 20,
    },
]


def atomic_to_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a CSV completely before replacing the existing destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")

    try:
        df.to_csv(temporary_path, index=False)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def atomic_joblib_dump(obj, path: Path) -> None:
    """Write a joblib artifact completely before replacing its destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")

    try:
        joblib.dump(obj, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def smape_positive_actual(actual, forecast) -> float:
    """Calculate sMAPE only where actual revenue is positive."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    positive_actual = actual > 0

    if positive_actual.sum() == 0:
        return np.nan

    actual_positive = actual[positive_actual]
    forecast_positive = forecast[positive_actual]

    denominator = (
        np.abs(actual_positive)
        + np.abs(forecast_positive)
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        individual_smape = np.where(
            denominator == 0,
            0.0,
            2.0
            * np.abs(forecast_positive - actual_positive)
            / denominator,
        )

    return float(np.mean(individual_smape))


def metrics_block(
    model_name,
    actual,
    forecast,
    firm_ids,
    fallback_observations=np.nan,
) -> dict:
    """Calculate the official metrics for one forecasting method."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    firm_ids = pd.Series(firm_ids, dtype="string")

    if len(actual) != len(forecast):
        raise ValueError(
            f"Actual and forecast lengths differ for {model_name}."
        )

    absolute_error = np.abs(forecast - actual)
    squared_error = (forecast - actual) ** 2
    positive_actual = actual > 0

    positive_actual_count = int(positive_actual.sum())
    nonpositive_actual_count = int((~positive_actual).sum())

    if positive_actual_count == 0:
        median_ape = np.nan
        within_5 = np.nan
        within_10 = np.nan
        within_20 = np.nan
    else:
        absolute_percentage_error = (
            absolute_error[positive_actual]
            / actual[positive_actual]
        )

        median_ape = float(
            np.median(absolute_percentage_error)
        )
        within_5 = float(
            np.mean(absolute_percentage_error <= 0.05)
        )
        within_10 = float(
            np.mean(absolute_percentage_error <= 0.10)
        )
        within_20 = float(
            np.mean(absolute_percentage_error <= 0.20)
        )

    return {
        "model": model_name,
        "test_observations": int(len(actual)),
        "test_firms": int(firm_ids.nunique()),
        "positive_actual_observations": positive_actual_count,
        "nonpositive_actual_observations": nonpositive_actual_count,
        "mae_usd": float(np.mean(absolute_error)),
        "mae_usd_millions": float(
            np.mean(absolute_error) / 1_000_000
        ),
        "rmse_usd": float(np.sqrt(np.mean(squared_error))),
        "rmse_usd_millions": float(
            np.sqrt(np.mean(squared_error)) / 1_000_000
        ),
        "median_absolute_percentage_error_pct": (
            median_ape * 100
        ),
        "smape_positive_actual": smape_positive_actual(
            actual,
            forecast,
        ),
        "within_5_pct": within_5 * 100,
        "within_10_pct": within_10 * 100,
        "within_20_pct": within_20 * 100,
        "negative_forecasts": int((forecast < 0).sum()),
        "zero_forecasts": int((forecast == 0).sum()),
        "fallback_observations": fallback_observations,
        "percentage_metric_denominator": (
            "actual_revenue_positive_only"
        ),
    }


def sector_from_sic(sic) -> str:
    """Convert a four-digit SIC code into a fixed broad sector."""
    try:
        sic_number = int(float(sic))
    except (ValueError, TypeError):
        return "other"

    sector_buckets = [
        (100, 1000, "agriculture"),
        (1000, 1500, "mining"),
        (1500, 1800, "construction"),
        (2000, 4000, "manufacturing"),
        (4000, 5000, "utilities_transport"),
        (5000, 5200, "wholesale"),
        (5200, 6000, "retail"),
        (6000, 6800, "finance"),
        (7000, 9000, "services"),
    ]

    for lower_bound, upper_bound, sector_name in sector_buckets:
        if lower_bound <= sic_number < upper_bound:
            return sector_name

    return "other"


def validate_input_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Validate row counts, splits, identifiers, and forecast timing."""
    required_columns = [
        "cik",
        "company_name",
        "sic",
        "period_date",
        "calendar_period",
        "filed_date",
        "target_period_date",
        "target_calendar_period",
        "target_revenue_next_qtr",
        "model_split",
        "revenue_usd",
        "naive_forecast_next_qtr",
        "seasonal_naive_forecast_next_qtr",
        *NUMERIC_FEATURES,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Modeling dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    if len(df) != EXPECTED_TOTAL_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_ROWS:,} timing-valid rows, "
            f"but found {len(df):,}."
        )

    actual_split_counts = (
        df["model_split"]
        .value_counts()
        .to_dict()
    )

    if actual_split_counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(
            "Unexpected split counts.\n"
            f"Expected: {EXPECTED_SPLIT_COUNTS}\n"
            f"Actual:   {actual_split_counts}"
        )

    duplicate_count = df.duplicated(
        ["cik", "period_date"]
    ).sum()

    if duplicate_count > 0:
        raise ValueError(
            f"Found {duplicate_count:,} duplicate CIK-period rows."
        )

    df = df.copy()
    df["filed_date"] = pd.to_datetime(
        df["filed_date"],
        errors="coerce",
    )
    df["target_period_date"] = pd.to_datetime(
        df["target_period_date"],
        errors="coerce",
    )

    if (
        df["filed_date"].isna().any()
        or df["target_period_date"].isna().any()
    ):
        raise ValueError(
            "Missing or invalid filing/target dates found."
        )

    computed_timing_valid = (
        df["filed_date"]
        < df["target_period_date"]
    )

    if not computed_timing_valid.all():
        invalid_count = int((~computed_timing_valid).sum())
        raise ValueError(
            f"Input contains {invalid_count:,} timing-invalid rows."
        )

    if "forecast_timing_valid" in df.columns:
        stored_timing_valid = (
            df["forecast_timing_valid"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("true")
        )

        if not stored_timing_valid.equals(computed_timing_valid):
            raise ValueError(
                "Stored forecast_timing_valid values disagree with dates."
            )

    df["forecast_timing_valid"] = computed_timing_valid

    missing_target = df["target_revenue_next_qtr"].isna().sum()

    if missing_target > 0:
        raise ValueError(
            f"Found {missing_target:,} missing target values."
        )

    return df


def build_features(df: pd.DataFrame):
    """Build a reproducible numeric and fixed-sector feature matrix."""
    df = df.copy()
    df["sector"] = df["sic"].apply(sector_from_sic)
    df["sector"] = pd.Categorical(
        df["sector"],
        categories=SECTOR_CATEGORIES,
    )

    numeric_matrix = df[NUMERIC_FEATURES].apply(
        pd.to_numeric,
        errors="coerce",
    )

    sector_matrix = pd.get_dummies(
        df["sector"],
        prefix="sector",
        dtype=float,
    )

    expected_sector_columns = [
        f"sector_{sector}"
        for sector in SECTOR_CATEGORIES
    ]

    sector_matrix = sector_matrix.reindex(
        columns=expected_sector_columns,
        fill_value=0.0,
    )

    feature_matrix = pd.concat(
        [numeric_matrix, sector_matrix],
        axis=1,
    ).astype(float)

    return feature_matrix, df


def main(project_root: Path, data_path: Path) -> None:
    report_table_dir = project_root / "outputs" / "report_tables"
    model_dir = project_root / "models"

    metrics_path = (
        report_table_dir
        / "log_growth_random_forest_metrics.csv"
    )
    predictions_path = (
        report_table_dir
        / "log_growth_random_forest_predictions_test_2025.csv"
    )
    model_path = (
        model_dir
        / "log_growth_random_forest_final.pkl"
    )

    print("=" * 80)
    print("FINAL LOG-GROWTH RANDOM FOREST")
    print("=" * 80)
    print(f"Project root: {project_root}")
    print(f"Input dataset: {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(
            f"Model-ready dataset not found:\n{data_path}"
        )

    df = pd.read_csv(
        data_path,
        dtype={
            "cik": "string",
            "sic": "string",
            "adsh": "string",
        },
        low_memory=False,
    )

    df = validate_input_dataset(df)

    print("\nInput validation passed:")
    print(f"  Total rows: {len(df):,}")
    print(
        "  Train / validation / test: "
        f"{EXPECTED_SPLIT_COUNTS['train_2021_2023']:,} / "
        f"{EXPECTED_SPLIT_COUNTS['validation_2024']:,} / "
        f"{EXPECTED_SPLIT_COUNTS['test_2025']:,}"
    )
    print("  Timing-invalid rows: 0")

    # The log-growth target is defined only where both current and future
    # historical revenue are positive. The future target is used only to create
    # historical training labels, never to route validation/test predictions.
    df["pos_pair"] = (
        (df["revenue_usd"] > 0)
        & (df["target_revenue_next_qtr"] > 0)
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        revenue_ratio = (
            df["target_revenue_next_qtr"]
            / df["revenue_usd"]
        )

        df["log_growth"] = np.where(
            df["pos_pair"],
            np.log(revenue_ratio.where(revenue_ratio > 0)),
            np.nan,
        )

    feature_matrix, df = build_features(df)

    train_mask = df["model_split"].eq("train_2021_2023")
    validation_mask = df["model_split"].eq("validation_2024")
    test_mask = df["model_split"].eq("test_2025")
    fit_mask = train_mask & df["pos_pair"]

    # All imputation values are learned from eligible training rows only.
    training_medians = (
        feature_matrix.loc[fit_mask]
        .median()
        .fillna(0.0)
    )

    feature_matrix = (
        feature_matrix
        .fillna(training_medians)
        .fillna(0.0)
    )

    best_parameters = None
    best_validation_smape = np.inf

    print("\nHyperparameter selection:")

    for parameters in CANDIDATE_PARAMETERS:
        candidate_model = RandomForestRegressor(
            random_state=RANDOM_SEED,
            n_jobs=-1,
            **parameters,
        )

        candidate_model.fit(
            feature_matrix.loc[fit_mask],
            df.loc[fit_mask, "log_growth"],
        )

        validation = df.loc[validation_mask].copy()
        validation_features = feature_matrix.loc[validation_mask]

        predicted_log_growth = candidate_model.predict(
            validation_features
        )

        model_forecast = (
            validation["revenue_usd"].to_numpy()
            * np.exp(predicted_log_growth)
        )

        fallback_forecast = validation[
            "seasonal_naive_forecast_next_qtr"
        ].to_numpy()

        current_revenue_positive = (
            validation["revenue_usd"].to_numpy() > 0
        )

        validation_forecast = np.where(
            current_revenue_positive,
            model_forecast,
            fallback_forecast,
        )

        validation_forecast = np.clip(
            validation_forecast,
            0,
            None,
        )

        validation_smape = smape_positive_actual(
            validation["target_revenue_next_qtr"],
            validation_forecast,
        )

        print(
            f"  validation sMAPE {validation_smape:.4f} "
            "(actual > 0)  "
            f"params={parameters}"
        )

        if validation_smape < best_validation_smape:
            best_parameters = parameters
            best_validation_smape = validation_smape

    print(
        f"Selected params: {best_parameters} "
        f"(validation sMAPE {best_validation_smape:.4f}, "
        "actual > 0)"
    )

    final_model = RandomForestRegressor(
        random_state=RANDOM_SEED,
        n_jobs=-1,
        **best_parameters,
    )

    final_model.fit(
        feature_matrix.loc[fit_mask],
        df.loc[fit_mask, "log_growth"],
    )

    test = df.loc[test_mask].copy()
    test_features = feature_matrix.loc[test_mask]

    predicted_log_growth_test = final_model.predict(
        test_features
    )

    model_forecast_test = (
        test["revenue_usd"].to_numpy()
        * np.exp(predicted_log_growth_test)
    )

    fallback_forecast_test = test[
        "seasonal_naive_forecast_next_qtr"
    ].to_numpy()

    # This routing uses current revenue only. It does not inspect the unknown
    # next-quarter target and therefore does not create target leakage.
    current_revenue_positive_test = (
        test["revenue_usd"].to_numpy() > 0
    )
    fallback_used_test = ~current_revenue_positive_test

    improved_forecast = np.where(
        current_revenue_positive_test,
        model_forecast_test,
        fallback_forecast_test,
    )

    improved_forecast = np.clip(
        improved_forecast,
        0,
        None,
    )

    test["model_name"] = MODEL_NAME
    test["model_version"] = MODEL_VERSION
    test["predicted_log_growth"] = predicted_log_growth_test
    test["model_forecast_before_routing"] = model_forecast_test
    test["fallback_forecast"] = fallback_forecast_test
    test["fallback_used"] = fallback_used_test
    test["pred_improved"] = improved_forecast
    test["actual_positive_for_percentage_metrics"] = (
        test["target_revenue_next_qtr"] > 0
    )
    test["forecast_error"] = (
        test["pred_improved"]
        - test["target_revenue_next_qtr"]
    )
    test["absolute_error"] = test["forecast_error"].abs()

    test["absolute_percentage_error"] = np.where(
        test["actual_positive_for_percentage_metrics"],
        test["absolute_error"]
        / test["target_revenue_next_qtr"],
        np.nan,
    )

    test["within_5_pct"] = np.where(
        test["actual_positive_for_percentage_metrics"],
        test["absolute_percentage_error"] <= 0.05,
        pd.NA,
    )
    test["within_10_pct"] = np.where(
        test["actual_positive_for_percentage_metrics"],
        test["absolute_percentage_error"] <= 0.10,
        pd.NA,
    )
    test["within_20_pct"] = np.where(
        test["actual_positive_for_percentage_metrics"],
        test["absolute_percentage_error"] <= 0.20,
        pd.NA,
    )

    actual_revenue = test["target_revenue_next_qtr"].to_numpy()
    firm_ids = test["cik"]

    metrics_df = pd.DataFrame(
        [
            metrics_block(
                "seasonal_naive_baseline",
                actual_revenue,
                test["seasonal_naive_forecast_next_qtr"],
                firm_ids,
            ),
            metrics_block(
                "last_quarter_naive_baseline",
                actual_revenue,
                test["naive_forecast_next_qtr"],
                firm_ids,
            ),
            metrics_block(
                MODEL_NAME,
                actual_revenue,
                test["pred_improved"],
                firm_ids,
                fallback_observations=int(
                    test["fallback_used"].sum()
                ),
            ),
        ]
    )

    baseline_mae = metrics_df.loc[
        metrics_df["model"].eq("seasonal_naive_baseline"),
        "mae_usd",
    ].iloc[0]
    final_mae = metrics_df.loc[
        metrics_df["model"].eq(MODEL_NAME),
        "mae_usd",
    ].iloc[0]

    mae_reduction_pct = (
        (baseline_mae - final_mae)
        / baseline_mae
        * 100
    )

    metrics_df["mae_reduction_vs_seasonal_pct"] = np.where(
        metrics_df["model"].eq(MODEL_NAME),
        mae_reduction_pct,
        np.nan,
    )
    metrics_df["selected_hyperparameters"] = np.where(
        metrics_df["model"].eq(MODEL_NAME),
        str(best_parameters),
        "",
    )
    metrics_df["random_seed"] = np.where(
        metrics_df["model"].eq(MODEL_NAME),
        RANDOM_SEED,
        np.nan,
    )
    metrics_df["model_version"] = np.where(
        metrics_df["model"].eq(MODEL_NAME),
        MODEL_VERSION,
        "",
    )

    prediction_columns = [
        "cik",
        "company_name",
        "sic",
        "period_date",
        "calendar_period",
        "fy",
        "fp",
        "form",
        "filed_date",
        "adsh",
        "target_period_date",
        "forecast_timing_valid",
        "target_calendar_period",
        "model_split",
        "is_derived_q4",
        "revenue_source",
        "revenue_usd",
        "target_revenue_next_qtr",
        "naive_forecast_next_qtr",
        "seasonal_naive_forecast_next_qtr",
        "predicted_log_growth",
        "model_forecast_before_routing",
        "fallback_forecast",
        "fallback_used",
        "pred_improved",
        "actual_positive_for_percentage_metrics",
        "forecast_error",
        "absolute_error",
        "absolute_percentage_error",
        "within_5_pct",
        "within_10_pct",
        "within_20_pct",
        "model_name",
        "model_version",
    ]

    prediction_columns = [
        column
        for column in prediction_columns
        if column in test.columns
    ]

    prediction_output = test[prediction_columns].copy()

    sector_encoding_columns = [
        column
        for column in feature_matrix.columns
        if column.startswith("sector_")
    ]

    model_bundle = {
        "model": final_model,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "selected_hyperparameters": best_parameters,
        "feature_columns": feature_matrix.columns.tolist(),
        "numeric_features": NUMERIC_FEATURES,
        "sector_categories": SECTOR_CATEGORIES,
        "sector_encoding_columns": sector_encoding_columns,
        "unrecognized_sector_handling": (
            "all sector dummy columns set to zero"
        ),
        "imputation_values": training_medians.to_dict(),
        "random_seed": RANDOM_SEED,
        "training_split": "train_2021_2023",
        "training_target_period": "2021-2023 target quarters",
        "validation_split": "validation_2024",
        "test_split": "test_2025",
        "training_candidate_rows": int(train_mask.sum()),
        "training_rows_used": int(fit_mask.sum()),
        "validation_rows": int(validation_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "test_firms": int(test["cik"].nunique()),
        "forecast_timing_rule": (
            "filed_date < target_period_date"
        ),
        "forecast_routing_rule": (
            "log-growth model when current revenue > 0; "
            "otherwise seasonal-naive fallback"
        ),
        "percentage_metric_denominator": (
            "actual_revenue_positive_only"
        ),
        "validation_selection_metric": (
            "sMAPE on observations with positive actual revenue"
        ),
    }

    atomic_to_csv(metrics_df, metrics_path)
    atomic_to_csv(prediction_output, predictions_path)
    atomic_joblib_dump(model_bundle, model_path)

    display_columns = [
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

    pd.set_option(
        "display.float_format",
        lambda value: f"{value:,.3f}",
    )

    print("\n" + "=" * 80)
    print("OFFICIAL FINAL METRICS — TEST_2025")
    print("=" * 80)
    print(metrics_df[display_columns].to_string(index=False))

    print(
        "\nPercentage-metric denominator: "
        f"{int((actual_revenue > 0).sum()):,} observations "
        "where actual revenue is positive."
    )
    print(
        "sMAPE, median percentage error, and hit rates use "
        "that positive-actual denominator."
    )
    print(
        "Final MAE reduction vs seasonal baseline: "
        f"{mae_reduction_pct:.2f}%"
    )
    print(
        "Fallback observations: "
        f"{int(test['fallback_used'].sum()):,}"
    )
    print(
        "Negative final forecasts: "
        f"{int((test['pred_improved'] < 0).sum()):,}"
    )

    print("\n" + "=" * 80)
    print("OUTPUTS SAVED")
    print("=" * 80)
    print(f"Metrics:     {metrics_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Model:       {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "project_root",
        nargs="?",
        default=None,
        help="Optional project root. Defaults to the script's parent folder.",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Optional model-ready CSV path.",
    )

    arguments = parser.parse_args()

    script_directory = Path(__file__).resolve().parent

    if arguments.project_root is not None:
        resolved_project_root = Path(
            arguments.project_root
        ).expanduser().resolve()
    elif script_directory.name == "scripts":
        resolved_project_root = script_directory.parent
    else:
        resolved_project_root = script_directory

    if arguments.data is not None:
        resolved_data_path = Path(
            arguments.data
        ).expanduser().resolve()
    else:
        resolved_data_path = (
            resolved_project_root
            / "data"
            / "processed"
            / "model_ready_sec_macro_2021_2025.csv"
        )

    main(
        resolved_project_root,
        resolved_data_path,
    )
