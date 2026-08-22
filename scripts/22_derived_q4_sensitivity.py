"""
Derived fiscal-Q4 sensitivity analysis
================================================

Compares final test performance between:

1. Directly reported current-quarter SEC revenue
2. Current-quarter fiscal-Q4 revenue derived as:
   annual revenue - Q1 - Q2 - Q3

Run from the project root:

    python scripts/22_derived_q4_sensitivity.py
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "report_tables"
    / "log_growth_random_forest_predictions_test_2025.csv"
)

MODEL_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_ready_sec_macro_2021_2025.csv"
)

OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "report_tables"

SENSITIVITY_OUTPUT = (
    OUTPUT_DIRECTORY / "derived_q4_sensitivity_test_2025.csv"
)

COUNTS_OUTPUT = (
    OUTPUT_DIRECTORY / "derived_q4_counts_by_split.csv"
)


def parse_boolean(series):
    """Convert Boolean, 0/1, or text Boolean values to True/False."""
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )


def calculate_metrics(group_name, data, total_test_rows):
    """Calculate final-model and seasonal-baseline metrics."""
    actual = pd.to_numeric(
        data["target_revenue_next_qtr"],
        errors="coerce",
    ).to_numpy(dtype=float)

    forecast = pd.to_numeric(
        data["pred_improved"],
        errors="coerce",
    ).to_numpy(dtype=float)

    seasonal = pd.to_numeric(
        data["seasonal_naive_forecast_next_qtr"],
        errors="coerce",
    ).to_numpy(dtype=float)

    if (
        np.isnan(actual).any()
        or np.isnan(forecast).any()
        or np.isnan(seasonal).any()
    ):
        raise ValueError(
            f"Missing numeric values found in group: {group_name}"
        )

    absolute_error = np.abs(forecast - actual)
    squared_error = np.square(forecast - actual)
    seasonal_absolute_error = np.abs(seasonal - actual)

    positive_actual = actual > 0
    positive_actual_values = actual[positive_actual]
    positive_forecasts = forecast[positive_actual]

    absolute_percentage_error = (
        np.abs(positive_forecasts - positive_actual_values)
        / positive_actual_values
    )

    smape_denominator = (
        np.abs(positive_actual_values)
        + np.abs(positive_forecasts)
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        individual_smape = np.where(
            smape_denominator == 0,
            0.0,
            (
                2.0
                * np.abs(
                    positive_forecasts - positive_actual_values
                )
                / smape_denominator
            ),
        )

    final_mae = float(np.mean(absolute_error))
    seasonal_mae = float(np.mean(seasonal_absolute_error))

    mae_reduction = (
        (seasonal_mae - final_mae)
        / seasonal_mae
        * 100
    )

    fallback_count = int(
        parse_boolean(data["fallback_used"]).sum()
    )

    return {
        "sensitivity_group": group_name,
        "test_observations": len(data),
        "test_firms": data["cik"].nunique(),
        "share_of_test_pct": len(data) / total_test_rows * 100,
        "positive_actual_observations": int(
            positive_actual.sum()
        ),
        "mae_usd_millions": final_mae / 1e6,
        "rmse_usd_millions": (
            float(np.sqrt(np.mean(squared_error))) / 1e6
        ),
        "median_absolute_percentage_error_pct": (
            float(np.median(absolute_percentage_error)) * 100
        ),
        "smape_positive_actual": float(
            np.mean(individual_smape)
        ),
        "within_5_pct": float(
            np.mean(absolute_percentage_error <= 0.05) * 100
        ),
        "within_10_pct": float(
            np.mean(absolute_percentage_error <= 0.10) * 100
        ),
        "within_20_pct": float(
            np.mean(absolute_percentage_error <= 0.20) * 100
        ),
        "seasonal_baseline_mae_usd_millions": seasonal_mae / 1e6,
        "mae_reduction_vs_seasonal_pct": mae_reduction,
        "fallback_observations": fallback_count,
        "negative_forecasts": int((forecast < 0).sum()),
    }


def build_split_counts(model_data):
    """Count derived fiscal-Q4 feature rows in each split."""
    model_data = model_data.copy()

    model_data["derived_q4_boolean"] = parse_boolean(
        model_data["is_derived_q4"]
    )

    split_order = [
        "train_2021_2023",
        "validation_2024",
        "test_2025",
    ]

    split_counts = (
        model_data.groupby("model_split", as_index=False)
        .agg(
            rows=("cik", "size"),
            unique_ciks=("cik", "nunique"),
            derived_q4_rows=("derived_q4_boolean", "sum"),
        )
    )

    split_counts["directly_reported_rows"] = (
        split_counts["rows"]
        - split_counts["derived_q4_rows"]
    )

    split_counts["derived_q4_share_pct"] = (
        split_counts["derived_q4_rows"]
        / split_counts["rows"]
        * 100
    )

    split_counts["sort_order"] = (
        split_counts["model_split"]
        .map(
            {
                split_name: position
                for position, split_name
                in enumerate(split_order)
            }
        )
    )

    split_counts = (
        split_counts.sort_values("sort_order")
        .drop(columns="sort_order")
        .reset_index(drop=True)
    )

    total_row = pd.DataFrame(
        [
            {
                "model_split": "total",
                "rows": len(model_data),
                "unique_ciks": model_data["cik"].nunique(),
                "derived_q4_rows": int(
                    model_data["derived_q4_boolean"].sum()
                ),
                "directly_reported_rows": int(
                    (~model_data["derived_q4_boolean"]).sum()
                ),
                "derived_q4_share_pct": float(
                    model_data["derived_q4_boolean"].mean()
                    * 100
                ),
            }
        ]
    )

    return pd.concat(
        [split_counts, total_row],
        ignore_index=True,
    )


def main():
    print("=" * 80)
    print("STEP 9: DERIVED FISCAL-Q4 SENSITIVITY ANALYSIS")
    print("=" * 80)

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n{PREDICTIONS_PATH}"
        )

    if not MODEL_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Modeling dataset not found:\n{MODEL_DATA_PATH}"
        )

    predictions = pd.read_csv(PREDICTIONS_PATH)
    model_data = pd.read_csv(MODEL_DATA_PATH)

    required_prediction_columns = {
        "cik",
        "is_derived_q4",
        "target_revenue_next_qtr",
        "seasonal_naive_forecast_next_qtr",
        "pred_improved",
        "fallback_used",
        "forecast_timing_valid",
    }

    missing_prediction_columns = (
        required_prediction_columns
        - set(predictions.columns)
    )

    if missing_prediction_columns:
        raise ValueError(
            "Prediction file is missing columns: "
            f"{sorted(missing_prediction_columns)}"
        )

    if len(predictions) != 12924:
        raise ValueError(
            "Expected 12,924 final test predictions, "
            f"but found {len(predictions):,}."
        )

    timing_valid = parse_boolean(
        predictions["forecast_timing_valid"]
    )

    if not timing_valid.all():
        raise ValueError(
            "Timing-invalid observations remain in predictions."
        )

    if len(model_data) != 48802:
        raise ValueError(
            "Expected 48,802 modeling rows, "
            f"but found {len(model_data):,}."
        )

    required_model_columns = {
        "cik",
        "model_split",
        "is_derived_q4",
    }

    missing_model_columns = (
        required_model_columns - set(model_data.columns)
    )

    if missing_model_columns:
        raise ValueError(
            "Modeling dataset is missing columns: "
            f"{sorted(missing_model_columns)}"
        )

    predictions["derived_q4_boolean"] = parse_boolean(
        predictions["is_derived_q4"]
    )

    sensitivity_groups = [
        (
            "all_test_observations",
            predictions,
        ),
        (
            "direct_sec_current_quarter",
            predictions.loc[
                ~predictions["derived_q4_boolean"]
            ],
        ),
        (
            "derived_fiscal_q4_current_quarter",
            predictions.loc[
                predictions["derived_q4_boolean"]
            ],
        ),
    ]

    sensitivity_results = pd.DataFrame(
        [
            calculate_metrics(
                group_name,
                group_data,
                len(predictions),
            )
            for group_name, group_data
            in sensitivity_groups
        ]
    )

    split_counts = build_split_counts(model_data)

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    sensitivity_results.to_csv(
        SENSITIVITY_OUTPUT,
        index=False,
    )

    split_counts.to_csv(
        COUNTS_OUTPUT,
        index=False,
    )

    pd.set_option(
        "display.float_format",
        lambda value: f"{value:,.3f}",
    )

    print("\nDerived-Q4 feature counts by split:\n")
    print(split_counts.to_string(index=False))

    print("\nTest-set sensitivity metrics:\n")
    print(sensitivity_results.to_string(index=False))

    print("\n" + "=" * 80)
    print("OUTPUTS SAVED")
    print("=" * 80)
    print(f"Sensitivity metrics: {SENSITIVITY_OUTPUT}")
    print(f"Split counts:        {COUNTS_OUTPUT}")
    print("\nStep 9 analysis completed successfully.")


if __name__ == "__main__":
    main()
