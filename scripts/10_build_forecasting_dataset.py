from pathlib import Path
import sys
import numpy as np
import pandas as pd


# ============================================================
# Script 10: Build Forecasting Dataset
# Project: Panel Forecasting of Firm Revenue Using SEC + Macro Data
# Period: 2021 Q1 - 2025 Q4
# ============================================================

MIN_FIRM_OBS = 8

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
    / "sec_revenue_panel_2021_2025_q4_enhanced.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecasting_dataset_2021_2025.csv"
)

BASELINE_SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecasting_dataset_baseline_summary.csv"
)


def signed_log(series: pd.Series) -> pd.Series:
    """
    Signed log transform that works for positive, zero, and negative revenue.

    Formula:
        sign(x) * log1p(abs(x))
    """
    return np.sign(series) * np.log1p(np.abs(series))


def add_calendar_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add clean calendar fields.

    quarter_index is used to make sure lags and targets are truly consecutive.
    Example:
        2021Q1 -> 2021 * 4 + 1
        2021Q2 -> 2021 * 4 + 2
    """
    df = df.copy()

    if "period_date" not in df.columns:
        raise ValueError("Missing required column: period_date")

    df["period_date"] = pd.to_datetime(df["period_date"], errors="coerce")

    if df["period_date"].isna().any():
        bad_dates = df["period_date"].isna().sum()
        raise ValueError(f"period_date has {bad_dates:,} missing or invalid dates.")

    fallback_year = df["period_date"].dt.year
    fallback_quarter = df["period_date"].dt.quarter

    year = pd.Series(np.nan, index=df.index)
    quarter = pd.Series(np.nan, index=df.index)

    if "calendar_period" in df.columns:
        cp = (
            df["calendar_period"]
            .astype("string")
            .str.upper()
            .str.replace(" ", "", regex=False)
        )

        extracted = cp.str.extract(r"(?P<year>\d{4})Q(?P<quarter>[1-4])")

        year = pd.to_numeric(extracted["year"], errors="coerce")
        quarter = pd.to_numeric(extracted["quarter"], errors="coerce")

    year = year.fillna(fallback_year).astype(int)
    quarter = quarter.fillna(fallback_quarter).astype(int)

    df["calendar_year"] = year
    df["calendar_quarter"] = quarter
    df["calendar_period"] = (
        df["calendar_year"].astype(str)
        + "Q"
        + df["calendar_quarter"].astype(str)
    )

    df["quarter_index"] = df["calendar_year"] * 4 + df["calendar_quarter"]

    return df


def quarter_label_from_index(qidx_series: pd.Series):
    """
    Convert quarter_index back to calendar period fields.

    Example:
        8097 -> 2024Q1
    """
    qidx = pd.to_numeric(qidx_series, errors="coerce")

    year = ((qidx - 1) // 4).astype("Int64")
    quarter = (((qidx - 1) % 4) + 1).astype("Int64")

    label = pd.Series(pd.NA, index=qidx_series.index, dtype="object")
    mask = qidx.notna()

    label.loc[mask] = (
        year.loc[mask].astype(str)
        + "Q"
        + quarter.loc[mask].astype(str)
    )

    return label, year, quarter


def safe_growth(current: pd.Series, lagged: pd.Series) -> pd.Series:
    """
    Revenue growth using absolute lagged revenue in denominator.

    Formula:
        (current - lagged) / abs(lagged)

    If lagged revenue is zero, growth is undefined and temporarily set to NaN.
    We later fill it with 0 and keep an undefined-growth flag.
    """
    out = pd.Series(np.nan, index=current.index, dtype="float64")

    mask = lagged.notna() & (lagged != 0)

    out.loc[mask] = (
        current.loc[mask] - lagged.loc[mask]
    ) / lagged.loc[mask].abs()

    return out


def baseline_metrics(
    df: pd.DataFrame,
    forecast_col: str,
    baseline_name: str,
) -> pd.DataFrame:
    """
    Calculate simple baseline forecast metrics by split.
    """
    rows = []

    for split_name, group in df.groupby("model_split", dropna=False):
        valid = group[["target_revenue_next_qtr", forecast_col]].dropna()

        if len(valid) == 0:
            rows.append(
                {
                    "baseline": baseline_name,
                    "model_split": split_name,
                    "rows_used": 0,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "median_absolute_error": np.nan,
                }
            )
            continue

        error = valid["target_revenue_next_qtr"] - valid[forecast_col]

        rows.append(
            {
                "baseline": baseline_name,
                "model_split": split_name,
                "rows_used": len(valid),
                "mae": error.abs().mean(),
                "rmse": np.sqrt((error ** 2).mean()),
                "median_absolute_error": error.abs().median(),
            }
        )

    return pd.DataFrame(rows)


def main():
    print("=" * 80)
    print("SCRIPT 10: BUILD FORECASTING DATASET")
    print("=" * 80)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found:\n{INPUT_PATH}")

    print("\nLoading input file:")
    print(INPUT_PATH)

    df = pd.read_csv(
        INPUT_PATH,
        dtype={
            "cik": "string",
            "company_name": "string",
            "sic": "string",
            "calendar_period": "string",
            "fp": "string",
            "form": "string",
            "adsh": "string",
            "quarter_folder": "string",
            "tag": "string",
            "stmt_values": "string",
            "uom": "string",
            "accepted": "string",
            "revenue_source": "string",
            "derivation_method": "string",
        },
        low_memory=False,
    )

    df.columns = [col.strip().lower() for col in df.columns]

    print(f"\nInitial rows: {len(df):,}")
    print(f"Initial columns: {len(df.columns):,}")

    # ------------------------------------------------------------
    # Required columns
    # ------------------------------------------------------------
    required_cols = ["cik", "period_date", "revenue_usd"]

    missing_required = [col for col in required_cols if col not in df.columns]

    if missing_required:
        print("\nAvailable columns:")
        for i, col in enumerate(df.columns):
            print(f"  {i}: {col}")

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_required)
        )

    df["cik"] = df["cik"].astype("string").str.strip()
    df["period_date"] = pd.to_datetime(df["period_date"], errors="coerce")
    df["revenue_usd"] = pd.to_numeric(df["revenue_usd"], errors="coerce")

    if df["revenue_usd"].isna().any():
        missing_revenue = df["revenue_usd"].isna().sum()
        raise ValueError(
            f"revenue_usd has {missing_revenue:,} missing or invalid values."
        )

    # ------------------------------------------------------------
    # Calendar fields and duplicate validation
    # ------------------------------------------------------------
    df = add_calendar_fields(df)

    duplicate_count = df.duplicated(["cik", "period_date"]).sum()

    print(f"\nDuplicate CIK + period_date rows: {duplicate_count:,}")

    if duplicate_count > 0:
        raise ValueError(
            "Duplicate CIK + period_date rows found. "
            "Stop and inspect the panel first."
        )

    # ------------------------------------------------------------
    # Sort firm panel
    # ------------------------------------------------------------
    df = df.sort_values(["cik", "period_date"]).reset_index(drop=True)

    print(f"Unique CIKs before history filter: {df['cik'].nunique():,}")

    # ------------------------------------------------------------
    # Keep firms with enough history
    # ------------------------------------------------------------
    df["firm_obs_total"] = df.groupby("cik")["period_date"].transform("count")

    df = df[df["firm_obs_total"] >= MIN_FIRM_OBS].copy()

    print(f"\nMinimum firm observations required: {MIN_FIRM_OBS}")
    print(f"Rows after firm history filter: {len(df):,}")
    print(f"Unique CIKs after firm history filter: {df['cik'].nunique():,}")

    # ------------------------------------------------------------
    # Signed log revenue
    # ------------------------------------------------------------
    df["signed_log_revenue"] = signed_log(df["revenue_usd"])

    # ------------------------------------------------------------
    # Lag features
    # ------------------------------------------------------------
    df = df.sort_values(["cik", "period_date"]).reset_index(drop=True)

    g = df.groupby("cik", sort=False)

    lag_list = [1, 2, 3, 4]

    for lag in lag_list:
        df[f"revenue_lag_{lag}"] = g["revenue_usd"].shift(lag)
        df[f"signed_log_revenue_lag_{lag}"] = g["signed_log_revenue"].shift(lag)
        df[f"quarter_index_lag_{lag}"] = g["quarter_index"].shift(lag)

        valid_lag = df[f"quarter_index_lag_{lag}"].eq(
            df["quarter_index"] - lag
        )

        df.loc[~valid_lag, f"revenue_lag_{lag}"] = np.nan
        df.loc[~valid_lag, f"signed_log_revenue_lag_{lag}"] = np.nan

    # ------------------------------------------------------------
    # Growth features
    # ------------------------------------------------------------
    df["revenue_growth_lag_1"] = safe_growth(
        df["revenue_usd"],
        df["revenue_lag_1"],
    )

    df["revenue_growth_lag_4"] = safe_growth(
        df["revenue_usd"],
        df["revenue_lag_4"],
    )

    # Flags for cases where growth is undefined because lagged revenue is zero
    df["revenue_growth_lag_1_undefined"] = (
        df["revenue_lag_1"].notna()
        & df["revenue_lag_1"].eq(0)
    ).astype(int)

    df["revenue_growth_lag_4_undefined"] = (
        df["revenue_lag_4"].notna()
        & df["revenue_lag_4"].eq(0)
    ).astype(int)

    # Fill undefined growth with 0 after creating flags
    df["revenue_growth_lag_1"] = df["revenue_growth_lag_1"].fillna(0)
    df["revenue_growth_lag_4"] = df["revenue_growth_lag_4"].fillna(0)

    # Signed-log changes are safer for zero and negative revenue
    df["signed_log_revenue_change_lag_1"] = (
        df["signed_log_revenue"]
        - df["signed_log_revenue_lag_1"]
    )

    df["signed_log_revenue_change_lag_4"] = (
        df["signed_log_revenue"]
        - df["signed_log_revenue_lag_4"]
    )

    # ------------------------------------------------------------
    # Target: next-quarter revenue
    # ------------------------------------------------------------
    df["target_revenue_next_qtr"] = g["revenue_usd"].shift(-1)
    df["target_signed_log_revenue_next_qtr"] = g["signed_log_revenue"].shift(-1)
    df["target_period_date"] = g["period_date"].shift(-1)
    df["target_quarter_index"] = g["quarter_index"].shift(-1)

    valid_next_target = df["target_quarter_index"].eq(
        df["quarter_index"] + 1
    )

    target_cols = [
        "target_revenue_next_qtr",
        "target_signed_log_revenue_next_qtr",
        "target_period_date",
        "target_quarter_index",
    ]

    for col in target_cols:
        df.loc[~valid_next_target, col] = np.nan

    (
        df["target_calendar_period"],
        df["target_calendar_year"],
        df["target_calendar_quarter"],
    ) = quarter_label_from_index(df["target_quarter_index"])

    # ------------------------------------------------------------
    # Split labels
    # Important:
    # Since each row predicts next-quarter revenue,
    # the split is based on the TARGET quarter.
    # ------------------------------------------------------------
    df["model_split"] = pd.NA

    target_year = df["target_calendar_year"]

    mask_train = ((target_year >= 2021) & (target_year <= 2023)).fillna(False)
    mask_validation = (target_year == 2024).fillna(False)
    mask_test = (target_year == 2025).fillna(False)

    df.loc[mask_train, "model_split"] = "train_2021_2023"
    df.loc[mask_validation, "model_split"] = "validation_2024"
    df.loc[mask_test, "model_split"] = "test_2025"

    # ------------------------------------------------------------
    # Baseline forecast columns
    # ------------------------------------------------------------

    # Naive baseline:
    # Predict next-quarter revenue using current-quarter revenue.
    df["naive_forecast_next_qtr"] = df["revenue_usd"]

    # Seasonal naive baseline:
    # Current row t predicts t+1.
    # Same quarter last year for target t+1 is t-3.
    df["seasonal_naive_forecast_next_qtr"] = df["revenue_lag_3"]

    # ------------------------------------------------------------
    # Build modeling-ready dataset
    # ------------------------------------------------------------
    required_model_cols = [
        "target_revenue_next_qtr",
        "model_split",
        "revenue_lag_1",
        "revenue_lag_2",
        "revenue_lag_3",
        "revenue_lag_4",
        "signed_log_revenue_lag_1",
        "signed_log_revenue_lag_2",
        "signed_log_revenue_lag_3",
        "signed_log_revenue_lag_4",
        "signed_log_revenue_change_lag_1",
        "signed_log_revenue_change_lag_4",
        "seasonal_naive_forecast_next_qtr",
    ]

    modeling_df = df.dropna(subset=required_model_cols).copy()

    helper_cols = [f"quarter_index_lag_{lag}" for lag in lag_list]

    modeling_df = modeling_df.drop(
        columns=helper_cols,
        errors="ignore",
    )

    # ------------------------------------------------------------
    # Reorder important columns to the front
    # ------------------------------------------------------------
    front_cols = [
        "cik",
        "company_name",
        "sic",
        "period_date",
        "calendar_period",
        "calendar_year",
        "calendar_quarter",
        "quarter_index",
        "fy",
        "fp",
        "form",
        "filed_date",
        "tag",
        "revenue_usd",
        "revenue_millions_usd",
        "signed_log_revenue",
        "target_period_date",
        "target_calendar_period",
        "target_calendar_year",
        "target_calendar_quarter",
        "target_quarter_index",
        "target_revenue_next_qtr",
        "target_signed_log_revenue_next_qtr",
        "model_split",
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
        "naive_forecast_next_qtr",
        "seasonal_naive_forecast_next_qtr",
        "is_derived_q4",
        "revenue_source",
        "derivation_method",
        "annual_revenue_usd",
        "q1_q3_revenue_usd",
        "firm_obs_total",
        "adsh",
        "quarter_folder",
        "stmt_values",
        "qtrs",
        "uom",
        "accepted",
    ]

    front_cols = [col for col in front_cols if col in modeling_df.columns]
    other_cols = [col for col in modeling_df.columns if col not in front_cols]

    modeling_df = modeling_df[front_cols + other_cols]

    # Clean date formatting for CSV output
    for date_col in ["period_date", "filed_date", "target_period_date"]:
        if date_col in modeling_df.columns:
            modeling_df[date_col] = pd.to_datetime(
                modeling_df[date_col],
                errors="coerce",
            ).dt.strftime("%Y-%m-%d")

    # ------------------------------------------------------------
    # Save output
    # ------------------------------------------------------------
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    modeling_df.to_csv(OUTPUT_PATH, index=False)

    # ------------------------------------------------------------
    # Validation checks
    # ------------------------------------------------------------
    missing_target_output = modeling_df["target_revenue_next_qtr"].isna().sum()
    duplicate_output = modeling_df.duplicated(["cik", "period_date"]).sum()

    split_summary = (
        modeling_df.groupby("model_split")
        .agg(
            rows=("cik", "size"),
            unique_ciks=("cik", "nunique"),
            min_feature_period=("calendar_period", "min"),
            max_feature_period=("calendar_period", "max"),
            min_target_period=("target_calendar_period", "min"),
            max_target_period=("target_calendar_period", "max"),
            derived_q4_feature_rows=(
                "is_derived_q4",
                lambda x: int((x == 1).sum()),
            ),
        )
        .reset_index()
    )

    split_order = {
        "train_2021_2023": 1,
        "validation_2024": 2,
        "test_2025": 3,
    }

    split_summary["sort_order"] = split_summary["model_split"].map(split_order)

    split_summary = (
        split_summary
        .sort_values("sort_order")
        .drop(columns="sort_order")
    )

    feature_missing_check = [
        "revenue_lag_1",
        "revenue_lag_2",
        "revenue_lag_3",
        "revenue_lag_4",
        "revenue_growth_lag_1",
        "revenue_growth_lag_4",
        "signed_log_revenue_lag_1",
        "signed_log_revenue_lag_2",
        "signed_log_revenue_lag_3",
        "signed_log_revenue_lag_4",
        "signed_log_revenue_change_lag_1",
        "signed_log_revenue_change_lag_4",
        "naive_forecast_next_qtr",
        "seasonal_naive_forecast_next_qtr",
    ]

    feature_missing_check = [
        col for col in feature_missing_check if col in modeling_df.columns
    ]

    missing_feature_counts = (
        modeling_df[feature_missing_check]
        .isna()
        .sum()
        .reset_index()
    )

    missing_feature_counts.columns = ["feature", "missing_rows"]

    # ------------------------------------------------------------
    # Baseline metrics
    # ------------------------------------------------------------
    baseline_summary = pd.concat(
        [
            baseline_metrics(
                modeling_df,
                forecast_col="naive_forecast_next_qtr",
                baseline_name="naive_current_revenue",
            ),
            baseline_metrics(
                modeling_df,
                forecast_col="seasonal_naive_forecast_next_qtr",
                baseline_name="seasonal_naive_same_quarter_last_year",
            ),
        ],
        ignore_index=True,
    )

    baseline_summary.to_csv(BASELINE_SUMMARY_PATH, index=False)

    # ------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("OUTPUT SAVED")
    print("=" * 80)
    print(OUTPUT_PATH)

    print("\nBaseline summary saved:")
    print(BASELINE_SUMMARY_PATH)

    print("\n" + "=" * 80)
    print("FINAL MODELING DATASET VALIDATION")
    print("=" * 80)

    print(f"\nFinal modeling rows: {len(modeling_df):,}")
    print(f"Final unique CIKs: {modeling_df['cik'].nunique():,}")
    print(f"Duplicate CIK + period_date rows in output: {duplicate_output:,}")
    print(f"Missing target_revenue_next_qtr rows in output: {missing_target_output:,}")

    print("\nSplit summary:")
    print(split_summary.to_string(index=False))

    print("\nMissing feature counts:")
    print(missing_feature_counts.to_string(index=False))

    print("\nBaseline feasibility and metrics:")
    print(baseline_summary.to_string(index=False))

    print("\nGrowth undefined flags:")
    print(
        modeling_df[
            [
                "revenue_growth_lag_1_undefined",
                "revenue_growth_lag_4_undefined",
            ]
        ]
        .sum()
        .to_string()
    )

    if missing_target_output != 0:
        raise ValueError(
            "Output still has missing target values. Stop and inspect."
        )

    if duplicate_output != 0:
        raise ValueError(
            "Output has duplicate CIK + period_date rows. Stop and inspect."
        )

    print("\nScript 10 completed successfully.")


if __name__ == "__main__":
    main()