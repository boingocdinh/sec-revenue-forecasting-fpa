from pathlib import Path
import sys
import pandas as pd


# ============================================================
# Script 13: Merge SEC Forecasting Dataset with FRED Macro Data
# Project: Panel Forecasting of Firm Revenue Using SEC + Macro Data
#
# Important modeling decision:
# - Merge macro data using feature-quarter calendar_period.
# - Do NOT merge using target_calendar_period because that would use
#   next-quarter macro information and create leakage.
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

if SCRIPT_DIR.name == "scripts":
    PROJECT_ROOT = SCRIPT_DIR.parent
else:
    PROJECT_ROOT = SCRIPT_DIR

if len(sys.argv) > 1:
    PROJECT_ROOT = Path(sys.argv[1]).expanduser().resolve()

SEC_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecasting_dataset_2021_2025.csv"
)

MACRO_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "macro_quarterly.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_ready_sec_macro_2021_2025.csv"
)

SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "report_tables"
    / "macro_merge_summary.csv"
)


def main():
    print("=" * 80)
    print("SCRIPT 13: MERGE SEC FORECASTING DATASET WITH MACRO DATA")
    print("=" * 80)

    if not SEC_INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing SEC forecasting dataset:\n{SEC_INPUT_PATH}")

    if not MACRO_INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing macro quarterly dataset:\n{MACRO_INPUT_PATH}")

    print("\nLoading SEC forecasting dataset:")
    print(SEC_INPUT_PATH)

    sec = pd.read_csv(SEC_INPUT_PATH, low_memory=False)

    print(f"  SEC rows: {len(sec):,}")
    print(f"  SEC columns: {len(sec.columns):,}")

    print("\nLoading macro quarterly dataset:")
    print(MACRO_INPUT_PATH)

    macro = pd.read_csv(MACRO_INPUT_PATH, low_memory=False)

    print(f"  Macro rows: {len(macro):,}")
    print(f"  Macro columns: {len(macro.columns):,}")

    # ------------------------------------------------------------
    # Basic required column checks
    # ------------------------------------------------------------
    sec_required = [
        "cik",
        "period_date",
        "calendar_period",
        "model_split",
        "target_revenue_next_qtr",
    ]

    macro_required = [
        "calendar_period",
        "cpi",
        "fed_funds_rate",
        "unemployment_rate",
        "cpi_qoq_change",
        "cpi_qoq_pct_change",
        "fed_funds_rate_qoq_change",
        "unemployment_rate_qoq_change",
    ]

    missing_sec = [col for col in sec_required if col not in sec.columns]
    missing_macro = [col for col in macro_required if col not in macro.columns]

    if missing_sec:
        raise ValueError("SEC file missing columns: " + ", ".join(missing_sec))

    if missing_macro:
        raise ValueError("Macro file missing columns: " + ", ".join(missing_macro))

    # ------------------------------------------------------------
    # Clean merge keys
    # ------------------------------------------------------------
    sec["calendar_period"] = sec["calendar_period"].astype(str).str.strip()
    macro["calendar_period"] = macro["calendar_period"].astype(str).str.strip()

    # Keep only macro columns needed for modeling
    macro_model_cols = [
        "calendar_period",
        "cpi",
        "fed_funds_rate",
        "unemployment_rate",
        "cpi_qoq_change",
        "cpi_qoq_pct_change",
        "fed_funds_rate_qoq_change",
        "unemployment_rate_qoq_change",
    ]

    macro_model = macro[macro_model_cols].copy()

    # Check one row per macro quarter
    macro_dup_count = macro_model.duplicated(["calendar_period"]).sum()

    print(f"\nDuplicate calendar_period rows in macro file: {macro_dup_count:,}")

    if macro_dup_count > 0:
        raise ValueError("Macro file has duplicate calendar_period rows. Stop and inspect.")

    # ------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------
    before_rows = len(sec)

    merged = sec.merge(
        macro_model,
        on="calendar_period",
        how="left",
        validate="many_to_one",
    )

    after_rows = len(merged)

    print(f"\nRows before merge: {before_rows:,}")
    print(f"Rows after merge:  {after_rows:,}")

    if before_rows != after_rows:
        raise ValueError("Row count changed after macro merge. Stop and inspect.")

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------
    macro_feature_cols = [
        "cpi",
        "fed_funds_rate",
        "unemployment_rate",
        "cpi_qoq_change",
        "cpi_qoq_pct_change",
        "fed_funds_rate_qoq_change",
        "unemployment_rate_qoq_change",
    ]

    missing_macro_counts = merged[macro_feature_cols].isna().sum()

    duplicate_cik_period = merged.duplicated(["cik", "period_date"]).sum()

    print(f"\nDuplicate CIK + period_date rows after merge: {duplicate_cik_period:,}")

    print("\nMissing macro values after merge:")
    print(missing_macro_counts.to_string())

    if duplicate_cik_period > 0:
        raise ValueError("Duplicate CIK + period_date rows found after merge.")

    if missing_macro_counts.sum() > 0:
        missing_periods = (
            merged.loc[
                merged[macro_feature_cols].isna().any(axis=1),
                "calendar_period",
            ]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        print("\nCalendar periods with missing macro values:")
        print(missing_periods)

        raise ValueError("Missing macro values found after merge. Stop and inspect.")

    # ------------------------------------------------------------
    # Split summary
    # ------------------------------------------------------------
    split_summary = (
        merged.groupby("model_split")
        .agg(
            rows=("cik", "size"),
            unique_ciks=("cik", "nunique"),
            min_feature_period=("calendar_period", "min"),
            max_feature_period=("calendar_period", "max"),
            min_target_period=("target_calendar_period", "min"),
            max_target_period=("target_calendar_period", "max"),
            mean_cpi=("cpi", "mean"),
            mean_fed_funds_rate=("fed_funds_rate", "mean"),
            mean_unemployment_rate=("unemployment_rate", "mean"),
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

    macro_by_period = (
        merged.groupby("calendar_period")
        .agg(
            rows=("cik", "size"),
            unique_ciks=("cik", "nunique"),
            cpi=("cpi", "first"),
            fed_funds_rate=("fed_funds_rate", "first"),
            unemployment_rate=("unemployment_rate", "first"),
        )
        .reset_index()
        .sort_values("calendar_period")
    )

    summary_rows = [
        {
            "metric": "sec_rows_before_merge",
            "value": before_rows,
        },
        {
            "metric": "rows_after_merge",
            "value": after_rows,
        },
        {
            "metric": "duplicate_cik_period_rows_after_merge",
            "value": duplicate_cik_period,
        },
        {
            "metric": "missing_cpi_rows",
            "value": int(missing_macro_counts["cpi"]),
        },
        {
            "metric": "missing_fed_funds_rate_rows",
            "value": int(missing_macro_counts["fed_funds_rate"]),
        },
        {
            "metric": "missing_unemployment_rate_rows",
            "value": int(missing_macro_counts["unemployment_rate"]),
        },
    ]

    summary_df = pd.DataFrame(summary_rows)

    # ------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    merged.to_csv(OUTPUT_PATH, index=False)
    summary_df.to_csv(SUMMARY_OUTPUT_PATH, index=False)

    print("\n" + "=" * 80)
    print("OUTPUTS SAVED")
    print("=" * 80)

    print("\nMacro-enhanced modeling dataset:")
    print(OUTPUT_PATH)

    print("\nMacro merge summary:")
    print(SUMMARY_OUTPUT_PATH)

    print("\n" + "=" * 80)
    print("MERGE VALIDATION SUMMARY")
    print("=" * 80)

    print("\nOverall summary:")
    print(summary_df.to_string(index=False))

    print("\nSplit summary:")
    print(split_summary.to_string(index=False))

    print("\nMacro values by feature period:")
    print(macro_by_period.to_string(index=False))

    print("\nImportant:")
    print("  Macro variables were joined using feature-quarter calendar_period.")
    print("  This avoids leakage from the target quarter.")

    print("\nScript 13 completed successfully.")


if __name__ == "__main__":
    main()