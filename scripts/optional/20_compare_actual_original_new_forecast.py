from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Script 20: Compare Actual Revenue vs Original YOY vs New Corrected Forecast
#
# Compare 3 numbers:
# 1. Actual 2025 revenue
# 2. Original YOY adjusted seasonal forecast
# 3. New YOY + residual correction forecast
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

if SCRIPT_DIR.name == "optional" and SCRIPT_DIR.parent.name == "scripts":
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
elif SCRIPT_DIR.name == "scripts":
    PROJECT_ROOT = SCRIPT_DIR.parent
else:
    PROJECT_ROOT = SCRIPT_DIR

if len(sys.argv) > 1:
    PROJECT_ROOT = Path(sys.argv[1]).expanduser().resolve()

INPUT_PATH = PROJECT_ROOT / "outputs" / "report_tables" / "yoy_macro_residual_predictions.csv"

OUTPUT_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "report_tables" / "actual_vs_original_vs_new_10_company_summary.csv"
OUTPUT_DETAIL_PATH = PROJECT_ROOT / "outputs" / "report_tables" / "actual_vs_original_vs_new_10_company_detail.csv"
PLOT_PATH = PROJECT_ROOT / "outputs" / "charts" / "actual_vs_original_vs_new_10_company.png"

BEST_MODEL_NAME = "yoy_plus_random_forest_macro_controls_residual"

COMPANY_KEYWORDS = [
    "WALMART",
    "AMAZON",
    "APPLE",
    "ALPHABET",
    "MICROSOFT",
    "COSTCO",
    "NVIDIA",
    "META",
    "HOME DEPOT",
    "TESLA",
]


def accuracy(actual, predicted):
    if actual == 0:
        return np.nan
    return max(0, 1 - abs(actual - predicted) / abs(actual))


def find_company_rows(df, keyword):
    return df[
        df["company_name"]
        .astype(str)
        .str.upper()
        .str.contains(keyword, na=False)
    ].copy()


def main():
    print("=" * 80)
    print("SCRIPT 20: ACTUAL VS ORIGINAL YOY VS NEW CORRECTED FORECAST")
    print("=" * 80)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing file:\n{INPUT_PATH}\nRun Script 19 first."
        )

    (PROJECT_ROOT / "outputs" / "report_tables").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "outputs" / "charts").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH, low_memory=False)

    print("\nLoaded residual prediction file:")
    print(INPUT_PATH)

    print(f"\nRows loaded: {len(df):,}")

    required_cols = [
        "cik",
        "company_name",
        "target_calendar_period",
        "target_revenue_next_qtr",
        "pred_yoy_adjusted_seasonal",
        "corrected_prediction",
        "model",
        "split",
    ]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    # Keep only selected best residual model and 2025 test
    df = df[
        (df["model"] == BEST_MODEL_NAME)
        & (df["split"] == "test_2025")
    ].copy()

    print(f"\nRows after filtering to best model and 2025 test: {len(df):,}")
    print(f"Unique CIKs: {df['cik'].nunique():,}")

    if df.empty:
        raise ValueError("No rows found for selected model and test_2025.")

    selected_rows = []
    missing_keywords = []

    for keyword in COMPANY_KEYWORDS:
        company_df = find_company_rows(df, keyword)

        if company_df.empty:
            missing_keywords.append(keyword)
            continue

        company_counts = (
            company_df.groupby(["cik", "company_name"])
            .agg(
                quarters=("target_calendar_period", "nunique"),
                total_actual_revenue=("target_revenue_next_qtr", "sum"),
            )
            .reset_index()
            .sort_values(
                ["quarters", "total_actual_revenue"],
                ascending=[False, False],
            )
        )

        chosen_cik = company_counts.iloc[0]["cik"]
        chosen_name = company_counts.iloc[0]["company_name"]

        chosen_rows = company_df[company_df["cik"] == chosen_cik].copy()
        chosen_rows = chosen_rows.sort_values("target_calendar_period")

        print(f"\nSelected for '{keyword}':")
        print(f"  CIK: {chosen_cik}")
        print(f"  Company: {chosen_name}")
        print(f"  Quarters: {chosen_rows['target_calendar_period'].nunique()}")

        selected_rows.append(chosen_rows)

    if missing_keywords:
        print("\nWARNING: Missing company keywords:")
        for keyword in missing_keywords:
            print(f"  - {keyword}")

    selected = pd.concat(selected_rows, ignore_index=True)

    # Rename for clarity
    selected["actual_revenue"] = selected["target_revenue_next_qtr"]
    selected["original_yoy_forecast"] = selected["pred_yoy_adjusted_seasonal"]
    selected["new_corrected_forecast"] = selected["corrected_prediction"]

    selected["original_abs_error"] = (
        selected["actual_revenue"] - selected["original_yoy_forecast"]
    ).abs()

    selected["new_abs_error"] = (
        selected["actual_revenue"] - selected["new_corrected_forecast"]
    ).abs()

    selected["original_accuracy"] = selected.apply(
        lambda row: accuracy(row["actual_revenue"], row["original_yoy_forecast"]),
        axis=1,
    )

    selected["new_accuracy"] = selected.apply(
        lambda row: accuracy(row["actual_revenue"], row["new_corrected_forecast"]),
        axis=1,
    )

    selected["new_model_better"] = selected["new_abs_error"] < selected["original_abs_error"]

    # Convert to millions
    money_cols = [
        "actual_revenue",
        "original_yoy_forecast",
        "new_corrected_forecast",
        "original_abs_error",
        "new_abs_error",
    ]

    for col in money_cols:
        selected[col + "_millions"] = selected[col] / 1_000_000

    # Detail output
    detail_cols = [
        "cik",
        "company_name",
        "target_calendar_period",
        "actual_revenue_millions",
        "original_yoy_forecast_millions",
        "new_corrected_forecast_millions",
        "original_abs_error_millions",
        "new_abs_error_millions",
        "original_accuracy",
        "new_accuracy",
        "new_model_better",
    ]

    detail = selected[detail_cols].copy()
    detail.to_csv(OUTPUT_DETAIL_PATH, index=False)

    # Annual company summary
    summary = (
        selected.groupby(["cik", "company_name"])
        .agg(
            quarters=("target_calendar_period", "nunique"),
            actual_2025_revenue_millions=("actual_revenue_millions", "sum"),
            original_yoy_2025_forecast_millions=("original_yoy_forecast_millions", "sum"),
            new_corrected_2025_forecast_millions=("new_corrected_forecast_millions", "sum"),
            original_mae_millions=("original_abs_error_millions", "mean"),
            new_mae_millions=("new_abs_error_millions", "mean"),
            original_avg_quarterly_accuracy=("original_accuracy", "mean"),
            new_avg_quarterly_accuracy=("new_accuracy", "mean"),
            quarters_new_better=("new_model_better", "sum"),
        )
        .reset_index()
    )

    summary["original_annual_abs_error_millions"] = (
        summary["actual_2025_revenue_millions"]
        - summary["original_yoy_2025_forecast_millions"]
    ).abs()

    summary["new_annual_abs_error_millions"] = (
        summary["actual_2025_revenue_millions"]
        - summary["new_corrected_2025_forecast_millions"]
    ).abs()

    summary["original_annual_accuracy"] = summary.apply(
        lambda row: accuracy(
            row["actual_2025_revenue_millions"],
            row["original_yoy_2025_forecast_millions"],
        ),
        axis=1,
    )

    summary["new_annual_accuracy"] = summary.apply(
        lambda row: accuracy(
            row["actual_2025_revenue_millions"],
            row["new_corrected_2025_forecast_millions"],
        ),
        axis=1,
    )

    summary["better_annual_model"] = np.where(
        summary["new_annual_abs_error_millions"]
        < summary["original_annual_abs_error_millions"],
        "New corrected forecast",
        "Original YOY forecast",
    )

    summary = summary.sort_values(
        "actual_2025_revenue_millions",
        ascending=False,
    )

    summary.to_csv(OUTPUT_SUMMARY_PATH, index=False)

    # Display clean table
    display = summary.copy()

    pct_cols = [
        "original_avg_quarterly_accuracy",
        "new_avg_quarterly_accuracy",
        "original_annual_accuracy",
        "new_annual_accuracy",
    ]

    for col in pct_cols:
        display[col] = (display[col] * 100).round(2)

    money_display_cols = [
        "actual_2025_revenue_millions",
        "original_yoy_2025_forecast_millions",
        "new_corrected_2025_forecast_millions",
        "original_annual_abs_error_millions",
        "new_annual_abs_error_millions",
        "original_mae_millions",
        "new_mae_millions",
    ]

    for col in money_display_cols:
        display[col] = display[col].round(2)

    print("\n" + "=" * 80)
    print("10-COMPANY COMPARISON: ACTUAL VS ORIGINAL YOY VS NEW CORRECTED")
    print("=" * 80)

    print(
        display[
            [
                "company_name",
                "actual_2025_revenue_millions",
                "original_yoy_2025_forecast_millions",
                "new_corrected_2025_forecast_millions",
                "original_annual_abs_error_millions",
                "new_annual_abs_error_millions",
                "original_annual_accuracy",
                "new_annual_accuracy",
                "better_annual_model",
            ]
        ].to_string(index=False)
    )

    # Overall summary
    actual_total = summary["actual_2025_revenue_millions"].sum()
    original_total = summary["original_yoy_2025_forecast_millions"].sum()
    new_total = summary["new_corrected_2025_forecast_millions"].sum()

    original_total_error = abs(actual_total - original_total)
    new_total_error = abs(actual_total - new_total)

    original_total_accuracy = accuracy(actual_total, original_total)
    new_total_accuracy = accuracy(actual_total, new_total)

    original_avg_q_acc = summary["original_avg_quarterly_accuracy"].mean()
    new_avg_q_acc = summary["new_avg_quarterly_accuracy"].mean()

    print("\n" + "=" * 80)
    print("OVERALL 10-COMPANY TOTAL")
    print("=" * 80)

    print(f"Actual total 2025 revenue:           {actual_total:,.2f} million USD")
    print(f"Original YOY forecast total:         {original_total:,.2f} million USD")
    print(f"New corrected forecast total:        {new_total:,.2f} million USD")
    print(f"Original annual total error:         {original_total_error:,.2f} million USD")
    print(f"New corrected annual total error:    {new_total_error:,.2f} million USD")
    print(f"Original annual total accuracy:      {original_total_accuracy * 100:.2f}%")
    print(f"New corrected annual total accuracy: {new_total_accuracy * 100:.2f}%")
    print(f"Original avg quarterly accuracy:     {original_avg_q_acc * 100:.2f}%")
    print(f"New corrected avg quarterly accuracy:{new_avg_q_acc * 100:.2f}%")

    # Plot
    plot_df = summary.copy()
    labels = (
        plot_df["company_name"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace(" INC", "", regex=False)
        .str.replace(" CORP", "", regex=False)
        .str[:18]
    )

    x = np.arange(len(plot_df))
    width = 0.25

    plt.figure(figsize=(16, 7))

    plt.bar(
        x - width,
        plot_df["actual_2025_revenue_millions"],
        width,
        label="Actual 2025 revenue",
    )

    plt.bar(
        x,
        plot_df["original_yoy_2025_forecast_millions"],
        width,
        label="Original YOY forecast",
    )

    plt.bar(
        x + width,
        plot_df["new_corrected_2025_forecast_millions"],
        width,
        label="New corrected forecast",
    )

    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("2025 revenue, millions USD")
    plt.title("Actual Revenue vs Original YOY Forecast vs New Corrected Forecast")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=300)
    plt.close()

    print("\n" + "=" * 80)
    print("OUTPUTS SAVED")
    print("=" * 80)
    print(f"Summary table: {OUTPUT_SUMMARY_PATH}")
    print(f"Detail table:  {OUTPUT_DETAIL_PATH}")
    print(f"Plot:          {PLOT_PATH}")

    print("\nScript 20 completed successfully.")


if __name__ == "__main__":
    main()