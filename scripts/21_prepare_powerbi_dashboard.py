"""Prepare final Power BI datasets from the frozen log-growth model outputs.

This script replaces the former residual-model export. Run from project root:
    python scripts/21_prepare_powerbi_dashboard.py

It writes the 15-company showcase, an all-eligible-company export, a selected
company audit file, the official metrics table, and refresh checks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FINAL_MODEL = "log_growth_random_forest"
FINAL_MODEL_DISPLAY = "Log-Growth Random Forest"
BASELINE_DISPLAY = "Seasonal Naive Baseline"
TEST_SPLIT = "test_2025"
BIG_COUNT = MID_COUNT = SMALL_COUNT = 5
SMALL_LOWER_PCT = 0.10
SMALL_UPPER_PCT = 0.25
HIGH_RISK_APE = 0.15


def project_root_from_script() -> Path:
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent if script_dir.name == "scripts" else script_dir


def clean_company_label(name: object) -> str:
    text = str(name).upper()
    for old, new in [
        (" /NEW", ""), (",", ""), (".", ""), (" INC", ""),
        (" CORP", ""), (" CORPORATION", ""), (" COMPANY", ""),
        (" LTD", ""), (" PLC", ""), (" CLASS A", ""),
        (" CL A", ""), (" HOLDINGS", ""), (" THE", ""),
    ]:
        text = text.replace(old, new)
    return text.strip()


def quarter_sort_key(period: object) -> float:
    text = str(period).strip().upper()
    try:
        year_text, quarter_text = text.split("Q")
        year, quarter = int(year_text), int(quarter_text)
    except (ValueError, TypeError):
        return np.nan
    return year * 4 + quarter if quarter in (1, 2, 3, 4) else np.nan


def quarter_start_date(period: object) -> pd.Timestamp:
    if pd.isna(quarter_sort_key(period)):
        return pd.NaT
    year_text, quarter_text = str(period).strip().upper().split("Q")
    return pd.Timestamp(
        year=int(year_text),
        month={1: 1, 2: 4, 3: 7, 4: 10}[int(quarter_text)],
        day=1,
    )


def build_company_universe(predictions: pd.DataFrame) -> pd.DataFrame:
    data = predictions.copy()
    needed = [
        "target_revenue_next_qtr",
        "seasonal_naive_forecast_next_qtr",
        "pred_improved",
    ]
    data["_row_complete"] = data[needed].notna().all(axis=1)
    universe = data.groupby(["cik", "company_name"], as_index=False).agg(
        quarters=("target_calendar_period", "nunique"),
        actual_2025_revenue=("target_revenue_next_qtr", "sum"),
        complete_rows=("_row_complete", "sum"),
        total_rows=("_row_complete", "size"),
    )
    universe = universe[
        (universe["quarters"] == 4)
        & (universe["complete_rows"] == 4)
        & (universe["total_rows"] == 4)
        & (universe["actual_2025_revenue"] > 0)
    ].copy()
    universe["actual_2025_revenue_musd"] = universe["actual_2025_revenue"] / 1e6
    universe["revenue_rank_2025"] = (
        universe["actual_2025_revenue"].rank(ascending=False, method="first").astype(int)
    )
    universe["revenue_percentile_2025"] = universe[
        "actual_2025_revenue"
    ].rank(pct=True, method="average")
    universe["company_label"] = universe["company_name"].map(clean_company_label)
    return universe.sort_values(
        ["actual_2025_revenue", "cik"], ascending=[False, True]
    ).reset_index(drop=True)


def select_showcase(universe: pd.DataFrame) -> pd.DataFrame:
    if len(universe) < 15:
        raise ValueError(f"Only {len(universe)} eligible firms; at least 15 are required.")
    used: set[str] = set()
    blocks: list[pd.DataFrame] = []

    big = universe.nlargest(BIG_COUNT, "actual_2025_revenue").copy()
    big["firm_size_group"] = "Big"
    big["selection_reason"] = "Top 5 by 2025 actual revenue"
    used.update(big["cik"])
    blocks.append(big)

    remaining = universe[~universe["cik"].isin(used)].copy()
    remaining["_median_distance"] = (
        remaining["revenue_percentile_2025"] - 0.50
    ).abs()
    mid = remaining.nsmallest(
        MID_COUNT, ["_median_distance", "revenue_rank_2025"]
    ).drop(columns="_median_distance")
    mid["firm_size_group"] = "Mid"
    mid["selection_reason"] = "Closest to median revenue percentile"
    used.update(mid["cik"])
    blocks.append(mid)

    band = universe[
        (~universe["cik"].isin(used))
        & (universe["revenue_percentile_2025"] >= SMALL_LOWER_PCT)
        & (universe["revenue_percentile_2025"] <= SMALL_UPPER_PCT)
    ].sort_values("actual_2025_revenue", ascending=False)
    if len(band) < SMALL_COUNT:
        raise ValueError("Too few firms in the documented 10th-25th percentile band.")
    positions = np.linspace(0, len(band) - 1, SMALL_COUNT).round().astype(int)
    small = band.iloc[positions].copy()
    small["firm_size_group"] = "Small"
    small["selection_reason"] = "Spread across 10th-25th revenue percentile"
    blocks.append(small)

    selected = pd.concat(blocks, ignore_index=True)
    if selected["cik"].nunique() != 15:
        raise ValueError("Showcase selection did not produce 15 distinct firms.")
    return selected


def add_universe_groups(
    universe: pd.DataFrame, selected: pd.DataFrame
) -> pd.DataFrame:
    all_key = universe[[
        "cik", "company_label", "revenue_rank_2025",
        "revenue_percentile_2025", "actual_2025_revenue_musd",
    ]].copy()

    def group_from_percentile(value: float) -> str:
        if value >= 0.80:
            return "Big"
        if 0.40 <= value < 0.60:
            return "Mid"
        if 0.10 <= value <= 0.25:
            return "Small"
        return "Other"

    all_key["firm_size_group"] = all_key["revenue_percentile_2025"].map(
        group_from_percentile
    )
    all_key["selection_reason"] = "Eligible four-quarter firm"
    all_key["is_showcase"] = 0
    overlay = selected.set_index("cik")[["firm_size_group", "selection_reason"]]
    all_key = all_key.set_index("cik")
    all_key.update(overlay)
    all_key.loc[overlay.index, "is_showcase"] = 1
    return all_key.reset_index()


def merge_history_features(
    predictions: pd.DataFrame, feature_path: Path
) -> pd.DataFrame:
    features = pd.read_csv(feature_path, dtype={"cik": "string"}, low_memory=False)
    keys = ["cik", "period_date", "calendar_period", "target_calendar_period"]
    wanted = [
        "revenue_lag_1", "revenue_lag_2", "revenue_lag_3", "revenue_lag_4",
        "cpi", "fed_funds_rate", "unemployment_rate", "cpi_qoq_change",
        "cpi_qoq_pct_change", "fed_funds_rate_qoq_change",
        "unemployment_rate_qoq_change", "firm_obs_total",
    ]
    small = features[keys + [c for c in wanted if c in features.columns]].drop_duplicates(keys)
    if small.duplicated(keys).any():
        raise ValueError("Feature data has duplicate merge keys.")
    merged = predictions.merge(small, on=keys, how="left", validate="one_to_one")
    if len(merged) != len(predictions):
        raise ValueError("History-feature merge changed the prediction row count.")
    return merged


def validate_frozen_inputs(
    predictions: pd.DataFrame, metrics: pd.DataFrame
) -> list[dict[str, object]]:
    model_row = metrics.loc[metrics["model"] == FINAL_MODEL]
    if len(model_row) != 1:
        raise ValueError(f"Metrics must contain exactly one {FINAL_MODEL!r} row.")
    official = model_row.iloc[0]
    checks: list[dict[str, object]] = []

    def add(name: str, expected: object, actual: object, passed: bool) -> None:
        checks.append({
            "check": name, "expected": expected, "actual": actual,
            "status": "PASS" if passed else "FAIL",
        })

    add("test observations", int(official["test_observations"]), len(predictions),
        len(predictions) == int(official["test_observations"]))
    add("test firms", int(official["test_firms"]), predictions["cik"].nunique(),
        predictions["cik"].nunique() == int(official["test_firms"]))
    positive_count = int((predictions["target_revenue_next_qtr"] > 0).sum())
    add("positive-actual denominator", int(official["positive_actual_observations"]),
        positive_count, positive_count == int(official["positive_actual_observations"]))
    computed_mae = float(predictions["absolute_error"].mean() / 1e6)
    add("final model MAE ($M)", float(official["mae_usd_millions"]), computed_mae,
        np.isclose(computed_mae, official["mae_usd_millions"], atol=1e-9))
    fallback_count = int(predictions["fallback_used"].fillna(False).astype(bool).sum())
    add("fallback observations", int(official["fallback_observations"]), fallback_count,
        fallback_count == int(official["fallback_observations"]))
    negative_count = int((predictions["pred_improved"] < 0).sum())
    add("negative final forecasts", int(official["negative_forecasts"]), negative_count,
        negative_count == int(official["negative_forecasts"]))
    invalid_count = int((~predictions["forecast_timing_valid"].astype(bool)).sum())
    add("timing-invalid rows", 0, invalid_count, invalid_count == 0)
    split_values = sorted(predictions["model_split"].dropna().unique().tolist())
    add("prediction split", TEST_SPLIT, ", ".join(split_values), split_values == [TEST_SPLIT])
    model_values = sorted(predictions["model_name"].dropna().unique().tolist())
    add("model name", FINAL_MODEL, ", ".join(model_values), model_values == [FINAL_MODEL])
    failed = [row for row in checks if row["status"] == "FAIL"]
    if failed:
        raise ValueError("Frozen input reconciliation failed:\n" + pd.DataFrame(failed).to_string(index=False))
    return checks


def prepare_wide_rows(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["target_quarter_sort"] = out["target_calendar_period"].map(quarter_sort_key)
    out["target_quarter_start_date"] = out["target_calendar_period"].map(quarter_start_date)
    out["current_revenue_musd"] = out["revenue_usd"] / 1e6
    out["actual_revenue_musd"] = out["target_revenue_next_qtr"] / 1e6
    out["seasonal_baseline_forecast_musd"] = out["seasonal_naive_forecast_next_qtr"] / 1e6
    out["final_model_forecast_musd"] = out["pred_improved"] / 1e6
    out["seasonal_baseline_variance_musd"] = (
        out["seasonal_baseline_forecast_musd"] - out["actual_revenue_musd"]
    )
    out["final_model_variance_musd"] = (
        out["final_model_forecast_musd"] - out["actual_revenue_musd"]
    )
    out["seasonal_baseline_absolute_error_musd"] = out[
        "seasonal_baseline_variance_musd"
    ].abs()
    out["final_model_absolute_error_musd"] = out["final_model_variance_musd"].abs()
    positive = out["actual_revenue_musd"] > 0
    out["final_model_absolute_percentage_error"] = np.where(
        positive, out["final_model_absolute_error_musd"] / out["actual_revenue_musd"], np.nan
    )
    out["final_model_accuracy_pct"] = np.where(
        positive, np.maximum(0.0, 1.0 - out["final_model_absolute_percentage_error"]) * 100, np.nan
    )
    out["forecast_risk_flag"] = np.select(
        [~positive, out["final_model_absolute_percentage_error"] >= HIGH_RISK_APE],
        ["Not scored: actual <= 0", "High risk"], default="Low risk",
    )
    for column in ["revenue_lag_1", "revenue_lag_2", "revenue_lag_3", "revenue_lag_4"]:
        if column in out.columns:
            out[f"{column}_musd"] = out[column] / 1e6
    out = out.sort_values(["cik", "target_quarter_sort"]).reset_index(drop=True)
    first_actual = out.groupby("cik")["actual_revenue_musd"].transform("first")
    out["actual_revenue_index_base100"] = np.where(
        first_actual != 0, out["actual_revenue_musd"] / first_actual * 100, np.nan
    )
    out["model_name_technical"] = FINAL_MODEL
    out["model_name_display"] = FINAL_MODEL_DISPLAY
    out["percentage_metric_denominator_note"] = "Actual revenue > 0 only"
    return out


def reshape_scenarios(wide: pd.DataFrame) -> pd.DataFrame:
    identifiers = [
        "cik", "company_name", "company_label", "sic", "firm_size_group",
        "selection_reason", "revenue_rank_2025", "revenue_percentile_2025",
        "actual_2025_revenue_musd", "is_showcase", "period_date", "calendar_period",
        "target_period_date", "target_calendar_period", "target_quarter_sort",
        "target_quarter_start_date", "form", "filed_date", "forecast_timing_valid",
        "is_derived_q4", "revenue_source", "current_revenue_musd", "actual_revenue_musd",
        "seasonal_baseline_forecast_musd", "final_model_forecast_musd",
        "final_model_variance_musd", "final_model_absolute_error_musd",
        "final_model_absolute_percentage_error", "final_model_accuracy_pct",
        "within_5_pct", "within_10_pct", "within_20_pct", "fallback_used",
        "forecast_risk_flag", "actual_revenue_index_base100", "model_name_technical",
        "model_name_display", "model_version", "percentage_metric_denominator_note",
        "cpi", "fed_funds_rate", "unemployment_rate", "cpi_qoq_change",
        "cpi_qoq_pct_change", "fed_funds_rate_qoq_change",
        "unemployment_rate_qoq_change", "revenue_lag_1_musd", "revenue_lag_2_musd",
        "revenue_lag_3_musd", "revenue_lag_4_musd", "firm_obs_total",
    ]
    identifiers = [column for column in identifiers if column in wide.columns]
    scenarios = [
        ("Actual Revenue", "actual_revenue_musd", None, None, None),
        (BASELINE_DISPLAY, "seasonal_baseline_forecast_musd",
         "seasonal_baseline_variance_musd", "seasonal_baseline_absolute_error_musd", None),
        (f"{FINAL_MODEL_DISPLAY} Forecast", "final_model_forecast_musd",
         "final_model_variance_musd", "final_model_absolute_error_musd",
         "final_model_accuracy_pct"),
    ]
    blocks: list[pd.DataFrame] = []
    for order, (name, value, variance, error, accuracy) in enumerate(scenarios, start=1):
        block = wide[identifiers].copy()
        block["forecast_scenario"] = name
        block["forecast_scenario_sort"] = order
        block["scenario_value_musd"] = wide[value]
        block["scenario_variance_musd"] = wide[variance] if variance else np.nan
        block["scenario_absolute_error_musd"] = wide[error] if error else np.nan
        block["scenario_accuracy_pct"] = wide[accuracy] if accuracy else np.nan
        blocks.append(block)
    final = pd.concat(blocks, ignore_index=True).sort_values([
        "revenue_rank_2025", "company_label", "target_quarter_sort", "forecast_scenario_sort"
    ]).reset_index(drop=True)
    for column in [c for c in final.columns if c.endswith("_musd")]:
        final[column] = final[column].round(3)
    for column in ["revenue_percentile_2025", "final_model_absolute_percentage_error"]:
        if column in final.columns:
            final[column] = final[column].round(6)
    for column in ["final_model_accuracy_pct", "scenario_accuracy_pct", "actual_revenue_index_base100"]:
        if column in final.columns:
            final[column] = final[column].round(3)
    return final


def prepare_metrics_for_powerbi(metrics: pd.DataFrame) -> pd.DataFrame:
    names = {
        "seasonal_naive_baseline": BASELINE_DISPLAY,
        "last_quarter_naive_baseline": "Last-Quarter Naive Baseline",
        FINAL_MODEL: FINAL_MODEL_DISPLAY,
    }
    output = metrics.copy()
    output.insert(1, "model_display_name", output["model"].map(names))
    output.insert(2, "is_final_model", (output["model"] == FINAL_MODEL).astype(int))
    output["percentage_metric_note"] = (
        "Median APE, sMAPE, and hit rates use actual revenue > 0 only"
    )
    return output


def parse_args() -> argparse.Namespace:
    root = project_root_from_script()
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    prediction_path = args.predictions or root / "outputs/report_tables/log_growth_random_forest_predictions_test_2025.csv"
    metrics_path = args.metrics or root / "outputs/report_tables/log_growth_random_forest_metrics.csv"
    feature_path = args.features or root / "data/processed/model_ready_sec_macro_2021_2025.csv"
    output_dir = args.output_dir or root / "outputs/powerbi"

    print("=" * 80)
    print("SCRIPT 21: FINAL LOG-GROWTH POWER BI DATASETS")
    print("=" * 80)
    print(f"Predictions: {prediction_path}")
    print(f"Metrics:     {metrics_path}")
    print(f"Features:    {feature_path}")
    for path in [prediction_path, metrics_path, feature_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    predictions = pd.read_csv(prediction_path, dtype={"cik": "string"}, low_memory=False)
    metrics = pd.read_csv(metrics_path, low_memory=False)
    required = [
        "cik", "company_name", "sic", "period_date", "calendar_period",
        "target_period_date", "target_calendar_period", "model_split", "form",
        "filed_date", "forecast_timing_valid", "is_derived_q4", "revenue_source",
        "revenue_usd", "target_revenue_next_qtr", "seasonal_naive_forecast_next_qtr",
        "pred_improved", "absolute_error", "fallback_used", "within_5_pct",
        "within_10_pct", "within_20_pct", "model_name", "model_version",
    ]
    missing = [column for column in required if column not in predictions.columns]
    if missing:
        raise ValueError("Prediction file is missing: " + ", ".join(missing))

    checks = validate_frozen_inputs(predictions, metrics)
    print("\nFrozen-input reconciliation: PASS")
    print(pd.DataFrame(checks).to_string(index=False))
    predictions = merge_history_features(predictions, feature_path)
    universe = build_company_universe(predictions)
    selected = select_showcase(universe)
    company_key = add_universe_groups(universe, selected)

    wide_all = predictions.merge(company_key, on="cik", how="inner", validate="many_to_one")
    if len(wide_all) != len(universe) * 4:
        raise ValueError("Eligible all-company export is not exactly four rows per firm.")
    wide_all = prepare_wide_rows(wide_all)
    wide_showcase = wide_all[wide_all["is_showcase"] == 1].copy()
    if len(wide_showcase) != 60 or wide_showcase["cik"].nunique() != 15:
        raise ValueError("Showcase export is not exactly 15 firms x 4 quarters.")

    long_showcase = reshape_scenarios(wide_showcase)
    long_all = reshape_scenarios(wide_all)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "Compatibility dashboard CSV": output_dir / "powerbi_forecast_data.csv",
        "Showcase dashboard CSV": output_dir / "powerbi_forecast_dashboard_15_companies.csv",
        "All eligible firms CSV": output_dir / "powerbi_forecast_dashboard_all_companies.csv",
        "Selected-company audit": output_dir / "powerbi_selected_15_company_list.csv",
        "Full-test metrics table": output_dir / "powerbi_model_metrics.csv",
        "Refresh checks": output_dir / "powerbi_refresh_checks.csv",
    }
    long_showcase.to_csv(paths["Compatibility dashboard CSV"], index=False)
    long_showcase.to_csv(paths["Showcase dashboard CSV"], index=False)
    long_all.to_csv(paths["All eligible firms CSV"], index=False)
    selected.to_csv(paths["Selected-company audit"], index=False)
    prepare_metrics_for_powerbi(metrics).to_csv(paths["Full-test metrics table"], index=False)
    pd.DataFrame(checks).to_csv(paths["Refresh checks"], index=False)

    print("\nSelected showcase companies:")
    print(selected[[
        "firm_size_group", "revenue_rank_2025", "company_label",
        "actual_2025_revenue_musd", "revenue_percentile_2025", "selection_reason",
    ]].to_string(index=False))
    print("\n" + "=" * 80)
    print("OUTPUTS SAVED")
    print("=" * 80)
    for label, path in paths.items():
        print(f"{label + ':':29s} {path}")
    print(f"\nShowcase: {len(long_showcase):,} scenario rows (15 firms x 4 quarters x 3 scenarios)")
    print(f"All eligible: {len(long_all):,} scenario rows ({wide_all['cik'].nunique():,} firms x 4 quarters x 3 scenarios)")
    print(f"Model label: {FINAL_MODEL_DISPLAY}")
    print("Script 21 completed successfully.")


if __name__ == "__main__":
    main()
