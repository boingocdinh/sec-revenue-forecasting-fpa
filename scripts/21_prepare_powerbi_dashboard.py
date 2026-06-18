"""
============================================================
Script 21: Prepare Final Power BI / Public Dashboard Dataset
Project: Panel Forecasting of Firm Revenue Using SEC + Macro Data
Final model: yoy_plus_random_forest_macro_controls_residual

Purpose
-------
Produce ONE clean, public, shareable CSV for Power BI:

    outputs/powerbi/powerbi_forecast_dashboard_15_companies.csv

The dataset contains exactly 15 companies (5 Big / 5 Mid / 5 Small),
selected programmatically and reproducibly from the 2025 test-set
predictions of the final model. No companies are hand-picked.

Everything Power BI needs is precomputed:
  - actual revenue
  - YOY baseline forecast
  - Random Forest adjustment (predicted residual)
  - YOY + Random Forest adjusted forecast
  - variance, absolute error, accuracy
  - forecast risk flag
  - firm size group + selection reason
  - company ranking + percentile
  - target quarter + quarter sorting
  - revenue index (base 100)
  - macro variables
  - revenue lag history
  - technical model name
  - scenario column for line charts

Selection rule (reproducible, defensible)
------------------------------------------
1. Start from the final model 2025 test predictions.
2. Keep firms with all 4 target quarters in 2025, positive total
   actual 2025 revenue, and no missing actual/forecast values.
3. Compute actual_2025_revenue_musd, revenue_rank_2025,
   revenue_percentile_2025.
4. Select:
     Big   = top 5 by total actual 2025 revenue
     Mid   = 5 firms closest to the 50th revenue percentile
     Small = 5 firms in the 10th-25th percentile band
             (representative lower-distribution firms, not the
              extreme/noisy absolute bottom)
============================================================
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np


# ============================================================
# Robust project-root resolution
# Works whether this file lives in <root>/ or <root>/scripts/
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent

if SCRIPT_DIR.name == "optional" and SCRIPT_DIR.parent.name == "scripts":
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
elif SCRIPT_DIR.name == "scripts":
    PROJECT_ROOT = SCRIPT_DIR.parent
else:
    PROJECT_ROOT = SCRIPT_DIR

# CLI:  python 21_...py [project_root] [--all]
#   --all  export EVERY eligible firm (showcase 15 tagged is_showcase=1)
EXPORT_ALL = "--all" in sys.argv
_pos_args = [a for a in sys.argv[1:] if not a.startswith("--")]
if _pos_args:
    PROJECT_ROOT = Path(_pos_args[0]).expanduser().resolve()


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
PREDICTIONS_PATH = (
    PROJECT_ROOT / "outputs" / "report_tables" / "yoy_macro_residual_predictions.csv"
)
FEATURE_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "model_ready_sec_macro_2021_2025.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "powerbi"
OUTPUT_PATH = OUTPUT_DIR / "powerbi_forecast_dashboard_15_companies.csv"
ALL_OUTPUT_PATH = OUTPUT_DIR / "powerbi_forecast_dashboard_all_companies.csv"
COMPANY_LIST_PATH = OUTPUT_DIR / "powerbi_selected_15_company_list.csv"

BEST_MODEL_NAME = "yoy_plus_random_forest_macro_controls_residual"

# Group sizes
BIG_COUNT = 5
MID_COUNT = 5
SMALL_COUNT = 5

# Small-firm percentile band (avoid extreme noisy bottom)
SMALL_LOWER_PCT = 0.10
SMALL_UPPER_PCT = 0.25

# Forecast risk threshold: |variance| as a share of actual
FORECAST_RISK_THRESHOLD = 0.15  # 15% absolute error => high risk


# ============================================================
# Helpers
# ============================================================
def clean_company_label(name):
    text = str(name).upper()
    replacements = [
        (" /NEW", ""),
        (",", ""),
        (".", ""),
        (" INC", ""),
        (" CORP", ""),
        (" CORPORATION", ""),
        (" COMPANY", ""),
        (" LTD", ""),
        (" PLC", ""),
        (" CLASS A", ""),
        (" CL A", ""),
        (" HOLDINGS", ""),
        (" THE", ""),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text.strip()


def quarter_sort_key(period):
    """True ordinal sort key: year*4 + quarter (e.g. 2025Q1 -> 8101)."""
    text = str(period).strip().upper()
    if "Q" not in text:
        return np.nan
    try:
        year = int(text.split("Q")[0])
        quarter = int(text.split("Q")[1])
    except (ValueError, IndexError):
        return np.nan
    return year * 4 + quarter


def quarter_start_date(period):
    text = str(period).strip().upper()
    if "Q" not in text:
        return pd.NaT
    try:
        year = int(text.split("Q")[0])
        quarter = int(text.split("Q")[1])
    except (ValueError, IndexError):
        return pd.NaT
    start_month = {1: 1, 2: 4, 3: 7, 4: 10}
    return pd.Timestamp(year=year, month=start_month[quarter], day=1)


def accuracy_pct(actual, predicted):
    """
    Forecast accuracy as a percentage, floored at 0.
        accuracy = max(0, 1 - |actual - predicted| / |actual|) * 100
    Returns NaN when actual is 0.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    out = np.full_like(actual, np.nan, dtype=float)
    mask = actual != 0
    out[mask] = np.maximum(
        0.0,
        1.0 - np.abs(actual[mask] - predicted[mask]) / np.abs(actual[mask]),
    ) * 100.0
    return out


# ============================================================
# Step 1-3: build eligible company universe and rank it
# ============================================================
def build_company_universe(df):
    """
    df is already filtered to BEST_MODEL_NAME and test_2025.
    Returns one row per firm with size metrics, keeping only firms that
    have all 4 quarters, positive revenue, and no missing actual/forecast.
    """
    # Flag rows with any missing actual or forecast value
    needed = ["target_revenue_next_qtr", "pred_yoy_adjusted_seasonal", "corrected_prediction"]
    df = df.copy()
    df["_row_complete"] = df[needed].notna().all(axis=1)

    grp = df.groupby(["cik", "company_name"])
    universe = grp.agg(
        quarters=("target_calendar_period", "nunique"),
        actual_2025_revenue=("target_revenue_next_qtr", "sum"),
        complete_rows=("_row_complete", "sum"),
        total_rows=("_row_complete", "size"),
    ).reset_index()

    # Eligibility: 4 distinct quarters, all rows complete, positive revenue
    universe = universe[
        (universe["quarters"] == 4)
        & (universe["complete_rows"] == universe["total_rows"])
        & (universe["total_rows"] == 4)
        & (universe["actual_2025_revenue"] > 0)
    ].copy()

    universe["actual_2025_revenue_musd"] = universe["actual_2025_revenue"] / 1_000_000

    # Rank (1 = largest) and percentile (0-1, higher = larger)
    universe["revenue_rank_2025"] = (
        universe["actual_2025_revenue"].rank(ascending=False, method="dense").astype(int)
    )
    universe["revenue_percentile_2025"] = universe["actual_2025_revenue"].rank(pct=True)
    universe["company_label"] = universe["company_name"].apply(clean_company_label)

    return universe.sort_values("actual_2025_revenue", ascending=False).reset_index(drop=True)


def select_15_companies(universe):
    """
    Programmatic 5/5/5 selection. No forced names.
    Returns selected firms with firm_size_group and selection_reason.
    """
    if len(universe) < (BIG_COUNT + MID_COUNT + SMALL_COUNT):
        raise ValueError(
            f"Only {len(universe)} eligible firms; need at least "
            f"{BIG_COUNT + MID_COUNT + SMALL_COUNT}."
        )

    selected = []
    used = set()

    # --- Big: top 5 by total actual 2025 revenue ---
    big = universe[~universe["cik"].isin(used)].nlargest(
        BIG_COUNT, "actual_2025_revenue"
    ).copy()
    big["firm_size_group"] = "Big"
    big["selection_reason"] = "Top 5 by 2025 actual revenue"
    used.update(big["cik"])
    selected.append(big)

    # --- Mid: 5 firms closest to the 50th percentile ---
    remaining = universe[~universe["cik"].isin(used)].copy()
    remaining["dist_to_median"] = (remaining["revenue_percentile_2025"] - 0.50).abs()
    mid = remaining.nsmallest(MID_COUNT, "dist_to_median").copy()
    mid = mid.drop(columns="dist_to_median")
    mid["firm_size_group"] = "Mid"
    mid["selection_reason"] = "Closest to median revenue percentile"
    used.update(mid["cik"])
    selected.append(mid)

    # --- Small: 10th-25th percentile band, representative ---
    band = universe[
        (~universe["cik"].isin(used))
        & (universe["revenue_percentile_2025"] >= SMALL_LOWER_PCT)
        & (universe["revenue_percentile_2025"] <= SMALL_UPPER_PCT)
    ].copy()

    if len(band) >= SMALL_COUNT:
        # Spread evenly across the band for representativeness
        band = band.sort_values("actual_2025_revenue", ascending=False).reset_index(drop=True)
        positions = np.linspace(0, len(band) - 1, SMALL_COUNT).round().astype(int)
        small = band.iloc[sorted(set(positions))].copy()
        # If rounding collapsed duplicates, top up from the band
        if len(small) < SMALL_COUNT:
            extra = band[~band["cik"].isin(small["cik"])].head(SMALL_COUNT - len(small))
            small = pd.concat([small, extra], ignore_index=True)
    else:
        # Widen the band if too few firms qualify
        wide = universe[
            (~universe["cik"].isin(used))
            & (universe["revenue_percentile_2025"] >= 0.05)
            & (universe["revenue_percentile_2025"] <= 0.35)
        ].sort_values("actual_2025_revenue", ascending=False)
        small = wide.head(SMALL_COUNT).copy()

    small["firm_size_group"] = "Small"
    small["selection_reason"] = "Lower revenue percentile representative"
    used.update(small["cik"])
    selected.append(small)

    out = pd.concat(selected, ignore_index=True)
    return out.head(BIG_COUNT + MID_COUNT + SMALL_COUNT).copy()


# ============================================================
# Step 4: merge in lag/macro history from model-ready file
# ============================================================
def merge_history_features(pred_df):
    if not FEATURE_DATA_PATH.exists():
        print(f"\nNote: feature file not found, skipping history merge:\n  {FEATURE_DATA_PATH}")
        return pred_df

    print(f"\nMerging history/macro features from:\n  {FEATURE_DATA_PATH}")
    features = pd.read_csv(FEATURE_DATA_PATH, low_memory=False)
    features["cik"] = features["cik"].astype(str)

    merge_keys = ["cik", "period_date", "calendar_period", "target_calendar_period"]
    wanted = [
        "revenue_lag_1", "revenue_lag_2", "revenue_lag_3", "revenue_lag_4",
        "revenue_growth_lag_1", "revenue_growth_lag_4",
        "cpi", "fed_funds_rate", "unemployment_rate",
        "cpi_qoq_change", "cpi_qoq_pct_change",
        "fed_funds_rate_qoq_change", "unemployment_rate_qoq_change",
        "calendar_year", "calendar_quarter", "is_derived_q4", "firm_obs_total",
    ]
    cols = [c for c in merge_keys if c in features.columns]
    cols += [c for c in wanted if c in features.columns and c not in pred_df.columns]
    features_small = features[cols].drop_duplicates(merge_keys)

    before = len(pred_df)
    merged = pred_df.merge(features_small, on=merge_keys, how="left", validate="many_to_one")
    if len(merged) != before:
        raise ValueError("Row count changed after history feature merge.")
    return merged


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 80)
    print("SCRIPT 21: PREPARE FINAL POWER BI DASHBOARD DATASET (15 COMPANIES)")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Missing predictions file:\n{PREDICTIONS_PATH}\nRun Script 19 first."
        )

    df = pd.read_csv(PREDICTIONS_PATH, low_memory=False)
    print(f"\nLoaded predictions: {len(df):,} rows")

    required = [
        "cik", "company_name", "sic", "period_date", "calendar_period",
        "target_period_date", "target_calendar_period", "target_revenue_next_qtr",
        "pred_yoy_adjusted_seasonal", "corrected_prediction", "predicted_residual",
        "revenue_usd", "model", "split",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    df["cik"] = df["cik"].astype(str)

    # Filter to final model + 2025 test
    df = df[(df["model"] == BEST_MODEL_NAME) & (df["split"] == "test_2025")].copy()
    print(f"Rows for final model + test_2025: {len(df):,}  (firms: {df['cik'].nunique():,})")
    if df.empty:
        raise ValueError("No rows for the final model on test_2025.")

    # ---- Build universe, rank, select 15 ----
    universe = build_company_universe(df)
    print(f"Eligible firms (4 quarters, complete, positive): {len(universe):,}")

    selected = select_15_companies(universe)
    print("\nSelected 15 companies:")
    print(
        selected[
            ["firm_size_group", "revenue_rank_2025", "company_label",
             "actual_2025_revenue_musd", "revenue_percentile_2025", "selection_reason"]
        ].to_string(index=False)
    )

    selected_key = selected[
        ["cik", "company_label", "firm_size_group", "selection_reason",
         "revenue_rank_2025", "revenue_percentile_2025", "actual_2025_revenue_musd"]
    ].copy()

    # ---- Merge history/macro ----
    df = merge_history_features(df)

    if EXPORT_ALL:
        # Every eligible firm. Showcase 15 tagged is_showcase=1.
        # Firms outside the showcase still get a size group by percentile,
        # and selection_reason = "Eligible firm (not in showcase 15)".
        all_key = universe[
            ["cik", "company_label", "revenue_rank_2025",
             "revenue_percentile_2025", "actual_2025_revenue_musd"]
        ].copy()

        def size_from_pct(p):
            if p >= 0.80:
                return "Big"
            if 0.40 <= p < 0.60:
                return "Mid"
            if 0.10 <= p <= 0.25:
                return "Small"
            return "Other"

        all_key["firm_size_group"] = all_key["revenue_percentile_2025"].apply(size_from_pct)
        all_key["selection_reason"] = "Eligible firm (not in showcase 15)"

        # Overlay the exact showcase group/reason for the 15
        showcase_map = selected.set_index("cik")[["firm_size_group", "selection_reason"]]
        all_key = all_key.set_index("cik")
        all_key.update(showcase_map)  # showcase group/reason win
        all_key["is_showcase"] = (all_key.index.isin(selected["cik"])).astype(int)
        all_key = all_key.reset_index()

        out = df.merge(all_key, on="cik", how="inner")
        out_path = ALL_OUTPUT_PATH
        export_label = "ALL eligible firms"
    else:
        out = df.merge(selected_key, on="cik", how="inner")
        out["is_showcase"] = 1
        out_path = OUTPUT_PATH
        export_label = "showcase 15 firms"

    # ---- Quarter ordering fields ----
    out["target_quarter_sort"] = out["target_calendar_period"].apply(quarter_sort_key)
    out["target_quarter_start_date"] = out["target_calendar_period"].apply(quarter_start_date)

    # ---- Money fields (millions USD) with business-readable names ----
    out["actual_revenue_musd"] = out["target_revenue_next_qtr"] / 1_000_000
    out["yoy_baseline_forecast_musd"] = out["pred_yoy_adjusted_seasonal"] / 1_000_000
    out["random_forest_adjustment_musd"] = out["predicted_residual"] / 1_000_000
    out["yoy_random_forest_adjusted_forecast_musd"] = out["corrected_prediction"] / 1_000_000
    out["current_revenue_musd"] = out["revenue_usd"] / 1_000_000

    # ---- Variance, error, accuracy (per scenario) ----
    out["yoy_baseline_variance_musd"] = (
        out["actual_revenue_musd"] - out["yoy_baseline_forecast_musd"]
    )
    out["yoy_random_forest_adjusted_variance_musd"] = (
        out["actual_revenue_musd"] - out["yoy_random_forest_adjusted_forecast_musd"]
    )

    out["yoy_baseline_abs_error_musd"] = out["yoy_baseline_variance_musd"].abs()
    out["yoy_random_forest_adjusted_abs_error_musd"] = (
        out["yoy_random_forest_adjusted_variance_musd"].abs()
    )

    out["yoy_baseline_accuracy_pct"] = accuracy_pct(
        out["actual_revenue_musd"], out["yoy_baseline_forecast_musd"]
    )
    out["yoy_random_forest_adjusted_accuracy_pct"] = accuracy_pct(
        out["actual_revenue_musd"], out["yoy_random_forest_adjusted_forecast_musd"]
    )

    # ---- Forecast risk flag (based on adjusted forecast) ----
    risk_ratio = np.where(
        out["actual_revenue_musd"] != 0,
        out["yoy_random_forest_adjusted_abs_error_musd"] / out["actual_revenue_musd"].abs(),
        np.nan,
    )
    out["forecast_abs_error_ratio"] = risk_ratio
    out["forecast_risk_flag"] = np.where(
        np.isnan(risk_ratio), "Unknown",
        np.where(risk_ratio >= FORECAST_RISK_THRESHOLD, "High risk", "Low risk"),
    )

    # ---- Revenue lag history in millions (if present) ----
    for col in ["revenue_lag_1", "revenue_lag_2", "revenue_lag_3", "revenue_lag_4"]:
        if col in out.columns:
            out[col + "_musd"] = out[col] / 1_000_000

    # ---- Revenue index base 100 (per firm, vs first available quarter) ----
    out = out.sort_values(["cik", "target_quarter_sort"]).reset_index(drop=True)
    first_rev = out.groupby("cik")["actual_revenue_musd"].transform("first")
    out["actual_revenue_index_base100"] = np.where(
        first_rev != 0, out["actual_revenue_musd"] / first_rev * 100.0, np.nan
    )

    # ---- Technical model name + scenario column for line charts ----
    out["model_name_technical"] = BEST_MODEL_NAME

    # ============================================================
    # Reshape to long "scenario" form so Power BI line charts can
    # plot Actual / YOY Baseline / YOY+RF Adjusted on one axis.
    # Each company-quarter produces 3 scenario rows.
    # ============================================================
    id_cols = [
        "cik", "company_name", "company_label", "sic",
        "firm_size_group", "selection_reason",
        "revenue_rank_2025", "revenue_percentile_2025", "actual_2025_revenue_musd",
        "calendar_period", "target_calendar_period",
        "target_quarter_sort", "target_quarter_start_date",
        "current_revenue_musd", "actual_revenue_index_base100",
        "forecast_risk_flag", "forecast_abs_error_ratio",
        "model_name_technical", "is_showcase",
        # macro
        "cpi", "fed_funds_rate", "unemployment_rate",
        "cpi_qoq_change", "cpi_qoq_pct_change",
        "fed_funds_rate_qoq_change", "unemployment_rate_qoq_change",
        # lag history
        "revenue_lag_1_musd", "revenue_lag_2_musd",
        "revenue_lag_3_musd", "revenue_lag_4_musd",
        # quality
        "is_derived_q4", "firm_obs_total",
    ]
    id_cols = [c for c in id_cols if c in out.columns]

    scenarios = [
        ("Actual",
         "actual_revenue_musd", None, None, None),
        ("YOY Baseline Forecast",
         "yoy_baseline_forecast_musd",
         "yoy_baseline_variance_musd",
         "yoy_baseline_abs_error_musd",
         "yoy_baseline_accuracy_pct"),
        ("YOY + Random Forest Adjusted Forecast",
         "yoy_random_forest_adjusted_forecast_musd",
         "yoy_random_forest_adjusted_variance_musd",
         "yoy_random_forest_adjusted_abs_error_musd",
         "yoy_random_forest_adjusted_accuracy_pct"),
    ]

    long_frames = []
    for scen_name, val_col, var_col, abserr_col, acc_col in scenarios:
        block = out[id_cols].copy()
        block["forecast_scenario"] = scen_name
        block["actual_revenue_musd"] = out["actual_revenue_musd"]
        block["yoy_baseline_forecast_musd"] = out["yoy_baseline_forecast_musd"]
        block["random_forest_adjustment_musd"] = out["random_forest_adjustment_musd"]
        block["yoy_random_forest_adjusted_forecast_musd"] = (
            out["yoy_random_forest_adjusted_forecast_musd"]
        )
        block["scenario_value_musd"] = out[val_col]
        block["scenario_variance_musd"] = out[var_col] if var_col else np.nan
        block["scenario_absolute_error_musd"] = out[abserr_col] if abserr_col else np.nan
        block["scenario_accuracy_pct"] = out[acc_col] if acc_col else np.nan
        long_frames.append(block)

    final = pd.concat(long_frames, ignore_index=True)

    # Stable sort for Power BI
    scenario_order = {
        "Actual": 1,
        "YOY Baseline Forecast": 2,
        "YOY + Random Forest Adjusted Forecast": 3,
    }
    final["_scen_order"] = final["forecast_scenario"].map(scenario_order)
    final = final.sort_values(
        ["revenue_rank_2025", "company_label", "target_quarter_sort", "_scen_order"]
    ).drop(columns="_scen_order").reset_index(drop=True)

    # Round money/percent fields for a tidy public file
    money_like = [c for c in final.columns if c.endswith("_musd")]
    for c in money_like:
        final[c] = final[c].round(3)
    for c in ["scenario_accuracy_pct"]:
        final[c] = final[c].round(2)
    if "forecast_abs_error_ratio" in final.columns:
        final["forecast_abs_error_ratio"] = final["forecast_abs_error_ratio"].round(4)
    if "revenue_percentile_2025" in final.columns:
        final["revenue_percentile_2025"] = final["revenue_percentile_2025"].round(4)
    if "actual_revenue_index_base100" in final.columns:
        final["actual_revenue_index_base100"] = final["actual_revenue_index_base100"].round(2)

    # ---- Save ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final.to_csv(out_path, index=False)
    selected.to_csv(COMPANY_LIST_PATH, index=False)

    n_firms = final["cik"].nunique()

    print("\n" + "=" * 80)
    print("OUTPUT SAVED")
    print("=" * 80)
    print(f"Dashboard CSV ({export_label}):  {out_path}")
    print(f"Showcase 15 company list:          {COMPANY_LIST_PATH}")
    print(f"\nRows: {len(final):,}  ({n_firms} firms x up to 4 quarters x 3 scenarios)")
    print(f"Columns: {len(final.columns):,}")
    if EXPORT_ALL:
        print(f"Showcase firms tagged is_showcase=1: {int((selected['cik'].nunique()))}")

    print("\nShowcase group counts:")
    print(selected["firm_size_group"].value_counts().to_string())

    print("\nScript 21 completed successfully.")


if __name__ == "__main__":
    main()
