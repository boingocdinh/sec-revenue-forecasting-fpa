from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# Script 18: Macro-Revenue Relationship Regression
#
# Research questions:
# 1. Do macro variables explain next-quarter revenue change?
# 2. After controlling for firm revenue history, do macro variables still matter?
# 3. Do macro variables have statistically meaningful coefficients?
# 4. Do macro variables improve R² after revenue-history controls?
# 5. Are signs reasonable?
#
# Important:
# - This is diagnostic / explanatory analysis.
# - This is NOT causal inference.
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

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_ready_sec_macro_2021_2025.csv"
)

RESULTS_DIR = PROJECT_ROOT / "outputs" / "report_tables"

MODEL_COMPARISON_PATH = RESULTS_DIR / "macro_relationship_model_comparison.csv"
COEFFICIENTS_PATH = RESULTS_DIR / "macro_relationship_coefficients.csv"
NOTES_PATH = RESULTS_DIR / "macro_relationship_notes.md"
PLOT_PATH = RESULTS_DIR / "macro_relationship_coefficients.png"

# Use all modeling rows because this is relationship analysis,
# not model tuning.
USE_SPLITS = [
    "train_2021_2023",
    "validation_2024",
    "test_2025",
]


def winsorize_series(s, lower=0.01, upper=0.99):
    """
    Clip extreme values to reduce outlier impact.
    """
    s = pd.to_numeric(s, errors="coerce")

    low = s.quantile(lower)
    high = s.quantile(upper)

    return s.clip(lower=low, upper=high)


def add_regression_features(df):
    """
    Create regression target and controls.
    """
    df = df.copy()

    # ------------------------------------------------------------
    # Main target:
    # signed-log next-quarter revenue change
    #
    # This works better than raw revenue change because revenue is skewed,
    # and it handles zero / negative revenue.
    # ------------------------------------------------------------
    df["target_signed_log_change_next_qtr"] = (
        df["target_signed_log_revenue_next_qtr"]
        - df["signed_log_revenue"]
    )

    # Raw dollar change, used for descriptive interpretation only
    df["target_revenue_delta_next_qtr"] = (
        df["target_revenue_next_qtr"]
        - df["revenue_usd"]
    )

    # Percent-style change, only valid if current revenue is not zero
    df["target_revenue_delta_rate_next_qtr"] = np.where(
        df["revenue_usd"].abs() > 0,
        df["target_revenue_delta_next_qtr"] / df["revenue_usd"].abs(),
        np.nan,
    )

    # SIC numeric / industry group
    if "sic" in df.columns:
        df["sic_numeric"] = pd.to_numeric(df["sic"], errors="coerce")
        df["sic_2digit"] = np.floor(df["sic_numeric"] / 100)
    else:
        df["sic_numeric"] = np.nan
        df["sic_2digit"] = np.nan

    # Revenue-history controls
    df["revenue_qoq_change"] = df["revenue_usd"] - df["revenue_lag_1"]
    df["revenue_yoy_change"] = df["revenue_usd"] - df["revenue_lag_4"]

    df["signed_log_qoq_change"] = (
        df["signed_log_revenue"] - df["signed_log_revenue_lag_1"]
    )

    df["signed_log_yoy_change"] = (
        df["signed_log_revenue"] - df["signed_log_revenue_lag_4"]
    )

    lag_cols = [
        "revenue_lag_1",
        "revenue_lag_2",
        "revenue_lag_3",
        "revenue_lag_4",
    ]

    df["revenue_lag_mean_1_4"] = df[lag_cols].mean(axis=1)
    df["revenue_lag_median_1_4"] = df[lag_cols].median(axis=1)

    df["current_to_lag_mean_ratio"] = np.where(
        df["revenue_lag_mean_1_4"].abs() > 0,
        df["revenue_usd"] / df["revenue_lag_mean_1_4"].abs(),
        np.nan,
    )

    df = df.replace([np.inf, -np.inf], np.nan)

    # Winsorize extreme regression variables
    winsor_cols = [
        "target_signed_log_change_next_qtr",
        "target_revenue_delta_rate_next_qtr",
        "revenue_growth_lag_1",
        "revenue_growth_lag_4",
        "signed_log_qoq_change",
        "signed_log_yoy_change",
        "current_to_lag_mean_ratio",
    ]

    for col in winsor_cols:
        if col in df.columns:
            df[col + "_w"] = winsorize_series(df[col])

    return df


def get_sign_interpretation(variable, coef, pvalue):
    """
    Human-readable interpretation of macro coefficient sign.
    """
    if coef > 0:
        direction = "positive"
        tendency = "increase"
    elif coef < 0:
        direction = "negative"
        tendency = "decrease"
    else:
        direction = "zero"
        tendency = "no clear change"

    significant_05 = pvalue < 0.05
    significant_10 = pvalue < 0.10

    if variable == "unemployment_rate_qoq_change":
        expected = "negative"
        plain_name = "unemployment rate change"
    elif variable == "fed_funds_rate_qoq_change":
        expected = "negative"
        plain_name = "interest rate change"
    elif variable == "cpi_qoq_pct_change":
        expected = "ambiguous"
        plain_name = "CPI inflation change"
    else:
        expected = "unknown"
        plain_name = variable

    if expected == "ambiguous":
        reasonableness = (
            "ambiguous because nominal revenue can rise with prices even if real demand falls"
        )
    elif direction == expected:
        reasonableness = "matches expected sign"
    else:
        reasonableness = "does not match expected sign"

    if significant_05:
        sig_text = "statistically significant at 5%"
    elif significant_10:
        sig_text = "marginally significant at 10%"
    else:
        sig_text = "not statistically significant"

    interpretation = (
        f"{plain_name}: coefficient is {direction}. "
        f"When this macro variable rises, next-quarter revenue change tends to {tendency}, "
        f"holding revenue-history controls constant. "
        f"It is {sig_text}; sign assessment: {reasonableness}."
    )

    return direction, expected, reasonableness, sig_text, interpretation


def main():
    print("=" * 80)
    print("SCRIPT 18: MACRO-REVENUE RELATIONSHIP REGRESSION")
    print("=" * 80)

    try:
        import statsmodels.formula.api as smf
    except ImportError:
        raise ImportError(
            "statsmodels is not installed.\n\n"
            "Run:\n"
            ".\\capstone_env\\Scripts\\python.exe -m pip install statsmodels"
        )

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file:\n{INPUT_PATH}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading model-ready SEC + macro dataset:")
    print(INPUT_PATH)

    df = pd.read_csv(INPUT_PATH, low_memory=False)

    print(f"\nRows loaded: {len(df):,}")
    print(f"Columns loaded: {len(df.columns):,}")

    df = add_regression_features(df)

    # ------------------------------------------------------------
    # Keep selected splits
    # ------------------------------------------------------------
    reg = df[df["model_split"].isin(USE_SPLITS)].copy()

    print("\nRegression sample splits:")
    print(reg["model_split"].value_counts().to_string())

    # ------------------------------------------------------------
    # Define variables
    # ------------------------------------------------------------
    target = "target_signed_log_change_next_qtr_w"

    macro_vars = [
        "cpi_qoq_pct_change",
        "fed_funds_rate_qoq_change",
        "unemployment_rate_qoq_change",
    ]

    revenue_history_controls = [
        "signed_log_revenue",
        "signed_log_revenue_lag_1",
        "signed_log_revenue_lag_3",
        "signed_log_revenue_lag_4",
        "signed_log_qoq_change_w",
        "signed_log_yoy_change_w",
        "revenue_growth_lag_1_w",
        "revenue_growth_lag_4_w",
        "current_to_lag_mean_ratio_w",
        "is_derived_q4",
    ]

    required_cols = (
        [target]
        + macro_vars
        + revenue_history_controls
        + ["calendar_quarter", "sic_2digit"]
    )

    missing_cols = [col for col in required_cols if col not in reg.columns]

    if missing_cols:
        raise ValueError("Missing required columns: " + ", ".join(missing_cols))

    reg = reg.dropna(subset=required_cols).copy()

    # To avoid too many rare industry dummy groups
    reg["sic_2digit"] = reg["sic_2digit"].fillna(-1).astype(int)
    industry_counts = reg["sic_2digit"].value_counts()
    common_industries = industry_counts[industry_counts >= 100].index

    reg["sic_2digit_grouped"] = np.where(
        reg["sic_2digit"].isin(common_industries),
        reg["sic_2digit"].astype(str),
        "Other",
    )

    print(f"\nRows used in regression after cleaning: {len(reg):,}")
    print(f"Unique CIKs: {reg['cik'].nunique():,}")
    print(f"Unique feature quarters: {reg['calendar_period'].nunique():,}")
    print(f"Industry groups used: {reg['sic_2digit_grouped'].nunique():,}")

    # ------------------------------------------------------------
    # Regression formulas
    # ------------------------------------------------------------
    macro_part = " + ".join(macro_vars)
    control_part = " + ".join(revenue_history_controls)

    # Model A: macro only
    formula_a = f"{target} ~ {macro_part}"

    # Model B: revenue history controls only
    formula_b = (
        f"{target} ~ {control_part} "
        f"+ C(calendar_quarter) + C(sic_2digit_grouped)"
    )

    # Model C: revenue history controls + macro variables
    formula_c = (
        f"{target} ~ {control_part} + {macro_part} "
        f"+ C(calendar_quarter) + C(sic_2digit_grouped)"
    )

    print("\nFitting Model A: Macro only...")
    model_a = smf.ols(formula_a, data=reg).fit(cov_type="HC3")

    print("Fitting Model B: Revenue history controls only...")
    model_b = smf.ols(formula_b, data=reg).fit(cov_type="HC3")

    print("Fitting Model C: Revenue history controls + macro variables...")
    model_c = smf.ols(formula_c, data=reg).fit(cov_type="HC3")

    # ------------------------------------------------------------
    # Model comparison
    # ------------------------------------------------------------
    model_rows = []

    for name, model, description in [
        ("Model A", model_a, "Macro only"),
        ("Model B", model_b, "Revenue history controls only"),
        ("Model C", model_c, "Revenue history controls + macro variables"),
    ]:
        model_rows.append(
            {
                "model": name,
                "description": description,
                "nobs": int(model.nobs),
                "r2": model.rsquared,
                "adj_r2": model.rsquared_adj,
                "aic": model.aic,
                "bic": model.bic,
            }
        )

    comparison = pd.DataFrame(model_rows)

    r2_b = comparison.loc[comparison["model"] == "Model B", "r2"].iloc[0]
    r2_c = comparison.loc[comparison["model"] == "Model C", "r2"].iloc[0]

    adj_r2_b = comparison.loc[comparison["model"] == "Model B", "adj_r2"].iloc[0]
    adj_r2_c = comparison.loc[comparison["model"] == "Model C", "adj_r2"].iloc[0]

    r2_improvement = r2_c - r2_b
    adj_r2_improvement = adj_r2_c - adj_r2_b

    comparison["r2_improvement_vs_model_b"] = np.nan
    comparison["adj_r2_improvement_vs_model_b"] = np.nan

    comparison.loc[
        comparison["model"] == "Model C",
        "r2_improvement_vs_model_b",
    ] = r2_improvement

    comparison.loc[
        comparison["model"] == "Model C",
        "adj_r2_improvement_vs_model_b",
    ] = adj_r2_improvement

    comparison.to_csv(MODEL_COMPARISON_PATH, index=False)

    # ------------------------------------------------------------
    # Macro coefficient table from Model C
    # ------------------------------------------------------------
    coef_rows = []

    target_std = reg[target].std()

    for var in macro_vars:
        coef = model_c.params.get(var, np.nan)
        std_err = model_c.bse.get(var, np.nan)
        pvalue = model_c.pvalues.get(var, np.nan)
        conf_low, conf_high = model_c.conf_int().loc[var]

        var_std = reg[var].std()
        one_std_effect = coef * var_std
        standardized_effect = one_std_effect / target_std if target_std != 0 else np.nan

        (
            direction,
            expected_sign,
            reasonableness,
            significance_text,
            interpretation,
        ) = get_sign_interpretation(var, coef, pvalue)

        coef_rows.append(
            {
                "variable": var,
                "coefficient": coef,
                "std_error_hc3": std_err,
                "p_value_hc3": pvalue,
                "conf_low_95": conf_low,
                "conf_high_95": conf_high,
                "significant_at_05": pvalue < 0.05,
                "significant_at_10": pvalue < 0.10,
                "direction": direction,
                "expected_sign": expected_sign,
                "sign_assessment": reasonableness,
                "significance_text": significance_text,
                "one_std_macro_effect_on_target": one_std_effect,
                "standardized_effect_size": standardized_effect,
                "plain_english_interpretation": interpretation,
            }
        )

    coef_df = pd.DataFrame(coef_rows)

    coef_df.to_csv(COEFFICIENTS_PATH, index=False)

    # ------------------------------------------------------------
    # Joint test: do macro variables jointly matter in Model C?
    # ------------------------------------------------------------
    try:
        joint_test = model_c.f_test(
            "cpi_qoq_pct_change = 0, "
            "fed_funds_rate_qoq_change = 0, "
            "unemployment_rate_qoq_change = 0"
        )

        joint_f_stat = float(joint_test.fvalue)
        joint_p_value = float(joint_test.pvalue)

    except Exception as e:
        joint_f_stat = np.nan
        joint_p_value = np.nan
        print(f"\nCould not compute joint F-test: {e}")

    # ------------------------------------------------------------
    # Plot macro coefficients
    # ------------------------------------------------------------
    plot_df = coef_df.copy()

    plt.figure(figsize=(10, 5))

    x = np.arange(len(plot_df))

    plt.bar(
        x,
        plot_df["coefficient"],
        yerr=[
            plot_df["coefficient"] - plot_df["conf_low_95"],
            plot_df["conf_high_95"] - plot_df["coefficient"],
        ],
        capsize=5,
    )

    plt.axhline(0, linestyle="--", linewidth=1)

    labels = [
        "CPI QoQ pct change",
        "Fed funds QoQ change",
        "Unemployment QoQ change",
    ]

    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Coefficient on signed-log revenue change")
    plt.title("Macro Coefficients After Revenue-History Controls")
    plt.tight_layout()

    plt.savefig(PLOT_PATH, dpi=300)
    plt.close()

    # ------------------------------------------------------------
    # Write notes
    # ------------------------------------------------------------
    notes = []

    notes.append("# Macro-Revenue Relationship Regression Notes")
    notes.append("")
    notes.append("## Research Question")
    notes.append("")
    notes.append(
        "Do quarterly macroeconomic variables explain next-quarter firm revenue change "
        "after controlling for each firm's recent revenue pattern?"
    )
    notes.append("")
    notes.append("## Regression Target")
    notes.append("")
    notes.append(
        "The dependent variable is the winsorized signed-log change from current revenue "
        "to next-quarter revenue:"
    )
    notes.append("")
    notes.append("`target_signed_log_change_next_qtr = target_signed_log_revenue_next_qtr - signed_log_revenue`")
    notes.append("")
    notes.append(
        "This target is used because raw revenue is highly skewed and can include zero or negative values."
    )
    notes.append("")
    notes.append("## Model Design")
    notes.append("")
    notes.append("- Model A: macro variables only")
    notes.append("- Model B: firm revenue-history controls only")
    notes.append("- Model C: revenue-history controls plus macro variables")
    notes.append("")
    notes.append("Revenue-history controls include lagged signed-log revenue, recent growth, year-over-year change, seasonality, and industry groups.")
    notes.append("")
    notes.append("## Model Comparison")
    notes.append("")
    for _, row in comparison.iterrows():
        notes.append(
            f"- {row['model']} ({row['description']}): "
            f"R² = {row['r2']:.6f}, adjusted R² = {row['adj_r2']:.6f}"
        )

    notes.append("")
    notes.append(
        f"Model C R² improvement over Model B: {r2_improvement:.8f}"
    )
    notes.append(
        f"Model C adjusted R² improvement over Model B: {adj_r2_improvement:.8f}"
    )
    notes.append("")
    notes.append("## Macro Coefficients from Model C")
    notes.append("")

    for _, row in coef_df.iterrows():
        notes.append(
            f"- `{row['variable']}`: coefficient = {row['coefficient']:.6f}, "
            f"p-value = {row['p_value_hc3']:.6f}. "
            f"{row['plain_english_interpretation']}"
        )

    notes.append("")
    notes.append("## Joint Macro Test")
    notes.append("")
    notes.append(
        f"Joint F-test p-value for all macro variables = {joint_p_value:.6f}"
    )
    notes.append("")
    notes.append("## Interpretation Warning")
    notes.append("")
    notes.append(
        "This regression identifies association, not causation. Also, macro variables are common "
        "to all firms within the same quarter, so statistical significance should be interpreted carefully."
    )

    NOTES_PATH.write_text("\n".join(notes), encoding="utf-8")

    # ------------------------------------------------------------
    # Print answers
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(comparison.to_string(index=False))

    print("\n" + "=" * 80)
    print("MACRO COEFFICIENTS AFTER REVENUE-HISTORY CONTROLS")
    print("=" * 80)

    display_coef = coef_df[
        [
            "variable",
            "coefficient",
            "std_error_hc3",
            "p_value_hc3",
            "significant_at_05",
            "direction",
            "expected_sign",
            "sign_assessment",
            "standardized_effect_size",
        ]
    ].copy()

    print(display_coef.to_string(index=False))

    print("\n" + "=" * 80)
    print("DIRECT ANSWERS")
    print("=" * 80)

    print("\n1. Do macro variables explain next-quarter revenue change?")
    print(f"   Macro-only Model A R²: {model_a.rsquared:.6f}")

    print("\n2. Do macro variables improve R² after revenue-history controls?")
    print(f"   Revenue-history Model B R²: {r2_b:.6f}")
    print(f"   Revenue-history + macro Model C R²: {r2_c:.6f}")
    print(f"   R² improvement: {r2_improvement:.8f}")
    print(f"   Adjusted R² improvement: {adj_r2_improvement:.8f}")

    if r2_improvement > 0:
        print("   Answer: Yes, macro variables improve R² slightly.")
    else:
        print("   Answer: No, macro variables do not improve R².")

    print("\n3. Are macro coefficients statistically meaningful?")
    for _, row in coef_df.iterrows():
        print(
            f"   {row['variable']}: p = {row['p_value_hc3']:.6f}, "
            f"{row['significance_text']}"
        )

    print("\n4. Are signs reasonable?")
    for _, row in coef_df.iterrows():
        print(
            f"   {row['variable']}: {row['direction']} coefficient; "
            f"{row['sign_assessment']}"
        )

    print("\n5. Do macro variables jointly matter?")
    print(f"   Joint F-test p-value: {joint_p_value:.6f}")

    if joint_p_value < 0.05:
        print("   Answer: The macro variables are jointly statistically significant at 5%.")
    elif joint_p_value < 0.10:
        print("   Answer: The macro variables are jointly marginally significant at 10%.")
    else:
        print("   Answer: The macro variables are not jointly statistically significant.")

    print("\n" + "=" * 80)
    print("OUTPUTS SAVED")
    print("=" * 80)
    print(f"Model comparison: {MODEL_COMPARISON_PATH}")
    print(f"Coefficients:     {COEFFICIENTS_PATH}")
    print(f"Coefficient plot: {PLOT_PATH}")
    print(f"Notes:            {NOTES_PATH}")

    print("\nScript 18 completed successfully.")


if __name__ == "__main__":
    main()