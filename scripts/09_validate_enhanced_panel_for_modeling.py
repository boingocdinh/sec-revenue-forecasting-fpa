from pathlib import Path
import sys
import pandas as pd
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent

if SCRIPT_DIR.name == "scripts":
    PROJECT_ROOT = SCRIPT_DIR.parent
else:
    PROJECT_ROOT = SCRIPT_DIR

if len(sys.argv) > 1:
    PROJECT_ROOT = Path(sys.argv[1]).expanduser().resolve()


REQUIRED_INPUT = "sec_revenue_panel_2021_2025_q4_enhanced.csv"


def load_panel(project_root: Path):
    panel_path = project_root / "data" / "processed" / REQUIRED_INPUT

    if not panel_path.exists():
        raise FileNotFoundError(
            f"Missing enhanced panel:\n{panel_path}\n"
            "Run 08_build_revenue_panel_with_derived_q4.py first."
        )

    panel = pd.read_csv(
        panel_path,
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

    return panel, panel_path


def clean_types(panel: pd.DataFrame):
    df = panel.copy()

    date_cols = ["period_date", "filed_date"]

    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    numeric_cols = [
        "calendar_year",
        "calendar_quarter",
        "fy",
        "tag_priority",
        "qtrs",
        "revenue_usd",
        "revenue_millions_usd",
        "is_derived_q4",
        "annual_revenue_usd",
        "q1_q3_revenue_usd",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def make_missing_summary(df: pd.DataFrame):
    rows = []

    for col in df.columns:
        missing = int(df[col].isna().sum())
        missing_pct = missing / len(df) if len(df) else np.nan

        rows.append(
            {
                "column": col,
                "missing_count": missing,
                "missing_pct": missing_pct,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["missing_count", "column"],
        ascending=[False, True],
    )


def make_period_summary(df: pd.DataFrame):
    summary = (
        df.groupby("calendar_period", dropna=False)
        .agg(
            rows=("revenue_usd", "size"),
            unique_ciks=("cik", "nunique"),
            reported_rows=("is_derived_q4", lambda x: int((x == 0).sum())),
            derived_q4_rows=("is_derived_q4", lambda x: int((x == 1).sum())),
            negative_revenue_rows=("revenue_usd", lambda x: int((x < 0).sum())),
            zero_revenue_rows=("revenue_usd", lambda x: int((x == 0).sum())),
            missing_revenue_rows=("revenue_usd", lambda x: int(x.isna().sum())),
            min_revenue_millions=("revenue_millions_usd", "min"),
            median_revenue_millions=("revenue_millions_usd", "median"),
            mean_revenue_millions=("revenue_millions_usd", "mean"),
            max_revenue_millions=("revenue_millions_usd", "max"),
        )
        .reset_index()
        .sort_values("calendar_period")
    )

    return summary


def make_duplicate_check(df: pd.DataFrame):
    dup = (
        df.groupby(["cik", "period_date"], dropna=False)
        .agg(
            rows=("revenue_usd", "size"),
            unique_values=("revenue_usd", "nunique"),
            sources=("revenue_source", lambda x: "|".join(sorted(set(x.dropna().astype(str))))),
            forms=("form", lambda x: "|".join(sorted(set(x.dropna().astype(str))))),
            min_revenue_usd=("revenue_usd", "min"),
            max_revenue_usd=("revenue_usd", "max"),
        )
        .reset_index()
    )

    dup = dup[dup["rows"] > 1].copy()

    return dup


def make_firm_history_summary(df: pd.DataFrame):
    firm_summary = (
        df.groupby("cik", dropna=False)
        .agg(
            company_name=("company_name", "last"),
            sic=("sic", "last"),
            observations=("revenue_usd", "size"),
            first_period_date=("period_date", "min"),
            last_period_date=("period_date", "max"),
            first_calendar_period=("calendar_period", "first"),
            last_calendar_period=("calendar_period", "last"),
            reported_rows=("is_derived_q4", lambda x: int((x == 0).sum())),
            derived_q4_rows=("is_derived_q4", lambda x: int((x == 1).sum())),
            negative_revenue_rows=("revenue_usd", lambda x: int((x < 0).sum())),
            zero_revenue_rows=("revenue_usd", lambda x: int((x == 0).sum())),
            min_revenue_usd=("revenue_usd", "min"),
            median_revenue_usd=("revenue_usd", "median"),
            max_revenue_usd=("revenue_usd", "max"),
        )
        .reset_index()
    )

    firm_summary["has_at_least_4_obs"] = firm_summary["observations"] >= 4
    firm_summary["has_at_least_8_obs"] = firm_summary["observations"] >= 8
    firm_summary["has_at_least_12_obs"] = firm_summary["observations"] >= 12
    firm_summary["has_at_least_16_obs"] = firm_summary["observations"] >= 16

    return firm_summary.sort_values(
        ["observations", "cik"],
        ascending=[False, True],
    )


def make_firm_history_distribution(firm_summary: pd.DataFrame):
    bins = [
        0,
        1,
        3,
        7,
        11,
        15,
        20,
        1000,
    ]

    labels = [
        "1 observation",
        "2-3 observations",
        "4-7 observations",
        "8-11 observations",
        "12-15 observations",
        "16-20 observations",
        "21+ observations",
    ]

    firm_summary = firm_summary.copy()

    firm_summary["history_bucket"] = pd.cut(
        firm_summary["observations"],
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True,
    )

    dist = (
        firm_summary.groupby("history_bucket", observed=False)
        .agg(
            firms=("cik", "nunique"),
            median_observations=("observations", "median"),
        )
        .reset_index()
    )

    return dist


def make_negative_zero_detail(df: pd.DataFrame):
    detail = df[
        df["revenue_usd"].isna()
        | df["revenue_usd"].lt(0)
        | df["revenue_usd"].eq(0)
    ].copy()

    keep_cols = [
        "cik",
        "company_name",
        "sic",
        "calendar_period",
        "period_date",
        "fy",
        "fp",
        "form",
        "filed_date",
        "tag",
        "revenue_usd",
        "revenue_millions_usd",
        "is_derived_q4",
        "revenue_source",
        "annual_revenue_usd",
        "q1_q3_revenue_usd",
        "adsh",
    ]

    for col in keep_cols:
        if col not in detail.columns:
            detail[col] = pd.NA

    return detail[keep_cols].sort_values(
        ["calendar_period", "revenue_usd"],
        ascending=[True, True],
    )


def make_large_outlier_detail(df: pd.DataFrame):
    valid = df[df["revenue_usd"].notna()].copy()

    if valid.empty:
        return pd.DataFrame()

    q1 = valid["revenue_usd"].quantile(0.25)
    q3 = valid["revenue_usd"].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 3 * iqr
    upper_bound = q3 + 3 * iqr

    outliers = valid[
        (valid["revenue_usd"] < lower_bound)
        | (valid["revenue_usd"] > upper_bound)
    ].copy()

    outliers["iqr_lower_bound"] = lower_bound
    outliers["iqr_upper_bound"] = upper_bound

    keep_cols = [
        "cik",
        "company_name",
        "sic",
        "calendar_period",
        "period_date",
        "fy",
        "fp",
        "form",
        "filed_date",
        "tag",
        "revenue_usd",
        "revenue_millions_usd",
        "is_derived_q4",
        "revenue_source",
        "adsh",
        "iqr_lower_bound",
        "iqr_upper_bound",
    ]

    for col in keep_cols:
        if col not in outliers.columns:
            outliers[col] = pd.NA

    return outliers[keep_cols].sort_values(
        "revenue_usd",
        ascending=False,
    )


def make_modeling_readiness_summary(df: pd.DataFrame, firm_summary: pd.DataFrame):
    total_rows = len(df)
    total_firms = df["cik"].nunique()

    duplicate_check = make_duplicate_check(df)

    rows = [
        {
            "metric": "total_rows",
            "value": total_rows,
        },
        {
            "metric": "unique_ciks",
            "value": total_firms,
        },
        {
            "metric": "duplicate_cik_period_rows",
            "value": len(duplicate_check),
        },
        {
            "metric": "missing_revenue_rows",
            "value": int(df["revenue_usd"].isna().sum()),
        },
        {
            "metric": "negative_revenue_rows",
            "value": int((df["revenue_usd"] < 0).sum()),
        },
        {
            "metric": "zero_revenue_rows",
            "value": int((df["revenue_usd"] == 0).sum()),
        },
        {
            "metric": "derived_q4_rows",
            "value": int((df["is_derived_q4"] == 1).sum()),
        },
        {
            "metric": "reported_rows",
            "value": int((df["is_derived_q4"] == 0).sum()),
        },
        {
            "metric": "firms_with_at_least_4_obs",
            "value": int(firm_summary["has_at_least_4_obs"].sum()),
        },
        {
            "metric": "firms_with_at_least_8_obs",
            "value": int(firm_summary["has_at_least_8_obs"].sum()),
        },
        {
            "metric": "firms_with_at_least_12_obs",
            "value": int(firm_summary["has_at_least_12_obs"].sum()),
        },
        {
            "metric": "firms_with_at_least_16_obs",
            "value": int(firm_summary["has_at_least_16_obs"].sum()),
        },
    ]

    return pd.DataFrame(rows)


def make_train_val_test_coverage(df: pd.DataFrame):
    temp = df.copy()

    temp["split"] = pd.NA

    temp.loc[
        temp["calendar_period"].between("2021Q1", "2023Q4"),
        "split",
    ] = "train_2021_2023"

    temp.loc[
        temp["calendar_period"].between("2024Q1", "2024Q4"),
        "split",
    ] = "validation_2024"

    temp.loc[
        temp["calendar_period"].between("2025Q1", "2025Q4"),
        "split",
    ] = "test_2025"

    coverage = (
        temp.groupby("split", dropna=False)
        .agg(
            rows=("revenue_usd", "size"),
            unique_ciks=("cik", "nunique"),
            reported_rows=("is_derived_q4", lambda x: int((x == 0).sum())),
            derived_q4_rows=("is_derived_q4", lambda x: int((x == 1).sum())),
            negative_revenue_rows=("revenue_usd", lambda x: int((x < 0).sum())),
            zero_revenue_rows=("revenue_usd", lambda x: int((x == 0).sum())),
        )
        .reset_index()
    )

    return coverage


def main():
    project_root = PROJECT_ROOT

    print("=" * 80)
    print("Validate Enhanced Revenue Panel for Modeling")
    print("=" * 80)
    print(f"Project root: {project_root}")

    panel, panel_path = load_panel(project_root)
    df = clean_types(panel)

    output_dir = project_root / "outputs" / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    missing_summary = make_missing_summary(df)
    period_summary = make_period_summary(df)
    duplicate_check = make_duplicate_check(df)
    firm_summary = make_firm_history_summary(df)
    firm_dist = make_firm_history_distribution(firm_summary)
    neg_zero_detail = make_negative_zero_detail(df)
    outlier_detail = make_large_outlier_detail(df)
    readiness = make_modeling_readiness_summary(df, firm_summary)
    split_coverage = make_train_val_test_coverage(df)

    missing_summary.to_csv(output_dir / "modeling_missing_value_summary.csv", index=False)
    period_summary.to_csv(output_dir / "modeling_period_summary.csv", index=False)
    duplicate_check.to_csv(output_dir / "modeling_duplicate_check.csv", index=False)
    firm_summary.to_csv(output_dir / "modeling_firm_history_summary.csv", index=False)
    firm_dist.to_csv(output_dir / "modeling_firm_history_distribution.csv", index=False)
    neg_zero_detail.to_csv(output_dir / "modeling_negative_zero_revenue_detail.csv", index=False)
    outlier_detail.to_csv(output_dir / "modeling_large_outlier_detail.csv", index=False)
    readiness.to_csv(output_dir / "modeling_readiness_summary.csv", index=False)
    split_coverage.to_csv(output_dir / "modeling_train_validation_test_coverage.csv", index=False)

    print("\nInput panel:")
    print(f"  {panel_path}")

    print("\nOutput folder:")
    print(f"  {output_dir}")

    print("\nFiles created:")
    print("  1. modeling_missing_value_summary.csv")
    print("  2. modeling_period_summary.csv")
    print("  3. modeling_duplicate_check.csv")
    print("  4. modeling_firm_history_summary.csv")
    print("  5. modeling_firm_history_distribution.csv")
    print("  6. modeling_negative_zero_revenue_detail.csv")
    print("  7. modeling_large_outlier_detail.csv")
    print("  8. modeling_readiness_summary.csv")
    print("  9. modeling_train_validation_test_coverage.csv")

    print("\nModeling readiness summary:")
    print(readiness.to_string(index=False))

    print("\nTrain / validation / test coverage:")
    print(split_coverage.to_string(index=False))

    print("\nFirm history distribution:")
    print(firm_dist.to_string(index=False))

    print("\nPeriod summary:")
    print(period_summary.to_string(index=False))

    print("\nDuplicate check:")
    if duplicate_check.empty:
        print("  Good: no duplicate CIK + period_date rows.")
    else:
        print(f"  WARNING: {len(duplicate_check):,} duplicate CIK-period rows remain.")

    print("\nImportant:")
    print("  This script only validates the panel for modeling.")
    print("  It does not remove negative, zero, or outlier values yet.")


if __name__ == "__main__":
    main()