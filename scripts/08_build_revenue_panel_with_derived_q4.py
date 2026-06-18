from pathlib import Path
import sys
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent

if SCRIPT_DIR.name == "scripts":
    PROJECT_ROOT = SCRIPT_DIR.parent
else:
    PROJECT_ROOT = SCRIPT_DIR

if len(sys.argv) > 1:
    PROJECT_ROOT = Path(sys.argv[1]).expanduser().resolve()


def parse_date_column(series):
    return pd.to_datetime(series, errors="coerce")


def make_calendar_period(date_series):
    year = date_series.dt.year.astype("Int64").astype(str)
    quarter = date_series.dt.quarter.astype("Int64").astype(str)
    return year + "Q" + quarter


def load_inputs(project_root: Path):
    quarterly_path = project_root / "data" / "processed" / "sec_revenue_panel_2021_2025.csv"

    q4_diag_path = (
        project_root
        / "outputs"
        / "diagnostics"
        / "q4_derivation_diagnostic_by_cik_fy.csv"
    )

    if not quarterly_path.exists():
        raise FileNotFoundError(
            f"Missing quarterly panel:\n{quarterly_path}\n"
            "Run 06b_extract_sec_revenue_panel_with_2026q1_source.py first."
        )

    if not q4_diag_path.exists():
        raise FileNotFoundError(
            f"Missing Q4 diagnostic file:\n{q4_diag_path}\n"
            "Run 07b_diagnose_q4_coverage_with_2026q1_source.py first."
        )

    quarterly = pd.read_csv(
        quarterly_path,
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
        },
        low_memory=False,
    )

    q4_diag = pd.read_csv(
        q4_diag_path,
        dtype={
            "cik": "string",
            "company_name": "string",
            "annual_form": "string",
            "annual_adsh": "string",
            "annual_tag": "string",
            "q4_problem_type": "string",
        },
        low_memory=False,
    )

    return quarterly, q4_diag


def prepare_reported_panel(quarterly: pd.DataFrame):
    reported = quarterly.copy()

    reported["period_date"] = parse_date_column(reported["period_date"])
    reported["filed_date"] = parse_date_column(reported["filed_date"])

    reported["fy"] = pd.to_numeric(reported["fy"], errors="coerce")
    reported["calendar_year"] = pd.to_numeric(reported["calendar_year"], errors="coerce")
    reported["calendar_quarter"] = pd.to_numeric(reported["calendar_quarter"], errors="coerce")
    reported["tag_priority"] = pd.to_numeric(reported["tag_priority"], errors="coerce")
    reported["qtrs"] = pd.to_numeric(reported["qtrs"], errors="coerce")
    reported["revenue_usd"] = pd.to_numeric(reported["revenue_usd"], errors="coerce")
    reported["revenue_millions_usd"] = pd.to_numeric(
        reported["revenue_millions_usd"],
        errors="coerce",
    )

    reported["is_derived_q4"] = 0
    reported["revenue_source"] = "reported_sec_qtrs1"
    reported["derivation_method"] = ""
    reported["annual_revenue_usd"] = pd.NA
    reported["q1_q3_revenue_usd"] = pd.NA

    return reported


def build_derived_q4_rows(q4_diag: pd.DataFrame, reported: pd.DataFrame):
    diag = q4_diag.copy()

    numeric_cols = [
        "fy",
        "q1_available",
        "q2_available",
        "q3_available",
        "q4_standalone_available",
        "q1_q3_count",
        "q1_q3_revenue_usd",
        "standalone_q4_revenue_usd",
        "annual_revenue_usd",
        "annual_revenue_millions_usd",
        "has_annual_revenue",
        "has_complete_q1_q3",
        "can_derive_q4",
        "derived_q4_revenue_usd",
        "derived_q4_revenue_millions_usd",
    ]

    for col in numeric_cols:
        if col in diag.columns:
            diag[col] = pd.to_numeric(diag[col], errors="coerce")

    diag["annual_period_date"] = parse_date_column(diag["annual_period_date"])
    diag["annual_filed_date"] = parse_date_column(diag["annual_filed_date"])

    eligible = diag[
        diag["q4_problem_type"].eq("missing_standalone_q4_but_derivable")
        & diag["can_derive_q4"].eq(1)
        & diag["derived_q4_revenue_usd"].notna()
        & diag["annual_period_date"].notna()
    ].copy()

    eligible = eligible[
        (eligible["annual_period_date"] >= pd.Timestamp("2021-01-01"))
        & (eligible["annual_period_date"] <= pd.Timestamp("2025-12-31"))
    ].copy()

    if eligible.empty:
        return pd.DataFrame(columns=reported.columns)

    cik_sic = (
        reported.dropna(subset=["sic"])
        .sort_values(["cik", "period_date"])
        .drop_duplicates("cik", keep="last")[["cik", "sic"]]
    )

    eligible = eligible.merge(cik_sic, on="cik", how="left")

    derived = pd.DataFrame()

    derived["cik"] = eligible["cik"]
    derived["company_name"] = eligible["company_name"].astype("string").str.strip()
    derived["sic"] = eligible["sic"]

    derived["period_date"] = eligible["annual_period_date"]
    derived["calendar_year"] = derived["period_date"].dt.year
    derived["calendar_quarter"] = derived["period_date"].dt.quarter
    derived["calendar_period"] = make_calendar_period(derived["period_date"])

    derived["fy"] = eligible["fy"]
    derived["fp"] = "FY_DERIVED_Q4"

    derived["form"] = eligible["annual_form"]
    derived["filed_date"] = eligible["annual_filed_date"]
    derived["adsh"] = eligible["annual_adsh"]
    derived["quarter_folder"] = pd.NA

    derived["tag"] = eligible["annual_tag"]
    derived["tag_priority"] = pd.NA
    derived["stmt_values"] = "IS"
    derived["qtrs"] = 1
    derived["uom"] = "USD"

    derived["revenue_usd"] = eligible["derived_q4_revenue_usd"]
    derived["revenue_millions_usd"] = derived["revenue_usd"] / 1_000_000

    derived["accepted"] = pd.NA

    derived["is_derived_q4"] = 1
    derived["revenue_source"] = "derived_q4_from_annual_minus_q1_q3"
    derived["derivation_method"] = "annual_qtrs4_revenue_minus_reported_q1_q2_q3"
    derived["annual_revenue_usd"] = eligible["annual_revenue_usd"]
    derived["q1_q3_revenue_usd"] = eligible["q1_q3_revenue_usd"]

    for col in reported.columns:
        if col not in derived.columns:
            derived[col] = pd.NA

    derived = derived[reported.columns].copy()

    return derived


def combine_and_validate(reported: pd.DataFrame, derived: pd.DataFrame):
    combined = pd.concat([reported, derived], ignore_index=True)

    combined["period_date"] = parse_date_column(combined["period_date"])
    combined["filed_date"] = parse_date_column(combined["filed_date"])
    combined["revenue_usd"] = pd.to_numeric(combined["revenue_usd"], errors="coerce")
    combined["revenue_millions_usd"] = combined["revenue_usd"] / 1_000_000

    combined = combined.sort_values(
        ["cik", "period_date", "is_derived_q4", "filed_date"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)

    duplicate_check = (
        combined.groupby(["cik", "period_date"], dropna=False)
        .agg(
            rows=("revenue_usd", "size"),
            unique_values=("revenue_usd", "nunique"),
            sources=("revenue_source", lambda x: "|".join(sorted(set(x.dropna().astype(str))))),
            min_revenue_usd=("revenue_usd", "min"),
            max_revenue_usd=("revenue_usd", "max"),
        )
        .reset_index()
    )

    duplicate_check = duplicate_check[duplicate_check["rows"] > 1].copy()

    return combined, duplicate_check


def make_period_summary(panel: pd.DataFrame):
    summary = (
        panel.groupby("calendar_period", dropna=False)
        .agg(
            rows=("revenue_usd", "size"),
            unique_ciks=("cik", "nunique"),
            reported_rows=("is_derived_q4", lambda x: int((x == 0).sum())),
            derived_q4_rows=("is_derived_q4", lambda x: int((x == 1).sum())),
            min_revenue_millions=("revenue_millions_usd", "min"),
            median_revenue_millions=("revenue_millions_usd", "median"),
            max_revenue_millions=("revenue_millions_usd", "max"),
        )
        .reset_index()
        .sort_values("calendar_period")
    )

    return summary


def make_source_summary(panel: pd.DataFrame):
    summary = (
        panel.groupby(["calendar_period", "revenue_source"], dropna=False)
        .agg(
            rows=("revenue_usd", "size"),
            unique_ciks=("cik", "nunique"),
        )
        .reset_index()
        .sort_values(["calendar_period", "revenue_source"])
    )

    return summary


def make_derived_quality_summary(derived: pd.DataFrame):
    if derived.empty:
        return pd.DataFrame()

    d = derived.copy()
    d["derived_q4_negative"] = d["revenue_usd"] < 0
    d["derived_q4_zero"] = d["revenue_usd"] == 0

    summary = (
        d.groupby("calendar_period", dropna=False)
        .agg(
            derived_rows=("revenue_usd", "size"),
            negative_derived_q4=("derived_q4_negative", "sum"),
            zero_derived_q4=("derived_q4_zero", "sum"),
            min_derived_q4_millions=("revenue_millions_usd", "min"),
            median_derived_q4_millions=("revenue_millions_usd", "median"),
            max_derived_q4_millions=("revenue_millions_usd", "max"),
        )
        .reset_index()
        .sort_values("calendar_period")
    )

    return summary


def main():
    project_root = PROJECT_ROOT

    print("=" * 80)
    print("Build SEC Revenue Panel with Derived Q4 Rows")
    print("=" * 80)
    print(f"Project root: {project_root}")

    quarterly, q4_diag = load_inputs(project_root)

    reported = prepare_reported_panel(quarterly)
    derived = build_derived_q4_rows(q4_diag, reported)

    combined, duplicate_check = combine_and_validate(reported, derived)

    processed_dir = project_root / "data" / "processed"
    diagnostic_dir = project_root / "outputs" / "diagnostics"

    processed_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_dir.mkdir(parents=True, exist_ok=True)

    enhanced_path = processed_dir / "sec_revenue_panel_2021_2025_q4_enhanced.csv"
    period_summary_path = processed_dir / "sec_revenue_panel_2021_2025_q4_enhanced_summary_by_period.csv"
    source_summary_path = processed_dir / "sec_revenue_panel_2021_2025_q4_enhanced_source_summary.csv"

    derived_rows_path = diagnostic_dir / "derived_q4_rows_added.csv"
    duplicate_check_path = diagnostic_dir / "q4_enhanced_duplicate_check.csv"
    derived_quality_path = diagnostic_dir / "derived_q4_quality_summary.csv"

    period_summary = make_period_summary(combined)
    source_summary = make_source_summary(combined)
    derived_quality = make_derived_quality_summary(derived)

    combined.to_csv(enhanced_path, index=False)
    period_summary.to_csv(period_summary_path, index=False)
    source_summary.to_csv(source_summary_path, index=False)

    derived.to_csv(derived_rows_path, index=False)
    duplicate_check.to_csv(duplicate_check_path, index=False)
    derived_quality.to_csv(derived_quality_path, index=False)

    print("\n" + "=" * 80)
    print("Enhanced panel complete.")
    print("=" * 80)

    print("\nMain output:")
    print(f"  {enhanced_path}")

    print("\nAdditional outputs:")
    print(f"  {period_summary_path}")
    print(f"  {source_summary_path}")
    print(f"  {derived_rows_path}")
    print(f"  {duplicate_check_path}")
    print(f"  {derived_quality_path}")

    print("\nRow counts:")
    print(f"  Original reported panel rows: {len(reported):,}")
    print(f"  Derived Q4 rows added: {len(derived):,}")
    print(f"  Enhanced panel rows: {len(combined):,}")
    print(f"  Enhanced unique CIKs: {combined['cik'].nunique():,}")

    print("\nDuplicate check:")
    if duplicate_check.empty:
        print("  Good: no duplicate CIK + period_date rows after adding derived Q4.")
    else:
        print(f"  WARNING: {len(duplicate_check):,} duplicate CIK-period rows remain.")
        print("  Inspect q4_enhanced_duplicate_check.csv")

    print("\nEnhanced summary by calendar period:")
    print(period_summary.to_string(index=False))

    print("\nDerived Q4 quality summary:")
    if derived_quality.empty:
        print("  No derived rows were created.")
    else:
        print(derived_quality.to_string(index=False))

    print("\nNote:")
    print("  is_derived_q4 = 0 means directly reported SEC quarterly revenue.")
    print("  is_derived_q4 = 1 means Q4 revenue derived from annual revenue minus Q1-Q3.")


if __name__ == "__main__":
    main()