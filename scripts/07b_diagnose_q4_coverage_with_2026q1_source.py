from pathlib import Path
import re
import sys
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent

if SCRIPT_DIR.name == "scripts":
    PROJECT_ROOT = SCRIPT_DIR.parent
else:
    PROJECT_ROOT = SCRIPT_DIR

if len(sys.argv) > 1:
    PROJECT_ROOT = Path(sys.argv[1]).expanduser().resolve()


REVENUE_TAG_PRIORITY = {
    "RevenueFromContractWithCustomerExcludingAssessedTax": 1,
    "Revenues": 2,
    "RevenueFromContractWithCustomerIncludingAssessedTax": 3,
    "Revenue": 4,
}

REVENUE_TAGS = list(REVENUE_TAG_PRIORITY.keys())

ANNUAL_FORMS = {
    "10-K",
    "10-K/A",
    "10-KT",
}

FORM_PRIORITY_FOR_TIES = {
    "10-K/A": 1,
    "10-KT": 2,
    "10-K": 3,
}

REQUIRED_FILES = {"sub.txt", "num.txt", "tag.txt", "pre.txt"}


def find_sec_quarter_folders(project_root: Path):
    folders = []

    for folder in project_root.rglob("*"):
        if not folder.is_dir():
            continue

        file_names = {p.name.lower() for p in folder.iterdir() if p.is_file()}

        if not REQUIRED_FILES.issubset(file_names):
            continue

        match = re.search(
            r"(20\d{2})\s*[-_ ]?\s*q([1-4])",
            folder.name,
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        year = int(match.group(1))
        quarter = int(match.group(2))

        if ((2021 <= year <= 2025 and 1 <= quarter <= 4) or (year == 2026 and quarter == 1)):
            quarter_label = f"{year}Q{quarter}"
            folders.append((quarter_label, folder))

    def sort_key(item):
        label = item[0]
        match = re.fullmatch(r"(20\d{2})Q([1-4])", label)
        return int(match.group(1)), int(match.group(2))

    return sorted(folders, key=sort_key)


def read_sec_file(path: Path, usecols, dtype=None):
    return pd.read_csv(
        path,
        sep="\t",
        usecols=lambda col: col in usecols,
        dtype=dtype,
        low_memory=False,
    )


def ensure_columns(df: pd.DataFrame, columns):
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def parse_sec_date(series):
    return pd.to_datetime(
        series.astype("Int64").astype(str),
        format="%Y%m%d",
        errors="coerce",
    )


def load_annual_revenue_candidates(quarter_label: str, folder: Path):
    print(f"\nReading annual candidates from {quarter_label}: {folder}")

    num_cols = [
        "adsh",
        "tag",
        "version",
        "coreg",
        "segments",
        "ddate",
        "qtrs",
        "uom",
        "value",
    ]

    num = read_sec_file(
        folder / "num.txt",
        num_cols,
        dtype={
            "adsh": "string",
            "tag": "string",
            "version": "string",
            "coreg": "string",
            "segments": "string",
            "uom": "string",
        },
    )

    num = ensure_columns(num, num_cols)
    num = num[num["tag"].isin(REVENUE_TAGS)].copy()

    if num.empty:
        return pd.DataFrame()

    tag_cols = [
        "tag",
        "version",
        "custom",
        "abstract",
        "datatype",
    ]

    tag_meta = read_sec_file(
        folder / "tag.txt",
        tag_cols,
        dtype={
            "tag": "string",
            "version": "string",
            "datatype": "string",
        },
    )

    tag_meta = ensure_columns(tag_meta, tag_cols)

    tag_meta = (
        tag_meta[tag_meta["tag"].isin(REVENUE_TAGS)]
        .drop_duplicates(["tag", "version"])
        .copy()
    )

    pre_cols = [
        "adsh",
        "tag",
        "version",
        "stmt",
    ]

    pre = read_sec_file(
        folder / "pre.txt",
        pre_cols,
        dtype={
            "adsh": "string",
            "tag": "string",
            "version": "string",
            "stmt": "string",
        },
    )

    pre = ensure_columns(pre, pre_cols)
    pre = pre[pre["tag"].isin(REVENUE_TAGS)].copy()

    if pre.empty:
        pre_summary = pd.DataFrame(
            columns=["adsh", "tag", "version", "stmt_values", "has_is_stmt"]
        )
    else:
        pre_summary = (
            pre.groupby(["adsh", "tag", "version"], dropna=False)
            .agg(
                stmt_values=(
                    "stmt",
                    lambda x: "|".join(sorted(set(x.dropna().astype(str)))),
                ),
                has_is_stmt=(
                    "stmt",
                    lambda x: "IS" in set(x.dropna().astype(str)),
                ),
            )
            .reset_index()
        )

    sub_cols = [
        "adsh",
        "cik",
        "name",
        "sic",
        "form",
        "period",
        "fy",
        "fp",
        "filed",
        "accepted",
    ]

    sub = read_sec_file(
        folder / "sub.txt",
        sub_cols,
        dtype={
            "adsh": "string",
            "cik": "string",
            "name": "string",
            "sic": "string",
            "form": "string",
            "fp": "string",
            "accepted": "string",
        },
    )

    sub = ensure_columns(sub, sub_cols)

    df = num.merge(tag_meta, on=["tag", "version"], how="left")
    df = df.merge(pre_summary, on=["adsh", "tag", "version"], how="left")
    df = df.merge(sub, on="adsh", how="left")

    df["quarter_folder"] = quarter_label

    df["stmt_values"] = df["stmt_values"].fillna("")
    df["has_is_stmt"] = df["has_is_stmt"].fillna(False).astype(bool)

    df["qtrs"] = pd.to_numeric(df["qtrs"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["custom"] = pd.to_numeric(df["custom"], errors="coerce")
    df["abstract"] = pd.to_numeric(df["abstract"], errors="coerce")
    df["ddate"] = pd.to_numeric(df["ddate"], errors="coerce")
    df["period"] = pd.to_numeric(df["period"], errors="coerce")
    df["filed"] = pd.to_numeric(df["filed"], errors="coerce")
    df["fy"] = pd.to_numeric(df["fy"], errors="coerce")

    df["tag_priority"] = df["tag"].map(REVENUE_TAG_PRIORITY)

    return df


def apply_annual_revenue_rule(df: pd.DataFrame):
    df = df.copy()

    df["coreg_clean"] = df["coreg"].fillna("").astype(str).str.strip()
    df["segments_clean"] = df["segments"].fillna("").astype(str).str.strip()

    df["pass_annual_revenue_rule"] = (
        df["tag"].isin(REVENUE_TAGS)
        & df["has_is_stmt"]
        & df["qtrs"].eq(4)
        & df["uom"].eq("USD")
        & df["custom"].eq(0)
        & df["abstract"].eq(0)
        & df["datatype"].eq("monetary")
        & df["value"].notna()
        & df["coreg_clean"].eq("")
        & df["segments_clean"].eq("")
        & df["ddate"].eq(df["period"])
        & df["form"].isin(ANNUAL_FORMS)
    )

    selected = df[df["pass_annual_revenue_rule"]].copy()

    selected = selected.sort_values(
        ["quarter_folder", "adsh", "ddate", "tag_priority", "value"],
        ascending=[True, True, True, True, False],
    )

    selected_by_filing = selected.drop_duplicates(
        ["adsh", "ddate"],
        keep="first",
    ).copy()

    selected_by_filing["form_priority_for_ties"] = (
        selected_by_filing["form"].map(FORM_PRIORITY_FOR_TIES).fillna(99)
    )

    selected_by_filing = selected_by_filing.sort_values(
        [
            "cik",
            "period",
            "filed",
            "form_priority_for_ties",
            "tag_priority",
            "adsh",
        ],
        ascending=[
            True,
            True,
            False,
            True,
            True,
            False,
        ],
    )

    annual_final = selected_by_filing.drop_duplicates(
        ["cik", "period"],
        keep="first",
    ).copy()

    return annual_final


def clean_annual_panel(annual_raw: pd.DataFrame):
    annual = annual_raw.copy()

    annual["period_date"] = parse_sec_date(annual["period"])
    annual["filed_date"] = parse_sec_date(annual["filed"])

    annual["annual_revenue_usd"] = annual["value"]
    annual["annual_revenue_millions_usd"] = annual["annual_revenue_usd"] / 1_000_000
    annual["company_name"] = annual["name"].astype("string").str.strip()

    keep_cols = [
        "cik",
        "company_name",
        "sic",
        "fy",
        "fp",
        "period",
        "period_date",
        "form",
        "filed_date",
        "adsh",
        "quarter_folder",
        "tag",
        "tag_priority",
        "qtrs",
        "uom",
        "annual_revenue_usd",
        "annual_revenue_millions_usd",
        "accepted",
    ]

    annual = ensure_columns(annual, keep_cols)
    annual = annual[keep_cols].copy()

    annual = annual[
        (annual["period_date"] >= pd.Timestamp("2021-01-01"))
        & (annual["period_date"] <= pd.Timestamp("2025-12-31"))
    ].copy()

    annual = annual.sort_values(["cik", "fy", "period_date"]).reset_index(drop=True)

    return annual


def build_q4_diagnostic(quarterly_panel: pd.DataFrame, annual_panel: pd.DataFrame):
    q = quarterly_panel.copy()
    a = annual_panel.copy()

    q["fy"] = pd.to_numeric(q["fy"], errors="coerce")
    q["revenue_usd"] = pd.to_numeric(q["revenue_usd"], errors="coerce")

    # Count available quarterly rows by company fiscal year
    q_summary = (
        q.groupby(["cik", "fy"], dropna=False)
        .agg(
            company_name=("company_name", "first"),
            q1_available=("fp", lambda x: int("Q1" in set(x.dropna().astype(str)))),
            q2_available=("fp", lambda x: int("Q2" in set(x.dropna().astype(str)))),
            q3_available=("fp", lambda x: int("Q3" in set(x.dropna().astype(str)))),
            q4_standalone_available=("fp", lambda x: int("FY" in set(x.dropna().astype(str)))),
            q1_q3_revenue_usd=(
                "revenue_usd",
                lambda s: s[
                    q.loc[s.index, "fp"].astype(str).isin(["Q1", "Q2", "Q3"])
                ].sum(),
            ),
            q1_q3_count=(
                "fp",
                lambda x: int(x.astype(str).isin(["Q1", "Q2", "Q3"]).sum()),
            ),
            standalone_q4_revenue_usd=(
                "revenue_usd",
                lambda s: s[
                    q.loc[s.index, "fp"].astype(str).eq("FY")
                ].sum(),
            ),
        )
        .reset_index()
    )

    annual_small = a[
        [
            "cik",
            "fy",
            "period_date",
            "form",
            "filed_date",
            "adsh",
            "tag",
            "annual_revenue_usd",
            "annual_revenue_millions_usd",
        ]
    ].copy()

    annual_small = annual_small.rename(
        columns={
            "period_date": "annual_period_date",
            "form": "annual_form",
            "filed_date": "annual_filed_date",
            "adsh": "annual_adsh",
            "tag": "annual_tag",
        }
    )

    diag = q_summary.merge(
        annual_small,
        on=["cik", "fy"],
        how="outer",
    )

    diag["has_annual_revenue"] = diag["annual_revenue_usd"].notna().astype(int)

    diag["has_complete_q1_q3"] = (
        diag["q1_available"].fillna(0).eq(1)
        & diag["q2_available"].fillna(0).eq(1)
        & diag["q3_available"].fillna(0).eq(1)
    ).astype(int)

    diag["can_derive_q4"] = (
        diag["has_annual_revenue"].eq(1)
        & diag["has_complete_q1_q3"].eq(1)
    ).astype(int)

    diag["derived_q4_revenue_usd"] = (
        diag["annual_revenue_usd"] - diag["q1_q3_revenue_usd"]
    )

    diag["derived_q4_revenue_millions_usd"] = (
        diag["derived_q4_revenue_usd"] / 1_000_000
    )

    diag["q4_problem_type"] = "other"

    diag.loc[
        diag["q4_standalone_available"].fillna(0).eq(1),
        "q4_problem_type",
    ] = "standalone_q4_already_available"

    diag.loc[
        diag["q4_standalone_available"].fillna(0).eq(0)
        & diag["can_derive_q4"].eq(1),
        "q4_problem_type",
    ] = "missing_standalone_q4_but_derivable"

    diag.loc[
        diag["q4_standalone_available"].fillna(0).eq(0)
        & diag["has_annual_revenue"].eq(0),
        "q4_problem_type",
    ] = "missing_annual_revenue"

    diag.loc[
        diag["q4_standalone_available"].fillna(0).eq(0)
        & diag["has_annual_revenue"].eq(1)
        & diag["has_complete_q1_q3"].eq(0),
        "q4_problem_type",
    ] = "annual_available_but_incomplete_q1_q3"

    diag = diag.sort_values(["fy", "cik"]).reset_index(drop=True)

    return diag


def make_period_summary(quarterly_panel: pd.DataFrame):
    summary = (
        quarterly_panel.groupby("calendar_period", dropna=False)
        .agg(
            rows=("revenue_usd", "size"),
            unique_ciks=("cik", "nunique"),
            median_revenue_millions=("revenue_millions_usd", "median"),
        )
        .reset_index()
        .sort_values("calendar_period")
    )

    return summary


def make_annual_summary(annual_panel: pd.DataFrame):
    if annual_panel.empty:
        return pd.DataFrame()

    summary = (
        annual_panel.groupby("fy", dropna=False)
        .agg(
            annual_rows=("annual_revenue_usd", "size"),
            unique_ciks=("cik", "nunique"),
            median_annual_revenue_millions=("annual_revenue_millions_usd", "median"),
        )
        .reset_index()
        .sort_values("fy")
    )

    return summary


def make_q4_diagnostic_summary(q4_diag: pd.DataFrame):
    summary = (
        q4_diag.groupby(["fy", "q4_problem_type"], dropna=False)
        .agg(
            company_years=("cik", "nunique"),
        )
        .reset_index()
        .sort_values(["fy", "q4_problem_type"])
    )

    return summary


def main():
    project_root = PROJECT_ROOT

    print("=" * 80)
    print("Diagnose Q4 Coverage and Annual Revenue Availability")
    print("=" * 80)
    print(f"Project root: {project_root}")

    quarterly_path = project_root / "data" / "processed" / "sec_revenue_panel_2021_2025.csv"

    if not quarterly_path.exists():
        raise FileNotFoundError(
            f"Cannot find quarterly panel: {quarterly_path}\n"
            "Run 06_extract_sec_revenue_panel.py first."
        )

    quarterly_panel = pd.read_csv(
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
        },
        low_memory=False,
    )

    quarter_folders = find_sec_quarter_folders(project_root)

    if not quarter_folders:
        raise FileNotFoundError("No SEC quarter folders found.")

    all_annual_candidates = []

    for quarter_label, folder in quarter_folders:
        qdf = load_annual_revenue_candidates(quarter_label, folder)

        if qdf.empty:
            print(f"  {quarter_label}: no annual revenue candidates found.")
            continue

        all_annual_candidates.append(qdf)
        print(f"  {quarter_label}: annual candidate rows = {len(qdf):,}")

    if not all_annual_candidates:
        raise ValueError("No annual revenue candidates found.")

    annual_full = pd.concat(all_annual_candidates, ignore_index=True)
    annual_raw = apply_annual_revenue_rule(annual_full)
    annual_panel = clean_annual_panel(annual_raw)

    q4_diagnostic = build_q4_diagnostic(quarterly_panel, annual_panel)

    period_summary = make_period_summary(quarterly_panel)
    annual_summary = make_annual_summary(annual_panel)
    q4_diag_summary = make_q4_diagnostic_summary(q4_diagnostic)

    output_dir = project_root / "outputs" / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    annual_panel_path = output_dir / "annual_revenue_qtrs4_panel.csv"
    q4_diagnostic_path = output_dir / "q4_derivation_diagnostic_by_cik_fy.csv"
    period_summary_path = output_dir / "quarterly_panel_period_summary.csv"
    annual_summary_path = output_dir / "annual_revenue_summary_by_fy.csv"
    q4_diag_summary_path = output_dir / "q4_derivation_diagnostic_summary_by_fy.csv"

    annual_panel.to_csv(annual_panel_path, index=False)
    q4_diagnostic.to_csv(q4_diagnostic_path, index=False)
    period_summary.to_csv(period_summary_path, index=False)
    annual_summary.to_csv(annual_summary_path, index=False)
    q4_diag_summary.to_csv(q4_diag_summary_path, index=False)

    print("\n" + "=" * 80)
    print("Q4 diagnostic complete.")
    print("=" * 80)

    print("\nFiles created:")
    print(f"  {annual_panel_path}")
    print(f"  {q4_diagnostic_path}")
    print(f"  {period_summary_path}")
    print(f"  {annual_summary_path}")
    print(f"  {q4_diag_summary_path}")

    print("\nQuarterly panel period summary:")
    print(period_summary.to_string(index=False))

    print("\nAnnual qtrs=4 revenue summary by fiscal year:")
    print(annual_summary.to_string(index=False))

    print("\nQ4 diagnostic summary by fiscal year:")
    print(q4_diag_summary.to_string(index=False))

    print("\nImportant:")
    print("  This script only diagnoses Q4 coverage.")
    print("  It does not modify the main quarterly revenue panel yet.")


if __name__ == "__main__":
    main()