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


# ============================================================
# SEC Revenue Extraction Rule v1.3
# ============================================================

REVENUE_TAG_PRIORITY = {
    "RevenueFromContractWithCustomerExcludingAssessedTax": 1,
    "Revenues": 2,
    "RevenueFromContractWithCustomerIncludingAssessedTax": 3,
    "Revenue": 4,
}

REVENUE_TAGS = list(REVENUE_TAG_PRIORITY.keys())

CORE_FORMS = {
    "10-Q",
    "10-Q/A",
    "10-K",
    "10-K/A",
    "10-QT",
    "10-KT",
}

FORM_PRIORITY_FOR_TIES = {
    "10-Q/A": 1,
    "10-K/A": 1,
    "10-QT": 2,
    "10-KT": 2,
    "10-Q": 3,
    "10-K": 3,
}

REQUIRED_FILES = {"sub.txt", "num.txt", "tag.txt", "pre.txt"}


# ============================================================
# Helper functions
# ============================================================

def find_sec_quarter_folders(project_root: Path):
    """
    Find folders that contain SEC Financial Statement Data Set files:
    sub.txt, num.txt, tag.txt, pre.txt

    Only keeps folders with names like 2021Q1, 2021_Q1, 2021-Q1, etc.
    between 2021Q1 and 2025Q4.
    """
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
    """
    Read SEC txt file safely using only the columns we need.
    """
    return pd.read_csv(
        path,
        sep="\t",
        usecols=lambda col: col in usecols,
        dtype=dtype,
        low_memory=False,
    )


def ensure_columns(df: pd.DataFrame, columns):
    """
    If an optional SEC column is missing, create it as NA.
    This keeps the script stable across quarters.
    """
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def parse_sec_date(series):
    """
    Convert SEC integer date like 20210331 into pandas datetime.
    """
    return pd.to_datetime(
        series.astype("Int64").astype(str),
        format="%Y%m%d",
        errors="coerce",
    )


# ============================================================
# Load one quarter
# ============================================================

def load_quarter_revenue_candidates(quarter_label: str, folder: Path):
    print(f"\nReading {quarter_label}: {folder}")

    # ----------------------------
    # 1. Read num.txt
    # ----------------------------
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

    # Keep only approved revenue tags early for speed
    num = num[num["tag"].isin(REVENUE_TAGS)].copy()

    if num.empty:
        return pd.DataFrame()

    # ----------------------------
    # 2. Read tag.txt metadata
    # ----------------------------
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

    # ----------------------------
    # 3. Read pre.txt to identify Income Statement rows
    # ----------------------------
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

    # ----------------------------
    # 4. Read sub.txt filing metadata
    # ----------------------------
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

    # ----------------------------
    # 5. Merge SEC files
    # ----------------------------
    df = num.merge(tag_meta, on=["tag", "version"], how="left")
    df = df.merge(pre_summary, on=["adsh", "tag", "version"], how="left")
    df = df.merge(sub, on="adsh", how="left")

    df["quarter_folder"] = quarter_label

    # ----------------------------
    # 6. Clean types
    # ----------------------------
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


# ============================================================
# Apply Revenue Selection Rule v1.3
# ============================================================

def apply_revenue_rule_v13(df: pd.DataFrame):
    df = df.copy()

    df["coreg_clean"] = df["coreg"].fillna("").astype(str).str.strip()
    df["segments_clean"] = df["segments"].fillna("").astype(str).str.strip()

    # Original revenue validity rule
    df["pass_revenue_rule_v1"] = (
        df["tag"].isin(REVENUE_TAGS)
        & df["has_is_stmt"]
        & df["qtrs"].eq(1)
        & df["uom"].eq("USD")
        & df["custom"].eq(0)
        & df["abstract"].eq(0)
        & df["datatype"].eq("monetary")
        & df["value"].notna()
    )

    # Selection filters
    df["pass_selection_filters_v12"] = (
        df["pass_revenue_rule_v1"]
        & df["coreg_clean"].eq("")
        & df["segments_clean"].eq("")
        & df["ddate"].eq(df["period"])
        & df["form"].isin(CORE_FORMS)
    )

    selected = df[df["pass_selection_filters_v12"]].copy()

    # ------------------------------------------------------------
    # Step 1: same filing + same ddate
    # Choose best revenue tag by priority
    # ------------------------------------------------------------
    selected = selected.sort_values(
        ["quarter_folder", "adsh", "ddate", "tag_priority", "value"],
        ascending=[True, True, True, True, False],
    )

    selected_by_filing = selected.drop_duplicates(
        ["adsh", "ddate"],
        keep="first",
    ).copy()

    # ------------------------------------------------------------
    # Step 2: same CIK + same period
    # Choose latest filed version
    # ------------------------------------------------------------
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

    final = selected_by_filing.drop_duplicates(
        ["cik", "period"],
        keep="first",
    ).copy()

    return selected, selected_by_filing, final


# ============================================================
# Final panel cleaning
# ============================================================

def clean_final_panel(final: pd.DataFrame):
    panel = final.copy()

    # Convert dates
    panel["period_date"] = parse_sec_date(panel["period"])
    panel["ddate_date"] = parse_sec_date(panel["ddate"])
    panel["filed_date"] = parse_sec_date(panel["filed"])

    # Calendar period variables based on reporting period
    panel["calendar_year"] = panel["period_date"].dt.year
    panel["calendar_quarter"] = panel["period_date"].dt.quarter
    panel["calendar_period"] = (
        panel["calendar_year"].astype("Int64").astype(str)
        + "Q"
        + panel["calendar_quarter"].astype("Int64").astype(str)
    )

    # Revenue variables
    panel["revenue_usd"] = panel["value"]
    panel["revenue_millions_usd"] = panel["revenue_usd"] / 1_000_000

    # Clean company name
    panel["company_name"] = panel["name"].astype("string").str.strip()

    # Keep only actual panel period 2021Q1 through 2025Q4
    # Important: this uses period_date, not quarter_folder.
    panel = panel[
        (panel["period_date"] >= pd.Timestamp("2021-01-01"))
        & (panel["period_date"] <= pd.Timestamp("2025-12-31"))
    ].copy()

    # Final column order
    final_cols = [
        "cik",
        "company_name",
        "sic",
        "calendar_period",
        "calendar_year",
        "calendar_quarter",
        "period_date",
        "fy",
        "fp",
        "form",
        "filed_date",
        "adsh",
        "quarter_folder",
        "tag",
        "tag_priority",
        "stmt_values",
        "qtrs",
        "uom",
        "revenue_usd",
        "revenue_millions_usd",
        "accepted",
    ]

    panel = ensure_columns(panel, final_cols)
    panel = panel[final_cols].copy()

    # Sort final panel
    panel = panel.sort_values(
        ["cik", "period_date", "filed_date"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    return panel


def make_extraction_summary(panel: pd.DataFrame):
    if panel.empty:
        return pd.DataFrame()

    summary = (
        panel.groupby("calendar_period", dropna=False)
        .agg(
            rows=("revenue_usd", "size"),
            unique_ciks=("cik", "nunique"),
            min_revenue_millions=("revenue_millions_usd", "min"),
            median_revenue_millions=("revenue_millions_usd", "median"),
            max_revenue_millions=("revenue_millions_usd", "max"),
        )
        .reset_index()
        .sort_values("calendar_period")
    )

    return summary


def make_tag_usage_summary(panel: pd.DataFrame):
    if panel.empty:
        return pd.DataFrame()

    summary = (
        panel.groupby(["calendar_period", "tag"], dropna=False)
        .agg(
            rows=("revenue_usd", "size"),
            unique_ciks=("cik", "nunique"),
        )
        .reset_index()
        .sort_values(["calendar_period", "tag"])
    )

    return summary


def make_duplicate_check(panel: pd.DataFrame):
    if panel.empty:
        return pd.DataFrame()

    dup = (
        panel.groupby(["cik", "period_date"], dropna=False)
        .agg(
            rows=("revenue_usd", "size"),
            unique_values=("revenue_usd", "nunique"),
            min_revenue_usd=("revenue_usd", "min"),
            max_revenue_usd=("revenue_usd", "max"),
            adsh_list=("adsh", lambda x: "|".join(sorted(set(x.dropna().astype(str))))),
            forms=("form", lambda x: "|".join(sorted(set(x.dropna().astype(str))))),
        )
        .reset_index()
    )

    dup = dup[dup["rows"] > 1].copy()

    return dup


# ============================================================
# Main
# ============================================================

def main():
    project_root = PROJECT_ROOT

    print("=" * 80)
    print("Extract SEC Revenue Panel: 2021Q1 to 2025Q4")
    print("Rule: Revenue Selection Rule v1.3")
    print("=" * 80)
    print(f"Project root: {project_root}")

    quarter_folders = find_sec_quarter_folders(project_root)

    if not quarter_folders:
        raise FileNotFoundError(
            "No SEC quarter folders found. Run this script from the project root."
        )

    print(f"\nFound {len(quarter_folders)} SEC quarter folders:")

    for label, folder in quarter_folders:
        print(f"  - {label}: {folder}")

    if len(quarter_folders) != 21:
        print("\nWARNING: Expected 21 source folders from 2021Q1 to 2026Q1.")
        print("Please check if any quarter folder is missing or named differently.")

    all_candidates = []

    for quarter_label, folder in quarter_folders:
        qdf = load_quarter_revenue_candidates(quarter_label, folder)

        if qdf.empty:
            print(f"  {quarter_label}: no approved revenue-tag rows found.")
            continue

        all_candidates.append(qdf)

        print(f"  {quarter_label}: candidate rows = {len(qdf):,}")

    if not all_candidates:
        raise ValueError("No revenue candidates found in any SEC quarter folder.")

    full = pd.concat(all_candidates, ignore_index=True)

    print("\nApplying Revenue Selection Rule v1.3...")

    selected_raw, selected_by_filing, final_raw = apply_revenue_rule_v13(full)
    panel = clean_final_panel(final_raw)

    processed_dir = project_root / "data" / "processed"
    validation_dir = project_root / "outputs" / "diagnostics"

    processed_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)

    panel_path = processed_dir / "sec_revenue_panel_2021_2025.csv"
    summary_path = processed_dir / "sec_revenue_panel_2021_2025_summary_by_period.csv"
    tag_summary_path = processed_dir / "sec_revenue_panel_2021_2025_tag_usage.csv"
    duplicate_check_path = validation_dir / "sec_revenue_panel_duplicate_check.csv"

    summary = make_extraction_summary(panel)
    tag_summary = make_tag_usage_summary(panel)
    duplicate_check = make_duplicate_check(panel)

    panel.to_csv(panel_path, index=False)
    summary.to_csv(summary_path, index=False)
    tag_summary.to_csv(tag_summary_path, index=False)
    duplicate_check.to_csv(duplicate_check_path, index=False)

    print("\n" + "=" * 80)
    print("Extraction complete.")
    print("=" * 80)

    print("\nMain output:")
    print(f"  {panel_path}")

    print("\nAdditional outputs:")
    print(f"  {summary_path}")
    print(f"  {tag_summary_path}")
    print(f"  {duplicate_check_path}")

    print("\nRow counts:")
    print(f"  Raw approved-tag candidates: {len(full):,}")
    print(f"  Rows after v1.2 filters: {len(selected_raw):,}")
    print(f"  Rows after one row per filing/date: {len(selected_by_filing):,}")
    print(f"  Rows after latest filing per CIK/period: {len(final_raw):,}")
    print(f"  Final panel rows after 2021-2025 period filter: {len(panel):,}")
    print(f"  Final unique CIKs: {panel['cik'].nunique():,}")

    print("\nDuplicate check:")
    if duplicate_check.empty:
        print("  Good: no duplicate CIK + period_date rows in final panel.")
    else:
        print(f"  WARNING: {len(duplicate_check):,} duplicate CIK-period rows remain.")
        print("  Inspect sec_revenue_panel_duplicate_check.csv")

    print("\nSummary by calendar period:")
    if summary.empty:
        print("  No summary available.")
    else:
        print(summary.to_string(index=False))

    print("\nNote:")
    print("  The final panel uses period_date/calendar_period as the time variable.")
    print("  quarter_folder only tells us which SEC download folder the filing came from.")


if __name__ == "__main__":
    main()