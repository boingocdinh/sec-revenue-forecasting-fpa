from pathlib import Path
import sys
import pandas as pd
import numpy as np


# ============================================================
# Script 12: Build Quarterly FRED Macro Dataset from Local CSVs
# No API key required. No internet required after manual download.
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

if SCRIPT_DIR.name == "scripts":
    PROJECT_ROOT = SCRIPT_DIR.parent
else:
    PROJECT_ROOT = SCRIPT_DIR

if len(sys.argv) > 1:
    PROJECT_ROOT = Path(sys.argv[1]).expanduser().resolve()

FRED_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "fred"

# Backward-compatible fallback to the legacy folder.
if not FRED_RAW_DIR.exists():
    _legacy_fred = PROJECT_ROOT / "data_raw" / "fred"
    if _legacy_fred.exists():
        print(
            f"WARNING: using legacy FRED path {_legacy_fred}.\n"
            f"         Please move files to {FRED_RAW_DIR}."
        )
        FRED_RAW_DIR = _legacy_fred

RAW_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "macro_raw.csv"
QUARTERLY_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "macro_quarterly.csv"

START_DATE = "2021-01-01"
END_DATE = "2025-12-31"

SERIES_FILES = {
    "CPIAUCSL": {
        "filename": "CPIAUCSL.csv",
        "final_name": "cpi",
    },
    "FEDFUNDS": {
        "filename": "FEDFUNDS.csv",
        "final_name": "fed_funds_rate",
    },
    "UNRATE": {
        "filename": "UNRATE.csv",
        "final_name": "unemployment_rate",
    },
}


def read_fred_csv(series_id, filename):
    path = FRED_RAW_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file:\n{path}\n\n"
            f"Please download {series_id} from FRED and save it as {filename}."
        )

    df = pd.read_csv(path)

    df.columns = [col.strip() for col in df.columns]

    # FRED CSV can use either observation_date or DATE
    if "observation_date" in df.columns:
        date_col = "observation_date"
    elif "DATE" in df.columns:
        date_col = "DATE"
    else:
        raise ValueError(
            f"Could not find date column in {filename}. "
            f"Columns found: {list(df.columns)}"
        )

    if series_id in df.columns:
        value_col = series_id
    else:
        possible_value_cols = [col for col in df.columns if col != date_col]

        if len(possible_value_cols) != 1:
            raise ValueError(
                f"Could not identify value column in {filename}. "
                f"Columns found: {list(df.columns)}"
            )

        value_col = possible_value_cols[0]

    out = df[[date_col, value_col]].copy()
    out = out.rename(columns={date_col: "date", value_col: "value"})

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out["series_id"] = series_id

    out = out[
        (out["date"] >= pd.Timestamp(START_DATE))
        & (out["date"] <= pd.Timestamp(END_DATE))
    ].copy()

    out = out.dropna(subset=["date", "value"])
    out = out[["date", "series_id", "value"]]
    out = out.sort_values("date").reset_index(drop=True)

    return out


def make_calendar_period(date_series):
    return date_series.dt.year.astype(str) + "Q" + date_series.dt.quarter.astype(str)


def main():
    print("=" * 80)
    print("SCRIPT 12: BUILD QUARTERLY FRED MACRO DATASET FROM LOCAL CSVs")
    print("=" * 80)

    (PROJECT_ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)

    all_series = []

    print("\nReading local FRED files from:")
    print(FRED_RAW_DIR)

    for series_id, info in SERIES_FILES.items():
        df = read_fred_csv(series_id, info["filename"])
        all_series.append(df)

        print(
            f"  {series_id}: {len(df):,} observations "
            f"from {df['date'].min().date()} to {df['date'].max().date()}"
        )

    raw = pd.concat(all_series, ignore_index=True)
    raw = raw.sort_values(["series_id", "date"]).reset_index(drop=True)

    raw.to_csv(RAW_OUTPUT_PATH, index=False)

    print("\nRaw macro file saved:")
    print(RAW_OUTPUT_PATH)

    raw["calendar_period"] = make_calendar_period(raw["date"])

    quarterly_long = (
        raw.groupby(["calendar_period", "series_id"], as_index=False)
        .agg(
            quarterly_mean=("value", "mean"),
            monthly_obs=("value", "count"),
            first_month=("date", "min"),
            last_month=("date", "max"),
        )
    )

    quarterly_wide = (
        quarterly_long
        .pivot(index="calendar_period", columns="series_id", values="quarterly_mean")
        .reset_index()
    )

    rename_map = {
        series_id: info["final_name"]
        for series_id, info in SERIES_FILES.items()
    }

    quarterly_wide = quarterly_wide.rename(columns=rename_map)

    extracted = quarterly_wide["calendar_period"].str.extract(
        r"(?P<calendar_year>\d{4})Q(?P<calendar_quarter>[1-4])"
    )

    quarterly_wide["calendar_year"] = pd.to_numeric(
        extracted["calendar_year"],
        errors="coerce"
    ).astype(int)

    quarterly_wide["calendar_quarter"] = pd.to_numeric(
        extracted["calendar_quarter"],
        errors="coerce"
    ).astype(int)

    quarterly_wide["quarter_index"] = (
        quarterly_wide["calendar_year"] * 4
        + quarterly_wide["calendar_quarter"]
    )

    quarterly_wide = quarterly_wide.sort_values("quarter_index").reset_index(drop=True)

    quarterly_wide = quarterly_wide[
        [
            "calendar_period",
            "calendar_year",
            "calendar_quarter",
            "quarter_index",
            "cpi",
            "fed_funds_rate",
            "unemployment_rate",
        ]
    ]

    # Macro change features
    quarterly_wide["cpi_qoq_change"] = quarterly_wide["cpi"].diff()
    quarterly_wide["cpi_qoq_pct_change"] = quarterly_wide["cpi"].pct_change()
    quarterly_wide["fed_funds_rate_qoq_change"] = quarterly_wide["fed_funds_rate"].diff()
    quarterly_wide["unemployment_rate_qoq_change"] = quarterly_wide["unemployment_rate"].diff()

    change_cols = [
        "cpi_qoq_change",
        "cpi_qoq_pct_change",
        "fed_funds_rate_qoq_change",
        "unemployment_rate_qoq_change",
    ]

    quarterly_wide[change_cols] = quarterly_wide[change_cols].fillna(0)

    quarterly_wide.to_csv(QUARTERLY_OUTPUT_PATH, index=False)

    expected_periods = [
        f"{year}Q{quarter}"
        for year in range(2021, 2026)
        for quarter in range(1, 5)
    ]

    actual_periods = quarterly_wide["calendar_period"].tolist()

    missing_periods = sorted(set(expected_periods) - set(actual_periods))
    extra_periods = sorted(set(actual_periods) - set(expected_periods))

    macro_cols = ["cpi", "fed_funds_rate", "unemployment_rate"]
    missing_values = quarterly_wide[macro_cols].isna().sum()

    print("\nQuarterly macro file saved:")
    print(QUARTERLY_OUTPUT_PATH)

    print("\nQuarterly macro summary:")
    print(f"  Rows: {len(quarterly_wide):,}")
    print(f"  First quarter: {quarterly_wide['calendar_period'].min()}")
    print(f"  Last quarter:  {quarterly_wide['calendar_period'].max()}")

    print("\nMissing expected quarters:")
    if missing_periods:
        print(missing_periods)
    else:
        print("  None")

    print("\nExtra quarters:")
    if extra_periods:
        print(extra_periods)
    else:
        print("  None")

    print("\nMissing values by macro column:")
    print(missing_values.to_string())

    print("\nMonthly observations per quarter and series:")
    print(
        quarterly_long
        .pivot(index="calendar_period", columns="series_id", values="monthly_obs")
        .reset_index()
        .to_string(index=False)
    )

    print("\nQuarterly macro preview:")
    print(quarterly_wide.to_string(index=False))

    if missing_periods:
        raise ValueError("Missing expected quarters. Stop and inspect.")

    if missing_values.sum() > 0:
        raise ValueError("Macro file has missing macro values. Stop and inspect.")

    print("\nScript 12 completed successfully.")


if __name__ == "__main__":
    main()