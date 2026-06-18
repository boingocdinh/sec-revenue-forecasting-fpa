-- ============================================================================
-- Revenue Forecasting Project — SQL Data Layer (MySQL)
-- Author: Boi Ngoc Dinh
--
-- Purpose: Load the SEC + macro revenue panel into MySQL and reproduce the core
-- data-preparation steps (filtering, deduplication, aggregation, YoY growth,
-- and the SEC-to-macro join) as SQL instead of pandas.
--
-- Reconciliation note: the forecast-accuracy summary in Section 7 is designed to
-- reproduce the Python pipeline's reported result of ~$184.3M mean absolute error
-- for the seasonal baseline on the 2025 test set. If this query returns a
-- materially different number, the SQL and Python definitions have diverged and
-- should be reconciled before trusting either.
--
-- How to run in MySQL Workbench:
--   1. Run Section 1 to create the database and table.
--   2. Import the CSV (Section 2 notes).
--   3. Run the queries in Sections 3-7 one at a time.
-- ============================================================================


-- ============================================================================
-- SECTION 1 — Create database and table
-- NOTE: This DROPs and recreates the table on every run. That is intentional for
-- a personal/reproducible project, but be aware it is destructive — never point
-- this at a shared database without checking what 'revenue_panel' currently holds.
-- ============================================================================
CREATE DATABASE IF NOT EXISTS revenue_forecasting;
USE revenue_forecasting;

DROP TABLE IF EXISTS revenue_panel;

CREATE TABLE revenue_panel (
    cik                          BIGINT,
    company_name                 VARCHAR(255),
    sic                          INT,
    calendar_period              VARCHAR(10),    -- e.g. '2023Q4'
    calendar_year                INT,
    calendar_quarter             INT,
    form                         VARCHAR(20),
    filed_date                   DATE,
    adsh                         VARCHAR(30),    -- SEC accession number (unique per filing)
    revenue_usd                  DOUBLE,
    revenue_millions_usd         DOUBLE,
    uom                          VARCHAR(10),
    model_split                  VARCHAR(20),
    target_calendar_period       VARCHAR(10),
    target_revenue_next_qtr      DOUBLE,
    revenue_lag_4                DOUBLE,
    seasonal_naive_forecast_next_qtr DOUBLE,
    is_derived_q4                INT,
    cpi                          DOUBLE,
    fed_funds_rate               DOUBLE,
    unemployment_rate            DOUBLE
);


-- ============================================================================
-- SECTION 2 — Loading the data
-- OPTION A (easiest): right-click 'revenue_panel' -> Table Data Import Wizard ->
--   select the CSV -> map the columns above -> finish. Slow for ~51k rows; wait.
-- OPTION B (faster): LOAD DATA LOCAL INFILE (needs SET GLOBAL local_infile = 1;).
-- ============================================================================
SELECT COUNT(*) AS total_rows FROM revenue_panel;
-- Expected after a full load: 51,168 rows.


-- ============================================================================
-- SECTION 3 — Filtering: keep only clean, usable revenue rows
--
-- DATA NOTE: 'revenue_usd > 0' removes 1,572 rows (3.1% of 51,168). These are
-- firm-quarters with zero or negative reported revenue in SEC data (e.g. certain
-- financial firms, reporting artifacts). They are real records, not errors — in
-- the modeling pipeline they are NOT deleted but routed to a seasonal-baseline
-- fallback. They are excluded here only because percentage/growth math is
-- undefined on non-positive revenue.
-- ============================================================================
SELECT *
FROM revenue_panel
WHERE revenue_usd > 0
  AND uom = 'USD'
  AND target_revenue_next_qtr IS NOT NULL;


-- ============================================================================
-- SECTION 4 — Deduplication: one row per company per quarter
-- Keep the latest filed version. TIEBREAKER: if two filings share the same
-- filed_date, fall back to the accession number (adsh) so the result is
-- DETERMINISTIC — the same row wins every time the query runs.
-- ============================================================================
WITH ranked AS (
    SELECT
        cik,
        company_name,
        calendar_period,
        revenue_usd,
        filed_date,
        adsh,
        ROW_NUMBER() OVER (
            PARTITION BY cik, calendar_period
            ORDER BY filed_date DESC, adsh DESC   -- adsh breaks filed_date ties
        ) AS rn
    FROM revenue_panel
    WHERE revenue_usd > 0
)
SELECT cik, company_name, calendar_period, revenue_usd, filed_date, adsh
FROM ranked
WHERE rn = 1;


-- ============================================================================
-- SECTION 5 — Aggregation: revenue by sector and by quarter
-- ============================================================================
SELECT
    sic,
    calendar_period,
    COUNT(DISTINCT cik)                 AS num_companies,
    ROUND(SUM(revenue_millions_usd), 1) AS total_revenue_mm,
    ROUND(AVG(revenue_millions_usd), 1) AS avg_revenue_mm
FROM revenue_panel
WHERE revenue_usd > 0
GROUP BY sic, calendar_period
ORDER BY calendar_period, total_revenue_mm DESC;


-- ============================================================================
-- SECTION 6 — Year-over-year growth using a window function (LAG)
-- ============================================================================
WITH firm_quarter AS (
    SELECT
        cik,
        company_name,
        calendar_period,
        calendar_year,
        calendar_quarter,
        revenue_usd,
        LAG(revenue_usd, 4) OVER (
            PARTITION BY cik
            ORDER BY calendar_year, calendar_quarter
        ) AS revenue_same_qtr_last_year
    FROM revenue_panel
    WHERE revenue_usd > 0
)
SELECT
    cik,
    company_name,
    calendar_period,
    revenue_usd,
    revenue_same_qtr_last_year,
    ROUND(
        (revenue_usd - revenue_same_qtr_last_year)
        / NULLIF(revenue_same_qtr_last_year, 0) * 100, 1
    ) AS yoy_growth_pct
FROM firm_quarter
WHERE revenue_same_qtr_last_year IS NOT NULL
ORDER BY cik, calendar_period;


-- ============================================================================
-- SECTION 7 — Macro context + forecast-accuracy summary (reconciliation query)
-- Summarizes the seasonal baseline's error on the 2025 test set, by quarter.
-- The overall mean_abs_error_mm here should reconcile to the Python pipeline's
-- ~$184.3M seasonal-baseline MAE. (Run without GROUP BY for the single overall
-- number; grouped below for the by-quarter view.)
-- ============================================================================
SELECT
    target_calendar_period                         AS quarter,
    COUNT(*)                                       AS num_forecasts,
    ROUND(AVG(ABS(seasonal_naive_forecast_next_qtr - target_revenue_next_qtr))
          / 1000000, 1)                            AS mean_abs_error_mm,
    ROUND(AVG(cpi), 1)                             AS avg_cpi,
    ROUND(AVG(fed_funds_rate), 2)                  AS avg_fed_funds_rate
FROM revenue_panel
WHERE model_split = 'test_2025'
  AND target_revenue_next_qtr > 0
GROUP BY target_calendar_period
ORDER BY target_calendar_period;

-- Overall reconciliation check (should be close to ~$184.3M):
SELECT
    ROUND(AVG(ABS(seasonal_naive_forecast_next_qtr - target_revenue_next_qtr))
          / 1000000, 1) AS overall_baseline_mae_mm
FROM revenue_panel
WHERE model_split = 'test_2025'
  AND target_revenue_next_qtr > 0;

-- ============================================================================
-- End of script.
-- ============================================================================
