"""Phase 4 — Preprocessing.

Takes the validated raw parquet (data/processed/household_power_raw.parquet,
written by scripts/validate_data.py) and produces the hourly series the rest
of the project is built on.

Two things happen here, in order, and the order matters for leakage:

1. Fill missing per-minute values. Missing values are marked '?' in the raw
   file (parsed as NaN by validate_data.py) at a measured rate of 1.25%
   across all seven numeric columns (docs/data_validation.md). This script
   checks whether missingness co-occurs across columns (whole readings
   dropped) or is independent per column, then forward-fills only — never
   backward-fill or interpolation, since either would let a future value
   leak into a "past" row.

2. Resample minute-level data to hourly. Global_active_power is an average
   power reading (kW) for that minute; the hourly value is the mean over
   the 60 minutes in that hour, which is numerically the average hourly
   power draw (kW) — this is the forecasting target. Sub-metering columns
   are watt-hours per minute, so they're summed (not averaged) to get
   total watt-hours for the hour.

Run: python scripts/preprocess.py
"""

from pathlib import Path

import pandas as pd

RAW_PARQUET = Path("data/processed/household_power_raw.parquet")
OUT_PARQUET = Path("data/processed/hourly.parquet")

MEAN_COLS = ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity"]
SUM_COLS = ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]

df = pd.read_parquet(RAW_PARQUET)
df = df.sort_values("datetime").set_index("datetime")

# --- Step 1: check whether missingness co-occurs across columns ---
missing_mask = df[MEAN_COLS + SUM_COLS].isna()
rows_any_missing = missing_mask.any(axis=1).sum()
rows_all_missing = missing_mask.all(axis=1).sum()
print(f"[1] Rows with at least one missing value: {rows_any_missing:,}")
print(f"    Rows where ALL numeric columns are missing: {rows_all_missing:,}")
if rows_any_missing != rows_all_missing:
    print("    NOTE: missingness does NOT perfectly co-occur across columns — "
          "the 'whole reading dropped' assumption in docs/data_validation.md "
          "does not fully hold; filling is still done per-column, so this is "
          "informational, not a blocker.")

# --- Step 2: forward-fill only (no backward-fill, no interpolation) ---
n_before = df[MEAN_COLS + SUM_COLS].isna().sum().sum()
df[MEAN_COLS + SUM_COLS] = df[MEAN_COLS + SUM_COLS].ffill()
n_after = df[MEAN_COLS + SUM_COLS].isna().sum().sum()
print(f"[2] Missing values before ffill: {n_before:,}, after: {n_after:,}")
if n_after > 0:
    # Only possible if the series starts with missing values (nothing to
    # forward-fill from). Report it rather than silently dropping rows.
    print(f"    WARNING: {n_after:,} values still missing after ffill "
          f"(likely at the very start of the series). These rows are dropped.")
    df = df.dropna(subset=MEAN_COLS + SUM_COLS)

# --- Step 3: resample to hourly ---
hourly = pd.DataFrame(index=pd.DatetimeIndex([], name="datetime"))
hourly = df[MEAN_COLS].resample("h").mean()
hourly[SUM_COLS] = df[SUM_COLS].resample("h").sum()
hourly["n_minutes"] = df["Global_active_power"].resample("h").count()

# Hours built from a partial 60 minutes (first/last hour of the series, or
# any hour that had unfillable missing minutes dropped above) are flagged,
# not silently kept as if they were as reliable as a full hour.
incomplete_hours = (hourly["n_minutes"] < 60).sum()
print(f"[3] Hourly rows: {len(hourly):,}")
print(f"    Date range: {hourly.index.min()} to {hourly.index.max()}")
print(f"    Hours built from fewer than 60 minutes: {incomplete_hours:,}")

hourly = hourly.rename(columns={"Global_active_power": "target_mean_kw"})
hourly = hourly.reset_index()

Path("data/processed").mkdir(parents=True, exist_ok=True)
hourly.to_parquet(OUT_PARQUET, index=False)
print(f"Saved {OUT_PARQUET} ({len(hourly):,} rows)")
