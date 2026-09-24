"""Phase 3 — Data validation.

Runs directly against the raw file downloaded by scripts/download_data.py,
before any cleaning or resampling. The goal is to know exactly what's in
the data before deciding how to handle it, not to assume.

Checks:
1. Row count and date range.
2. Missing-value rate, both overall and restricted to the target column
   (Global_active_power) since that's what gets forecast.
3. Timestamp gaps: the file is supposed to be one row per minute; report
   how many minutes are actually missing from the sequence entirely
   (as opposed to present but marked '?').
4. Exact duplicate rows.
5. Basic range sanity checks on the numeric columns (the dataset
   documentation gives expected units; this just checks nothing is wildly
   out of range, e.g. negative power draw).

Run: python scripts/validate_data.py
"""

from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/household_power_consumption.txt")

NUMERIC_COLS = [
    "Global_active_power", "Global_reactive_power", "Voltage",
    "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
]

df = pd.read_csv(
    RAW_PATH, sep=";", na_values=["?"], low_memory=False,
)
df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")

print(f"[1] Row count: {len(df):,}")
print(f"    Date range: {df['datetime'].min()} to {df['datetime'].max()}")

print("[2] Missing-value rate per column:")
for col in NUMERIC_COLS:
    n_missing = df[col].isna().sum()
    print(f"    {col}: {n_missing:,} missing ({n_missing / len(df):.2%})")

full_range = pd.date_range(df["datetime"].min(), df["datetime"].max(), freq="min")
present = set(df["datetime"])
missing_minutes = [t for t in full_range if t not in present]
print(f"[3] Expected minutes in range: {len(full_range):,}")
print(f"    Minutes present as rows:   {df['datetime'].nunique():,}")
print(f"    Minutes missing entirely (no row at all): {len(missing_minutes):,}")
if missing_minutes:
    print(f"    First few missing: {missing_minutes[:5]}")

n_dupes = df.duplicated(subset=["datetime"]).sum()
print(f"[4] Duplicate timestamp rows: {n_dupes:,}")

print("[5] Range sanity check (min / max per numeric column):")
for col in NUMERIC_COLS:
    print(f"    {col}: min={df[col].min()}  max={df[col].max()}")

df.to_parquet("data/processed/household_power_raw.parquet", index=False)
print("Saved data/processed/household_power_raw.parquet")
