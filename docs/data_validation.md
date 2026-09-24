# Data Validation — Phase 3

Measured directly by `scripts/validate_data.py` against the raw file
downloaded by `scripts/download_data.py`. Numbers below are copied from an
actual run, not estimated.

## Results (measured)

```
[1] Row count: 2,075,259
    Date range: 2006-12-16 17:24:00 to 2010-11-26 21:02:00
[2] Missing-value rate per column:
    Global_active_power: 25,979 missing (1.25%)
    Global_reactive_power: 25,979 missing (1.25%)
    Voltage: 25,979 missing (1.25%)
    Global_intensity: 25,979 missing (1.25%)
    Sub_metering_1: 25,979 missing (1.25%)
    Sub_metering_2: 25,979 missing (1.25%)
    Sub_metering_3: 25,979 missing (1.25%)
[3] Expected minutes in range: 2,075,259
    Minutes present as rows:   2,075,259
    Minutes missing entirely (no row at all): 0
[4] Duplicate timestamp rows: 0
[5] Range sanity check (min / max per numeric column):
    Global_active_power: min=0.076  max=11.122
    Global_reactive_power: min=0.0  max=1.39
    Voltage: min=223.2  max=254.15
    Global_intensity: min=0.2  max=48.4
    Sub_metering_1: min=0.0  max=88.0
    Sub_metering_2: min=0.0  max=80.0
    Sub_metering_3: min=0.0  max=31.0
```

## What this means for preprocessing

- **Every minute in the date range has a row.** There are no entirely
  missing timestamps to interpolate a whole row for — every gap is a `?`
  inside an otherwise-present row. This simplifies preprocessing: no need
  to reindex/insert missing minutes, only to fill missing values within
  existing rows.
- **All seven numeric columns are missing at exactly the same rate and
  count (25,979 rows, 1.25%)** — strongly suggesting these are missing in
  whole rows (i.e. entire readings dropped for those minutes), not
  independently missing per column. This will be a check made directly in
  preprocessing rather than assumed from this rate alone.
- **No duplicate timestamps** — no need for a de-duplication step before
  resampling to hourly.
- **All values are within physically sane ranges** for household mains
  power (0–11 kW active power, ~230V voltage, matching French household
  mains) — no evidence of unit errors or corrupted rows from this check.

## Handling decision for Phase 4 (preprocessing)

Missing values are filled with **forward-fill only** (each `?` replaced
with the most recent known value strictly before it), never
backward-fill or an interpolation that uses future values, since a
backward or centered fill would leak future information into a training
row's features. This is the same leakage-avoidance discipline as the lag
features described in the design doc — it isn't only a "modeling" concern,
it starts at the missing-value handling step in preprocessing.

## Phase 4 results (measured, `scripts/preprocess.py`)

```
[1] Rows with at least one missing value: 25,979
    Rows where ALL numeric columns are missing: 25,979
[2] Missing values before ffill: 181,853, after: 0
[3] Hourly rows: 34,589
    Date range: 2006-12-16 17:00:00 to 2010-11-26 21:00:00
    Hours built from fewer than 60 minutes: 2
```

The "all columns missing together" hypothesis above is confirmed exactly:
25,979 rows have at least one missing value, and the same 25,979 rows have
*every* column missing — no partial/independent missingness. 181,853 total
missing cells (25,979 rows × 7 columns) were forward-filled with 0
remaining afterward, so no rows needed to be dropped for being unfillable.
Only 2 of the 34,589 hourly rows are built from fewer than 60 minutes (the
series starts at 17:24 and ends at 21:02, so the first and last hours are
naturally partial) — both are flagged in the output rather than silently
treated as equally reliable as a full hour.
