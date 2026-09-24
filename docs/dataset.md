# Dataset

**Source:** Hebrail, G. & Berard, A. (2006). Individual Household Electric
Power Consumption [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C58K54

**License:** Creative Commons Attribution 4.0 International (CC BY 4.0), as
stated on UCI's dataset page. Sharing and adaptation are permitted with
attribution — this file is that attribution.

**What it is:** one-minute-resolution electricity consumption measurements
for a single household in Sceaux, France (a Paris suburb), December 2006
through November 2010. Columns: global active power (kW), global reactive
power (kW), voltage (V), global intensity (A), and three sub-metering
channels (kitchen, laundry room, water heater/AC), each in watt-hours.

**Why this dataset and not the originally-proposed PJM Kaggle dataset:**
the PJM dataset is a re-upload of PJM Interconnection's grid data. PJM's
own data portal requires agreeing to its own data license agreement, and no
equivalently clear open license could be confirmed for the Kaggle listing.
This dataset has a license confirmed directly on its own primary source
(UCI), so it's the one this project is built on. Consequence: this project
forecasts **household-level** short-term load, not regional grid load — a
smaller-scale but still real and common forecasting task (smart meters,
demand response, home energy management).

## Getting the file

```
python scripts/download_data.py
```

This downloads directly from UCI's own static file host (not Kaggle),
extracts it into `data/raw/household_power_consumption.txt`, and prints
the file sizes so you can confirm the download completed correctly.

## Getting the file if the script can't reach the network

If `scripts/download_data.py` can't complete the download (network policy,
offline environment, etc.), get it manually:

1. Visit https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
2. Download the zip from the "Download" button.
3. Unzip it into `data/raw/`, so `data/raw/household_power_consumption.txt`
   exists.

## Known data quality issues (see docs/data_validation.md for measured numbers)

The UCI dataset description notes that some measurements are missing,
marked `?` in the raw file. `scripts/validate_data.py` measures the actual
missing-value rate and any entirely-missing timestamps directly from the
downloaded file — see docs/data_validation.md for the real numbers, not an
estimate.
