"""Phase 2 — Data acquisition.

Downloads the UCI "Individual Household Electric Power Consumption" dataset
directly from UCI's own static file host (not a Kaggle mirror, so the
license is the one confirmed on UCI's dataset page: CC BY 4.0).

Source: Hebrail, G. & Berard, A. (2006). Individual Household Electric
Power Consumption [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C58K54

Run: python scripts/download_data.py
"""

import zipfile
from pathlib import Path

import requests

URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
RAW_DIR = Path("data/raw")
ZIP_PATH = RAW_DIR / "household_power_consumption.zip"

RAW_DIR.mkdir(parents=True, exist_ok=True)

print(f"Downloading {URL}")
resp = requests.get(URL, timeout=120)
resp.raise_for_status()
ZIP_PATH.write_bytes(resp.content)
print(f"Saved {ZIP_PATH} ({ZIP_PATH.stat().st_size:,} bytes)")

with zipfile.ZipFile(ZIP_PATH) as zf:
    names = zf.namelist()
    print(f"Archive contains: {names}")
    zf.extractall(RAW_DIR)

extracted = list(RAW_DIR.glob("*.txt"))
print(f"Extracted: {extracted}")
for f in extracted:
    print(f"  {f.name}: {f.stat().st_size:,} bytes")
