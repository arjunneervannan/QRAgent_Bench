from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

FF_HELP = (
    "\nExpected CSV at data/ff25_daily.csv with columns:"
    "\n  - Date or date in YYYYMMDD"
    "\n  - 25 portfolio return columns (either % or decimal)."
    "\nTip: Download the daily 25 Portfolios formed on Size and Book-to-Market, then save as ff25_daily.csv"
)

def load_ff25_daily(path: str | Path = "data/ff25_daily.csv") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}{FF_HELP}")
    # Read generously with low_memory=False to avoid mixed type warnings
    df = pd.read_csv(path, low_memory=False)
    # Parse YYYYMMDD or YYYYMM
    fmt = "%Y%m%d"
    df["date"] = pd.to_datetime(df["date"].astype(str), format=fmt, errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date")

    # Ensure we have 25 columns of returns
    # Coerce to numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(axis=1, how="all").dropna(how="any")

    if df.shape[1] < 10:
        raise ValueError(f"Parsed only {df.shape[1]} numeric columns; expected 25.{FF_HELP}")

    df /= 100.0

    # Standardize column names
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    return df.sort_index()
