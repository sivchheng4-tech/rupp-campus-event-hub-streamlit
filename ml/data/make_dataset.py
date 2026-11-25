"""Cleaning: data/raw/events.csv  ->  data/processed/clean.csv

Run:  python -m ml.data.make_dataset
Every cleaning rule lives here, so the raw file is never edited by hand.
"""
import re

import numpy as np
import pandas as pd

from ml.config import (CATEGORIES, DROP_COLUMNS, INPUT_COLUMNS, PROCESSED_DATA_PATH,
                       RAW_DATA_PATH, TARGET_COLUMN, VENUES)

_TIME_RE = re.compile(r"^(\d{1,2})(?::(\d{2}))?\s*(AM|PM)?$")
_BOOL = {"yes": 1, "y": 1, "true": 1, "1": 1, "free": 1,
         "no": 0, "n": 0, "false": 0, "0": 0, "paid": 0}


def parse_hour(value) -> float:
    """'14:00' -> 14, '2:00 PM' -> 14, anything unreadable -> NaN."""
    m = _TIME_RE.match(str(value).strip().upper())
    if not m:
        return np.nan
    hour, ampm = int(m.group(1)), m.group(3)
    if ampm == "PM" and hour < 12:
        hour += 12
    if ampm == "AM" and hour == 12:
        hour = 0
    return float(hour) if 0 <= hour <= 23 else np.nan


def _to_bool01(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().map(_BOOL)


def clean(raw: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    n0 = len(df)

    df = df.drop_duplicates()
    n_dups = n0 - len(df)

    # text columns: trim, fix case, keep only known values
    df["category"] = df["category"].astype(str).str.strip().str.title()
    df["venue"] = df["venue"].astype(str).str.strip().str.title()
    df.loc[~df["category"].isin(CATEGORIES), "category"] = np.nan
    df.loc[~df["venue"].isin(VENUES), "venue"] = np.nan

    df["is_free"] = _to_bool01(df["is_free"])
    df["exam_week"] = _to_bool01(df["exam_week"])
    df["start_hour"] = df["start_time"].map(parse_hour)
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")

    for col in ["ticket_price", "capacity", "promo_days", "speaker_rating", TARGET_COLUMN]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # business rules
    df.loc[df["is_free"] == 1, "ticket_price"] = 0
    df["ticket_price"] = df["ticket_price"].fillna(0).clip(lower=0)
    df.loc[~df["speaker_rating"].between(1, 5), "speaker_rating"] = np.nan   # missing is OK
    df["promo_days"] = df["promo_days"].clip(0, 60)
    df.loc[~df["capacity"].between(10, 5000), "capacity"] = np.nan

    # rows we cannot use
    required = ["category", "venue", "event_date", "start_hour", "is_free",
                "capacity", "promo_days", "exam_week", TARGET_COLUMN]
    before = len(df)
    df = df.dropna(subset=required)
    n_unusable = before - len(df)

    df = df[df[TARGET_COLUMN] >= 0]
    df[TARGET_COLUMN] = np.minimum(df[TARGET_COLUMN], df["capacity"])   # cannot exceed seats
    df["event_date"] = df["event_date"].dt.strftime("%Y-%m-%d")

    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    out = df[INPUT_COLUMNS + [TARGET_COLUMN]].reset_index(drop=True)
    for col in ["start_hour", "is_free", "capacity", "promo_days", "exam_week", TARGET_COLUMN]:
        out[col] = out[col].round().astype(int)

    if verbose:
        print(f"Rows in raw file       : {n0}")
        print(f"Duplicates removed     : {n_dups}")
        print(f"Unusable rows removed  : {n_unusable}")
        print(f"Rows in clean data     : {len(out)}")
        print(f"Missing speaker_rating : {int(out['speaker_rating'].isna().sum())} (kept, imputed later)")
    return out


def main() -> pd.DataFrame:
    raw = pd.read_csv(RAW_DATA_PATH)
    out = clean(raw, verbose=True)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Saved -> {PROCESSED_DATA_PATH}")
    return out


if __name__ == "__main__":
    main()
