"""Create a SAMPLE raw dataset (synthetic, deliberately dirty) for a campus event hub.

Run:  python -m ml.data.make_raw
Replace data/raw/events.csv with your real data when you have it.
"""
import numpy as np
import pandas as pd

from ml.config import CATEGORIES, RAW_DATA_PATH, VENUES

VENUE_CAPACITY = {  # (low, high) seats
    "Classroom": (30, 80), "Library Hall": (80, 150), "Main Hall": (150, 300),
    "Auditorium": (300, 800), "Outdoor Field": (200, 1000),
}
CATEGORY_BASE = {"Academic": 0.45, "Career": 0.55, "Sports": 0.60,
                 "Cultural": 0.55, "Social": 0.65, "Workshop": 0.40}


def generate(n: int = 700, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    category = rng.choice(CATEGORIES, n)
    venue = rng.choice(VENUES, n)
    dates = pd.to_datetime("2025-09-01") + pd.to_timedelta(rng.integers(0, 270, n), unit="D")
    hour = rng.integers(8, 21, n)
    is_free = rng.random(n) < 0.65
    price = np.where(is_free, 0.0, np.round(rng.uniform(2, 15, n), 1))
    capacity = np.array([rng.integers(*VENUE_CAPACITY[v]) for v in venue])
    promo = rng.integers(0, 22, n)
    rating = np.round(rng.uniform(2.5, 5, n), 1)
    exam = (rng.random(n) < 0.12).astype(int)
    weekend = dates.dayofweek >= 5

    rate = (
        np.array([CATEGORY_BASE[c] for c in category])
        + 0.12 * is_free - 0.012 * price
        + 0.010 * np.minimum(promo, 14)
        + 0.05 * (rating - 3.5)
        - 0.22 * exam
        - 0.012 * np.abs(hour - 16)
        + 0.08 * (weekend & np.isin(category, ["Sports", "Social", "Cultural"]))
        - 0.10 * (weekend & np.isin(category, ["Academic", "Workshop"]))
        + rng.normal(0, 0.07, n)
    )
    attendance = np.round(capacity * np.clip(rate, 0.05, 1.0)).astype(int)

    df = pd.DataFrame({
        "event_id": [f"E{i:04d}" for i in range(1, n + 1)],
        "event_name": [f"{c} event {i}" for i, c in enumerate(category, 1)],
        "category": category, "venue": venue,
        "event_date": dates.strftime("%Y-%m-%d"),
        "start_time": [f"{h:02d}:00" for h in hour],
        "is_free": np.where(is_free, "Yes", "No"),
        "ticket_price": price, "capacity": capacity, "promo_days": promo,
        "speaker_rating": rating, "exam_week": exam, "attendance": attendance,
    })

    # ---- make it dirty, like real data -------------------------------------
    messy = rng.random(n) < 0.15
    df.loc[messy, "category"] = [c.upper() if rng.random() < 0.5 else f" {c.lower()} " for c in df.loc[messy, "category"]]
    df.loc[rng.random(n) < 0.08, "venue"] = df["venue"].str.lower()
    df.loc[rng.random(n) < 0.20, "is_free"] = df["is_free"].map({"Yes": "Y", "No": "N"})
    pm = rng.random(n) < 0.15
    df.loc[pm, "start_time"] = [f"{(int(t[:2]) - 1) % 12 + 1}:00 {'PM' if int(t[:2]) >= 12 else 'AM'}" for t in df.loc[pm, "start_time"]]
    df.loc[df["is_free"].isin(["Yes", "Y"]) & (rng.random(n) < 0.5), "ticket_price"] = np.nan
    df.loc[rng.random(n) < 0.12, "speaker_rating"] = np.nan
    df.loc[rng.random(n) < 0.01, "speaker_rating"] = 9.0            # impossible rating
    df.loc[rng.random(n) < 0.01, "capacity"] = -1                    # impossible capacity
    df.loc[rng.random(n) < 0.03, "event_date"] = None
    df.loc[rng.random(n) < 0.01, "attendance"] = np.nan
    df.loc[rng.random(n) < 0.02, "attendance"] = df["capacity"] + 40  # more people than seats
    df = pd.concat([df, df.sample(15, random_state=seed)], ignore_index=True)  # duplicates
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = generate()
    out.to_csv(RAW_DATA_PATH, index=False)
    print(f"Saved {len(out)} rows -> {RAW_DATA_PATH}")
