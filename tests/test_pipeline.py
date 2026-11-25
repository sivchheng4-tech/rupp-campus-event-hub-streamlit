import numpy as np
import pandas as pd

from ml import config as cfg
from ml import service
from ml.data.make_dataset import clean, parse_hour
from ml.data.make_raw import generate
from ml.features import add_features
from ml.train import fit_and_evaluate


def test_parse_hour():
    assert parse_hour("14:00") == 14
    assert parse_hour("2:00 PM") == 14
    assert parse_hour("12:00 AM") == 0
    assert np.isnan(parse_hour("soon"))


def test_clean_fixes_dirty_data():
    raw = generate(n=200)
    df = clean(raw)
    assert len(df) < len(raw)                                   # duplicates / bad rows removed
    assert list(df.columns) == cfg.INPUT_COLUMNS + [cfg.TARGET_COLUMN]
    assert set(df["category"]) <= set(cfg.CATEGORIES)
    assert (df["attendance"] <= df["capacity"]).all()
    assert df["speaker_rating"].dropna().between(1, 5).all()
    assert df.duplicated().sum() == 0


def test_add_features():
    out = add_features(pd.DataFrame({"event_date": ["2026-12-12"]}))   # a Saturday
    assert out.loc[0, "is_weekend"] == 1
    assert out.loc[0, "month"] == 12
    assert "event_date" not in out.columns


def test_training_and_prediction():
    df = clean(generate(n=300))
    pipeline, metrics, _, _ = fit_and_evaluate(df)
    assert metrics["test"]["rmse"] < metrics["baseline"]["test_rmse"]   # beats the baseline
    features = {"category": "Sports", "venue": "Outdoor Field", "event_date": "2026-12-12",
                "start_hour": 16, "is_free": 1, "ticket_price": 0, "capacity": 400,
                "promo_days": 10, "speaker_rating": "", "exam_week": 0}
    result = service.predict(features, model=pipeline)
    assert 0 <= result["predicted_attendance"] <= 400


def test_validation_errors():
    for bad in [{}, {"category": "Opera"}]:
        try:
            service.predict(bad, model=object())
        except ValueError:
            continue
        raise AssertionError("expected a ValueError")
