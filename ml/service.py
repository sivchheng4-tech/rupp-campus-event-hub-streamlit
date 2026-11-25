"""Functions the API calls. Keeps the web layer thin."""
import json
import math
from collections import deque
from datetime import date, datetime, timezone

import joblib
import pandas as pd

from ml import config as cfg

_model = None
_history = deque(maxlen=50)


class ModelNotTrainedError(RuntimeError):
    pass


def load_model(force: bool = False):
    global _model
    if _model is None or force:
        if not cfg.MODEL_PATH.exists():
            raise ModelNotTrainedError(
                "No trained model found. Run `python -m ml.data.make_dataset` and `python -m ml.train` first.")
        _model = joblib.load(cfg.MODEL_PATH)
    return _model


def get_schema() -> list:
    fields = [dict(f) for f in cfg.FEATURE_SCHEMA]
    for f in fields:
        if f["type"] == "date" and not f.get("default"):
            f["default"] = date.today().isoformat()
    return fields


def _validate(features: dict) -> dict:
    row = {}
    for f in cfg.FEATURE_SCHEMA:
        name, raw = f["name"], features.get(f["name"])
        if raw is None or (isinstance(raw, str) and raw.strip() == ""):
            if f["required"]:
                raise ValueError(f"'{name}' is required")
            row[name] = math.nan
            continue
        if f["type"] == "number":
            try:
                value = float(raw)
            except (TypeError, ValueError):
                raise ValueError(f"'{name}' must be a number")
            if not (f["min"] <= value <= f["max"]):
                raise ValueError(f"'{name}' must be between {f['min']} and {f['max']}")
            row[name] = value
        elif f["type"] == "select":
            match = [o["value"] for o in f["options"] if str(o["value"]) == str(raw)]
            if not match:
                allowed = ", ".join(str(o["value"]) for o in f["options"])
                raise ValueError(f"'{name}' must be one of: {allowed}")
            row[name] = match[0]
        elif f["type"] == "date":
            try:
                row[name] = date.fromisoformat(str(raw)[:10]).isoformat()
            except ValueError:
                raise ValueError(f"'{name}' must be a date like 2026-12-10")
    return row


def predict(features: dict, model=None) -> dict:
    row = _validate(features)
    model = model if model is not None else load_model()
    X = pd.DataFrame([row], columns=cfg.INPUT_COLUMNS)
    capacity = row["capacity"]
    pred = float(min(max(float(model.predict(X)[0]), 0.0), capacity))
    fill = pred / capacity
    result = {
        "predicted_attendance": int(round(pred)),
        "capacity": int(capacity),
        "fill_rate": round(fill, 3),
        "level": "low" if fill < 0.4 else "medium" if fill < 0.75 else "high",
    }
    _history.appendleft({"time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                         "features": features, "result": result})
    return result


def get_history() -> list:
    return list(_history)


def get_metrics() -> dict:
    if not cfg.METRICS_PATH.exists():
        raise ModelNotTrainedError("No metrics yet. Run `python -m ml.train` first.")
    return json.loads(cfg.METRICS_PATH.read_text(encoding="utf-8"))


def get_data(limit: int = 100) -> list:
    if not cfg.PROCESSED_DATA_PATH.exists():
        raise ModelNotTrainedError("No clean data yet. Run `python -m ml.data.make_dataset` first.")
    df = pd.read_csv(cfg.PROCESSED_DATA_PATH).head(limit)
    return json.loads(df.to_json(orient="records"))


def train() -> dict:
    from ml.train import run_training
    metrics = run_training(verbose=False)
    load_model(force=True)
    return metrics
