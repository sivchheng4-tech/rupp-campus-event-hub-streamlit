"""Compare models (CV) -> tune the best -> evaluate on the TEST set -> save.

Run:  python -m ml.train
"""
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from ml import config as cfg
from ml.evaluate import regression_metrics, save_figures
from ml.features import add_features, build_preprocessor


def build_pipeline(estimator) -> Pipeline:
    """features -> preprocessing -> model, saved as ONE object."""
    return Pipeline([
        ("features", FunctionTransformer(add_features, validate=False)),
        ("prep", build_preprocessor()),
        ("model", estimator),
    ])


def candidate_models() -> dict:
    rs = cfg.RANDOM_STATE
    return {
        "Ridge Regression": Ridge(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=rs),
        "Gradient Boosting": GradientBoostingRegressor(random_state=rs),
    }


PARAM_GRIDS = {
    "Ridge Regression": {"model__alpha": [0.1, 1.0, 10.0]},
    "Random Forest": {"model__max_depth": [None, 8, 16], "model__min_samples_leaf": [1, 3]},
    "Gradient Boosting": {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1],
                          "model__max_depth": [2, 3]},
}


def _native(value):
    return value.item() if isinstance(value, np.generic) else value


def fit_and_evaluate(df: pd.DataFrame, verbose: bool = False):
    """Everything except saving. Returns (pipeline, metrics, y_test, y_pred)."""
    X = df.drop(columns=[cfg.TARGET_COLUMN])
    y = df[cfg.TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE)
    cv = KFold(n_splits=cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE)

    def cv_rmse(estimator):
        scores = cross_val_score(build_pipeline(estimator), X_train, y_train,
                                 cv=cv, scoring=cfg.PRIMARY_METRIC)
        return float(-scores.mean()), float(scores.std())

    # 1) baseline + model comparison (training set only)
    if verbose:
        print(f"1) Comparing models ({cfg.CV_FOLDS}-fold CV on the training set, RMSE, lower is better):")
    base_mean, base_std = cv_rmse(DummyRegressor(strategy="mean"))
    if verbose:
        print(f"  {'Baseline (mean)':<20} rmse = {base_mean:.2f} +/- {base_std:.2f}")
    cv_results = []
    for name, est in candidate_models().items():
        mean, std = cv_rmse(est)
        cv_results.append({"model": name, "rmse_mean": mean, "rmse_std": std})
        if verbose:
            print(f"  {name:<20} rmse = {mean:.2f} +/- {std:.2f}")
    best_name = min(cv_results, key=lambda r: r["rmse_mean"])["model"]

    # 2) tune the best
    if verbose:
        print(f"2) Tuning best model: {best_name}")
    search = GridSearchCV(build_pipeline(candidate_models()[best_name]), PARAM_GRIDS[best_name],
                          cv=cv, scoring=cfg.PRIMARY_METRIC)
    search.fit(X_train, y_train)
    pipeline = search.best_estimator_

    # 3) the TEST set is used once, here
    y_pred = np.clip(pipeline.predict(X_test), 0, X_test["capacity"].to_numpy())
    test = regression_metrics(y_test, y_pred)
    baseline = build_pipeline(DummyRegressor(strategy="mean")).fit(X_train, y_train)
    base_test = regression_metrics(y_test, baseline.predict(X_test))
    if verbose:
        print(f"3) Test set: MAE {test['mae']:.1f} people | RMSE {test['rmse']:.1f} | R2 {test['r2']:.2f}"
              f"   (baseline RMSE {base_test['rmse']:.1f})")

    imp = permutation_importance(pipeline, X_test, y_test, n_repeats=5,
                                 random_state=cfg.RANDOM_STATE, scoring=cfg.PRIMARY_METRIC)
    importance = sorted(
        [{"feature": c, "importance": float(v)} for c, v in zip(X_test.columns, imp.importances_mean)],
        key=lambda r: r["importance"], reverse=True)

    metrics = {
        "task_type": cfg.TASK_TYPE, "target": cfg.TARGET_COLUMN,
        "primary_metric": cfg.PRIMARY_METRIC,
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "baseline": {"cv_rmse": base_mean, "test_rmse": base_test["rmse"], "test_mae": base_test["mae"]},
        "cv_results": cv_results,
        "best_model": best_name,
        "best_params": {k: _native(v) for k, v in search.best_params_.items()},
        "test": test,
        "feature_importance": importance,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return pipeline, metrics, y_test, y_pred


def write_model_card(m: dict) -> str:
    return f"""# Model card: event attendance

- **Task:** predict `{m['target']}` (number of people) for a campus event ({m['task_type']})
- **Model:** {m['best_model']} {m['best_params']}
- **Trained:** {m['trained_at']} on {m['n_train']} events, tested on {m['n_test']}
- **Test error:** on average the prediction is off by {m['test']['mae']:.1f} people (RMSE {m['test']['rmse']:.1f}, R2 {m['test']['r2']:.2f})
- **Baseline (always predict the mean):** RMSE {m['baseline']['test_rmse']:.1f}
- **Limits:** trained on one dataset; events very different from the training data (new venue types, huge capacity) will be predicted less reliably.
"""


def run_training(verbose: bool = True) -> dict:
    df = pd.read_csv(cfg.PROCESSED_DATA_PATH)
    pipeline, metrics, y_test, y_pred = fit_and_evaluate(df, verbose=verbose)
    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, cfg.MODEL_PATH)
    cfg.METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    cfg.MODEL_CARD_PATH.write_text(write_model_card(metrics), encoding="utf-8")
    save_figures(y_test, y_pred)
    if verbose:
        print(f"Saved model -> {cfg.MODEL_PATH}")
    return metrics


if __name__ == "__main__":
    run_training()
