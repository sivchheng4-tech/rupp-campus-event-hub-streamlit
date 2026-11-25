"""Metrics + figures."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml.config import FIGURES_DIR


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_figures(y_true, y_pred) -> list:
    """Actual-vs-predicted plot. Skipped quietly if matplotlib is not installed."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.5)
    lim = [0, max(float(np.max(y_true)), float(np.max(y_pred))) * 1.05]
    ax.plot(lim, lim, "r--", label="perfect prediction")
    ax.set_xlabel("Actual attendance")
    ax.set_ylabel("Predicted attendance")
    ax.set_title("Test set: actual vs predicted")
    ax.legend()
    path = FIGURES_DIR / "actual_vs_predicted.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return [path]
