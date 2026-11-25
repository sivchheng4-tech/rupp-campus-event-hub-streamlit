"""Central settings for the ML pipeline. Edit this file first when you change the data."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---- paths -------------------------------------------------------------------
RAW_DATA_PATH = ROOT / "data" / "raw" / "events.csv"
PROCESSED_DATA_PATH = ROOT / "data" / "processed" / "clean.csv"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
MODEL_CARD_PATH = MODELS_DIR / "model_card.md"
FIGURES_DIR = ROOT / "reports" / "figures"

# ---- problem definition ------------------------------------------------------
TASK_TYPE = "regression"          # this template: predict how many people attend an event
TARGET_COLUMN = "attendance"
DROP_COLUMNS = ["event_id", "event_name"]   # identifiers, not useful for the model
PRIMARY_METRIC = "neg_root_mean_squared_error"

TEST_SIZE = 0.2
CV_FOLDS = 5
RANDOM_STATE = 42

# ---- allowed values ----------------------------------------------------------
CATEGORIES = ["Academic", "Career", "Sports", "Cultural", "Social", "Workshop"]
VENUES = ["Classroom", "Library Hall", "Main Hall", "Auditorium", "Outdoor Field"]

# ---- model inputs (also builds the form in the web app) ----------------------
# type: select | number | date.  required=False means missing values are allowed
# (they are imputed by the model pipeline).
FEATURE_SCHEMA = [
    {"name": "category", "label": "Event category", "type": "select", "required": True,
     "options": [{"value": c, "label": c} for c in CATEGORIES], "default": "Academic"},
    {"name": "venue", "label": "Venue", "type": "select", "required": True,
     "options": [{"value": v, "label": v} for v in VENUES], "default": "Main Hall"},
    {"name": "event_date", "label": "Event date", "type": "date", "required": True, "default": ""},
    {"name": "start_hour", "label": "Start hour (0-23)", "type": "number", "required": True,
     "min": 6, "max": 22, "step": 1, "default": 14},
    {"name": "is_free", "label": "Ticket", "type": "select", "required": True,
     "options": [{"value": 1, "label": "Free"}, {"value": 0, "label": "Paid"}], "default": 1},
    {"name": "ticket_price", "label": "Ticket price (USD)", "type": "number", "required": True,
     "min": 0, "max": 100, "step": 0.5, "default": 0},
    {"name": "capacity", "label": "Venue capacity", "type": "number", "required": True,
     "min": 10, "max": 5000, "step": 10, "default": 200},
    {"name": "promo_days", "label": "Days promoted before event", "type": "number", "required": True,
     "min": 0, "max": 60, "step": 1, "default": 7},
    {"name": "speaker_rating", "label": "Speaker rating (1-5, optional)", "type": "number",
     "required": False, "min": 1, "max": 5, "step": 0.1, "default": 4},
    {"name": "exam_week", "label": "During exam week?", "type": "select", "required": True,
     "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}], "default": 0},
]
INPUT_COLUMNS = [f["name"] for f in FEATURE_SCHEMA]
