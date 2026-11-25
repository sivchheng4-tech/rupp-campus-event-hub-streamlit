"""Feature engineering + preprocessing."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL_FEATURES = ["category", "venue"]
NUMERIC_FEATURES = ["start_hour", "is_free", "ticket_price", "capacity", "promo_days",
                    "speaker_rating", "exam_week", "day_of_week", "month", "is_weekend"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """TODO: add your own features here. Runs inside the model pipeline, so the
    API gets the same features as training."""
    df = df.copy()
    date = pd.to_datetime(df["event_date"])
    df["day_of_week"] = date.dt.dayofweek          # 0 = Monday
    df["month"] = date.dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df.drop(columns=["event_date"])


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric, NUMERIC_FEATURES),
                              ("cat", categorical, CATEGORICAL_FEATURES)])
