"""
Classifies road surface from per-second telemetry signals.
Labels: highway | urban | rural | offroad
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

FEATURE_COLS = [
    "speed_kmh", "acceleration", "braking_force",
    "vibration_g", "engine_rpm", "throttle_pct",
]
TARGET_COL = "road_surface"


class RoadSurfaceClassifier:
    def __init__(self, n_estimators: int = 80, random_state: int = 42):
        self.model   = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state)
        self.encoder = LabelEncoder()

    def fit(self, df: pd.DataFrame) -> "RoadSurfaceClassifier":
        X = df[FEATURE_COLS].fillna(0)
        y = self.encoder.fit_transform(df[TARGET_COL])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        print("Road Surface Classifier Report:")
        print(classification_report(y_test, preds, target_names=self.encoder.classes_))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[FEATURE_COLS].fillna(0)
        return self.encoder.inverse_transform(self.model.predict(X))

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[FEATURE_COLS].fillna(0)
        probs = self.model.predict_proba(X)
        return pd.DataFrame(probs, columns=self.encoder.classes_)

    def save(self, path: str) -> None:
        joblib.dump({"model": self.model, "encoder": self.encoder}, path)

    @classmethod
    def load(cls, path: str) -> "RoadSurfaceClassifier":
        obj  = cls()
        data = joblib.load(path)
        obj.model, obj.encoder = data["model"], data["encoder"]
        return obj
