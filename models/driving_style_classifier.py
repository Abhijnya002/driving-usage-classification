"""
Classifies driving style from per-second telemetry.
Labels: aggressive | moderate | eco
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

FEATURE_COLS = [
    "speed_kmh", "acceleration", "braking_force",
    "engine_rpm", "throttle_pct", "fuel_rate_lph",
]
TARGET_COL = "driving_style"


class DrivingStyleClassifier:
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)),
        ])
        self.encoder = LabelEncoder()

    def fit(self, df: pd.DataFrame) -> "DrivingStyleClassifier":
        X = df[FEATURE_COLS].fillna(0)
        y = self.encoder.fit_transform(df[TARGET_COL])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.pipeline.fit(X_train, y_train)
        preds = self.pipeline.predict(X_test)
        print("Driving Style Classifier Report:")
        print(classification_report(y_test, preds, target_names=self.encoder.classes_))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[FEATURE_COLS].fillna(0)
        return self.encoder.inverse_transform(self.pipeline.predict(X))

    def save(self, path: str) -> None:
        joblib.dump({"pipeline": self.pipeline, "encoder": self.encoder}, path)

    @classmethod
    def load(cls, path: str) -> "DrivingStyleClassifier":
        obj  = cls()
        data = joblib.load(path)
        obj.pipeline, obj.encoder = data["pipeline"], data["encoder"]
        return obj
