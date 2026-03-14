"""
Classifies each vehicle into one of four usage archetypes:
  highway_cruiser | city_commuter | mixed_use | offroad_explorer
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from pathlib import Path

FEATURE_COLS = [
    "mean_speed", "std_speed", "max_speed",
    "mean_accel", "std_accel",
    "mean_brake", "max_brake",
    "mean_vibration", "max_vibration",
    "mean_rpm", "mean_throttle", "mean_fuel_rate",
    "secs_over_100", "secs_idle",
    "mean_jerk", "std_jerk",
    "hard_brake_rate", "harsh_accel_rate", "high_vib_rate",
]
TARGET_COL = "archetype"


class ArchetypeClassifier:
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.model   = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.encoder = LabelEncoder()

    def fit(self, df: pd.DataFrame) -> "ArchetypeClassifier":
        X = df[FEATURE_COLS].fillna(0)
        y = self.encoder.fit_transform(df[TARGET_COL])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        print("Archetype Classifier Report:")
        print(classification_report(y_test, preds, target_names=self.encoder.classes_))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[FEATURE_COLS].fillna(0)
        return self.encoder.inverse_transform(self.model.predict(X))

    def feature_importances(self) -> pd.Series:
        return (
            pd.Series(self.model.feature_importances_, index=FEATURE_COLS)
            .sort_values(ascending=False)
        )

    def save(self, path: str) -> None:
        joblib.dump({"model": self.model, "encoder": self.encoder}, path)

    @classmethod
    def load(cls, path: str) -> "ArchetypeClassifier":
        obj  = cls()
        data = joblib.load(path)
        obj.model, obj.encoder = data["model"], data["encoder"]
        return obj
