"""
End-to-end runner:
  1. Generate simulated fleet telemetry (if not present)
  2. Run PySpark feature engineering pipeline
  3. Train archetype, road-surface, and driving-style classifiers
  4. Output predictions + durability proxy scores
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

DATA_PATH    = Path("data/fleet_telemetry.csv")
OUTPUT_DIR   = Path("output")
MODELS_DIR   = Path("output/models")


def _generate_data() -> None:
    from data.generate_data import generate_fleet
    DATA_PATH.parent.mkdir(exist_ok=True)
    df = generate_fleet(n_vehicles=40, seconds_per_vehicle=1800)
    df.to_csv(DATA_PATH, index=False)
    print(f"Generated {len(df):,} rows -> {DATA_PATH}")


def _run_spark_pipeline() -> None:
    from pipeline.pipeline import run
    OUTPUT_DIR.mkdir(exist_ok=True)
    run(str(DATA_PATH), str(OUTPUT_DIR))


def _train_models() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    feat_path = OUTPUT_DIR / "vehicle_features.csv"
    if not feat_path.exists():
        print("vehicle_features.csv not found — run pipeline first.")
        sys.exit(1)

    features = pd.read_csv(feat_path)

    from models.archetype_classifier   import ArchetypeClassifier
    from models.road_surface_classifier import RoadSurfaceClassifier
    from models.driving_style_classifier import DrivingStyleClassifier

    arch  = ArchetypeClassifier().fit(features)
    arch.save(MODELS_DIR / "archetype_model.joblib")
    print("Feature importances (archetype):")
    print(arch.feature_importances().head(8).to_string())

    # Road-surface and driving-style classifiers train on raw per-second data
    raw = pd.read_csv(DATA_PATH, nrows=50_000)

    surf = RoadSurfaceClassifier().fit(raw)
    surf.save(MODELS_DIR / "road_surface_model.joblib")

    style = DrivingStyleClassifier().fit(raw)
    style.save(MODELS_DIR / "driving_style_model.joblib")


def _predict_and_score() -> None:
    feat_path = OUTPUT_DIR / "vehicle_features.csv"
    features  = pd.read_csv(feat_path)

    from models.archetype_classifier    import ArchetypeClassifier
    arch  = ArchetypeClassifier.load(MODELS_DIR / "archetype_model.joblib")
    features["predicted_archetype"] = arch.predict(features)

    # Durability proxy scores
    features["suspension_wear_idx"] = (
        features["mean_vibration"] * 2.0
        + features["high_vib_rate"] * 5.0
        + (features["secs_over_100"] / features["total_seconds"].clip(lower=1)) * 1.5
    ).clip(0, 10)

    features["brake_wear_idx"] = (
        features["mean_brake"] * 1.5
        + features["hard_brake_rate"] * 8.0
    ).clip(0, 10)

    features["engine_stress_idx"] = (
        features["mean_rpm"] / 1000.0
        + features["mean_throttle"] / 20.0
    ).clip(0, 10)

    out = OUTPUT_DIR / "predictions.csv"
    features[[
        "vehicle_id", "archetype", "predicted_archetype",
        "suspension_wear_idx", "brake_wear_idx", "engine_stress_idx",
    ]].to_csv(out, index=False)
    print(f"Predictions saved to {out}")
    print(features.groupby("predicted_archetype")[["suspension_wear_idx", "brake_wear_idx"]].mean().round(3))


def main() -> None:
    parser = argparse.ArgumentParser(description="Driving Usage Classification Pipeline")
    parser.add_argument("--generate", action="store_true", help="Generate simulated telemetry data")
    parser.add_argument("--pipeline", action="store_true", help="Run PySpark feature pipeline")
    parser.add_argument("--train",    action="store_true", help="Train classification models")
    parser.add_argument("--predict",  action="store_true", help="Run predictions and score durability")
    parser.add_argument("--all",      action="store_true", help="Run full end-to-end pipeline")
    args = parser.parse_args()

    if args.all:
        args.generate = args.pipeline = args.train = args.predict = True

    if args.generate:
        _generate_data()
    if args.pipeline:
        _run_spark_pipeline()
    if args.train:
        _train_models()
    if args.predict:
        _predict_and_score()

    if not any([args.generate, args.pipeline, args.train, args.predict]):
        parser.print_help()


if __name__ == "__main__":
    main()
