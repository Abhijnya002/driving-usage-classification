"""
Main PySpark pipeline entry point.
Runs ingestion -> feature engineering -> saves feature table.
"""

import argparse
from pathlib import Path

from pyspark.sql import SparkSession

from pipeline.data_ingestion import load_telemetry
from pipeline.feature_engineering import compute_trip_features, compute_event_features


def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("DrivingUsageClassification")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def run(input_path: str, output_dir: str) -> None:
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    raw = load_telemetry(spark, input_path)
    raw.cache()
    print(f"Loaded {raw.count():,} rows from {input_path}")

    trip_feats  = compute_trip_features(raw)
    event_feats = compute_event_features(raw)

    features = trip_feats.join(event_feats, on="vehicle_id", how="left")

    out = Path(output_dir)
    features.toPandas().to_csv(out / "vehicle_features.csv", index=False)
    print(f"Feature table saved to {out / 'vehicle_features.csv'}")

    raw.createOrReplaceTempView("telemetry")
    _run_sql_analytics(spark, out)

    spark.stop()


def _run_sql_analytics(spark: SparkSession, out: Path) -> None:
    summary = spark.sql("""
        SELECT
            road_surface,
            driving_style,
            COUNT(DISTINCT vehicle_id) AS vehicles,
            ROUND(AVG(speed_kmh), 2)   AS avg_speed,
            ROUND(AVG(vibration_g), 4) AS avg_vibration,
            ROUND(AVG(engine_rpm), 0)  AS avg_rpm
        FROM telemetry
        GROUP BY road_surface, driving_style
        ORDER BY road_surface, driving_style
    """)
    summary.show(truncate=False)
    summary.toPandas().to_csv(out / "surface_style_summary.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/fleet_telemetry.csv")
    parser.add_argument("--output", default="output")
    args = parser.parse_args()

    Path(args.output).mkdir(exist_ok=True)
    run(args.input, args.output)
