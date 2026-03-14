"""
Aggregates per-second telemetry into per-vehicle trip-level features
used as input to the sklearn classifiers.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def compute_trip_features(df: DataFrame) -> DataFrame:
    """Return one row per vehicle with aggregate driving features."""
    w = Window.partitionBy("vehicle_id").orderBy("timestamp")

    df = df.withColumn("jerk", F.col("acceleration") - F.lag("acceleration", 1).over(w))

    agg = (
        df.groupBy("vehicle_id", "archetype", "road_surface", "driving_style")
        .agg(
            F.mean("speed_kmh").alias("mean_speed"),
            F.stddev("speed_kmh").alias("std_speed"),
            F.max("speed_kmh").alias("max_speed"),
            F.mean("acceleration").alias("mean_accel"),
            F.stddev("acceleration").alias("std_accel"),
            F.mean("braking_force").alias("mean_brake"),
            F.max("braking_force").alias("max_brake"),
            F.mean("vibration_g").alias("mean_vibration"),
            F.max("vibration_g").alias("max_vibration"),
            F.mean("engine_rpm").alias("mean_rpm"),
            F.mean("throttle_pct").alias("mean_throttle"),
            F.mean("fuel_rate_lph").alias("mean_fuel_rate"),
            F.sum(F.when(F.col("speed_kmh") > 100, 1).otherwise(0)).alias("secs_over_100"),
            F.sum(F.when(F.col("speed_kmh") < 5, 1).otherwise(0)).alias("secs_idle"),
            F.mean("jerk").alias("mean_jerk"),
            F.stddev("jerk").alias("std_jerk"),
        )
    )
    return agg


def compute_event_features(df: DataFrame) -> DataFrame:
    """Flag hard-braking and harsh-acceleration events per vehicle."""
    events = (
        df.withColumn("hard_brake",   (F.col("braking_force") > 3.0).cast("int"))
          .withColumn("harsh_accel",  (F.col("acceleration")  > 2.5).cast("int"))
          .withColumn("high_vibration", (F.col("vibration_g") > 2.0).cast("int"))
          .groupBy("vehicle_id")
          .agg(
              F.sum("hard_brake").alias("hard_brake_events"),
              F.sum("harsh_accel").alias("harsh_accel_events"),
              F.sum("high_vibration").alias("high_vibration_events"),
              F.count("*").alias("total_seconds"),
          )
          .withColumn("hard_brake_rate",    F.col("hard_brake_events")   / F.col("total_seconds"))
          .withColumn("harsh_accel_rate",   F.col("harsh_accel_events")  / F.col("total_seconds"))
          .withColumn("high_vib_rate",      F.col("high_vibration_events") / F.col("total_seconds"))
    )
    return events
