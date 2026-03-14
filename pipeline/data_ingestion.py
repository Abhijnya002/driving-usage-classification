from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType, IntegerType, TimestampType,
)

TELEMETRY_SCHEMA = StructType([
    StructField("timestamp",      TimestampType(), True),
    StructField("vehicle_id",     StringType(),    True),
    StructField("archetype",      StringType(),    True),
    StructField("road_surface",   StringType(),    True),
    StructField("driving_style",  StringType(),    True),
    StructField("speed_kmh",      DoubleType(),    True),
    StructField("acceleration",   DoubleType(),    True),
    StructField("braking_force",  DoubleType(),    True),
    StructField("vibration_g",    DoubleType(),    True),
    StructField("engine_rpm",     IntegerType(),   True),
    StructField("throttle_pct",   DoubleType(),    True),
    StructField("fuel_rate_lph",  DoubleType(),    True),
])


def load_telemetry(spark: SparkSession, path: str) -> DataFrame:
    return (
        spark.read
        .option("header", "true")
        .schema(TELEMETRY_SCHEMA)
        .csv(path)
        .withColumn("date", F.to_date("timestamp"))
    )


def load_telemetry_parquet(spark: SparkSession, path: str) -> DataFrame:
    return spark.read.parquet(path)
