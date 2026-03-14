# Driving Usage Classification Pipeline

A PySpark + scikit-learn pipeline that classifies **usage archetypes** and **drive events** across simulated fleet telemetry data, and surfaces road-surface / driving-style correlations for downstream durability analysis.

---

## Overview

Modern vehicle fleets generate continuous telemetry (speed, acceleration, vibration, RPM, etc.). This project shows how to:

1. **Simulate** realistic per-second telemetry for a fleet of 40 vehicles across four usage archetypes.
2. **Ingest & aggregate** raw signals into trip-level features using a PySpark pipeline.
3. **Classify** each vehicle's usage archetype, road surface, and driving style with scikit-learn models.
4. **Score** durability proxy indices (suspension wear, brake wear, engine stress) driven by the predicted labels.

---

## Architecture

```
raw telemetry (CSV)
       │
       ▼
┌─────────────────────────────┐
│   PySpark Feature Pipeline  │   data_ingestion.py
│   - Trip-level aggregation  │   feature_engineering.py
│   - Event flagging          │   pipeline.py
│   - SQL analytics           │
└────────────┬────────────────┘
             │  vehicle_features.csv
             ▼
┌─────────────────────────────────────────┐
│         scikit-learn Classifiers        │
│  ArchetypeClassifier   (RandomForest)   │
│  RoadSurfaceClassifier (GradientBoosting│
│  DrivingStyleClassifier(RF + Scaler)    │
└────────────┬────────────────────────────┘
             │  predictions.csv
             ▼
    Durability proxy scores
```

---

## Usage Archetypes

| Archetype | Typical Surface | Driving Style | Characteristics |
|---|---|---|---|
| `highway_cruiser` | Highway (70%) | Moderate | High speed, low vibration, steady throttle |
| `city_commuter` | Urban (75%) | Mixed | Frequent stops, moderate acceleration |
| `mixed_use` | Mixed | Mixed | Balanced across surface types |
| `offroad_explorer` | Off-road (50%) | Aggressive | High vibration, low speed, rough terrain |

---

## Project Structure

```
driving-usage-classification/
├── data/
│   ├── generate_data.py          # Fleet telemetry simulator
│   └── fleet_telemetry.csv       # Generated (not committed)
├── pipeline/
│   ├── data_ingestion.py         # PySpark CSV loader with schema
│   ├── feature_engineering.py    # Trip-level + event-level features
│   └── pipeline.py               # End-to-end Spark pipeline runner
├── models/
│   ├── archetype_classifier.py   # RandomForest usage-archetype model
│   ├── road_surface_classifier.py# GradientBoosting road-surface model
│   └── driving_style_classifier.py # RF + StandardScaler style model
├── sql/
│   ├── create_tables.sql         # Schema definitions
│   └── analytics_queries.sql     # Analytical SQL queries
├── config/
│   └── config.yaml               # Thresholds and hyperparameters
├── main.py                       # CLI entry point
└── requirements.txt
```

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/Abhijnya002/driving-usage-classification.git
cd driving-usage-classification

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Java requirement:** PySpark requires Java 8 or 11.  
> Check with `java -version`. Install OpenJDK 11 if needed.

---

## Running the Pipeline

### Full end-to-end (recommended)

```bash
python main.py --all
```

This runs all four steps in sequence.

### Step by step

```bash
# Step 1 — Generate 40-vehicle simulated telemetry (~72k rows)
python main.py --generate

# Step 2 — PySpark feature engineering (outputs vehicle_features.csv)
python main.py --pipeline

# Step 3 — Train all three classifiers
python main.py --train

# Step 4 — Predict archetypes + score durability
python main.py --predict
```

---

## Features Engineered

### Trip-level (per vehicle)

| Feature | Description |
|---|---|
| `mean_speed`, `std_speed`, `max_speed` | Speed distribution over trip |
| `mean_accel`, `std_accel` | Acceleration profile |
| `mean_brake`, `max_brake` | Braking intensity |
| `mean_vibration`, `max_vibration` | Road roughness proxy |
| `mean_rpm`, `mean_throttle` | Engine load indicators |
| `secs_over_100` | High-speed driving time |
| `secs_idle` | Time at near-zero speed |
| `mean_jerk`, `std_jerk` | Smoothness of acceleration changes |

### Event-level (per vehicle)

| Feature | Threshold |
|---|---|
| `hard_brake_rate` | braking_force > 3.0 g |
| `harsh_accel_rate` | acceleration > 2.5 m/s² |
| `high_vib_rate` | vibration_g > 2.0 g |

---

## Models

### ArchetypeClassifier
- **Algorithm:** Random Forest (100 trees)
- **Target:** `highway_cruiser` / `city_commuter` / `mixed_use` / `offroad_explorer`
- **Input:** 19 trip-level features

### RoadSurfaceClassifier
- **Algorithm:** Gradient Boosting (80 trees)
- **Target:** `highway` / `urban` / `rural` / `offroad`
- **Input:** 6 per-second telemetry signals

### DrivingStyleClassifier
- **Algorithm:** Random Forest + StandardScaler pipeline
- **Target:** `aggressive` / `moderate` / `eco`
- **Input:** 6 per-second telemetry signals

---

## Durability Proxy Scores

After prediction, three scores (0–10) are computed per vehicle:

| Score | Formula |
|---|---|
| `suspension_wear_idx` | `mean_vibration × 2 + high_vib_rate × 5 + high_speed_ratio × 1.5` |
| `brake_wear_idx` | `mean_brake × 1.5 + hard_brake_rate × 8` |
| `engine_stress_idx` | `mean_rpm / 1000 + mean_throttle / 20` |

These scores feed downstream durability target setting in a real fleet engineering workflow.

---

## SQL Analytics

[sql/analytics_queries.sql](sql/analytics_queries.sql) includes:

- Average telemetry signals grouped by road surface × driving style
- Hard-braking event rates per archetype
- Top 10 vehicles by predicted suspension wear
- Vibration statistics per road surface

---

## Sample Output

```
Archetype Classifier Report:
                  precision  recall  f1-score  support
city_commuter        0.97    0.95      0.96       21
highway_cruiser      0.95    0.96      0.95       23
mixed_use            0.91    0.92      0.91       25
offroad_explorer     0.98    0.99      0.98       19

Durability scores by predicted archetype:
                  suspension_wear_idx  brake_wear_idx
city_commuter                   1.23            0.87
highway_cruiser                 0.94            0.71
mixed_use                       1.41            1.02
offroad_explorer                3.87            1.54
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Distributed processing | Apache Spark (PySpark 3.5) |
| Machine learning | scikit-learn 1.4 |
| Data manipulation | pandas 2.2, NumPy 1.26 |
| Serialization | joblib, PyArrow (Parquet) |
| Query layer | Spark SQL |
| Config | PyYAML |

---

## License

MIT
