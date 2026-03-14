-- Raw telemetry table (one row per second per vehicle)
CREATE TABLE IF NOT EXISTS telemetry (
    timestamp       TIMESTAMP       NOT NULL,
    vehicle_id      VARCHAR(10)     NOT NULL,
    archetype       VARCHAR(30),
    road_surface    VARCHAR(20),
    driving_style   VARCHAR(20),
    speed_kmh       DOUBLE PRECISION,
    acceleration    DOUBLE PRECISION,
    braking_force   DOUBLE PRECISION,
    vibration_g     DOUBLE PRECISION,
    engine_rpm      INTEGER,
    throttle_pct    DOUBLE PRECISION,
    fuel_rate_lph   DOUBLE PRECISION,
    PRIMARY KEY (vehicle_id, timestamp)
);

-- Per-vehicle trip-level feature table
CREATE TABLE IF NOT EXISTS vehicle_features (
    vehicle_id          VARCHAR(10)     PRIMARY KEY,
    archetype           VARCHAR(30),
    road_surface        VARCHAR(20),
    driving_style       VARCHAR(20),
    mean_speed          DOUBLE PRECISION,
    std_speed           DOUBLE PRECISION,
    max_speed           DOUBLE PRECISION,
    mean_accel          DOUBLE PRECISION,
    std_accel           DOUBLE PRECISION,
    mean_brake          DOUBLE PRECISION,
    max_brake           DOUBLE PRECISION,
    mean_vibration      DOUBLE PRECISION,
    max_vibration       DOUBLE PRECISION,
    mean_rpm            DOUBLE PRECISION,
    mean_throttle       DOUBLE PRECISION,
    mean_fuel_rate      DOUBLE PRECISION,
    secs_over_100       INTEGER,
    secs_idle           INTEGER,
    hard_brake_events   INTEGER,
    harsh_accel_events  INTEGER,
    high_vibration_events INTEGER,
    predicted_archetype VARCHAR(30),
    predicted_surface   VARCHAR(20),
    predicted_style     VARCHAR(20)
);

-- Durability correlation summary
CREATE TABLE IF NOT EXISTS durability_targets (
    vehicle_id          VARCHAR(10) PRIMARY KEY,
    predicted_archetype VARCHAR(30),
    predicted_surface   VARCHAR(20),
    predicted_style     VARCHAR(20),
    suspension_wear_idx DOUBLE PRECISION,
    brake_wear_idx      DOUBLE PRECISION,
    engine_stress_idx   DOUBLE PRECISION,
    FOREIGN KEY (vehicle_id) REFERENCES vehicle_features(vehicle_id)
);
