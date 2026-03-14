-- Average telemetry signals by road surface and driving style
SELECT
    road_surface,
    driving_style,
    COUNT(DISTINCT vehicle_id)      AS vehicle_count,
    ROUND(AVG(speed_kmh), 2)        AS avg_speed_kmh,
    ROUND(AVG(vibration_g), 4)      AS avg_vibration_g,
    ROUND(AVG(engine_rpm), 0)       AS avg_rpm,
    ROUND(AVG(fuel_rate_lph), 3)    AS avg_fuel_rate
FROM telemetry
GROUP BY road_surface, driving_style
ORDER BY road_surface, driving_style;

-- Hard-braking event rate per archetype
SELECT
    archetype,
    COUNT(*)                                                    AS total_seconds,
    SUM(CASE WHEN braking_force > 3.0 THEN 1 ELSE 0 END)       AS hard_brake_count,
    ROUND(
        SUM(CASE WHEN braking_force > 3.0 THEN 1 ELSE 0 END)
        / NULLIF(COUNT(*), 0)::DOUBLE PRECISION * 1000, 2
    )                                                           AS hard_brake_per_1k_sec
FROM telemetry
GROUP BY archetype
ORDER BY hard_brake_per_1k_sec DESC;

-- Vehicles with highest predicted suspension wear
SELECT
    vehicle_id,
    predicted_archetype,
    predicted_surface,
    ROUND(suspension_wear_idx, 3)   AS suspension_wear,
    ROUND(brake_wear_idx, 3)        AS brake_wear,
    ROUND(engine_stress_idx, 3)     AS engine_stress
FROM durability_targets
ORDER BY suspension_wear_idx DESC
LIMIT 10;

-- Correlation proxy: vibration vs road surface
SELECT
    road_surface,
    ROUND(AVG(vibration_g), 4)      AS mean_vib,
    ROUND(STDDEV(vibration_g), 4)   AS std_vib,
    ROUND(MAX(vibration_g), 4)      AS peak_vib
FROM telemetry
GROUP BY road_surface
ORDER BY mean_vib DESC;
