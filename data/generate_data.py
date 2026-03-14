"""
Simulates fleet telemetry data for N vehicles over a time window.
Outputs a CSV with one row per second of driving.
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROAD_SURFACES = ["highway", "urban", "rural", "offroad"]
DRIVING_STYLES = ["aggressive", "moderate", "eco"]
ARCHETYPES = ["highway_cruiser", "city_commuter", "mixed_use", "offroad_explorer"]

RNG = np.random.default_rng(42)


def _road_surface_params(surface: str) -> dict:
    params = {
        "highway":  dict(speed_mean=110, speed_std=15, accel_std=0.5, vibration_mean=0.3),
        "urban":    dict(speed_mean=35,  speed_std=12, accel_std=2.0, vibration_mean=0.6),
        "rural":    dict(speed_mean=70,  speed_std=20, accel_std=1.0, vibration_mean=0.8),
        "offroad":  dict(speed_mean=25,  speed_std=10, accel_std=1.5, vibration_mean=2.5),
    }
    return params[surface]


def _style_multiplier(style: str) -> dict:
    multipliers = {
        "aggressive": dict(speed=1.15, accel=1.8, brake=1.6),
        "moderate":   dict(speed=1.00, accel=1.0, brake=1.0),
        "eco":        dict(speed=0.90, accel=0.6, brake=0.7),
    }
    return multipliers[style]


def generate_vehicle_trip(
    vehicle_id: str,
    archetype: str,
    n_seconds: int = 3600,
) -> pd.DataFrame:
    archetype_surface_bias = {
        "highway_cruiser":   [0.70, 0.15, 0.10, 0.05],
        "city_commuter":     [0.10, 0.75, 0.12, 0.03],
        "mixed_use":         [0.30, 0.35, 0.30, 0.05],
        "offroad_explorer":  [0.10, 0.10, 0.30, 0.50],
    }
    archetype_style_bias = {
        "highway_cruiser":   [0.20, 0.55, 0.25],
        "city_commuter":     [0.30, 0.50, 0.20],
        "mixed_use":         [0.25, 0.50, 0.25],
        "offroad_explorer":  [0.40, 0.45, 0.15],
    }

    surface_probs = archetype_surface_bias[archetype]
    style_probs   = archetype_style_bias[archetype]

    surface = RNG.choice(ROAD_SURFACES, p=surface_probs)
    style   = RNG.choice(DRIVING_STYLES, p=style_probs)

    sp = _road_surface_params(surface)
    sm = _style_multiplier(style)

    speeds       = np.clip(RNG.normal(sp["speed_mean"] * sm["speed"], sp["speed_std"], n_seconds), 0, 200)
    accelerations = RNG.normal(0, sp["accel_std"] * sm["accel"], n_seconds)
    braking      = np.clip(-accelerations * sm["brake"], 0, None)
    vibration    = np.clip(RNG.normal(sp["vibration_mean"], 0.2, n_seconds), 0, None)
    engine_rpm   = np.clip(speeds * 50 + RNG.normal(0, 200, n_seconds), 500, 7000)
    throttle_pct = np.clip((speeds / 200) * 100 + RNG.normal(0, 5, n_seconds), 0, 100)
    fuel_rate    = throttle_pct * 0.05 + RNG.normal(0, 0.2, n_seconds)

    timestamps = pd.date_range("2026-01-15", periods=n_seconds, freq="s")

    return pd.DataFrame({
        "timestamp":       timestamps,
        "vehicle_id":      vehicle_id,
        "archetype":       archetype,
        "road_surface":    surface,
        "driving_style":   style,
        "speed_kmh":       speeds.round(2),
        "acceleration":    accelerations.round(3),
        "braking_force":   braking.round(3),
        "vibration_g":     vibration.round(4),
        "engine_rpm":      engine_rpm.round(0).astype(int),
        "throttle_pct":    throttle_pct.round(2),
        "fuel_rate_lph":   fuel_rate.round(3),
    })


def generate_fleet(n_vehicles: int = 40, seconds_per_vehicle: int = 1800) -> pd.DataFrame:
    frames = []
    for i in range(n_vehicles):
        vid       = f"VH{i+1:04d}"
        archetype = RNG.choice(ARCHETYPES)
        frames.append(generate_vehicle_trip(vid, archetype, seconds_per_vehicle))
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    out = Path(__file__).parent / "fleet_telemetry.csv"
    df  = generate_fleet(n_vehicles=40, seconds_per_vehicle=1800)
    df.to_csv(out, index=False)
    print(f"Generated {len(df):,} rows -> {out}")
