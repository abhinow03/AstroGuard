"""
Probabilistic synthetic health-variable generation for the mission timeline.

Takes the BioGears physiological segment (baseline + EVA + recovery) and
extends it to a 24–72 hour mission at 1-minute resolution by:
  1. Extracting representative stats from each BioGears phase.
  2. Building per-phase synthetic time-series from probability distributions.
  3. Applying event modifiers from the discrete-event engine.
  4. Applying microgravity baseline HR and SpO2 offsets.

Also provides:
  build_hydration_timeline() — per-minute hydration level [0, 1]
  build_food_timeline()      — per-minute caloric level [0, 1]
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from simulation.events import EventType, MissionEvent, get_event_at_minute


# ── Internal helpers ───────────────────────────────────────────────────────────

def _resample(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Linear interpolation of `arr` to exactly `target_len` points."""
    src_idx = np.linspace(0, len(arr) - 1, target_len)
    return np.interp(src_idx, np.arange(len(arr)), arr)


def _extract_phase(df: pd.DataFrame, col: str,
                   t_start_s: float, t_end_s: float) -> np.ndarray:
    """Return the values of `col` between t_start_s and t_end_s (seconds)."""
    mask = (df["Time_s"] >= t_start_s) & (df["Time_s"] < t_end_s)
    vals = df.loc[mask, col].dropna().values
    return vals if len(vals) > 0 else np.array([df[col].mean()])


# ── Hydration timeline ─────────────────────────────────────────────────────────

def build_hydration_timeline(
    events: List[MissionEvent],
    mission_hours: int,
    eva_intensity: float,
    water_intake_L_per_rest_hour: float = 0.25,
    seed: int = 0,
) -> np.ndarray:
    """
    Build a per-minute hydration level array [0, 1] for the mission.

    Rules:
      - Starts fully hydrated (1.0).
      - Drains during EVA at a rate proportional to sweat (intensity-scaled).
      - Drains slowly during rest from basal water loss.
      - Drains during stochastic dehydration events.
      - Recovers during rest/sleep from drinking (water_intake_L_per_rest_hour).

    Parameters
    ----------
    water_intake_L_per_rest_hour : how much the astronaut drinks when not in EVA.
        0.0 = no drinking, 0.25 = modest (250 mL/h), 0.5 = good hydration.
    """
    mission_min = mission_hours * 60
    hydration   = np.ones(mission_min)

    # Drain rates (per minute)
    drain_eva         = 0.0020 * (0.5 + eva_intensity)   # more sweat at high intensity
    drain_deh_per_int = 0.0015                            # dehydration event × intensity
    drain_basal       = 0.0003                            # unavoidable insensible loss

    # Recovery: 1 L/h restores ~0.5 units per hour → 0.5/60 per min per L/h
    recover_rate = water_intake_L_per_rest_hour * (0.5 / 60.0)

    for i in range(1, mission_min):
        ev = get_event_at_minute(events, i)
        h  = hydration[i - 1]

        if ev is not None and ev.event_type == EventType.EVA:
            h -= drain_eva
        elif ev is not None and ev.event_type == EventType.DEHYDRATION:
            h -= drain_deh_per_int * ev.intensity
            h -= drain_basal
        else:
            h -= drain_basal
            h += recover_rate   # drinking during rest / sleep

        hydration[i] = float(np.clip(h, 0.0, 1.0))

    return hydration


# ── Food / caloric timeline ────────────────────────────────────────────────────

def build_food_timeline(
    events: List[MissionEvent],
    mission_hours: int,
    eva_intensity: float,
    meals_per_day: float = 3.0,
    seed: int = 0,
) -> np.ndarray:
    """
    Build a per-minute caloric/food level array [0, 1] for the mission.

    Rules:
      - Starts at 0.8 (astronaut already ate before mission start).
      - Drains faster during EVA (higher metabolic burn).
      - Drains slowly at rest (basal metabolic rate).
      - Boosted at meal times (evenly spaced through day, not during EVA).

    Parameters
    ----------
    meals_per_day : number of full meals the astronaut eats per 24-hour day.
        1 = minimal, 2 = light, 3 = standard, 4 = optimal.
    """
    mission_min = mission_hours * 60
    food = np.full(mission_min, 0.8)

    drain_eva  = 0.0030 * (0.3 + eva_intensity)   # higher intensity burns more
    drain_rest = 0.0004                             # basal metabolic drain

    # Meals spaced evenly across each 24-hour day
    meal_interval_min = int(60 * 24 / max(meals_per_day, 1))
    # Each meal restores a fixed fraction — more meals = smaller each, same total
    meal_boost = min(0.45, 0.9 / max(meals_per_day, 1))

    for i in range(1, mission_min):
        ev = get_event_at_minute(events, i)
        f  = food[i - 1]

        if ev is not None and ev.event_type == EventType.EVA:
            f -= drain_eva
        else:
            f -= drain_rest

        # Meal — skip if astronaut is mid-EVA (can't eat in a space suit)
        if i % meal_interval_min == 0:
            if ev is None or ev.event_type != EventType.EVA:
                f += meal_boost

        food[i] = float(np.clip(f, 0.0, 1.0))

    return food


# ── Main mission-timeline builder ──────────────────────────────────────────────

def build_mission_timeline(
    biogears_df: pd.DataFrame,
    events: List[MissionEvent],
    mission_hours: int,
    eva_duration_min: float,
    recovery_min: float,
    mission_day: int = 0,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Construct a full mission time-series at 1-minute resolution.

    Microgravity effects applied:
      - Resting HR elevated by 1.5 bpm × mission_day (cap +18 bpm).
      - SpO2 baseline reduced by 0.002 × min(mission_day, 5) for cephalad
        fluid shift in the first 5 days (body adapts after that).

    Columns returned:
        minute, HeartRate, OxygenSaturation, CoreTemperature,
        RespirationRate, event_label
    """
    rng = np.random.default_rng(seed)
    mission_min = mission_hours * 60
    minutes = np.arange(mission_min)

    # ── Extract BioGears phase profiles (in seconds) ──────────────────────────
    from simulation.biogears import BG_EVA_DURATION_MIN, BG_RECOVERY_MIN
    baseline_end_s = 30.0
    eva_end_s      = baseline_end_s + BG_EVA_DURATION_MIN * 60.0
    total_s        = eva_end_s + BG_RECOVERY_MIN * 60.0

    def _phase(col: str, t0: float, t1: float) -> np.ndarray:
        return _extract_phase(biogears_df, col, t0, t1) if col in biogears_df.columns else np.array([0.0])

    bg_base_hr   = _phase("HeartRate",        0,              baseline_end_s)
    bg_eva_hr    = _phase("HeartRate",        baseline_end_s, eva_end_s)
    bg_rec_hr    = _phase("HeartRate",        eva_end_s,      total_s)

    bg_base_spo2 = _phase("OxygenSaturation", 0,              baseline_end_s)
    bg_eva_spo2  = _phase("OxygenSaturation", baseline_end_s, eva_end_s)
    bg_rec_spo2  = _phase("OxygenSaturation", eva_end_s,      total_s)

    bg_base_temp = _phase("CoreTemperature",  0,              baseline_end_s)
    bg_eva_temp  = _phase("CoreTemperature",  baseline_end_s, eva_end_s)
    bg_rec_temp  = _phase("CoreTemperature",  eva_end_s,      total_s)

    bg_base_rr   = _phase("RespirationRate",  0,              baseline_end_s)
    bg_eva_rr    = _phase("RespirationRate",  baseline_end_s, eva_end_s)
    bg_rec_rr    = _phase("RespirationRate",  eva_end_s,      total_s)

    # Reference baseline statistics
    hr_base   = float(bg_base_hr.mean())
    spo2_base = float(bg_base_spo2.mean())
    if spo2_base > 1.1:
        spo2_base /= 100.0
    temp_base = float(bg_base_temp.mean())
    rr_base   = float(bg_base_rr.mean())

    # ── Microgravity baseline adjustments ────────────────────────────────────
    # Cardiovascular deconditioning raises resting HR (cap at +18 bpm)
    mg_hr_offset   = min(1.5 * max(0, mission_day), 18.0)
    hr_base        = hr_base + mg_hr_offset

    # Cephalad fluid shift lowers SpO2 in first 5 days then stabilises
    mg_spo2_offset = -0.002 * min(max(0, mission_day), 5)
    spo2_base      = spo2_base + mg_spo2_offset

    # ── Initialise output arrays ──────────────────────────────────────────────
    hr_arr   = np.full(mission_min, hr_base)
    spo2_arr = np.full(mission_min, spo2_base)
    temp_arr = np.full(mission_min, temp_base)
    rr_arr   = np.full(mission_min, rr_base)

    # Baseline Gaussian noise
    hr_arr   += rng.normal(0, 2.0,   mission_min)
    spo2_arr += rng.normal(0, 0.003, mission_min)
    temp_arr += rng.normal(0, 0.05,  mission_min)
    rr_arr   += rng.normal(0, 0.5,   mission_min)

    # ── Apply event physiology ────────────────────────────────────────────────
    for ev in events:
        s   = max(0, ev.start_min)
        e   = min(ev.end_min, mission_min)
        dur = e - s
        if dur <= 0:
            continue

        if ev.event_type == EventType.EVA:
            hr_arr[s:e]   = _resample(bg_eva_hr,   dur) + rng.normal(0, 1.5, dur)
            spo2_arr[s:e] = _resample(bg_eva_spo2, dur) + rng.normal(0, 0.002, dur)
            temp_arr[s:e] = _resample(bg_eva_temp, dur) + rng.normal(0, 0.03,  dur)
            rr_arr[s:e]   = _resample(bg_eva_rr,   dur) + rng.normal(0, 0.3,   dur)

            # Recovery tail
            rec_start = e
            rec_end   = min(e + int(recovery_min), mission_min)
            rec_dur   = rec_end - rec_start
            if rec_dur > 0:
                hr_arr[rec_start:rec_end]   = _resample(bg_rec_hr,   rec_dur) + rng.normal(0, 1.0,  rec_dur)
                spo2_arr[rec_start:rec_end] = _resample(bg_rec_spo2, rec_dur) + rng.normal(0, 0.002, rec_dur)
                temp_arr[rec_start:rec_end] = _resample(bg_rec_temp, rec_dur) + rng.normal(0, 0.02,  rec_dur)
                rr_arr[rec_start:rec_end]   = _resample(bg_rec_rr,   rec_dur) + rng.normal(0, 0.2,   rec_dur)

        elif ev.event_type == EventType.SLEEP:
            hr_arr[s:e]   = rng.normal(58.0,  3.0,   dur)
            spo2_arr[s:e] = rng.normal(0.971, 0.003, dur)
            temp_arr[s:e] = rng.normal(36.4,  0.05,  dur)
            rr_arr[s:e]   = rng.normal(12.0,  0.5,   dur)

        elif ev.event_type == EventType.DEHYDRATION:
            drift_hr   = np.linspace(0, 15.0 * ev.intensity, dur)
            drift_spo2 = np.linspace(0, -0.008 * ev.intensity, dur)
            hr_arr[s:e]   += drift_hr   + rng.normal(0, 1.0,   dur)
            spo2_arr[s:e] += drift_spo2 + rng.normal(0, 0.002, dur)

    # ── Clip to physiological limits ──────────────────────────────────────────
    hr_arr   = np.clip(hr_arr,   35,   220)
    spo2_arr = np.clip(spo2_arr, 0.70, 1.00)
    temp_arr = np.clip(temp_arr, 35.0, 41.0)
    rr_arr   = np.clip(rr_arr,   6,    50)

    # ── Event label column ────────────────────────────────────────────────────
    labels = np.array(["Rest"] * mission_min, dtype=object)
    for ev in events:
        s = max(0, ev.start_min)
        e = min(ev.end_min, mission_min)
        labels[s:e] = ev.event_type.value

    return pd.DataFrame({
        "minute":           minutes,
        "HeartRate":        hr_arr,
        "OxygenSaturation": spo2_arr,
        "CoreTemperature":  temp_arr,
        "RespirationRate":  rr_arr,
        "event_label":      labels,
    })
