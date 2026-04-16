"""
Probabilistic synthetic health-variable generation for the mission timeline.

Takes the BioGears physiological segment (baseline + EVA + recovery) and
extends it to a 24–72 hour mission at 1-minute resolution by:
  1. Extracting representative stats from each BioGears phase.
  2. Building per-phase synthetic time-series from probability distributions.
  3. Applying event modifiers from the discrete-event engine.
"""
from __future__ import annotations

from typing import List, Tuple

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


# ── Main mission-timeline builder ──────────────────────────────────────────────

def build_mission_timeline(
    biogears_df: pd.DataFrame,
    events: List[MissionEvent],
    mission_hours: int,
    eva_duration_min: float,
    recovery_min: float,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Construct a full mission time-series at 1-minute resolution.

    Columns returned:
        minute, HeartRate, OxygenSaturation, CoreTemperature,
        RespirationRate, event_label
    """
    rng = np.random.default_rng(seed)
    mission_min = mission_hours * 60
    minutes = np.arange(mission_min)

    # ── Extract BioGears phase profiles (in seconds) ──────────────────────────
    # BioGears scenario uses 30s baseline, BG_EVA_DURATION_MIN exercise, BG_RECOVERY_MIN recovery.
    # health_vars resamples those shapes to fill the actual mission durations.
    from simulation.biogears import BG_EVA_DURATION_MIN, BG_RECOVERY_MIN
    baseline_end_s = 30.0
    eva_end_s      = baseline_end_s + BG_EVA_DURATION_MIN * 60.0
    total_s        = eva_end_s + BG_RECOVERY_MIN * 60.0

    def _phase(col: str, t0: float, t1: float) -> np.ndarray:
        return _extract_phase(biogears_df, col, t0, t1) if col in biogears_df.columns else np.array([0.0])

    # Raw BioGears phase arrays (seconds resolution)
    bg_base_hr   = _phase("HeartRate",        0,            baseline_end_s)
    bg_eva_hr    = _phase("HeartRate",        baseline_end_s, eva_end_s)
    bg_rec_hr    = _phase("HeartRate",        eva_end_s,    total_s)

    bg_base_spo2 = _phase("OxygenSaturation", 0,            baseline_end_s)
    bg_eva_spo2  = _phase("OxygenSaturation", baseline_end_s, eva_end_s)
    bg_rec_spo2  = _phase("OxygenSaturation", eva_end_s,    total_s)

    bg_base_temp = _phase("CoreTemperature",  0,            baseline_end_s)
    bg_eva_temp  = _phase("CoreTemperature",  baseline_end_s, eva_end_s)
    bg_rec_temp  = _phase("CoreTemperature",  eva_end_s,    total_s)

    bg_base_rr   = _phase("RespirationRate",  0,            baseline_end_s)
    bg_eva_rr    = _phase("RespirationRate",  baseline_end_s, eva_end_s)
    bg_rec_rr    = _phase("RespirationRate",  eva_end_s,    total_s)

    # Reference baseline statistics (Normal distribution parameters)
    hr_base   = float(bg_base_hr.mean())
    spo2_base = float(bg_base_spo2.mean()) if bg_base_spo2.mean() > 1 else float(bg_base_spo2.mean())
    # OxygenSaturation in BioGears is unitless 0–1
    if spo2_base > 1.1:          # stored as % (0–100)
        spo2_base /= 100.0
    temp_base = float(bg_base_temp.mean())
    rr_base   = float(bg_base_rr.mean())

    # ── Initialise output arrays ──────────────────────────────────────────────
    hr_arr   = np.full(mission_min, hr_base)
    spo2_arr = np.full(mission_min, spo2_base)
    temp_arr = np.full(mission_min, temp_base)
    rr_arr   = np.full(mission_min, rr_base)

    # Baseline Gaussian noise (probabilistic input requirement)
    hr_arr   += rng.normal(0, 2.0,    mission_min)
    spo2_arr += rng.normal(0, 0.003,  mission_min)
    temp_arr += rng.normal(0, 0.05,   mission_min)
    rr_arr   += rng.normal(0, 0.5,    mission_min)

    # ── Apply event physiology ────────────────────────────────────────────────
    for ev in events:
        s = max(0, ev.start_min)
        e = min(ev.end_min, mission_min)
        dur = e - s
        if dur <= 0:
            continue

        if ev.event_type == EventType.EVA:
            # Use BioGears exercise-phase shape, resampled to EVA duration
            hr_arr[s:e]   = _resample(bg_eva_hr, dur)   + rng.normal(0, 1.5, dur)
            spo2_arr[s:e] = _resample(bg_eva_spo2, dur) + rng.normal(0, 0.002, dur)
            temp_arr[s:e] = _resample(bg_eva_temp, dur) + rng.normal(0, 0.03, dur)
            rr_arr[s:e]   = _resample(bg_eva_rr, dur)   + rng.normal(0, 0.3, dur)

            # Recovery tail (if space left before next event)
            rec_start = e
            rec_end   = min(e + int(recovery_min), mission_min)
            rec_dur   = rec_end - rec_start
            if rec_dur > 0:
                hr_arr[rec_start:rec_end]   = _resample(bg_rec_hr,   rec_dur) + rng.normal(0, 1.0, rec_dur)
                spo2_arr[rec_start:rec_end] = _resample(bg_rec_spo2, rec_dur) + rng.normal(0, 0.002, rec_dur)
                temp_arr[rec_start:rec_end] = _resample(bg_rec_temp, rec_dur) + rng.normal(0, 0.02, rec_dur)
                rr_arr[rec_start:rec_end]   = _resample(bg_rec_rr,   rec_dur) + rng.normal(0, 0.2, rec_dur)

        elif ev.event_type == EventType.SLEEP:
            # Sleep physiology: lower HR, cooler core, stable SpO2
            hr_arr[s:e]   = rng.normal(58.0, 3.0, dur)
            spo2_arr[s:e] = rng.normal(0.971, 0.003, dur)
            temp_arr[s:e] = rng.normal(36.4, 0.05, dur)
            rr_arr[s:e]   = rng.normal(12.0, 0.5, dur)

        elif ev.event_type == EventType.DEHYDRATION:
            # Gradual HR drift upward; slight SpO2 drop
            drift_hr   = np.linspace(0, 15.0 * ev.intensity, dur)
            drift_spo2 = np.linspace(0, -0.008 * ev.intensity, dur)
            hr_arr[s:e]   += drift_hr   + rng.normal(0, 1.0, dur)
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
        "minute":          minutes,
        "HeartRate":       hr_arr,
        "OxygenSaturation": spo2_arr,
        "CoreTemperature": temp_arr,
        "RespirationRate": rr_arr,
        "event_label":     labels,
    })
