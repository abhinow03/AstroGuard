"""
Risk analytics and Monte Carlo simulation.

Provides:
  - Single-run analytics (peak, time-at-risk, trend, flags)
  - Monte Carlo runner that randomises EVA timing / dehydration across N runs
  - Aggregation into mean trajectory, percentile envelope, P(risk) curve
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from simulation.events import sample_events
from simulation.health_vars import build_mission_timeline, build_hydration_timeline, build_food_timeline
from simulation.fatigue import compute_fatigue


# ── Single-run analytics ───────────────────────────────────────────────────────

def single_run_analytics(
    fatigue: np.ndarray,
    threshold: float = 0.80,
    mission_hours: int = 48,
) -> Dict:
    """
    Compute summary statistics for one fatigue trajectory.

    Returns a dict with:
        peak_fatigue, peak_minute, time_at_risk_pct,
        trend_slope, trend_label, risk_periods
    """
    mission_min = len(fatigue)

    peak_idx   = int(np.argmax(fatigue))
    peak_val   = float(fatigue[peak_idx])
    at_risk    = np.sum(fatigue > threshold)
    pct_at_risk = float(at_risk / mission_min * 100)

    # Trend: linear regression on the last 20% of the mission
    tail_start = int(mission_min * 0.80)
    tail       = fatigue[tail_start:]
    x          = np.arange(len(tail), dtype=float)
    slope, _, _, _, _ = stats.linregress(x, tail)

    if slope > 0.00005:
        trend_label = "Rising"
    elif slope < -0.00005:
        trend_label = "Falling"
    else:
        trend_label = "Stable"

    # Continuous risk windows
    risk_periods: List[Tuple[int, int]] = []
    in_risk, start = False, 0
    for i, f in enumerate(fatigue):
        if f > threshold and not in_risk:
            in_risk, start = True, i
        elif f <= threshold and in_risk:
            risk_periods.append((start, i))
            in_risk = False
    if in_risk:
        risk_periods.append((start, mission_min))

    return {
        "peak_fatigue":     peak_val,
        "peak_minute":      peak_idx,
        "time_at_risk_pct": pct_at_risk,
        "trend_slope":      float(slope),
        "trend_label":      trend_label,
        "risk_periods":     risk_periods,
    }


def mission_status(analytics: Dict, threshold: float) -> Tuple[str, str]:
    """
    Return (status_label, colour) based on analytics.
    status_label: "SAFE" | "MONITOR" | "ABORT"
    """
    peak  = analytics["peak_fatigue"]
    trend = analytics["trend_label"]
    pct   = analytics["time_at_risk_pct"]

    if peak >= threshold and (trend == "Rising" or pct > 10):
        return "ABORT",   "#dc3545"
    elif peak >= threshold * 0.85:
        return "MONITOR", "#fd7e14"
    else:
        return "SAFE",    "#28a745"


# ── Monte Carlo ────────────────────────────────────────────────────────────────

def run_monte_carlo(
    biogears_df: pd.DataFrame,
    mission_hours: int,
    eva_intensity: float,
    eva_duration_min: float,
    recovery_min: float,
    n_sims: int = 100,
    threshold: float = 0.80,
    n_evas: int = 1,
    mission_day: int = 0,
    water_intake: float = 0.25,
    meals_per_day: float = 3.0,
    base_seed: int = 42,
) -> Dict:
    """
    Run `n_sims` independent simulations with stochastic event timing.

    Each simulation randomises:
      - EVA start time (Uniform)
      - EVA duration (Normal ± 0.5 h)
      - Dehydration event occurrence (Bernoulli 0.3)
      - Health-variable noise seed

    Returns a dict:
        time_axis        : np.ndarray  [mission_min]
        fatigue_matrix   : np.ndarray  [n_sims × mission_min]
        mean_fatigue     : np.ndarray  [mission_min]
        p5_fatigue       : np.ndarray  [mission_min]  (5th percentile)
        p95_fatigue      : np.ndarray  [mission_min]  (95th percentile)
        p_risk           : np.ndarray  [mission_min]  P(fatigue > threshold)
        max_per_sim      : np.ndarray  [n_sims]
        n_sims           : int
        threshold        : float
    """
    mission_min = mission_hours * 60
    fatigue_matrix = np.zeros((n_sims, mission_min))

    for i in range(n_sims):
        seed = base_seed + i * 7
        rng  = np.random.default_rng(seed)

        # Randomise EVA intensity slightly
        intensity_i = float(np.clip(eva_intensity + rng.normal(0, 0.05), 0.1, 0.95))

        events_i = sample_events(
            mission_hours=mission_hours,
            eva_intensity=intensity_i,
            n_evas=n_evas,
            seed=seed,
        )

        mission_df_i = build_mission_timeline(
            biogears_df=biogears_df,
            events=events_i,
            mission_hours=mission_hours,
            eva_duration_min=eva_duration_min,
            recovery_min=recovery_min,
            mission_day=mission_day,
            seed=seed + 1,
        )

        hydration_i = build_hydration_timeline(
            events=events_i,
            mission_hours=mission_hours,
            eva_intensity=intensity_i,
            water_intake_L_per_rest_hour=water_intake,
            seed=seed + 2,
        )

        food_i = build_food_timeline(
            events=events_i,
            mission_hours=mission_hours,
            eva_intensity=intensity_i,
            meals_per_day=meals_per_day,
            seed=seed + 3,
        )

        fatigue_i, _ = compute_fatigue(
            hr_arr=mission_df_i["HeartRate"].values,
            events=events_i,
            threshold=threshold,
            mission_day=mission_day,
            hydration_arr=hydration_i,
            food_arr=food_i,
        )
        fatigue_matrix[i] = fatigue_i

    time_axis   = np.arange(mission_min)
    mean_fat    = fatigue_matrix.mean(axis=0)
    p5_fat      = np.percentile(fatigue_matrix, 5,  axis=0)
    p95_fat     = np.percentile(fatigue_matrix, 95, axis=0)
    p_risk      = (fatigue_matrix > threshold).mean(axis=0)
    max_per_sim = fatigue_matrix.max(axis=1)

    return {
        "time_axis":      time_axis,
        "fatigue_matrix": fatigue_matrix,
        "mean_fatigue":   mean_fat,
        "p5_fatigue":     p5_fat,
        "p95_fatigue":    p95_fat,
        "p_risk":         p_risk,
        "max_per_sim":    max_per_sim,
        "n_sims":         n_sims,
        "threshold":      threshold,
    }


def mc_summary(mc: Dict) -> Dict:
    """High-level summary statistics from Monte Carlo results."""
    p_any_breach = float(np.mean(mc["max_per_sim"] > mc["threshold"]))
    mean_peak    = float(mc["max_per_sim"].mean())
    worst_case   = float(mc["max_per_sim"].max())

    return {
        "p_any_breach": p_any_breach,
        "mean_peak_fatigue": mean_peak,
        "worst_case_fatigue": worst_case,
        "n_sims": mc["n_sims"],
    }
