"""
Mission-level fatigue accumulation model.

Unified equation incorporating:
  - Microgravity deconditioning factor D = 1 + 0.008 × mission_day
  - Hydration penalty  (dehydration makes fatigue accumulate faster)
  - Food/caloric boost (good nutrition speeds recovery)
  - HR-normalised effort signal for EVA accumulation
  - Fixed sleep recovery, passive rest recovery

BioGears' own FatigueLevel is displayed alongside this model in the UI.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from simulation.events import EventType, MissionEvent, get_event_at_minute


# ── Base rate constants (per minute, Earth physiology, fully hydrated/fed) ──────
_RATE_EVA_ACCUMULATE  = 0.0080   # × D × hr_norm × hydration_penalty
_RATE_SLEEP_RECOVER   = 0.0060   # ÷ D × recovery_boost
_RATE_PASSIVE_RECOVER = 0.0020   # ÷ D × (1 − hr_norm) × recovery_boost


def compute_fatigue(
    hr_arr: np.ndarray,
    events: List[MissionEvent],
    threshold: float = 0.80,
    baseline_hr: float = 72.0,
    max_hr: float = 200.0,
    mission_day: int = 0,
    hydration_arr: Optional[np.ndarray] = None,
    food_arr: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Compute per-minute fatigue for the mission timeline.

    Parameters
    ----------
    hr_arr        : heart-rate array at 1-minute resolution
    events        : list of MissionEvent from the event engine
    threshold     : risk threshold (fatigue > threshold → at risk)
    baseline_hr   : resting HR used to normalise the effort signal
    max_hr        : theoretical maximum HR (upper bound for normalisation)
    mission_day   : days the astronaut has been in microgravity before EVA.
                    Drives deconditioning factor D = 1 + 0.008 × mission_day.
    hydration_arr : per-minute hydration level [0, 1]. 1 = fully hydrated.
                    Defaults to all-ones (no effect) if not provided.
    food_arr      : per-minute caloric/food level [0, 1]. 1 = well-fed.
                    Defaults to all-ones (no effect) if not provided.

    Fatigue update equations
    ------------------------
    D = 1 + 0.008 × mission_day                          (deconditioning)
    hydration_penalty = 1 + 0.4 × (1 − hydration[i])    (1.0–1.4)
    recovery_boost    = 0.6 + 0.4 × food[i]             (0.6–1.0)

    EVA:   Δ = +0.008 × D × hr_norm × hydration_penalty
    Sleep: Δ = −0.006 ÷ D × recovery_boost
    Rest:  Δ = −0.002 ÷ D × (1 − hr_norm) × recovery_boost

    Returns
    -------
    fatigue      : float array [0, 1], shape = (mission_min,)
    risk_periods : list of (start_min, end_min) tuples for continuous risk windows
    """
    mission_min = len(hr_arr)
    fatigue = np.zeros(mission_min)

    # Microgravity deconditioning — grows with days spent in space
    D = 1.0 + 0.008 * max(0, mission_day)

    # Default to neutral (no penalty / no boost) if arrays not supplied
    if hydration_arr is None:
        hydration_arr = np.ones(mission_min)
    if food_arr is None:
        food_arr = np.ones(mission_min)

    for i in range(1, mission_min):
        prev = fatigue[i - 1]

        # HR normalised to workload fraction above resting baseline
        hr_norm = float(np.clip(
            (hr_arr[i] - baseline_hr) / max(max_hr - baseline_hr, 1.0),
            0.0, 1.0,
        ))

        h = float(np.clip(hydration_arr[i], 0.0, 1.0))
        f = float(np.clip(food_arr[i], 0.0, 1.0))

        # Scalars derived from hydration and food
        hydration_penalty = 1.0 + 0.4 * (1.0 - h)   # range [1.0, 1.4]
        recovery_boost    = 0.6 + 0.4 * f             # range [0.6, 1.0]

        ev = get_event_at_minute(events, i)

        if ev is not None and ev.event_type == EventType.EVA:
            delta = _RATE_EVA_ACCUMULATE * D * hr_norm * hydration_penalty
        elif ev is not None and ev.event_type == EventType.SLEEP:
            delta = -(_RATE_SLEEP_RECOVER / D) * recovery_boost
        else:
            delta = -(_RATE_PASSIVE_RECOVER / D) * (1.0 - hr_norm) * recovery_boost

        fatigue[i] = float(np.clip(prev + delta, 0.0, 1.0))

    # ── Identify continuous risk windows ──────────────────────────────────────
    risk_periods: List[Tuple[int, int]] = []
    in_risk = False
    start = 0
    for i, f_val in enumerate(fatigue):
        if f_val > threshold and not in_risk:
            in_risk = True
            start = i
        elif f_val <= threshold and in_risk:
            risk_periods.append((start, i))
            in_risk = False
    if in_risk:
        risk_periods.append((start, mission_min))

    return fatigue, risk_periods


def normalise_biogears_fatigue(biogears_df, mission_min: int) -> np.ndarray:
    """
    Resample BioGears' own FatigueLevel (seconds resolution) to the
    mission timeline (minutes resolution, length = mission_min).
    Returns an array of zeros if the column is absent.
    """
    if "FatigueLevel" not in biogears_df.columns:
        return np.zeros(mission_min)

    bg_fatigue = biogears_df["FatigueLevel"].values.astype(float)

    if len(bg_fatigue) >= mission_min:
        resampled = np.interp(
            np.linspace(0, len(bg_fatigue) - 1, mission_min),
            np.arange(len(bg_fatigue)),
            bg_fatigue,
        )
    else:
        pad = np.zeros(mission_min)
        resampled_segment = np.interp(
            np.linspace(0, len(bg_fatigue) - 1, len(bg_fatigue)),
            np.arange(len(bg_fatigue)),
            bg_fatigue,
        )
        pad[:len(resampled_segment)] = resampled_segment
        resampled = pad

    return np.clip(resampled, 0.0, 1.0)
