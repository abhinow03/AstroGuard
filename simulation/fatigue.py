"""
Mission-level fatigue accumulation model.

Fatigue is a state variable in [0, 1] that:
  - rises during EVA proportional to normalised HR above baseline
  - falls faster during Sleep events
  - falls slowly during rest
  - is clipped to [0, 1] at every step

The model is deliberately simple so it can be explained and defended.
BioGears' own FatigueLevel is displayed alongside this model in the UI.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from simulation.events import EventType, MissionEvent, get_event_at_minute


# ── Rate constants (per minute) ────────────────────────────────────────────────
_RATE_EVA_ACCUMULATE  = 0.0080   # fatigue gained per min at full normalised HR
_RATE_SLEEP_RECOVER   = 0.0060   # fatigue lost per min during sleep
_RATE_PASSIVE_RECOVER = 0.0020   # fatigue lost per min during rest (scaled by 1 - hr_norm)


def compute_fatigue(
    hr_arr: np.ndarray,
    events: List[MissionEvent],
    threshold: float = 0.80,
    baseline_hr: float = 72.0,
    max_hr: float = 200.0,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Compute per-minute fatigue for the mission timeline.

    Parameters
    ----------
    hr_arr       : heart-rate array at 1-minute resolution
    events       : list of MissionEvent from the event engine
    threshold    : risk threshold (fatigue > threshold → at risk)
    baseline_hr  : resting HR used to normalise the effort signal
    max_hr       : theoretical maximum HR (upper bound for normalisation)

    Returns
    -------
    fatigue      : float array [0, 1], shape = (mission_min,)
    risk_periods : list of (start_min, end_min) tuples for continuous risk windows
    """
    mission_min = len(hr_arr)
    fatigue = np.zeros(mission_min)

    for i in range(1, mission_min):
        prev = fatigue[i - 1]

        # HR normalised to workload fraction above resting baseline
        hr_norm = float(np.clip(
            (hr_arr[i] - baseline_hr) / max(max_hr - baseline_hr, 1.0),
            0.0, 1.0
        ))

        ev = get_event_at_minute(events, i)

        if ev is not None and ev.event_type == EventType.EVA:
            delta = _RATE_EVA_ACCUMULATE * hr_norm
        elif ev is not None and ev.event_type == EventType.SLEEP:
            delta = -_RATE_SLEEP_RECOVER
        else:
            # Passive recovery – faster when HR is close to baseline
            delta = -_RATE_PASSIVE_RECOVER * (1.0 - hr_norm)

        fatigue[i] = float(np.clip(prev + delta, 0.0, 1.0))

    # ── Identify continuous risk windows ──────────────────────────────────────
    risk_periods: List[Tuple[int, int]] = []
    in_risk = False
    start = 0
    for i, f in enumerate(fatigue):
        if f > threshold and not in_risk:
            in_risk = True
            start = i
        elif f <= threshold and in_risk:
            risk_periods.append((start, i))
            in_risk = False
    if in_risk:
        risk_periods.append((start, mission_min))

    return fatigue, risk_periods


def normalise_biogears_fatigue(biogears_df, mission_min: int) -> np.ndarray:
    """
    Resample BioGears' own FatigueLevel (seconds resolution) to the
    mission timeline (minutes resolution, length = mission_min).

    If the column is absent, returns an array of zeros.
    """
    if "FatigueLevel" not in biogears_df.columns:
        return np.zeros(mission_min)

    bg_fatigue = biogears_df["FatigueLevel"].values.astype(float)

    # BioGears FatigueLevel covers only the EVA segment duration.
    # We zero-pad to mission length (fatigue was zero before EVA starts
    # and we let our own model take over after the segment ends).
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
