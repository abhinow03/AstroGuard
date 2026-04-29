"""
Mission-level fatigue accumulation model.

Fatigue is computed from five physiological drivers, all grounded in
BioGears patient data and NASA ISS research — no abstract sliders:

  1. Glycogen depletion/replenishment  (carbs_g, meals, lean mass)
  2. Hydration state with sodium penalty (water_L, sodium_mg)
  3. Sleep debt accumulation            (sleep_hours, mission_day)
  4. Protein recovery adequacy          (protein_g, weight_kg)
  5. Microgravity deconditioning        (MicrogravityFactors from patient.py)

When bg_calibration is supplied (from extract_biogears_calibration()), the model
uses BioGears-measured rates instead of the hardcoded constants:
  - eva_fatigue_rate_per_min      replaces _RATE_EVA_ACCUMULATE
  - glycogen_depletion_g_per_min  replaces _GLY_DEPLETION_PER_MIN
  - glycogen_start_g              replaces glycogen_max * 0.80
  - sweat_rate_mg_per_min         feeds _hydration_state() via actual water loss
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

from simulation.events import EventType, MissionEvent, get_event_at_minute

if TYPE_CHECKING:
    from simulation.patient import MicrogravityFactors, PatientProfile


# ── Base rate constants (per minute, Earth physiology baseline) ─────────────
_RATE_EVA_ACCUMULATE  = 0.0080   # scaled by VO2max degradation + all penalties
_RATE_SLEEP_RECOVER   = 0.0060   # scaled by VO2max factor, protein, sleep hours
_RATE_PASSIVE_RECOVER = 0.0020   # passive rest recovery


def compute_fatigue(
    hr_arr:             np.ndarray,
    events:             List[MissionEvent],
    threshold:          float = 0.80,
    # ── Patient profile (replaces bare baseline_hr / max_hr) ─────────────────
    patient:            Optional["PatientProfile"]        = None,
    mg_factors:         Optional["MicrogravityFactors"]   = None,
    # ── Macronutrient inputs (BioGears NutritionProfile units) ───────────────
    carb_g_per_meal:    float = 130.0,   # g  — BioGears Carbohydrate field
    protein_g_per_meal: float = 20.0,    # g  — BioGears Protein field
    meals_per_day:      float = 3.0,
    daily_water_L:      float = 2.0,     # L  — BioGears Water field
    sodium_mg_per_day:  float = 1000.0,  # mg — BioGears Sodium field (normalised)
    sleep_hours:        float = 8.0,     # hr — BioGears SleepAmount field
    # ── BioGears-measured calibration (from extract_biogears_calibration()) ──
    bg_calibration:     Optional[dict]                    = None,
    # ── Legacy / backward-compat fallback ────────────────────────────────────
    mission_day:        int   = 0,       # used only when mg_factors is None
    baseline_hr:        float = 72.0,    # used only when patient is None
    max_hr:             float = 200.0,   # used only when patient is None
    # ── Kept for API compat (ignored — glycogen replaces this) ───────────────
    hydration_arr:      Optional[np.ndarray] = None,
    food_arr:           Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """
    Compute per-minute fatigue and glycogen fraction for the mission timeline.

    When bg_calibration is supplied, the model uses BioGears-measured rates:
      - eva_fatigue_rate_per_min      → replaces _RATE_EVA_ACCUMULATE
      - glycogen_depletion_g_per_min  → replaces _GLY_DEPLETION_PER_MIN
      - glycogen_start_g              → replaces glycogen_max × 0.80
      - sweat_rate_mg_per_min         → feeds into hydration via actual sweat loss

    Returns
    -------
    fatigue          : float array [0, 1], shape = (mission_min,)
    glycogen_fraction: float array [0, 1], shape = (mission_min,)
    risk_periods     : list of (start_min, end_min) continuous risk windows
    """
    from simulation.patient import microgravity_factors as _mg_factors

    mission_min = len(hr_arr)

    # ── Resolve patient parameters ────────────────────────────────────────────
    if patient is not None:
        p_baseline_hr = patient.hr_baseline
        p_hr_max      = patient.hr_max
        p_lean_mass   = patient.lean_mass_kg
        p_weight_kg   = patient.weight_kg
    else:
        p_baseline_hr = baseline_hr
        p_hr_max      = max_hr
        p_lean_mass   = 60.9   # StandardMale default
        p_weight_kg   = 77.0

    # ── Resolve microgravity factors ──────────────────────────────────────────
    if mg_factors is None:
        mg = _mg_factors(mission_day)
    else:
        mg = mg_factors

    # Adjusted baselines after microgravity deconditioning
    hr_baseline_mg  = p_baseline_hr + mg.hr_offset
    vo2max_factor   = mg.vo2max_factor
    glycogen_max    = _glycogen_init(p_lean_mass, mg.glycogen_factor)

    # ── BioGears-calibrated rates (override defaults when available) ──────────
    if bg_calibration:
        rate_eva      = float(bg_calibration.get("eva_fatigue_rate_per_min",
                                                  _RATE_EVA_ACCUMULATE))
        gly_depl_rate = float(bg_calibration.get("glycogen_depletion_g_per_min",
                                                  _GLY_DEPLETION_PER_MIN))
        glycogen_g    = float(bg_calibration.get("glycogen_start_g",
                                                  glycogen_max * 0.80))
        # Actual sweat-adjusted hydration: if we lose more water than we drink, deficit worsens
        sweat_mg_min  = float(bg_calibration.get("sweat_rate_mg_per_min", 600.0))
        sweat_L_day   = sweat_mg_min * 60 * 24 / 1e6   # mg/min → L/day (full-day equivalent)
        hyd           = _hydration_state(daily_water_L, sodium_mg_per_day,
                                         sweat_loss_L_per_day=sweat_L_day)
    else:
        rate_eva      = _RATE_EVA_ACCUMULATE
        gly_depl_rate = _GLY_DEPLETION_PER_MIN
        glycogen_g    = glycogen_max * 0.80
        hyd           = _hydration_state(daily_water_L, sodium_mg_per_day)

    # ── Pre-compute mission-constant scalars ──────────────────────────────────
    hyd_penalty     = _hydration_penalty(hyd)
    prot_recovery   = _protein_recovery_factor(protein_g_per_meal, meals_per_day, p_weight_kg)
    sleep_penalty   = _sleep_debt_penalty(sleep_hours, mission_day if mg_factors is None else mg.mission_day)

    # ── Output arrays ─────────────────────────────────────────────────────────
    fatigue             = np.zeros(mission_min)
    glycogen_fraction   = np.zeros(mission_min)
    glycogen_fraction[0]= glycogen_g / max(glycogen_max, 1.0)

    for i in range(1, mission_min):
        prev      = fatigue[i - 1]
        hr_range  = max(p_hr_max - hr_baseline_mg, 1.0)
        hr_norm   = float(np.clip((hr_arr[i] - hr_baseline_mg) / hr_range, 0.0, 1.0))

        gly_frac  = float(np.clip(glycogen_g / glycogen_max, 0.0, 1.0))
        gly_penalty = 1.0 + 0.60 * (1.0 - gly_frac)   # empty glycogen = 60% harder

        ev = get_event_at_minute(events, i)

        if ev is not None and ev.event_type == EventType.EVA:
            delta      = (rate_eva
                          * (hr_norm / max(vo2max_factor, 0.01))
                          * hyd_penalty
                          * gly_penalty
                          * sleep_penalty)
            # Use BioGears-measured depletion rate scaled by current intensity
            glycogen_g = max(0.0, glycogen_g - gly_depl_rate * float(np.clip(ev.intensity, 0, 1)))

        elif ev is not None and ev.event_type == EventType.SLEEP:
            sleep_rate = _RATE_SLEEP_RECOVER * vo2max_factor * prot_recovery * (sleep_hours / 8.0)
            delta      = -sleep_rate
            glycogen_g = min(glycogen_max,
                             glycogen_g + _glycogen_replenish(glycogen_g, glycogen_max,
                                                              carb_g_per_meal, meals_per_day))
        else:
            delta      = (-_RATE_PASSIVE_RECOVER * (1.0 - hr_norm)
                          * prot_recovery / max(sleep_penalty, 0.01))
            glycogen_g = min(glycogen_max,
                             glycogen_g + _glycogen_replenish(glycogen_g, glycogen_max,
                                                              carb_g_per_meal, meals_per_day))

        fatigue[i]            = float(np.clip(prev + delta, 0.0, 1.0))
        glycogen_fraction[i]  = float(np.clip(glycogen_g / glycogen_max, 0.0, 1.0))

    # ── Continuous risk windows ───────────────────────────────────────────────
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

    return fatigue, glycogen_fraction, risk_periods


# ── Glycogen state engine ─────────────────────────────────────────────────────
#
# Glycogen is the primary muscle fuel during EVA. BioGears tracks MuscleGlycogen
# directly (output column). Here we model it as a running state variable that
# depletes during EVA and replenishes during rest when carbohydrates are available.
#
# Sources:
#   Brooks et al. "Exercise Physiology" 8th ed. — ~15 g glycogen / kg lean mass
#   Burke et al. (2011) IJSNEM — carbohydrate oxidation rates during aerobic work
#   Ivy et al. (1988) J Appl Physiol — post-exercise glycogen resynthesis rates

# Depletion: aerobic exercise at moderate intensity burns ~2.8 g/min of glycogen
# per unit EVA intensity (based on RER ~0.92 at 70% VO2max, carb contribution ~60%).
_GLY_DEPLETION_PER_MIN   = 2.8    # g / min / intensity_unit

# Replenishment: during rest with adequate carb intake, muscle glycogen resynthesises
# at ~5% of remaining deficit per hour (Ivy et al. 1988). We approximate this per
# minute as a fraction of daily carbohydrate absorbed (70% bioavailability assumed).
_GLY_ABSORPTION_FRACTION = 0.70   # fraction of ingested carbs absorbed by muscle
_GLY_RESYNTHESIS_RATE    = 5e-4   # fractional resynthesis per minute during rest


def _glycogen_init(lean_mass_kg: float, glycogen_factor: float = 1.0) -> float:
    """
    Initial glycogen store in grams.

    Capacity = lean_mass × 15 g/kg, then scaled by microgravity muscle_factor
    (muscle atrophy reduces total glycogen capacity over time in space).
    """
    return lean_mass_kg * 15.0 * glycogen_factor


def _glycogen_depletion(intensity: float) -> float:
    """Return grams of glycogen burned per minute at given EVA intensity."""
    return _GLY_DEPLETION_PER_MIN * float(np.clip(intensity, 0.0, 1.0))


def _glycogen_replenish(
    glycogen_g: float,
    glycogen_max: float,
    carb_g_per_meal: float,
    meals_per_day: float,
) -> float:
    """
    Return grams of glycogen added per minute during rest/sleep.

    Converts daily carbohydrate intake → per-minute absorbed carbs →
    glycogen resynthesis rate, capped so stores never exceed maximum.
    """
    if glycogen_g >= glycogen_max:
        return 0.0
    daily_carb_g       = carb_g_per_meal * meals_per_day
    per_min_absorbed_g = (daily_carb_g * _GLY_ABSORPTION_FRACTION) / (24 * 60)
    deficit_g          = glycogen_max - glycogen_g
    return min(per_min_absorbed_g, deficit_g * _GLY_RESYNTHESIS_RATE * 60)


# ── Hydration model ──────────────────────────────────────────────────────────
#
# In microgravity, aldosterone regulation is dysregulated and sodium retention
# is elevated, causing fluid to redistribute cephalad. Higher sodium intake
# increases the effective water requirement to maintain plasma osmolarity.
#
# Sources:
#   NASA TM-2015-218570 — "Water requirements in space"
#   Drummer et al. (2000) — sodium/fluid retention on ISS
#   NASA HSMO guidelines — minimum 2.0 L/day in space (vs 1.5 L ground)

_WATER_NEED_BASE_L        = 2.0    # L/day NASA minimum for spaceflight
_SODIUM_REFERENCE_MG      = 1500.0 # mg/day below which no extra water needed
_SODIUM_WATER_COUPLING    = 1.5e-4  # L extra water needed per mg sodium above reference


def _hydration_state(daily_water_L: float, sodium_mg_per_day: float,
                     sweat_loss_L_per_day: float = 0.0) -> float:
    """
    Compute instantaneous hydration level [0, 1] from daily intake and sodium.

    When sweat_loss_L_per_day is supplied (from BioGears SweatRate output),
    it is added to the effective water requirement so that high-intensity EVA
    sweat loss is accounted for with a measured value rather than a Python estimate.

    Returns 1.0 when water intake equals or exceeds requirement; falls linearly
    to 0.0 at zero water intake.
    """
    sodium_excess      = max(0.0, sodium_mg_per_day - _SODIUM_REFERENCE_MG)
    effective_need_L   = (_WATER_NEED_BASE_L
                          + sodium_excess * _SODIUM_WATER_COUPLING
                          + sweat_loss_L_per_day)
    return float(np.clip(daily_water_L / effective_need_L, 0.0, 1.0))


def _hydration_penalty(hydration: float) -> float:
    """
    Fatigue accumulation multiplier from dehydration.

    At full hydration (1.0) → no penalty (×1.0).
    At zero hydration     → maximum penalty (×1.4).
    Range [1.0, 1.4] matching the prior model — now driven by real water/sodium inputs.
    """
    return 1.0 + 0.4 * (1.0 - float(np.clip(hydration, 0.0, 1.0)))


# ── Sleep debt accumulation ───────────────────────────────────────────────────
#
# ISS astronauts average 6.5h sleep/night against a target of 8h (NASA policy).
# Chronic sleep restriction degrades cognitive and physical performance:
#   - ~3% impairment per hour of daily sleep debt (Czeisler et al. 2006, Science)
#   - Accumulates over the work week; partially cleared on "rest days" (Fridays on ISS)
#
# We model debt as weekly (7-day rolling) cumulative deficit, capped at 40% penalty
# to avoid unphysical values during extended missions.

_SLEEP_TARGET_H          = 8.0    # NASA target sleep hours per night
_SLEEP_DEBT_IMPAIR_RATE  = 0.025  # performance loss per hour of cumulative deficit
_SLEEP_DEBT_PENALTY_CAP  = 0.40   # maximum 40% increase in fatigue accumulation


def _sleep_debt_penalty(sleep_hours: float, mission_day: int) -> float:
    """
    Fatigue multiplier from accumulated sleep debt [1.0, 1.40].

    Debt accumulates over a 7-day rolling window (ISS work-week model).
    At 6.5h/night for 7 days: debt = 10.5h → penalty = 1 + min(0.4, 10.5×0.025) = 1.26.
    At 8h+/night: penalty = 1.0 (no debt).
    """
    daily_deficit_h  = max(0.0, _SLEEP_TARGET_H - float(sleep_hours))
    # Rolling 7-day window — debt doesn't compound indefinitely (weekend recovery effect)
    week_day         = (mission_day % 7) + 1
    cumulative_h     = daily_deficit_h * week_day
    return 1.0 + min(_SLEEP_DEBT_PENALTY_CAP, cumulative_h * _SLEEP_DEBT_IMPAIR_RATE)


# ── Protein recovery factor ───────────────────────────────────────────────────
#
# NASA recommendation: 1.6 g/kg/day protein for EVA astronauts (muscle maintenance
# in microgravity; standard RDA is 0.8 g/kg/day but EVA work increases demand).
# Adequate protein → faster muscle glycogen replenishment and lower baseline fatigue
# on subsequent mission days.
#
# Source: Stein & Blanc (2011) "Does protein supplementation prevent muscle
#   disuse atrophy and loss of strength?" — NASA Human Research Program.

_PROTEIN_TARGET_G_PER_KG = 1.6    # g protein per kg bodyweight per day (NASA EVA standard)


def _protein_recovery_factor(
    protein_g_per_meal: float,
    meals_per_day: float,
    weight_kg: float,
) -> float:
    """
    Sleep/rest recovery speed multiplier [0.70, 1.0] based on protein adequacy.

    At NASA-recommended 1.6 g/kg/day → factor = 1.0 (full recovery speed).
    At 50% of recommendation    → factor ≈ 0.85.
    At zero protein             → factor = 0.70 (30% slower recovery).
    """
    daily_protein_g  = float(protein_g_per_meal) * float(meals_per_day)
    target_g         = _PROTEIN_TARGET_G_PER_KG * float(weight_kg)
    adequacy         = float(np.clip(daily_protein_g / max(target_g, 1.0), 0.0, 1.0))
    return 0.70 + 0.30 * adequacy


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
