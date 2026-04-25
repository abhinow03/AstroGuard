"""
BioGears patient profile loader.

Parses patient XMLs from the BioGears patients/ directory into PatientProfile
dataclasses that carry the raw BioGears fields (name, sex, age, weight, height,
body-fat fraction, baseline vitals, sleep amount).

Derived physiological metrics (VO2max, glycogen capacity, fitness tier) and
microgravity deconditioning factors are added in later stages of this module.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

# ── BioGears paths ─────────────────────────────────────────────────────────────
BIOGEARS_PATIENTS_DIR   = Path("Z:/BIOGEARS/bin/patients")
BIOGEARS_NUTRITION_DIR  = Path("Z:/BIOGEARS/bin/nutrition")

_NS = "uri:/mil/tatrc/physiology/datamodel"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _val(elem: ET.Element, tag: str, default: float = 0.0) -> float:
    """Return the `value` attribute of the first matching child, or default."""
    child = elem.find(f"{{{_NS}}}{tag}")
    if child is None:
        return default
    return float(child.get("value", default))


def _val_unit(elem: ET.Element, tag: str, default: float = 0.0) -> tuple[float, str]:
    """Return (value, unit) for a child element. Returns (default, '') if missing."""
    child = elem.find(f"{{{_NS}}}{tag}")
    if child is None:
        return default, ""
    return float(child.get("value", default)), child.get("unit", "")


def _txt(elem: ET.Element, tag: str, default: str = "") -> str:
    """Return stripped text content of the first matching child, or default."""
    child = elem.find(f"{{{_NS}}}{tag}")
    if child is None:
        return default
    return (child.text or default).strip()


def _to_lb(value: float, unit: str) -> float:
    """Normalise weight to lb regardless of source unit."""
    if unit in ("kg", "kilogram"):
        return value / 0.453592
    return value  # already lb


def _to_in(value: float, unit: str) -> float:
    """Normalise height to inches regardless of source unit."""
    if unit in ("cm", "centimeter", "m"):
        return value / 2.54 if unit != "m" else value * 39.3701
    return value  # already in


# ── Raw patient dataclass ──────────────────────────────────────────────────────

@dataclass
class PatientProfile:
    """
    Direct mapping of BioGears patient XML fields.

    Units are preserved as-is from the XML (lb, in, 1/min, mmHg, hr) so that
    nothing is silently converted before the caller decides how to use them.
    Derived SI quantities (weight_kg, height_cm, …) are computed in __post_init__.
    """
    # ── Raw BioGears fields ────────────────────────────────────────────────────
    name:                str
    sex:                 str
    age_yr:              float
    weight_lb:           float
    height_in:           float
    body_fat_fraction:   float
    hr_baseline:         float   # 1/min  (resting HR in the patient file)
    rr_baseline:         float   # 1/min
    systolic_bp:         float   # mmHg
    diastolic_bp:        float   # mmHg
    sleep_amount_hr:     float   # hr  (only set explicitly in SleepDeprived; default 8 h)

    # ── Derived SI quantities (set by __post_init__) ──────────────────────────
    weight_kg:          float = 0.0
    height_cm:          float = 0.0
    bmi:                float = 0.0
    lean_mass_kg:       float = 0.0
    fat_mass_kg:        float = 0.0
    glycogen_capacity_g: float = 0.0
    hr_max:             float = 0.0
    vo2max_ml_kg_min:   float = 0.0
    fitness_tier:       str   = ""

    def __post_init__(self) -> None:
        # SI conversions
        self.weight_kg  = self.weight_lb * 0.453592
        self.height_cm  = self.height_in * 2.54
        h_m             = self.height_cm / 100.0

        # Body composition
        self.bmi            = self.weight_kg / (h_m ** 2)
        self.lean_mass_kg   = self.weight_kg * (1.0 - self.body_fat_fraction)
        self.fat_mass_kg    = self.weight_kg * self.body_fat_fraction

        # Glycogen capacity — skeletal muscle stores ~15 g glycogen per kg lean mass
        # (Brooks et al., Exercise Physiology, 8th ed.)
        self.glycogen_capacity_g = self.lean_mass_kg * 15.0

        # Max HR (Tanaka formula, more accurate than 220-age for adults)
        self.hr_max = 208.0 - 0.7 * self.age_yr

        # VO2max estimate — Uth-Sørensen-Overgaard-Pedersen (2004)
        # VO2max ≈ 15 × (HRmax / HRrest)
        self.vo2max_ml_kg_min = 15.0 * (self.hr_max / self.hr_baseline)

        # Fitness tier — NASA/ACSM sex-specific thresholds (mL/kg/min)
        self.fitness_tier = _classify_fitness(self.vo2max_ml_kg_min, self.sex)

    @property
    def display_name(self) -> str:
        return self.name.replace("_", " ")

    @property
    def pulse_pressure(self) -> float:
        return self.systolic_bp - self.diastolic_bp

    @property
    def daily_kcal_estimate(self) -> float:
        """Mifflin-St Jeor BMR × 1.375 (lightly active) as a baseline daily need."""
        if self.sex.lower() == "female":
            bmr = 10 * self.weight_kg + 6.25 * self.height_cm - 5 * self.age_yr - 161
        else:
            bmr = 10 * self.weight_kg + 6.25 * self.height_cm - 5 * self.age_yr + 5
        return bmr * 1.375


def _classify_fitness(vo2max: float, sex: str) -> str:
    """
    NASA/ACSM aerobic fitness tier by VO2max (mL/kg/min).

    Thresholds from ACSM's Guidelines for Exercise Testing and Prescription, 11th ed.
    NASA uses similar categories for astronaut fitness clearance.
    """
    if sex.lower() == "female":
        if vo2max >= 47: return "Elite"
        if vo2max >= 37: return "Good"
        if vo2max >= 28: return "Normal"
        return "Deconditioned"
    else:
        if vo2max >= 55: return "Elite"
        if vo2max >= 43: return "Good"
        if vo2max >= 35: return "Normal"
        return "Deconditioned"


# ── Parser ─────────────────────────────────────────────────────────────────────

def parse_patient(xml_path: Path) -> PatientProfile:
    """
    Parse one BioGears patient XML file into a PatientProfile.

    Handles both the legacy format (readOnly attribute, lb/in units) and the
    newer format used by some patients (direct value attribute, SI units).
    All weights are normalised to lb and heights to inches internally so that
    __post_init__ SI conversions are consistent across all profiles.
    """
    root = ET.parse(str(xml_path)).getroot()

    w_val, w_unit = _val_unit(root, "Weight", 170.0)
    h_val, h_unit = _val_unit(root, "Height", 71.0)

    return PatientProfile(
        name               = _txt(root, "Name",   xml_path.stem),
        sex                = _txt(root, "Sex",    "Male"),
        age_yr             = _val(root, "Age",    44.0),
        weight_lb          = _to_lb(w_val, w_unit),
        height_in          = _to_in(h_val, h_unit),
        body_fat_fraction  = _val(root, "BodyFatFraction", 0.21),
        hr_baseline        = _val(root, "HeartRateBaseline",                  72.0),
        rr_baseline        = _val(root, "RespirationRateBaseline",            16.0),
        systolic_bp        = _val(root, "SystolicArterialPressureBaseline",  114.0),
        diastolic_bp       = _val(root, "DiastolicArterialPressureBaseline",  73.5),
        sleep_amount_hr    = _val(root, "SleepAmount", 8.0),
    )


def load_patients(patients_dir: Path = BIOGEARS_PATIENTS_DIR) -> Dict[str, PatientProfile]:
    """
    Load every non-template patient XML from `patients_dir`.

    Returns a dict keyed by patient name (same as the <Name> tag in the XML).
    Files whose names contain 'Template' are skipped — they are BioGears internal
    scaffolds that are not meant to be used directly as simulation subjects.
    """
    patients: Dict[str, PatientProfile] = {}
    if not patients_dir.exists():
        return patients
    for xml_file in sorted(patients_dir.glob("*.xml")):
        if "Template" in xml_file.name:
            continue
        try:
            p = parse_patient(xml_file)
            patients[p.name] = p
        except Exception:
            pass
    return patients


def get_patient(name: str, patients_dir: Path = BIOGEARS_PATIENTS_DIR) -> Optional[PatientProfile]:
    """Return a single patient by name, or None if not found."""
    xml_path = patients_dir / f"{name}.xml"
    if not xml_path.exists():
        return None
    try:
        return parse_patient(xml_path)
    except Exception:
        return None


# ── Nutrition profiles ─────────────────────────────────────────────────────────

@dataclass
class NutritionProfile:
    """
    Direct mapping of a BioGears nutrition XML.

    The six fields here (Carbohydrate, Fat, Protein, Calcium, Sodium, Water)
    are the exact inputs BioGears passes to its metabolic engine — they map
    1-to-1 with what you would put in a ConsumeNutrients action in a scenario.

    This is the "elemental level" the fatigue model uses: no more abstract
    "meals per day" — these are the actual macronutrient substrates.
    """
    name:            str
    carbohydrate_g:  float   # g per meal/serving
    fat_g:           float   # g per meal/serving
    protein_g:       float   # g per meal/serving
    calcium_mg:      float   # mg per meal/serving
    sodium_mg:       float   # mg per meal/serving  (normalised from g or mg)
    water_L:         float   # L per meal/serving

    @property
    def calories(self) -> float:
        """Atwater factors: carbs 4 kcal/g, protein 4 kcal/g, fat 9 kcal/g."""
        return self.carbohydrate_g * 4.0 + self.protein_g * 4.0 + self.fat_g * 9.0

    @property
    def carb_pct(self) -> float:
        return (self.carbohydrate_g * 4.0 / self.calories * 100) if self.calories else 0.0

    @property
    def protein_pct(self) -> float:
        return (self.protein_g * 4.0 / self.calories * 100) if self.calories else 0.0

    @property
    def fat_pct(self) -> float:
        return (self.fat_g * 9.0 / self.calories * 100) if self.calories else 0.0


# BioGears reference nutrition — "StandardNutrition" per-meal values used as defaults
STANDARD_NUTRITION = NutritionProfile(
    name           = "StandardNutrition",
    carbohydrate_g = 130.0,
    fat_g          = 27.0,
    protein_g      = 20.0,
    calcium_mg     = 500.0,
    sodium_mg      = 1000.0,
    water_L        = 0.5,
)

# NASA ISS daily targets (for reference / slider annotations)
NASA_DAILY_TARGETS = {
    "carbohydrate_g": 370.0,   # ~50% of 3000 kcal
    "protein_g":      120.0,   # ~1.6 g/kg for 75kg astronaut
    "fat_g":          100.0,   # ~30% of 3000 kcal
    "calcium_mg":    1200.0,   # elevated for bone loss countermeasure
    "sodium_mg":     2300.0,   # NASA upper limit (fluid retention risk)
    "water_L":          2.0,   # minimum; suit adds ~0.5L loss per EVA hour
}


def parse_nutrition(xml_path: Path) -> NutritionProfile:
    """
    Parse one BioGears nutrition XML into a NutritionProfile.

    Handles the sodium unit ambiguity: some files store sodium in grams,
    others in milligrams — detected from the `unit` attribute and normalised
    to mg so downstream math is always in consistent units.
    """
    root = ET.parse(str(xml_path)).getroot()
    name = _txt(root, "Name", xml_path.stem)

    carb    = _val(root, "Carbohydrate", 0.0)
    fat     = _val(root, "Fat",          0.0)
    protein = _val(root, "Protein",      0.0)
    calcium = _val(root, "Calcium",      0.0)
    water   = _val(root, "Water",        0.0)

    # Sodium: unit varies across nutrition files (g vs mg)
    sodium_el = root.find(f"{{{_NS}}}Sodium")
    sodium_mg = 0.0
    if sodium_el is not None:
        raw  = float(sodium_el.get("value", 0.0))
        unit = sodium_el.get("unit", "mg")
        sodium_mg = raw * 1000.0 if unit == "g" else raw

    return NutritionProfile(
        name           = name,
        carbohydrate_g = carb,
        fat_g          = fat,
        protein_g      = protein,
        calcium_mg     = calcium,
        sodium_mg      = sodium_mg,
        water_L        = water,
    )


def load_nutrition_profiles(
    nutrition_dir: Path = BIOGEARS_NUTRITION_DIR,
) -> "Dict[str, NutritionProfile]":
    """
    Load all nutrition XMLs from the BioGears nutrition/ directory.

    Returns a dict keyed by profile name. Falls back gracefully to an empty
    dict if the directory does not exist (e.g. BioGears not installed).
    """
    profiles: Dict[str, NutritionProfile] = {}
    if not nutrition_dir.exists():
        return profiles
    for xml_file in sorted(nutrition_dir.glob("*.xml")):
        try:
            n = parse_nutrition(xml_file)
            profiles[n.name] = n
        except Exception:
            pass
    return profiles


# ── Microgravity deconditioning model ─────────────────────────────────────────

@dataclass
class MicrogravityFactors:
    """
    NASA ISS-calibrated deconditioning factors for a given number of days in space.

    Sources:
      - Trappe et al. (2004) "Exercise in space: human skeletal muscle after 6 months
        aboard the International Space Station." J Appl Physiol.
      - Scott et al. (2019) "Characterization of maximum aerobic capacity..." NASA HRP.
      - Shykoff (2019) "Cardiac output at rest and exercise..." NASA JSC.
      - NASA Human Research Roadmap — Human Research Program evidence reports.

    All factors are dimensionless fractions of the ground-baseline value
    (1.0 = no change) except hr_offset (absolute bpm) and spo2_offset (fraction).
    """
    mission_day:     int

    # VO2max retention — fraction of ground VO2max still available
    vo2max_factor:   float   # 1.0 → 0.72 over 180 days

    # Skeletal muscle mass retention — drives glycogen capacity
    muscle_factor:   float   # 1.0 → 0.80 (ISS countermeasures slow this)

    # Additional resting HR bpm imposed by microgravity (cardiac unloading)
    hr_offset:       float   # 0 → +15 bpm

    # SpO2 fraction change — fluid cephalad shift in first week only
    spo2_offset:     float   # 0 → −0.021 (days 1-7 only, then resolves)

    # Glycogen capacity multiplier (= muscle_factor; VO2max doesn't change storage)
    glycogen_factor: float

    def adjust_patient(self, p: PatientProfile) -> dict:
        """
        Return a dict of patient vitals adjusted for this mission day.

        Used by health_vars.py and fatigue.py to shift the patient's baselines
        without mutating the original PatientProfile object.
        """
        return {
            "hr_baseline":         p.hr_baseline + self.hr_offset,
            "vo2max_ml_kg_min":    p.vo2max_ml_kg_min * self.vo2max_factor,
            "glycogen_capacity_g": p.glycogen_capacity_g * self.glycogen_factor,
            "spo2_baseline_adj":   self.spo2_offset,   # added to base SpO2 in health_vars
        }


def microgravity_factors(mission_day: int) -> MicrogravityFactors:
    """
    Compute ISS-calibrated deconditioning factors for `mission_day` days in space.

    Deconditioning curve (two-phase):
      Phase 1 (days 1-30): VO2max loss ≈ 0.5 %/day  → rapid cardiovascular adaptation
      Phase 2 (days 31-180): VO2max loss ≈ 0.1 %/day → plateau as body re-adapts
      Floor at 72% (observed ~25-28% total VO2max loss after 6-month ISS missions).

    Muscle mass loss ≈ 0.2 %/day, floor at 80% (exercise countermeasures on ISS
    — ARED, CEVIS — significantly slow further loss after ~100 days).

    Resting HR increases ≈ 0.12 bpm/day due to reduced venous return and cardiac
    unloading; capped at +15 bpm consistent with ISS telemetry data.

    SpO2 effect is driven by the cephalad fluid shift in the first 7 days only;
    once the body has redistributed fluid volume, SpO2 returns to near-baseline.
    """
    d = max(0, int(mission_day))

    # ── VO2max (two-phase piecewise, NASA Scott et al. 2019) ──────────────────
    if d <= 30:
        vo2max_factor = max(0.72, 1.0 - 0.005 * d)
    else:
        vo2max_factor = max(0.72, 0.85 - 0.001 * (d - 30))

    # ── Skeletal muscle mass (Trappe et al. 2004) ─────────────────────────────
    muscle_factor = max(0.80, 1.0 - 0.002 * d)

    # ── Resting HR offset (ISS telemetry, Shykoff 2019) ──────────────────────
    hr_offset = min(15.0, 0.12 * d)

    # ── SpO2 cephalad fluid shift (first 7 days only) ────────────────────────
    # -0.003 per day for 7 days = max −0.021 (~98.5% → ~97.5% SpO2)
    spo2_offset = -0.003 * min(d, 7)

    return MicrogravityFactors(
        mission_day    = d,
        vo2max_factor  = round(vo2max_factor, 4),
        muscle_factor  = round(muscle_factor, 4),
        hr_offset      = round(hr_offset, 2),
        spo2_offset    = round(spo2_offset, 4),
        glycogen_factor= round(muscle_factor, 4),
    )
