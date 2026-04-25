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
