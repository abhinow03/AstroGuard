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


def _txt(elem: ET.Element, tag: str, default: str = "") -> str:
    """Return stripped text content of the first matching child, or default."""
    child = elem.find(f"{{{_NS}}}{tag}")
    if child is None:
        return default
    return (child.text or default).strip()


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

    @property
    def display_name(self) -> str:
        return self.name.replace("_", " ")


# ── Parser ─────────────────────────────────────────────────────────────────────

def parse_patient(xml_path: Path) -> PatientProfile:
    """Parse one BioGears patient XML file into a PatientProfile."""
    root = ET.parse(str(xml_path)).getroot()
    return PatientProfile(
        name               = _txt(root, "Name",   xml_path.stem),
        sex                = _txt(root, "Sex",    "Male"),
        age_yr             = _val(root, "Age",    44.0),
        weight_lb          = _val(root, "Weight", 170.0),
        height_in          = _val(root, "Height", 71.0),
        body_fat_fraction  = _val(root, "BodyFatFraction", 0.21),
        hr_baseline        = _val(root, "HeartRateBaseline",         72.0),
        rr_baseline        = _val(root, "RespirationRateBaseline",   16.0),
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
