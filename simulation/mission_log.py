"""
Mission log writer.

Saves a full simulation run to an XML file in mission_logs/ using the same
namespace, element names, xsi:type conventions, and DataRequest pattern as
the BioGears scenario XML produced by biogears.py.

The root element is <Scenario> so the file is structurally identical to a
BioGears scenario — the InitialParameters / DataRequests / Actions block is
byte-for-byte the same. Four extra sibling sections extend it with mission
configuration, physiology parameters, scheduled events, risk analytics, and
a physiological summary.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from simulation.events import EventType, MissionEvent

# ── Paths ─────────────────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).parent.parent / "mission_logs"

# ── XML namespaces (identical to BioGears) ────────────────────────────────────
_NS      = "uri:/mil/tatrc/physiology/datamodel"
_XSI     = "http://www.w3.org/2001/XMLSchema-instance"
_NS_MAP  = {"xmlns": _NS, "xmlns:xsi": _XSI}

# Register so ET doesn't add ns0: prefixes
ET.register_namespace("",    _NS)
ET.register_namespace("xsi", _XSI)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sub(parent: ET.Element, tag: str, attrib: dict | None = None, text: str | None = None) -> ET.Element:
    el = ET.SubElement(parent, tag, attrib=attrib or {})
    if text is not None:
        el.text = text
    return el


def _xsi_type(type_name: str) -> dict:
    return {f"{{{_XSI}}}type": type_name}


def _fmt(v: float, p: int = 4) -> str:
    return f"{v:.{p}f}"


# ── Public API ────────────────────────────────────────────────────────────────

def save_mission_log(
    eva_intensity:    float,
    eva_duration_min: float,
    recovery_min:     float,
    mission_hours:    int,
    n_evas:           int,
    threshold:        float,
    bg_mode:          str,
    n_sims:           int,
    mission_day:      int,
    water_intake:     float,
    meals_per_day:    float,
    events:           List[MissionEvent],
    mission_df:       pd.DataFrame,
    fatigue:          np.ndarray,
    hydration_arr:    np.ndarray,
    food_arr:         np.ndarray,
    analytics:        Dict,
    mc_sum:           Dict,
    status_label:     str,
    status_color:     str,
    bg_msg:           str,
) -> Path:
    """
    Write one mission log XML to mission_logs/ and return the file path.

    File name: MissionLog_YYYYMMDD_HHMMSS.xml
    """
    LOG_DIR.mkdir(exist_ok=True)
    ts       = datetime.now()
    ts_iso   = ts.strftime("%Y-%m-%dT%H:%M:%S")
    ts_file  = ts.strftime("%Y%m%d_%H%M%S")
    mission_id = f"MISSION-{ts_file}"

    # ── Root: <Scenario> with same namespace as BioGears ─────────────────────
    root = ET.Element(f"{{{_NS}}}Scenario", {
        f"{{{_XSI}}}schemaLocation": "",
        "contentVersion":            "AstroGuard_1.0",
        "timestamp":                 ts_iso,
        "missionID":                 mission_id,
    })

    _sub(root, f"{{{_NS}}}Name",        text=f"EVA_Mission_Scenario_{ts_file}")
    _sub(root, f"{{{_NS}}}Description", text=(
        f"AstroGuard Mission Log · {mission_hours}h mission · "
        f"EVA intensity {eva_intensity:.2f} · Day {mission_day} in microgravity"
    ))

    # ── InitialParameters (identical to BioGears) ────────────────────────────
    init = _sub(root, f"{{{_NS}}}InitialParameters")
    _sub(init, f"{{{_NS}}}PatientFile", text="StandardMale.xml")

    # ── DataRequests (identical block to BioGears scenario) ──────────────────
    dr_block = _sub(root, f"{{{_NS}}}DataRequests", {"SamplesPerSecond": "1"})

    _data_requests = [
        ("HeartRate",                "1/min",    "2"),
        ("OxygenSaturation",         "unitless", "4"),
        ("CoreTemperature",          "degC",     "2"),
        ("RespirationRate",          "1/min",    "2"),
        ("FatigueLevel",             "unitless", "4"),
        ("TotalWorkRateLevel",       "unitless", "4"),
        ("AchievedExerciseLevel",    "unitless", "4"),
        ("TotalMetabolicRate",       "W",        "2"),
        ("CardiacOutput",            "L/min",    "2"),
        ("SystolicArterialPressure", "mmHg",     "1"),
        ("DiastolicArterialPressure","mmHg",     "1"),
        ("SweatRate",                "mg/min",   "2"),
        ("MuscleGlycogen",           "g",        "2"),
    ]
    for name, unit, prec in _data_requests:
        _sub(dr_block, f"{{{_NS}}}DataRequest", {
            **_xsi_type("PhysiologyDataRequestData"),
            "Name":      name,
            "Unit":      unit,
            "Precision": prec,
        })

    # ── Actions (identical 3-phase block to BioGears scenario) ───────────────
    baseline_s = 30
    eva_s      = int(eva_duration_min * 60)
    rec_s      = int(recovery_min * 60)

    actions = _sub(root, f"{{{_NS}}}Actions")

    # Phase 1: baseline
    a1 = _sub(actions, f"{{{_NS}}}Action", _xsi_type("AdvanceTimeData"))
    _sub(a1, f"{{{_NS}}}Time", {"value": str(baseline_s), "unit": "s"})

    # Phase 2: EVA workload
    a2 = _sub(actions, f"{{{_NS}}}Action", _xsi_type("ExerciseData"))
    ge2 = _sub(a2, f"{{{_NS}}}GenericExercise")
    _sub(ge2, f"{{{_NS}}}Intensity", {"value": f"{eva_intensity:.3f}"})
    a2t = _sub(actions, f"{{{_NS}}}Action", _xsi_type("AdvanceTimeData"))
    _sub(a2t, f"{{{_NS}}}Time", {"value": str(eva_s), "unit": "s"})

    # Phase 3: recovery
    a3 = _sub(actions, f"{{{_NS}}}Action", _xsi_type("ExerciseData"))
    ge3 = _sub(a3, f"{{{_NS}}}GenericExercise")
    _sub(ge3, f"{{{_NS}}}Intensity", {"value": "0.000"})
    a3t = _sub(actions, f"{{{_NS}}}Action", _xsi_type("AdvanceTimeData"))
    _sub(a3t, f"{{{_NS}}}Time", {"value": str(rec_s), "unit": "s"})

    # ── MissionParameters ────────────────────────────────────────────────────
    mp = _sub(root, f"{{{_NS}}}MissionParameters")
    _sub(mp, f"{{{_NS}}}MissionDuration",        {"value": str(mission_hours), "unit": "h"})
    _sub(mp, f"{{{_NS}}}NumberOfEVAs",           {"value": str(n_evas)})
    _sub(mp, f"{{{_NS}}}RiskThreshold",          {"value": _fmt(threshold, 2)})
    _sub(mp, f"{{{_NS}}}BioGearsMode",           text=bg_mode.upper())
    _sub(mp, f"{{{_NS}}}BioGearsStatus",         text=bg_msg[:120])
    _sub(mp, f"{{{_NS}}}MonteCarloSimulations",  {"value": str(n_sims)})

    # ── PhysiologyParameters ─────────────────────────────────────────────────
    D           = 1.0 + 0.008 * mission_day
    hr_offset   = min(1.5 * mission_day, 18.0)
    spo2_offset = -0.002 * min(mission_day, 5)

    pp = _sub(root, f"{{{_NS}}}PhysiologyParameters")

    mg = _sub(pp, f"{{{_NS}}}MicrogravityDeconditioning")
    _sub(mg, f"{{{_NS}}}MissionDay",           {"value": str(mission_day)})
    _sub(mg, f"{{{_NS}}}DeconditioningFactor", {"value": _fmt(D, 3)})
    _sub(mg, f"{{{_NS}}}HRBaselineOffset",     {"value": _fmt(hr_offset, 1), "unit": "1/min"})
    _sub(mg, f"{{{_NS}}}SpO2BaselineOffset",   {"value": _fmt(spo2_offset, 4), "unit": "unitless"})

    nh = _sub(pp, f"{{{_NS}}}NutritionHydration")
    _sub(nh, f"{{{_NS}}}WaterIntake",     {"value": _fmt(water_intake, 2), "unit": "L/h"})
    _sub(nh, f"{{{_NS}}}MealsPerDay",    {"value": str(int(meals_per_day))})
    _sub(nh, f"{{{_NS}}}MinHydration",   {"value": _fmt(float(hydration_arr.min()), 4), "unit": "unitless"})
    _sub(nh, f"{{{_NS}}}FinalHydration", {"value": _fmt(float(hydration_arr[-1]),  4), "unit": "unitless"})
    _sub(nh, f"{{{_NS}}}MinFoodLevel",   {"value": _fmt(float(food_arr.min()), 4), "unit": "unitless"})
    _sub(nh, f"{{{_NS}}}FinalFoodLevel", {"value": _fmt(float(food_arr[-1]),  4), "unit": "unitless"})

    # ── MissionEvents ─────────────────────────────────────────────────────────
    me_block = _sub(root, f"{{{_NS}}}MissionEvents")
    _type_map = {
        EventType.EVA:         "EVAEventData",
        EventType.SLEEP:       "SleepEventData",
        EventType.DEHYDRATION: "DehydrationEventData",
    }
    for ev in events:
        attrib = {
            **_xsi_type(_type_map.get(ev.event_type, "MissionEventData")),
            "startMin":    str(ev.start_min),
            "endMin":      str(ev.end_min),
            "durationMin": str(ev.duration_min),
        }
        if ev.intensity > 0:
            attrib["intensity"] = _fmt(ev.intensity, 3)
        _sub(me_block, f"{{{_NS}}}Event", attrib)

    # ── RiskAnalytics ─────────────────────────────────────────────────────────
    ra = _sub(root, f"{{{_NS}}}RiskAnalytics")
    _sub(ra, f"{{{_NS}}}MissionStatus", {"color": status_color}, text=status_label)
    _sub(ra, f"{{{_NS}}}PeakFatigue",  {
        "value":    _fmt(analytics["peak_fatigue"], 4),
        "atMinute": str(analytics["peak_minute"]),
        "atHour":   _fmt(analytics["peak_minute"] / 60, 2),
    })
    _sub(ra, f"{{{_NS}}}TimeAtRisk",   {"value": _fmt(analytics["time_at_risk_pct"], 2), "unit": "%"})
    _sub(ra, f"{{{_NS}}}FatigueTrend", {"slope": _fmt(analytics["trend_slope"], 6)},
         text=analytics["trend_label"])
    _sub(ra, f"{{{_NS}}}MonteCarlo",   {
        "nSimulations":     str(mc_sum["n_sims"]),
        "probOfBreach":     _fmt(mc_sum["p_any_breach"], 4),
        "meanPeakFatigue":  _fmt(mc_sum["mean_peak_fatigue"], 4),
        "worstCaseFatigue": _fmt(mc_sum["worst_case_fatigue"], 4),
    })

    rp_block = _sub(ra, f"{{{_NS}}}RiskPeriods",
                    {"count": str(len(analytics["risk_periods"]))})
    for start, end in analytics["risk_periods"]:
        _sub(rp_block, f"{{{_NS}}}Period", {
            "startMin":    str(start),
            "endMin":      str(end),
            "durationMin": str(end - start),
            "startHour":   _fmt(start / 60, 2),
        })

    # ── PhysiologicalSummary (DataRequest-style, matches BioGears pattern) ───
    ps = _sub(root, f"{{{_NS}}}PhysiologicalSummary")

    hr   = mission_df["HeartRate"].values
    spo2 = mission_df["OxygenSaturation"].values
    temp = mission_df["CoreTemperature"].values

    # Baseline = mean of first 30 minutes (pre-EVA)
    base_n = min(30, len(hr))

    _physiological_vars = [
        ("HeartRate",        "1/min",    hr,          "baselineValue", "peakValue",  "finalValue"),
        ("OxygenSaturation", "unitless", spo2,        "baselineValue", "minValue",   "finalValue"),
        ("CoreTemperature",  "degC",     temp,        "baselineValue", "peakValue",  "finalValue"),
        ("FatigueLevel",     "unitless", fatigue,     None,            "peakValue",  "finalValue"),
        ("HydrationLevel",   "unitless", hydration_arr, None,          "minValue",   "finalValue"),
        ("FoodLevel",        "unitless", food_arr,    None,            "minValue",   "finalValue"),
    ]

    for name, unit, arr, base_key, extreme_key, final_key in _physiological_vars:
        attrib = {
            **_xsi_type("PhysiologyDataRequestData"),
            "Name": name,
            "Unit": unit,
            final_key: _fmt(float(arr[-1]), 4),
        }
        if base_key:
            attrib[base_key] = _fmt(float(arr[:base_n].mean()), 4)
        # extreme: peak for upward signals, min for downward
        if "min" in extreme_key.lower():
            attrib[extreme_key] = _fmt(float(arr.min()), 4)
        else:
            attrib[extreme_key] = _fmt(float(arr.max()), 4)
        _sub(ps, f"{{{_NS}}}DataRequest", attrib)

    # ── Write file ────────────────────────────────────────────────────────────
    _indent(root)
    tree = ET.ElementTree(root)
    out_path = LOG_DIR / f"MissionLog_{ts_file}.xml"
    tree.write(str(out_path), xml_declaration=True, encoding="UTF-8")
    return out_path


def _indent(elem: ET.Element, level: int = 0) -> None:
    """Add pretty-print indentation in-place (pure stdlib, Python 3.8 compat)."""
    pad = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = pad
        for child in elem:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = pad
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = pad
    if not level:
        elem.tail = "\n"
