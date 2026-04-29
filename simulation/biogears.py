"""
BioGears integration layer.

Generates a custom EVA scenario XML, runs bg-cli.exe via subprocess,
and parses the resulting CSV into a clean DataFrame.

Falls back to the bundled CardiovascularValidationResults.csv if
the BioGears executable is unavailable or the run fails.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BIOGEARS_BIN    = Path("Z:/BIOGEARS/bin")
BG_SCENARIO     = BIOGEARS_BIN / "bg-scenario.exe"
RUNS_DIR        = BIOGEARS_BIN / "runs"
SCENARIO_NAME   = "EVA_Mission_Scenario"
FALLBACK_CSV    = Path(__file__).parent.parent / "CardiovascularValidationResults.csv"
PRECOMPUTED_DIR = Path(__file__).parent.parent / "precomputed"

# BioGears scenario is intentionally SHORT for live demo (< 10 min total).
# The physiological SHAPE + calibration rates are captured here and then
# resampled/extended to fill the full mission duration in health_vars.py.
BG_EVA_DURATION_MIN  = 5    # minutes of BioGears exercise phase
BG_RECOVERY_MIN      = 1.5  # minutes of BioGears recovery phase


# ── Scenario XML generation ────────────────────────────────────────────────────

def _generate_scenario_xml(
    eva_intensity: float,
    patient_file:  str   = "StandardMale.xml",
    carb_g:        float = 130.0,
    fat_g:         float = 27.0,
    protein_g:     float = 20.0,
    sodium_g:      float = 1.0,
    water_L:       float = 0.5,
) -> str:
    """
    Build the BioGears scenario XML for a SHORT EVA capture run.

    Structure:
      30 s  baseline stabilisation
      ConsumeNutrientsData  ← pre-EVA meal (user-configured macros)
      30 s  post-meal wait
      BG_EVA_DURATION_MIN   exercise at eva_intensity
      BG_RECOVERY_MIN       cooldown (intensity 0)

    Total wall-clock run time: ~7.5 min (fast enough for live demo).
    """
    baseline_s   = 30
    post_meal_s  = 30
    eva_s        = int(BG_EVA_DURATION_MIN * 60)
    rec_s        = int(BG_RECOVERY_MIN * 60)

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Scenario xmlns="uri:/mil/tatrc/physiology/datamodel"
          xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
          contentVersion="BioGears_6.3.0-beta"
          xsi:schemaLocation="">
  <Name>{SCENARIO_NAME}</Name>
  <Description>EVA Workload Simulation — Astronaut Fatigue Digital Twin</Description>
  <InitialParameters>
    <PatientFile>{patient_file}</PatientFile>
  </InitialParameters>

  <!-- 1 sample/s keeps output manageable -->
  <DataRequests SamplesPerSecond="1">
    <DataRequest xsi:type="PhysiologyDataRequestData" Name="HeartRate"               Unit="1/min"    Precision="2"/>
    <DataRequest xsi:type="PhysiologyDataRequestData" Name="OxygenSaturation"        Unit="unitless" Precision="4"/>
    <DataRequest xsi:type="PhysiologyDataRequestData" Name="CoreTemperature"         Unit="degC"     Precision="2"/>
    <DataRequest xsi:type="PhysiologyDataRequestData" Name="RespirationRate"         Unit="1/min"    Precision="2"/>
    <DataRequest xsi:type="PhysiologyDataRequestData" Name="FatigueLevel"            Unit="unitless" Precision="4"/>
    <DataRequest xsi:type="PhysiologyDataRequestData" Name="TotalWorkRateLevel"      Unit="unitless" Precision="4"/>
    <DataRequest xsi:type="PhysiologyDataRequestData" Name="AchievedExerciseLevel"   Unit="unitless" Precision="4"/>
    <DataRequest xsi:type="PhysiologyDataRequestData" Name="TotalMetabolicRate"      Unit="W"        Precision="2"/>
    <DataRequest xsi:type="PhysiologyDataRequestData" Name="CardiacOutput"           Unit="L/min"    Precision="2"/>
    <DataRequest xsi:type="PhysiologyDataRequestData" Name="SystolicArterialPressure"  Unit="mmHg"  Precision="1"/>
    <DataRequest xsi:type="PhysiologyDataRequestData" Name="DiastolicArterialPressure" Unit="mmHg"  Precision="1"/>
    <DataRequest xsi:type="PhysiologyDataRequestData" Name="SweatRate"               Unit="mg/min"   Precision="2"/>
    <DataRequest xsi:type="PhysiologyDataRequestData" Name="MuscleGlycogen"          Unit="g"        Precision="2"/>
  </DataRequests>

  <Actions>
    <!-- Phase 1: baseline stabilisation -->
    <Action xsi:type="AdvanceTimeData">
      <Time value="{baseline_s}" unit="s"/>
    </Action>

    <!-- Phase 2: pre-EVA meal — BioGears models metabolic response -->
    <Action xsi:type="ConsumeNutrientsData">
      <Nutrition>
        <Carbohydrate value="{carb_g:.1f}" unit="g"/>
        <Fat value="{fat_g:.1f}" unit="g"/>
        <Protein value="{protein_g:.1f}" unit="g"/>
        <Sodium value="{sodium_g:.3f}" unit="g"/>
        <Water value="{water_L:.2f}" unit="L"/>
      </Nutrition>
    </Action>
    <Action xsi:type="AdvanceTimeData">
      <Time value="{post_meal_s}" unit="s"/>
    </Action>

    <!-- Phase 3: EVA workload -->
    <Action xsi:type="ExerciseData">
      <GenericExercise>
        <Intensity value="{eva_intensity:.3f}"/>
      </GenericExercise>
    </Action>
    <Action xsi:type="AdvanceTimeData">
      <Time value="{eva_s}" unit="s"/>
    </Action>

    <!-- Phase 4: recovery -->
    <Action xsi:type="ExerciseData">
      <GenericExercise>
        <Intensity value="0.0"/>
      </GenericExercise>
    </Action>
    <Action xsi:type="AdvanceTimeData">
      <Time value="{rec_s}" unit="s"/>
    </Action>
  </Actions>
</Scenario>
"""


# ── Run BioGears ───────────────────────────────────────────────────────────────

def _find_output_csv(before_mtime: float) -> Path | None:
    """
    Return the BioGears output CSV written after `before_mtime`.

    bg-scenario.exe writes to its own directory (BIOGEARS_BIN) with the
    naming convention: <ScenarioName_lowercase>Results.csv
    """
    # Primary: expected output name from bg-scenario.exe
    expected_name = f"{SCENARIO_NAME.lower()}Results.csv"
    expected = BIOGEARS_BIN / expected_name
    if expected.exists() and expected.stat().st_mtime > before_mtime:
        return expected

    # Secondary: check runs/ subdirectory
    if RUNS_DIR.exists():
        in_runs = RUNS_DIR / f"{SCENARIO_NAME}.csv"
        if in_runs.exists() and in_runs.stat().st_mtime > before_mtime:
            return in_runs

    # Fallback: newest CSV anywhere in BIOGEARS_BIN modified after start
    candidates = [
        p for p in BIOGEARS_BIN.glob("*.csv")
        if p.stat().st_mtime > before_mtime
    ]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return None


def run_biogears(
    eva_intensity:  float,
    patient_file:   str   = "StandardMale.xml",
    carb_g:         float = 130.0,
    fat_g:          float = 27.0,
    protein_g:      float = 20.0,
    sodium_g:       float = 1.0,
    water_L:        float = 0.5,
    timeout:        int   = 600,
    progress_callback=None,
) -> Tuple["pd.DataFrame | None", str]:
    """
    Write patient + nutrition-specific EVA scenario XML → call bg-scenario.exe → parse CSV.

    Uses bg-scenario.exe (NOT bg-cli.exe — that crashes on this system).
    Total scenario: ~7.5 min (baseline + pre-EVA meal + 5 min EVA + 1.5 min recovery).

    Returns (DataFrame | None, status_message).
    """
    if not BG_SCENARIO.exists():
        return None, f"bg-scenario.exe not found at {BG_SCENARIO}"

    scenario_path = BIOGEARS_BIN / f"{SCENARIO_NAME}.xml"
    xml = _generate_scenario_xml(eva_intensity, patient_file=patient_file,
                                 carb_g=carb_g, fat_g=fat_g, protein_g=protein_g,
                                 sodium_g=sodium_g, water_L=water_L)
    scenario_path.write_text(xml, encoding="utf-8")

    if progress_callback:
        progress_callback("BioGears scenario written — starting engine…")

    t_start = time.time()
    before_mtime = t_start - 1.0

    try:
        # Open BioGears in its own CMD window so the user can watch it run.
        # CREATE_NEW_CONSOLE (0x00000010) is Windows-only — pops a new terminal.
        proc = subprocess.Popen(
            [str(BG_SCENARIO), str(scenario_path)],
            creationflags=0x00000010,   # CREATE_NEW_CONSOLE
        )
        proc.wait(timeout=timeout)
        result_returncode = proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        return None, f"BioGears timed out after {timeout}s"
    except (FileNotFoundError, OSError) as exc:
        return None, f"Failed to launch BioGears: {exc}"

    elapsed = time.time() - t_start

    if result_returncode != 0:
        return None, f"BioGears exited with code {result_returncode}"

    csv_path = _find_output_csv(before_mtime)
    if csv_path is None:
        return None, "BioGears ran but no output CSV found in runs/"

    df = _parse_csv(csv_path)
    if df is None:
        return None, f"Could not parse BioGears output at {csv_path}"

    return df, f"BioGears OK — {len(df)} rows in {elapsed:.1f}s ({csv_path.name})"


# ── CSV parsing ────────────────────────────────────────────────────────────────

_COL_MAP = {
    "HeartRate":               "HeartRate",
    "OxygenSaturation":        "OxygenSaturation",
    "CoreTemperature":         "CoreTemperature",
    "RespirationRate":         "RespirationRate",
    "FatigueLevel":            "FatigueLevel",
    "TotalWorkRateLevel":      "TotalWorkRateLevel",
    "AchievedExerciseLevel":   "AchievedExerciseLevel",
    "TotalMetabolicRate":      "TotalMetabolicRate_W",
    "CardiacOutput":           "CardiacOutput",
    "SystolicArterialPressure":  "SBP",
    "DiastolicArterialPressure": "DBP",
    "SweatRate":               "SweatRate",
    "MuscleGlycogen":          "MuscleGlycogen",
}


def _parse_csv(csv_path: Path) -> pd.DataFrame | None:
    """Parse a BioGears output CSV and normalise column names."""
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    rename: dict[str, str] = {}
    for col in df.columns:
        col_stripped = col.split("(")[0].strip()
        if col_stripped.startswith("Time"):
            rename[col] = "Time_s"
        else:
            for key, canonical in _COL_MAP.items():
                if key in col:
                    rename[col] = canonical
                    break

    df = df.rename(columns=rename)

    if "Time_s" not in df.columns or "HeartRate" not in df.columns:
        return None

    df = df.dropna(subset=["Time_s", "HeartRate"])
    return df.reset_index(drop=True)


# ── Fallback CSV ───────────────────────────────────────────────────────────────

def _load_fallback(eva_intensity: float, eva_duration_min: float, recovery_min: float) -> pd.DataFrame:
    """
    Synthesise a realistic BioGears-like EVA segment when the executable
    is unavailable.

    The output Time_s axis covers the SAME structure that a real BioGears run
    would produce: 2 min baseline → eva_duration_min exercise → recovery_min
    recovery. This lets health_vars.py's phase-extraction logic work correctly.
    """
    baseline_s   = 120.0
    eva_s        = eva_duration_min * 60.0
    recovery_s   = recovery_min * 60.0
    total_s      = baseline_s + eva_s + recovery_s

    # 1 sample per second
    n       = int(total_s)
    time_s  = np.linspace(0, total_s, n)
    rng     = np.random.default_rng(42)

    # Phase mask: 0 = baseline, 1 = EVA, 2 = recovery
    phase = np.zeros(n)
    phase[(time_s >= baseline_s) & (time_s < baseline_s + eva_s)] = 1
    phase[time_s >= baseline_s + eva_s] = 2

    # HR profile
    # Baseline: ~72 bpm; EVA: ramp to 72 + intensity×100; recovery: decay back
    hr_baseline = 72.0
    hr_peak     = hr_baseline + eva_intensity * 100.0

    hr = np.where(
        phase == 0,
        hr_baseline,
        np.where(
            phase == 1,
            hr_baseline + (hr_peak - hr_baseline) * np.clip(
                (time_s - baseline_s) / max(eva_s * 0.3, 1), 0, 1),
            hr_peak * np.exp(-(time_s - (baseline_s + eva_s)) / max(recovery_s * 0.4, 1))
            + hr_baseline * (1 - np.exp(-(time_s - (baseline_s + eva_s)) / max(recovery_s * 0.4, 1))),
        ),
    )
    hr += rng.normal(0, 2.0, n)

    # Fatigue (rises during EVA, decays in recovery)
    fatigue_level = np.where(
        phase == 0, 0.0,
        np.where(
            phase == 1,
            eva_intensity * np.clip((time_s - baseline_s) / max(eva_s, 1), 0, 1) * 0.70,
            eva_intensity * 0.70 * np.exp(-(time_s - (baseline_s + eva_s)) / max(recovery_s * 0.5, 1)),
        )
    )

    # Work rate level
    work_rate = np.where(phase == 1, eva_intensity, 0.0)

    # Other signals
    spo2 = np.where(phase == 1,
                    0.98 - eva_intensity * 0.018,
                    0.980) + rng.normal(0, 0.002, n)
    temp = np.where(phase == 1,
                    37.0 + eva_intensity * 1.5 * np.clip((time_s - baseline_s) / max(eva_s, 1), 0, 1),
                    37.0) + rng.normal(0, 0.04, n)
    rr   = np.where(phase == 1, 16 + eva_intensity * 22, 16) + rng.normal(0, 0.5, n)

    return pd.DataFrame({
        "Time_s":              time_s,
        "HeartRate":           np.clip(hr, 40, 220),
        "OxygenSaturation":    np.clip(spo2, 0.70, 1.0),
        "CoreTemperature":     np.clip(temp, 35.0, 41.0),
        "RespirationRate":     np.clip(rr, 6, 50),
        "FatigueLevel":        np.clip(fatigue_level, 0, 1),
        "TotalWorkRateLevel":  np.clip(work_rate, 0, 1),
        "AchievedExerciseLevel": np.clip(work_rate, 0, 1),
        "TotalMetabolicRate_W":  80 + eva_intensity * 400 * np.clip(work_rate, 0, 1),
        "CardiacOutput":       np.clip(5.0 + eva_intensity * 15 * np.clip(work_rate, 0, 1), 5, 25),
        "SBP":                 np.clip(120 + eva_intensity * 40 * np.clip(work_rate, 0, 1), 100, 200),
        "DBP":                 np.clip(80  + eva_intensity * 10 * np.clip(work_rate, 0, 1), 60, 120),
        "SweatRate":           np.clip(eva_intensity * 500 * np.clip(work_rate, 0, 1), 0, 1000),
        "MuscleGlycogen":      np.clip(350 - eva_intensity * 100 * np.clip(work_rate, 0, 1), 50, 400),
    })


# ── Precomputed XML library ────────────────────────────────────────────────────
#
# Cache key includes patient + intensity + nutrition so different astronauts
# and meal configurations produce separate cached files.
#
# File naming: StandardMale_intensity_050_c130_p20.xml
#              Male_22_Fit_Soldier_intensity_050_c130_p20.xml
#
# Legacy files (intensity_050.xml) are still readable as a fallback.

import xml.etree.ElementTree as ET

_PC_NS  = "uri:/mil/tatrc/physiology/datamodel"
_PC_XSI = "http://www.w3.org/2001/XMLSchema-instance"
ET.register_namespace("",    _PC_NS)
ET.register_namespace("xsi", _PC_XSI)


def _precomputed_key(
    eva_intensity: float,
    patient_file:  str,
    carb_g:        float,
    protein_g:     float,
) -> str:
    """Build a unique filename stem for a BioGears run with specific patient + nutrition."""
    pname = Path(patient_file).stem
    return (
        f"{pname}_intensity_{int(round(eva_intensity * 100)):03d}"
        f"_c{int(carb_g)}_p{int(protein_g)}"
    )


def _save_precomputed_xml(df: pd.DataFrame, eva_intensity: float,
                          pc_key: str | None = None) -> Path:
    """
    Serialise a BioGears output DataFrame to an XML file in precomputed/.

    `pc_key` is the filename stem (from _precomputed_key). Falls back to the
    legacy intensity-only naming when not provided.

    Returns the path of the written file.
    """
    PRECOMPUTED_DIR.mkdir(exist_ok=True)
    if pc_key:
        fname = f"{pc_key}.xml"
    else:
        fname = f"intensity_{int(round(eva_intensity * 100)):03d}.xml"
    out   = PRECOMPUTED_DIR / fname

    root = ET.Element(f"{{{_PC_NS}}}BiogearsCaptureData", {
        f"{{{_PC_XSI}}}schemaLocation": "",
        "contentVersion": "AstroGuard_1.0",
        "evaIntensity":   f"{eva_intensity:.4f}",
        "rows":           str(len(df)),
        "columns":        str(len(df.columns)),
    })

    # Write column metadata
    cols_el = ET.SubElement(root, f"{{{_PC_NS}}}Columns")
    for col in df.columns:
        ET.SubElement(cols_el, f"{{{_PC_NS}}}Column", {"name": col})

    # Write row data
    rows_el = ET.SubElement(root, f"{{{_PC_NS}}}Rows")
    for _, row in df.iterrows():
        r_el = ET.SubElement(rows_el, f"{{{_PC_NS}}}Row")
        for col in df.columns:
            ET.SubElement(r_el, f"{{{_PC_NS}}}V", {"c": col}).text = f"{row[col]:.6g}"

    # Pretty-print
    _pc_indent(root)
    ET.ElementTree(root).write(str(out), xml_declaration=True, encoding="UTF-8")
    return out


def _pc_indent(elem: ET.Element, level: int = 0) -> None:
    """Minimal in-place pretty-printer (Python 3.8 compatible)."""
    pad = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = pad
        for child in elem:
            _pc_indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = pad
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = pad


def _load_precomputed_xml(xml_path: Path) -> "pd.DataFrame | None":
    """
    Parse a precomputed BioGears XML file back into a DataFrame.

    Reads the <Columns> block to recover column order, then reads each <Row>
    collecting <V c='ColName'>value</V> elements. Returns None on any parse
    error so callers can fall through gracefully.
    """
    try:
        root    = ET.parse(str(xml_path)).getroot()
        cols_el = root.find(f"{{{_PC_NS}}}Columns")
        if cols_el is None:
            return None
        columns = [c.get("name", "") for c in cols_el.findall(f"{{{_PC_NS}}}Column")]
        if not columns:
            return None

        rows_el = root.find(f"{{{_PC_NS}}}Rows")
        if rows_el is None:
            return None

        records = []
        for row_el in rows_el.findall(f"{{{_PC_NS}}}Row"):
            record: dict = {}
            for v_el in row_el.findall(f"{{{_PC_NS}}}V"):
                col = v_el.get("c", "")
                if col:
                    record[col] = float((v_el.text or "0").strip())
            records.append(record)

        if not records:
            return None

        df = pd.DataFrame(records, columns=columns)
        return df.dropna(subset=["Time_s", "HeartRate"]).reset_index(drop=True)
    except Exception:
        return None


def _load_precomputed(eva_intensity: float,
                      pc_key: str | None = None) -> Tuple["pd.DataFrame | None", str]:
    """
    Load a precomputed BioGears XML.

    Priority:
      1. Exact match by pc_key (patient + intensity + nutrition)
      2. Fallback to legacy intensity-only naming (intensity_050.xml)
      3. Fallback to CSV (migration path from older sessions)

    For the legacy intensity-only path, blends between two adjacent files when
    the requested intensity falls between two cached values.
    """
    if not PRECOMPUTED_DIR.exists():
        return None, "precomputed/ directory not found"

    # 1. Exact patient+nutrition match
    if pc_key:
        exact = PRECOMPUTED_DIR / f"{pc_key}.xml"
        if exact.exists():
            df = _load_precomputed_xml(exact)
            if df is not None and len(df) > 20:
                return df, f"Precomputed XML (exact) {pc_key}"

    # 2. Legacy intensity-only XML files
    xmls = sorted(PRECOMPUTED_DIR.glob("intensity_*.xml"))
    if not xmls:
        csvs = sorted(PRECOMPUTED_DIR.glob("intensity_*.csv"))
        if not csvs:
            return None, "No precomputed files — run Live BioGears at least once"
        xmls_or_csvs = csvs
        loader    = _parse_csv
        ext_label = "CSV (legacy)"
    else:
        xmls_or_csvs = xmls
        loader    = _load_precomputed_xml
        ext_label = "XML (legacy)"

    available: dict[float, Path] = {}
    for p in xmls_or_csvs:
        try:
            # Legacy files: intensity_050.xml → stem has exactly 2 underscore parts
            parts = p.stem.split("_")
            if len(parts) == 2:
                val = int(parts[1]) / 100.0
                available[val] = p
        except (IndexError, ValueError):
            continue

    if not available:
        return None, "No legacy precomputed files found"

    keys = sorted(available)

    if eva_intensity <= keys[0]:
        df = loader(available[keys[0]])
        return df, f"Precomputed {ext_label} intensity={keys[0]:.2f}"
    if eva_intensity >= keys[-1]:
        df = loader(available[keys[-1]])
        return df, f"Precomputed {ext_label} intensity={keys[-1]:.2f}"

    for j in range(len(keys) - 1):
        lo, hi = keys[j], keys[j + 1]
        if lo <= eva_intensity <= hi:
            alpha  = (eva_intensity - lo) / (hi - lo)
            df_lo  = loader(available[lo])
            df_hi  = loader(available[hi])
            if df_lo is None or df_hi is None:
                break
            n        = min(len(df_lo), len(df_hi))
            df_blend = df_lo.iloc[:n].copy().reset_index(drop=True)
            for col in df_lo.columns:
                if col != "Time_s" and col in df_hi.columns:
                    df_blend[col] = (
                        (1 - alpha) * df_lo[col].values[:n]
                        + alpha     * df_hi[col].values[:n]
                    )
            return df_blend, f"Precomputed {ext_label} blend {lo:.2f}->{hi:.2f} @ {eva_intensity:.2f}"

    return None, "Precomputed blend failed"


# ── Public API ─────────────────────────────────────────────────────────────────

def get_biogears_segment(
    eva_intensity:  float,
    eva_duration_min: float,
    recovery_min:   float,
    patient_file:   str   = "StandardMale.xml",
    carb_g:         float = 130.0,
    fat_g:          float = 27.0,
    protein_g:      float = 20.0,
    sodium_g:       float = 1.0,
    water_L:        float = 0.5,
    mode:           str   = "live",
    progress_callback=None,
) -> Tuple[pd.DataFrame, bool, str]:
    """
    Top-level function used by the Streamlit app.

    mode="csv"  — instant: loads from precomputed/ library, blending if needed.
                  Falls back to synth if no precomputed files exist.
    mode="live" — runs bg-scenario.exe with the selected patient + nutrition,
                  saves result to precomputed/ on success, falls back to synth if unavailable.

    Returns: (df, used_real_biogears, status_message)
    """
    pc_key = _precomputed_key(eva_intensity, patient_file, carb_g, protein_g)

    if mode == "csv":
        df, msg = _load_precomputed(eva_intensity, pc_key)
        if df is not None and len(df) > 20:
            return df, True, f"CSV MODE · {msg}"
        fallback_df = _load_fallback(eva_intensity, BG_EVA_DURATION_MIN, BG_RECOVERY_MIN)
        return fallback_df, False, f"CSV MODE · {msg} — using synth fallback"

    # Live mode
    df, msg = run_biogears(eva_intensity, patient_file=patient_file,
                           carb_g=carb_g, fat_g=fat_g, protein_g=protein_g,
                           sodium_g=sodium_g, water_L=water_L,
                           progress_callback=progress_callback)
    if df is not None and len(df) > 20:
        _save_precomputed_xml(df, eva_intensity, pc_key)   # cache with patient+nutrition key
        return df, True, msg

    fallback_df = _load_fallback(eva_intensity, BG_EVA_DURATION_MIN, BG_RECOVERY_MIN)
    fallback_msg = f"BioGears unavailable — using synthesised signals. ({msg})"
    return fallback_df, False, fallback_msg


def extract_biogears_calibration(df: pd.DataFrame, baseline_s: float = 30.0) -> dict:
    """
    Extract patient + nutrition-specific physiological rates from a BioGears run.

    The EVA phase starts after the baseline (30 s) + post-meal wait (30 s) = 60 s.
    All rates are per-minute, matching fatigue.py's 1-minute simulation tick.

    Returns
    -------
    eva_fatigue_rate_per_min      : BioGears FatigueLevel slope during EVA (0.008 fallback)
    glycogen_start_g              : MuscleGlycogen at EVA start
    glycogen_depletion_g_per_min  : MuscleGlycogen depletion rate during EVA (2.8 fallback)
    sweat_rate_mg_per_min         : mean SweatRate during EVA (600 mg/min fallback ≈ 0.6 L/h)
    """
    # EVA starts after baseline (30 s) + post-meal wait (30 s)
    eva_start_s = baseline_s + 30.0
    eva_df = df[df["Time_s"] > eva_start_s].copy()

    cal: dict = {}

    # ── BioGears FatigueLevel slope ──────────────────────────────────────────
    if "FatigueLevel" in df.columns and len(eva_df) > 5:
        fv = eva_df["FatigueLevel"].values.astype(float)
        duration_min = len(fv) / 60.0
        cal["eva_fatigue_rate_per_min"] = float(
            max(0.0, fv[-1] - fv[0]) / max(duration_min, 0.1)
        )
    else:
        cal["eva_fatigue_rate_per_min"] = 0.008

    # ── MuscleGlycogen depletion ─────────────────────────────────────────────
    if "MuscleGlycogen" in df.columns and len(eva_df) > 5:
        gv = eva_df["MuscleGlycogen"].values.astype(float)
        duration_min = len(gv) / 60.0
        cal["glycogen_start_g"] = float(gv[0])
        cal["glycogen_depletion_g_per_min"] = float(
            max(0.0, gv[0] - gv[-1]) / max(duration_min, 0.1)
        )
    else:
        cal["glycogen_start_g"] = 350.0
        cal["glycogen_depletion_g_per_min"] = 2.8

    # ── SweatRate ────────────────────────────────────────────────────────────
    if "SweatRate" in df.columns and len(eva_df) > 5:
        cal["sweat_rate_mg_per_min"] = float(eva_df["SweatRate"].mean())
    else:
        cal["sweat_rate_mg_per_min"] = 600.0

    return cal


def extract_segment_stats(df: pd.DataFrame, baseline_s: float = 30.0) -> dict:
    """
    Extract mean ± std of key physiological variables from
    the baseline and exercise segments of a BioGears run.
    """
    baseline = df[df["Time_s"] <= baseline_s]
    exercise = df[df["Time_s"] > baseline_s]

    def _stats(series: pd.Series) -> Tuple[float, float]:
        return float(series.mean()), float(series.std(ddof=0) + 1e-9)

    stats: dict = {}
    for col in ["HeartRate", "OxygenSaturation", "CoreTemperature", "RespirationRate"]:
        if col in df.columns:
            stats[f"baseline_{col}_mean"], stats[f"baseline_{col}_std"] = _stats(baseline[col])
            stats[f"exercise_{col}_mean"], stats[f"exercise_{col}_std"] = _stats(exercise[col])

    return stats
