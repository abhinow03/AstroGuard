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
BIOGEARS_BIN = Path("Z:/BIOGEARS/bin")
BG_SCENARIO  = BIOGEARS_BIN / "bg-scenario.exe"   # bg-scenario.exe works; bg-cli.exe crashes
RUNS_DIR     = BIOGEARS_BIN / "runs"
SCENARIO_NAME = "EVA_Mission_Scenario"
FALLBACK_CSV  = Path(__file__).parent.parent / "CardiovascularValidationResults.csv"

# BioGears scenario is intentionally SHORT (10 min EVA, 5 min recovery).
# The physiological SHAPE is captured here and then resampled to fill the
# full mission EVA duration in health_vars.py.
BG_EVA_DURATION_MIN  = 10   # minutes of BioGears exercise phase
BG_RECOVERY_MIN      = 5    # minutes of BioGears recovery phase


# ── Scenario XML generation ────────────────────────────────────────────────────

def _generate_scenario_xml(eva_intensity: float) -> str:
    """
    Build the BioGears scenario XML string for a SHORT EVA capture run.
    Duration is fixed (BG_EVA_DURATION_MIN / BG_RECOVERY_MIN) so the
    Streamlit app doesn't wait more than ~3-4 minutes for BioGears.
    """
    baseline_s = 30                          # 30s warm-up (stabilisation is separate)
    eva_s      = BG_EVA_DURATION_MIN * 60
    rec_s      = BG_RECOVERY_MIN * 60

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Scenario xmlns="uri:/mil/tatrc/physiology/datamodel"
          xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
          contentVersion="BioGears_6.3.0-beta"
          xsi:schemaLocation="">
  <Name>{SCENARIO_NAME}</Name>
  <Description>EVA Workload Simulation — Astronaut Fatigue Digital Twin</Description>
  <InitialParameters>
    <PatientFile>StandardMale.xml</PatientFile>
  </InitialParameters>

  <!-- 1 sample/s keeps output manageable (~few thousand rows) -->
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

    <!-- Phase 2: EVA workload -->
    <Action xsi:type="ExerciseData">
      <GenericExercise>
        <Intensity value="{eva_intensity:.3f}"/>
      </GenericExercise>
    </Action>
    <Action xsi:type="AdvanceTimeData">
      <Time value="{eva_s}" unit="s"/>
    </Action>

    <!-- Phase 3: recovery -->
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
    eva_intensity: float,
    timeout: int = 600,
    progress_callback=None,
) -> Tuple["pd.DataFrame | None", str]:
    """
    Write EVA scenario XML → call bg-scenario.exe → parse output CSV.

    Uses bg-scenario.exe (NOT bg-cli.exe — that crashes on this system).
    Scenario duration is fixed short (BG_EVA_DURATION_MIN / BG_RECOVERY_MIN)
    so the run completes in ~3-4 minutes.

    Returns (DataFrame | None, status_message).
    """
    if not BG_SCENARIO.exists():
        return None, f"bg-scenario.exe not found at {BG_SCENARIO}"

    scenario_path = BIOGEARS_BIN / f"{SCENARIO_NAME}.xml"
    xml = _generate_scenario_xml(eva_intensity)
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


# ── Public API ─────────────────────────────────────────────────────────────────

def get_biogears_segment(
    eva_intensity: float,
    eva_duration_min: float,   # kept for API compatibility; BioGears uses BG_EVA_DURATION_MIN
    recovery_min: float,       # kept for API compatibility; BioGears uses BG_RECOVERY_MIN
    progress_callback=None,
) -> Tuple[pd.DataFrame, bool, str]:
    """
    Top-level function used by the Streamlit app.

    Runs a short BioGears EVA capture (BG_EVA_DURATION_MIN minutes) to get
    the physiological shape at `eva_intensity`.  health_vars.py then
    resamples that shape to fill the actual mission EVA duration.

    Returns:
        (df, used_real_biogears, status_message)
    """
    df, msg = run_biogears(eva_intensity, progress_callback=progress_callback)
    if df is not None and len(df) > 20:
        return df, True, msg

    # Fallback: synthesise a realistic segment
    fallback_df = _load_fallback(eva_intensity, BG_EVA_DURATION_MIN, BG_RECOVERY_MIN)
    fallback_msg = ("BioGears unavailable — using synthesised physiological signals. "
                    f"(Reason: {msg})")
    return fallback_df, False, fallback_msg


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
