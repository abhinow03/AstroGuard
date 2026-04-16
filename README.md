# AstroGuard — Astronaut Health Digital Twin

> Real-time fatigue monitoring and injury risk analysis for long-duration spaceflight missions, powered by BioGears cardiovascular physiology and Monte Carlo simulation.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![BioGears](https://img.shields.io/badge/BioGears-8.2.0-green)
![Plotly](https://img.shields.io/badge/Plotly-5.x-purple)

---

## What it does

AstroGuard simulates an astronaut's physiological state over a 24–72 hour mission. It uses:

- **BioGears 8.2.0** to generate real cardiovascular data (heart rate, SpO₂, core temperature, fatigue level) for an EVA exercise scenario
- **Probabilistic health variables** — HR, SpO₂, core temperature, respiration rate modelled with Normal distributions and extended across the full mission timeline
- **Discrete-event engine** — stochastically schedules EVA, sleep, and dehydration events with randomised timing and intensity
- **Mission-level fatigue model** — accumulates and recovers fatigue per minute based on event type and normalised heart rate
- **Monte Carlo analysis** — 10–500 simulations with randomised event schedules to compute breach probability and worst-case fatigue
- **Streamlit dashboard** — interactive dark-themed UI with 4 tabs: Mission Overview, Risk Analysis, Phase Space, Monte Carlo

---

## Screenshots

| Mission Overview | Risk Analysis |
|---|---|
| 4-panel physiology chart (HR / SpO₂ / Temp / Fatigue) | Fatigue gauge + risk flag log |

| Phase Space | Monte Carlo |
|---|---|
| HR vs Fatigue trajectory coloured by time | Fatigue envelope + simulation heatmap |

---

## Project structure

```
AstroGuard/
├── app.py                              # Streamlit dashboard (entry point)
├── fatigue_model.py                    # Standalone legacy fatigue script
├── CardiovascularValidationResults.csv # Fallback CSV if BioGears unavailable
│
├── simulation/
│   ├── biogears.py      # Run BioGears via subprocess, parse output, fallback synth
│   ├── events.py        # Discrete-event engine (EVA, Sleep, Dehydration)
│   ├── health_vars.py   # 24–72h probabilistic mission timeline builder
│   └── fatigue.py       # Per-minute fatigue accumulation model
│
├── analytics/
│   └── risk.py          # Single-run analytics + Monte Carlo runner
│
└── visualization/
    └── charts.py        # All Plotly chart functions (dark space theme)
```

---

## Requirements

- Python 3.9 or higher
- BioGears 8.2.0 (optional — the app falls back to synthesised signals if unavailable)

### Python dependencies

```bash
pip install streamlit plotly scipy numpy pandas
```

Or install from the list at once:

```bash
pip install streamlit plotly scipy numpy pandas
```

---

## BioGears installation (optional but recommended)

BioGears provides real cardiovascular physiology for the EVA scenario. Without it the app still works using synthesised fallback signals.

1. Download **BioGears 8.2.0** from the official releases:
   [https://github.com/BioGearsEngine/core/releases](https://github.com/BioGearsEngine/core/releases)

2. Extract to a fixed path. The default path expected by the app is:
   ```
   Z:\BIOGEARS\
   ```
   Inside you should have:
   ```
   Z:\BIOGEARS\bin\bg-scenario.exe
   Z:\BIOGEARS\bin\patients\StandardMale.xml
   ```

3. If your BioGears is installed at a different path, open `simulation/biogears.py` and update this line near the top:
   ```python
   BIOGEARS_BIN = Path("Z:/BIOGEARS/bin")
   ```
   Change `Z:/BIOGEARS/bin` to wherever your `bin/` folder is, for example:
   ```python
   BIOGEARS_BIN = Path("C:/BioGears/bin")
   ```

4. The app generates a custom EVA scenario XML, runs `bg-scenario.exe`, and parses the output CSV automatically. A CMD window will open briefly while BioGears runs — this is expected.

> **Note:** BioGears only works natively on **Windows**. On Linux/macOS the app will automatically use synthesised fallback signals.

---

## Running the app

```bash
# Clone the repo
git clone https://github.com/abhinow03/AstroGuard.git
cd AstroGuard

# Install dependencies
pip install streamlit plotly scipy numpy pandas

# Run
streamlit run app.py
```

Open the URL printed in the terminal (default: `http://localhost:8501`).

---

## How to use

1. **Sidebar — BioGears EVA Scenario**
   - Set EVA intensity (0.1 = light jog, 0.9 = max exertion)
   - Set EVA duration and recovery duration
   - These parameters are sent directly to BioGears

2. **Sidebar — Mission Parameters**
   - Choose mission length: 24h / 48h / 72h
   - Set number of EVA events (1–3)
   - Set the fatigue risk threshold (default 0.80)

3. **Sidebar — Monte Carlo**
   - Choose number of simulations (10 to 500)

4. **Click "Run BioGears + Simulate"**
   - BioGears runs in the background (~3 min for the fixed 10-min EVA segment)
   - Results are cached — subsequent runs with the same BioGears parameters are instant

5. **Explore the 4 tabs**
   - **Mission Overview** — full physiological time-series with event shading
   - **Risk Analysis** — fatigue gauge, trend analysis, risk flag log
   - **Phase Space** — HR vs Fatigue trajectory
   - **Monte Carlo** — probabilistic envelope, heatmap, peak fatigue histogram

---

## How the physiology pipeline works

```
BioGears (EVA scenario, ~10 min)
    └── get_biogears_segment()
            │
            ▼
    biogears_df  (HR, SpO₂, CoreTemp, FatigueLevel at 1 Hz)
            │
            ▼
    build_mission_timeline()   ← sample_events() schedules EVA/Sleep/Dehydration
            │
            ▼
    mission_df  (1-min resolution, 24–72h)
            │
            ▼
    compute_fatigue()          ← per-minute accumulation model
            │
            ▼
    single_run_analytics()     ← peak, time-at-risk, trend slope
    run_monte_carlo()          ← 100 random mission variants
            │
            ▼
    charts.py                  ← Plotly figures → Streamlit dashboard
```

---

## Fatigue model

The fatigue index `f(t) ∈ [0, 1]` updates every mission minute:

| Phase | Rule |
|---|---|
| EVA | `f += 0.008 × hr_norm` |
| Sleep | `f -= 0.006` |
| Rest | `f -= 0.002 × (1 − hr_norm)` |

where `hr_norm = (HR − 35) / (220 − 35)`.

Mission status is determined by peak fatigue relative to threshold:
- **SAFE** — peak < threshold × 0.85
- **MONITOR** — peak < threshold
- **ABORT** — peak ≥ threshold

---

## Notes

- BioGears runs are cached by `@st.cache_data` keyed on `(eva_intensity, eva_duration_min, recovery_min)` — changing only mission hours or Monte Carlo N does not re-run BioGears
- The fixed BioGears scenario is always 10 min EVA + 5 min recovery (fast run ~3 min). The shape is then resampled to fill whatever EVA duration you set in the sidebar
- Each Monte Carlo simulation uses a different random seed, randomising EVA start time, duration, and whether a dehydration event occurs

---

## Built for

Semester 6 Digital Twins Mini-Project — RESPOND Basket 2025, RES-HSFC-2025-001
