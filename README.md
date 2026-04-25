# AstroGuard — Astronaut Health Digital Twin

A mission-control dashboard that simulates how an astronaut's body responds during a space mission. It uses a real physics-based physiology engine (BioGears), probabilistic health modelling, and Monte Carlo risk analysis to predict fatigue and flag when an astronaut is in danger.

---

## Table of Contents

1. [What is this project?](#1-what-is-this-project)
2. [The Big Picture — How everything connects](#2-the-big-picture--how-everything-connects)
3. [Step 1 — BioGears: the physiology engine](#3-step-1--biogears-the-physiology-engine)
4. [Step 2 — Mission events: what happens during the mission](#4-step-2--mission-events-what-happens-during-the-mission)
5. [Step 3 — Building the 24–72 hour health timeline](#5-step-3--building-the-2472-hour-health-timeline)
6. [Step 4 — Fatigue model: the math](#6-step-4--fatigue-model-the-math)
7. [Step 5 — Risk analytics and Monte Carlo simulation](#7-step-5--risk-analytics-and-monte-carlo-simulation)
8. [Every graph explained](#8-every-graph-explained)
9. [Every sidebar control explained](#9-every-sidebar-control-explained)
10. [The top metric cards explained](#10-the-top-metric-cards-explained)
11. [Mission Status: SAFE / MONITOR / ABORT](#11-mission-status-safe--monitor--abort)

---

## 1. What is this project?

When an astronaut performs a spacewalk (called an EVA — Extravehicular Activity), their body is under extreme stress. Their heart rate spikes, oxygen levels drop slightly, core body temperature rises, and they accumulate fatigue over time. If fatigue builds up too much without recovery, it can become life-threatening.

This project is a **digital twin** — a software copy of an astronaut's body — that lets you simulate different mission scenarios and see whether the astronaut is safe, needs to be monitored, or should abort the mission.

You can control:
- How intense the spacewalk is
- How long the spacewalk lasts
- How long the mission is (24, 48, or 72 hours)
- How many Monte Carlo "what-if" scenarios to run

The dashboard then shows you in real time what the astronaut's heart rate, oxygen saturation, body temperature, and fatigue index would look like over the entire mission.

---

## 2. The Big Picture — How everything connects

```
SIDEBAR CONTROLS
      │
      ▼
┌─────────────────┐
│  BioGears 8.2   │  ← Real physics engine simulates a short EVA (10 min)
│  (biogears.py)  │    and gives us the physiological "shape" of the response
└────────┬────────┘
         │  heart rate curve, SpO2, temperature, BioGears FatigueLevel
         ▼
┌─────────────────┐
│  Mission Events │  ← Stochastically schedules EVA, Sleep, Dehydration
│  (events.py)    │    across the full 24–72 hour mission
└────────┬────────┘
         │  list of when each event happens
         ▼
┌─────────────────────┐
│  Health Timeline    │  ← Takes BioGears shape + event schedule →
│  (health_vars.py)   │    builds minute-by-minute HR, SpO2, Temp, RR
└────────┬────────────┘
         │  1-minute-resolution arrays of physiological signals
         ▼
┌─────────────────┐
│  Fatigue Model  │  ← Uses HR at each minute to compute fatigue
│  (fatigue.py)   │    accumulation over the entire mission
└────────┬────────┘
         │  fatigue array [0 to 1] for every minute
         ▼
┌─────────────────┐
│  Risk Analytics │  ← Finds peak fatigue, time at risk, trend direction
│  (risk.py)      │    Runs Monte Carlo across N randomised missions
└────────┬────────┘
         │  statistics, probability curves, heatmaps
         ▼
┌─────────────────┐
│   Dashboard     │  ← All charts, metric cards, status banner
│   (app.py)      │
└─────────────────┘
```

Each step feeds the next. Nothing is calculated in isolation.

---

## 3. Step 1 — BioGears: the physiology engine

**File:** `simulation/biogears.py`

### What is BioGears?

BioGears is an open-source human physiology simulator developed for medical and military training. It models the cardiovascular system, respiratory system, musculoskeletal system, and more — all coupled together the way they work in a real human body. When you tell it "exercise at 50% intensity for 10 minutes", it simulates what happens to the heart, lungs, muscles, and blood chemistry second-by-second.

### What we ask BioGears to simulate

We create a short EVA scenario with three phases:

| Phase | Duration | What happens |
|-------|----------|-------------|
| Baseline stabilisation | 30 seconds | Astronaut at rest — body reaches steady state |
| EVA workload | 10 minutes | Exercise at the intensity you set |
| Recovery | 5 minutes | Exercise stops, body recovers |

The scenario is intentionally short (15 minutes total) so the simulation finishes quickly. The physiological *shape* — how heart rate ramps up, how fatigue grows, how oxygen saturation drops — is what we care about. We then stretch that shape to cover the real EVA duration you set in the sidebar.

### What BioGears outputs (per second)

| Signal | Unit | What it means |
|--------|------|--------------|
| HeartRate | beats/min | How fast the heart is beating |
| OxygenSaturation | 0–1 (unitless) | Fraction of haemoglobin carrying oxygen |
| CoreTemperature | °C | Internal body temperature (not skin) |
| RespirationRate | breaths/min | Breathing rate |
| FatigueLevel | 0–1 (unitless) | BioGears' own built-in fatigue estimate |
| TotalWorkRateLevel | 0–1 | Fraction of maximum work rate being done |
| TotalMetabolicRate | Watts | Total energy output of the body |
| CardiacOutput | L/min | Volume of blood the heart pumps per minute |
| SystolicArterialPressure | mmHg | Top number in blood pressure |
| DiastolicArterialPressure | mmHg | Bottom number in blood pressure |
| SweatRate | mg/min | How fast the astronaut is sweating |
| MuscleGlycogen | grams | Remaining fuel in the muscles |

### What if BioGears isn't installed?

If `bg-scenario.exe` isn't found or fails, the code automatically falls back to a **synthetic segment** generated entirely in Python. This synthetic fallback uses the same mathematical relationships (HR rises proportionally to intensity, SpO2 drops slightly, temperature rises, fatigue grows) so the dashboard still behaves realistically. The status badge on the dashboard will say **SYNTH FALLBACK** instead of **BIOGEARS LIVE**.

### Key equations in the fallback

```
hr_peak = 72 + intensity × 100          # peak heart rate during EVA
hr_rise = hr_baseline + (hr_peak - hr_baseline) × ramp    # ramp-up over 30% of EVA
hr_decay = hr_peak × exp(-t / τ) + hr_baseline × (1 - exp(-t / τ))   # exponential recovery

fatigue = intensity × min(t / eva_duration, 1) × 0.70    # rises linearly during EVA
fatigue_recovery = fatigue_peak × exp(-t / τ_recovery)    # decays exponentially
```

Where `intensity` is the EVA intensity you set (0.1 to 0.9).

---

## 4. Step 2 — Mission events: what happens during the mission

**File:** `simulation/events.py`

A real mission is not just one continuous spacewalk. There are periods of rest, sleep, and unexpected physiological stressors. This module randomly schedules three types of events across the full mission timeline.

### Event 1: EVA (Extravehicular Activity) — the spacewalk

This is the main workload event. It uses the intensity you set in the sidebar.

- **When it starts:** Randomly drawn from Uniform(6 hours, 14 hours) into each mission day. So if the mission is 48 hours, the first EVA starts somewhere between hour 6 and hour 14.
- **How long it lasts:** Randomly drawn from Normal(4 hours, 0.5 hours), clipped to between 2 and 6 hours. This means most EVAs last around 4 hours with some variation.
- **Effect on the body:** Uses the BioGears exercise shape (scaled to the EVA duration). Heart rate, temperature, and respiration all rise. SpO2 drops slightly.

### Event 2: Sleep / Recovery

Every 24 hours the astronaut sleeps. Sleep is physiologically important — it's when the body repairs itself and fatigue decreases fastest.

- **When it starts:** Around 22:00 (10 PM) each day, with ±1 hour of random jitter.
- **How long it lasts:** Randomly drawn from Uniform(7 hours, 9 hours).
- **Effect on the body:** Heart rate drops to ~58 bpm, core temperature drops to ~36.4°C, SpO2 normalises at ~97.1%, respiration slows to ~12 breaths/min.
- **Effect on fatigue:** Fatigue decreases at a rate of 0.006 per minute (the fastest recovery rate).

### Event 3: Dehydration episode (probabilistic)

Space suits are hot and the astronaut can't drink freely. Dehydration is a real risk.

- **Whether it happens:** 30% chance (Bernoulli(0.3)) — it may or may not occur.
- **When it starts:** Randomly drawn from Uniform(12 hours into mission, mission_length − 4 hours).
- **How long it lasts:** Randomly drawn from Uniform(2 hours, 5 hours).
- **Effect on the body:** Gradual heart rate drift upward (up to +15 bpm at full intensity), slight SpO2 drop. This mimics the cardiovascular strain of fluid loss.

The colour coding on all charts:
- **Orange shading** = EVA window
- **Blue shading** = Sleep window
- **Red shading** = Dehydration window

---

## 5. Step 3 — Building the 24–72 hour health timeline

**File:** `simulation/health_vars.py`

### The problem

BioGears only simulates 15 minutes. We need 24 to 72 hours of data (1,440 to 4,320 minutes) at 1-minute resolution.

### The solution

We extract the physiological "shapes" from BioGears — the curve of how HR rises during exercise, how it recovers afterward — and **resample** those shapes to fit each event's actual duration on the mission timeline.

Think of it like having a template waveform for how the heart responds to exercise, and stretching or squeezing that template to fit however long the actual EVA is.

### Step by step

1. Start with every variable at the resting baseline value (from BioGears' stabilisation phase).
2. Add small Gaussian noise at every minute to simulate natural body variation:
   - Heart rate: Normal(0, 2 bpm) per minute
   - SpO2: Normal(0, 0.003) per minute
   - Core temperature: Normal(0, 0.05°C) per minute
   - Respiration rate: Normal(0, 0.5 breaths/min) per minute
3. For each EVA window, replace the baseline signal with the BioGears EVA shape (resampled to EVA duration), plus noise.
4. After each EVA, apply the BioGears recovery shape (resampled to recovery duration).
5. For each Sleep window, replace the signal with sleep physiology (lower HR, lower temp, normal SpO2) drawn from Normal distributions.
6. For Dehydration, add a linear drift to HR and SpO2 over the episode's duration.
7. Clip all values to physiological limits (HR: 35–220, SpO2: 70%–100%, Temp: 35°C–41°C).

The result is a DataFrame with one row per minute, covering the entire mission.

---

## 6. Step 4 — Fatigue model: the math

**File:** `simulation/fatigue.py`

### What fatigue means here

Fatigue is a number from **0 to 1**:
- 0 = fully rested
- 1 = completely exhausted (mission abort territory)

It's calculated minute by minute, where each minute's fatigue depends on the previous minute's fatigue plus or minus a small amount based on what's happening.

### The heart rate normalisation step

Before calculating fatigue, we convert heart rate into a **normalised effort signal** (how hard the astronaut is working relative to their maximum):

```
hr_norm = (HR_current - HR_baseline) / (HR_max - HR_baseline)
```

With default values:
- `HR_baseline = 72 bpm` (resting heart rate)
- `HR_max = 200 bpm` (theoretical maximum)

So if the astronaut's heart rate is 136 bpm:

```
hr_norm = (136 - 72) / (200 - 72) = 64 / 128 = 0.5
```

This means they are working at 50% of their maximum effort. `hr_norm` is clipped to [0, 1].

### The fatigue update equation

At every minute `i`, fatigue updates as:

**During EVA:**
```
fatigue[i] = fatigue[i-1] + 0.0080 × hr_norm[i]
```
The harder the astronaut works (higher heart rate), the faster fatigue accumulates. At full effort (hr_norm = 1), fatigue rises by 0.008 per minute.

**During Sleep:**
```
fatigue[i] = fatigue[i-1] - 0.0060
```
Sleep always recovers fatigue at a fixed rate of 0.006 per minute, regardless of heart rate.

**During Rest (not EVA, not Sleep):**
```
fatigue[i] = fatigue[i-1] - 0.0020 × (1 - hr_norm[i])
```
Passive recovery is slow (0.002/min max) and is even slower if the heart rate is still elevated (e.g., after an EVA hasn't fully recovered yet).

**Always:** fatigue is clipped to [0, 1] — it can never go below 0 or above 1.

### How fast does fatigue change?

| Situation | Rate | Time to go from 0 to 0.8 at full effort |
|-----------|------|----------------------------------------|
| EVA, max intensity (hr_norm = 1.0) | +0.008/min | 100 minutes |
| EVA, half intensity (hr_norm = 0.5) | +0.004/min | 200 minutes |
| Sleep | −0.006/min | Recover 0.8 in ~133 minutes |
| Passive rest (hr_norm = 0) | −0.002/min | Recover 0.8 in 400 minutes |

### BioGears FatigueLevel vs our model

BioGears has its own internal fatigue estimate (the `FatigueLevel` column from the engine). We display both on the same chart:

- **Green solid line** = our fatigue model (calculated from HR using the equations above)
- **Teal dotted line** = BioGears' internal FatigueLevel (from its own muscle physiology model)

They are similar but not identical — our model is driven by heart rate and events, while BioGears' model is driven by muscle glycogen depletion, oxygen consumption, and metabolic rate at the muscle level. Seeing them agree is a good sign; divergence tells you something interesting about the physiology.

---

## 7. Step 5 — Risk analytics and Monte Carlo simulation

**File:** `analytics/risk.py`

### Single-run analytics

After one simulation, we compute:

| Metric | How it's calculated |
|--------|-------------------|
| **Peak Fatigue** | `max(fatigue_array)` |
| **Peak Minute** | `argmax(fatigue_array)` |
| **Time at Risk** | `count(fatigue > threshold) / total_minutes × 100%` |
| **Trend** | Linear regression (`scipy.stats.linregress`) on the last 20% of the mission. Slope > 0.00005 = Rising, < −0.00005 = Falling, else Stable |
| **Risk Windows** | Continuous time intervals where `fatigue > threshold` |

### Monte Carlo: simulating uncertainty

A single simulation uses one random seed — one possible version of the mission. In reality, we don't know exactly when the EVA will start, how tired the astronaut already is, or whether dehydration will occur. Monte Carlo simulation explores all those possibilities at once.

**What changes in each Monte Carlo run:**
- EVA start time (Uniform random)
- EVA duration (Normal ±0.5 hours)
- EVA intensity (Normal ±0.05 around your setting)
- Whether dehydration occurs (Bernoulli 0.3)
- All noise seeds for the health variables

**What we collect from N runs:**
- `fatigue_matrix`: an N × time_steps array of every fatigue trajectory
- `mean_fatigue[t]`: average fatigue at each minute across all runs
- `p5_fatigue[t]` and `p95_fatigue[t]`: the 5th and 95th percentile — the 90% confidence band
- `p_risk[t]`: fraction of simulations where fatigue exceeded the threshold at that minute
- `max_per_sim`: the peak fatigue reached in each individual simulation

### P(breach) — the key number

```
P(breach) = count(max_per_sim > threshold) / N
```

This is the probability that, across all possible mission scenarios we simulated, the astronaut's fatigue at some point exceeds the danger threshold. A value of 0.35 means "in 35% of our simulated scenarios, the astronaut would hit dangerous fatigue levels".

---

## 8. Every graph explained

### Tab 1: PHYSIO · OVERVIEW

This is the main four-panel chart showing the full mission health timeline.

#### Panel 1 — Heart Rate (orange)
- **What it shows:** The astronaut's heart rate in beats per minute across the entire mission.
- **What to look for:** Big orange spikes = EVA periods. Dips to ~58 bpm = sleep. A gradual rise without recovery = dehydration or fatigue accumulation.
- **Normal range:** ~65–80 bpm at rest. During heavy EVA it can reach 150–180 bpm.
- **Why it matters:** Heart rate is the primary input to our fatigue model. High HR for sustained periods = rapid fatigue accumulation.

#### Panel 2 — SpO₂ (blue)
- **What it shows:** Blood oxygen saturation — the percentage of haemoglobin in the blood that is carrying oxygen.
- **What to look for:** The line should stay above 95% almost always. Drops during EVA are normal (mild hypoxia from exertion). A sustained drop below 90% is dangerous.
- **Normal range:** 97–99% at rest. May dip to 94–96% during intense EVA.
- **Why it matters:** If SpO2 drops too low, the astronaut's muscles and brain don't get enough oxygen, accelerating fatigue and impairing judgement.

#### Panel 3 — Core Temperature (purple)
- **What it shows:** Internal body temperature in Celsius.
- **What to look for:** Rises during EVA as muscles generate heat. The space suit's thermal management system handles some of this, but prolonged EVA can cause hyperthermia. Drops slightly during sleep.
- **Normal range:** 36.5–37.5°C at rest. May reach 38.5–39°C at high EVA intensity.
- **Why it matters:** Hyperthermia (overheating) is a real risk during EVA and accelerates both physical and cognitive fatigue.

#### Panel 4 — Fatigue Index (green solid + teal dotted)
- **What it shows:** The computed fatigue index from 0 to 1.
  - Green solid line = our HR-based model
  - Teal dotted line = BioGears' own FatigueLevel
  - Red dashed horizontal line = your risk threshold
- **What to look for:** Green line rising above the red threshold line = danger zone. The line should drop during sleep periods. If the line never fully recovers between EVAs in a multi-EVA mission, cumulative fatigue is building up.
- **Coloured background bands:** Orange = EVA, Blue = Sleep, Red = Dehydration.

#### The BioGears Raw Segment chart (below the main chart)
This smaller chart shows only the raw 15-minute BioGears output — the actual engine output before we resample it. It lets you verify that BioGears ran correctly and see the exact physiological response shape that was captured.

---

### Tab 2: RISK · ANALYSIS

#### Risk Gauge (the circular dial)
- **What it shows:** The peak fatigue value from the simulation, displayed as a gauge from 0 to 1.
- **Green zone:** Peak fatigue below 70% of threshold — safe.
- **Orange zone:** Between 70% and 100% of threshold — monitor closely.
- **Red zone:** Peak fatigue above threshold — danger.
- The delta arrow shows how far above or below the threshold the peak fatigue is.

#### Risk Summary Table
Lists every continuous time window where fatigue exceeded the threshold, with start time, end time, and duration. If there are no entries, the astronaut never hit the danger zone.

#### Trend Analysis Text
Reports the linear regression result on the last 20% of the mission:
- **Rising:** Fatigue is still climbing at mission end — the astronaut would not be safe to continue.
- **Falling:** Recovery is occurring — fatigue is decreasing by end of mission.
- **Stable:** Fatigue has plateaued — neither improving nor worsening.

---

### Tab 3: DYNAMICS · PHASE

#### Phase Space: Heart Rate vs Fatigue
This is the most visually interesting chart and the hardest to understand at first.

- **Each dot** is one minute of the mission.
- **X-axis:** Heart rate at that minute.
- **Y-axis:** Fatigue index at that minute.
- **Colour:** Encodes mission time (blue = early, green = middle, orange = later, red = near end).

**What patterns mean:**

- **Cluster in bottom-left (low HR, low fatigue):** Resting or sleeping. Healthy.
- **Trail moving right and up (high HR, rising fatigue):** During EVA. Normal.
- **Trail moving left but staying high (HR dropping, fatigue staying high):** Post-EVA recovery — fatigue isn't gone even though exercise stopped.
- **Points in the red zone at the top:** Dangerous moments where fatigue exceeded the threshold.
- **Tight loops or spirals:** Repeated EVA-recovery cycles. In a 72-hour mission with multiple EVAs, you'll see several loops.

The shape of the cloud tells you about mission structure at a glance. A well-managed mission looks like a loop that always returns to the bottom-left. A dangerous mission has points piling up in the top-right red zone.

---

### Tab 4: MC · SIMULATION

#### Monte Carlo Envelope Chart
- **Green filled band:** The 90% confidence interval (5th to 95th percentile of fatigue at each minute across all N simulations). This band shows the range of likely outcomes.
- **Bright green line:** The mean fatigue trajectory — the average across all simulations.
- **Red dotted line (right axis, 0–100%):** P(risk) — the probability at each minute that a random simulation has fatigue above the threshold. When this climbs above 50%, more than half of all possible mission scenarios are dangerous.

**What to look for:**
- A narrow band = scenarios are consistent and predictable.
- A wide band = high uncertainty. Small changes in timing or intensity produce very different outcomes.
- The moment the red P(risk) line rises steeply is when the mission enters a window of high danger probability.

#### Fatigue Heatmap
- **Each row** is one Monte Carlo simulation.
- **Each column** is one minute of the mission.
- **Colour:** Green (low fatigue) → yellow → orange → red (high fatigue).

**What to look for:**
- Vertical stripes of orange/red appearing at similar times across many rows = that time window is consistently dangerous regardless of randomisation.
- Some rows much redder than others = high sensitivity to initial conditions (some random seeds produce much worse outcomes).
- If the bottom half of the heatmap is green and the top half is red, the mission is on the boundary — small changes in planning could make it safe or dangerous.

#### Peak Fatigue Histogram
- X-axis: peak fatigue reached in each Monte Carlo run.
- Y-axis: how many runs reached that peak.
- Red vertical line: the threshold.
- Bars to the right of the red line = dangerous runs.

---

## 9. Every sidebar control explained

### BioGears Scenario Section

#### EVA Intensity (slider: 0.1 – 0.9)
- **What it is:** A fraction of the astronaut's maximum physical capacity. 0.5 = working at 50% of maximum.
- **How it maps to heart rate:** `HR_peak = 72 + intensity × 100`. So intensity 0.5 → HR peaks at ~122 bpm. Intensity 0.9 → HR peaks at ~162 bpm.
- **What changes when you increase it:**
  - BioGears runs a harder exercise → higher HR output → steeper HR curve
  - SpO2 drops more (by up to 1.8% at intensity 0.9)
  - Core temperature rises more during EVA
  - Fatigue accumulates faster (`hr_norm` is higher → larger delta per minute)
  - In Monte Carlo, more scenarios breach the threshold
- **Realistic values:** ISS astronauts during EVA typically work at 0.4–0.7. Above 0.8 is extreme exertion.

#### EVA Duration (slider: 30 – 240 minutes)
- **What it is:** How long the EVA phase of the mission lasts, in minutes.
- **What changes when you increase it:**
  - The BioGears exercise shape is stretched to fill the full duration
  - The astronaut spends more time at elevated HR → more fatigue accumulates
  - There's a longer orange band on all charts
  - Peak fatigue is higher and reached sooner proportionally
  - In Monte Carlo, P(breach) increases significantly for long EVAs at moderate-high intensity
- **Realistic values:** Real ISS EVAs typically last 5–8 hours (300–480 min). 6.5 hours is average.

#### Recovery Duration (slider: 10 – 60 minutes)
- **What it is:** How many minutes of rest occur immediately after EVA before the astronaut resumes other activities.
- **What changes when you increase it:**
  - The BioGears recovery shape (exponential HR decay) is stretched over a longer period
  - More time is available for passive fatigue recovery after EVA
  - Fatigue index at end of recovery is lower
  - Less cumulative fatigue going into the next mission phase
- **Tip:** If you run a high-intensity EVA, you need a long recovery. Short recovery after intense EVA → dangerously high fatigue going into the next activity.

---

### Mission Parameters Section

#### Mission Duration (24h / 48h / 72h)
- **What it is:** Total length of the simulated mission.
- **What changes when you change it:**
  - **24h:** One EVA, one sleep period. Short mission. Fatigue usually recovers before mission end.
  - **48h:** Two sleep periods. Fatigue accumulation and recovery balance more clearly visible. Dehydration more likely to appear.
  - **72h:** Three days. Multiple EVAs, multiple sleep cycles. Cumulative fatigue can build across days if EVAs are intense. Best for seeing long-term trends.
- **Chart impact:** All time axes stretch accordingly. Hour labels on charts shift.

#### Risk Threshold (slider: 0.60 – 0.95)
- **What it is:** The fatigue level above which the system considers the astronaut at risk.
- **How it's used:**
  - Red dashed line on the Fatigue chart
  - Gauge zones
  - SAFE / MONITOR / ABORT status logic
  - Monte Carlo P(risk) calculation
- **What changes when you lower it (e.g., from 0.80 to 0.65):**
  - More minutes are flagged as "at risk" even for the same fatigue trajectory
  - P(breach) increases
  - Mission status is more likely to show MONITOR or ABORT
  - You are being more conservative — appropriate for high-stakes missions or inexperienced crew
- **What changes when you raise it (e.g., to 0.90):**
  - Only extreme fatigue is flagged
  - Appropriate only if the crew are highly trained and the mission allows for higher physiological cost

---

### Monte Carlo Section

#### N Simulations (10 / 50 / 100 / 500)
- **What it is:** How many randomised mission scenarios to run.
- **What changes:**
  - **10:** Fast (< 1 second). Results are rough — the percentile bands are noisy.
  - **50:** Good balance. Bands are fairly smooth.
  - **100 (default):** Standard. P(risk) curve is reliable.
  - **500:** Slowest (~5–10 seconds). Very smooth bands and highly accurate P(breach) estimate. Use this for final analysis.
- **Why it matters:** With 10 simulations, you might get P(breach) = 0.3 when the true value is 0.45. With 500, you get a much more accurate estimate of actual mission risk.

---

## 10. The top metric cards explained

These four cards update every time you run a simulation.

| Card | What it shows | What it means |
|------|--------------|---------------|
| **Peak Fatigue** | Maximum fatigue index reached during the mission | The single worst moment. If this is above threshold, danger occurred. |
| **Time at Risk** | Percentage of mission minutes where fatigue > threshold | Duration of the dangerous period, not just the peak. 5% of a 48h mission = 1.44 hours above threshold. |
| **P(Breach)** | Probability that fatigue exceeded threshold in Monte Carlo | Across all N randomised scenarios, how often did danger occur? |
| **Mission Status** | SAFE / MONITOR / ABORT | Derived from peak fatigue, trend, and time at risk (see section 11). |

---

## 11. Mission Status: SAFE / MONITOR / ABORT

The status is determined by this logic:

```
if peak_fatigue >= threshold AND (trend == "Rising" OR time_at_risk > 10%):
    status = "ABORT"           ← red banner, flashing alert

elif peak_fatigue >= threshold × 0.85:
    status = "MONITOR"         ← amber banner, caution alert

else:
    status = "SAFE"            ← no alert banner
```

**SAFE:** Peak fatigue is well below the threshold. The mission can continue as planned.

**MONITOR:** Peak fatigue is close to the threshold (within 15%) but hasn't crossed it, OR the trend is improving. Watch closely but no immediate action required.

**ABORT:** Fatigue crossed the threshold AND is either still rising or the astronaut has been in the danger zone for more than 10% of the mission. This represents a sustained dangerous physiological state. The crew should return to the spacecraft immediately.

---

## Running the dashboard

```bash
cd Miniproject
streamlit run app.py
```

Then open `http://localhost:8501` in your browser. Set your parameters in the left sidebar and click **EXECUTE SIMULATION**.
