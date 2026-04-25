"""
AstroGuard — Astronaut Health Digital Twin
==========================================
Run with:   streamlit run app.py
UI: HUD / Sci-Fi retro-futurism — UI UX Pro Max Style 11 + Style 7
"""
from __future__ import annotations

import time

import numpy as np
import streamlit as st

from analytics.risk import mc_summary, mission_status, run_monte_carlo, single_run_analytics
from simulation.biogears import get_biogears_segment
from simulation.mission_log import save_mission_log
from simulation.events import sample_events
from simulation.fatigue import compute_fatigue, normalise_biogears_fatigue
from simulation.health_vars import build_mission_timeline, build_hydration_timeline, build_food_timeline
from simulation.patient import load_patients, microgravity_factors
from visualization.charts import (
    make_risk_gauge,
    plot_biogears_raw,
    plot_mission_overview,
    plot_monte_carlo_envelope,
    plot_phase_space,
    plot_risk_heatmap,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AstroGuard · Mission Control",
    page_icon="🛰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── HUD Design System CSS ──────────────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700;800&display=swap" rel="stylesheet">

<style>
/* ══════════════════════════════════════════════════════════
   HUD DESIGN TOKENS  (UI UX Pro Max · Style 11 Retro-Futurism
   + Style 7 Dark OLED · space-medical palette)
   ══════════════════════════════════════════════════════════ */
:root {
  --hud-bg:       #030608;
  --hud-surface:  #060c18;
  --hud-card:     #080e1e;
  --hud-border:   #0c2040;
  --hud-grid:     rgba(0,212,255,0.04);
  --hud-scan:     rgba(0,212,255,0.018);

  --hud-cyan:     #00d4ff;
  --hud-green:    #00ff88;
  --hud-orange:   #ff6b00;
  --hud-red:      #ff1a3c;
  --hud-amber:    #ffaa00;
  --hud-purple:   #b44fff;
  --hud-blue:     #1a7fff;

  --hud-text:     #c0ddef;
  --hud-muted:    #1e4060;
  --hud-dim:      #0f2540;

  --mono: 'Share Tech Mono', 'Courier New', monospace;
  --ui:   'Exo 2', 'Segoe UI', sans-serif;

  --corner-size: 10px;
  --corner-w:    2px;
  --glow-cyan:   0 0 8px rgba(0,212,255,0.5);
  --glow-green:  0 0 8px rgba(0,255,136,0.5);
  --glow-red:    0 0 8px rgba(255,26,60,0.5);
  --glow-orange: 0 0 8px rgba(255,107,0,0.5);
}

/* ── A: Scanline overlay (C) ── zero-performance CSS only */
body::before {
  content: '';
  position: fixed;
  inset: 0;
  background: repeating-linear-gradient(
    0deg,
    var(--hud-scan) 0px,
    var(--hud-scan) 1px,
    transparent 1px,
    transparent 4px
  );
  pointer-events: none;
  z-index: 9990;
}

/* ── Moving scan line (C) ── */
body::after {
  content: '';
  position: fixed;
  left: 0; right: 0;
  height: 120px;
  top: 0;
  background: linear-gradient(
    to bottom,
    transparent 0%,
    rgba(0,212,255,0.04) 40%,
    rgba(0,212,255,0.08) 50%,
    rgba(0,212,255,0.04) 60%,
    transparent 100%
  );
  animation: hud-scan 12s linear infinite;
  pointer-events: none;
  z-index: 9989;
}
@keyframes hud-scan {
  0%   { transform: translateY(-120px); }
  100% { transform: translateY(110vh); }
}

/* ── CRT phosphor title glow (C) ── */
.hud-title-glow {
  text-shadow:
    0 0 6px rgba(0,212,255,0.9),
    0 0 20px rgba(0,212,255,0.4),
    0 0 40px rgba(0,212,255,0.15);
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
  background: var(--hud-bg) !important;
  color: var(--hud-text);
  font-family: var(--ui);
}
[data-testid="stMain"] { background: var(--hud-bg) !important; }
/* ── Style Streamlit header to match HUD ── */
header[data-testid="stHeader"] {
  background: rgba(4,8,20,0.95) !important;
  border-bottom: 1px solid rgba(0,212,255,0.2) !important;
  backdrop-filter: blur(8px);
}
[data-testid="stToolbar"] {
  background: transparent !important;
}
[data-testid="stToolbar"] button,
[data-testid="stToolbar"] a {
  color: var(--hud-cyan) !important;
  border-color: rgba(0,212,255,0.3) !important;
}
#MainMenu { visibility: hidden !important; }
[data-testid="block-container"] { padding-top: 0.25rem !important; }

/* ── Sidebar — mission control panel ── */
[data-testid="stSidebar"] {
  background: var(--hud-surface) !important;
  border-right: 1px solid var(--hud-border) !important;
}
[data-testid="stSidebar"]::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent, var(--hud-cyan), transparent);
}
/* Slider track */
[data-testid="stSidebar"] .stSlider > div > div > div {
  background: var(--hud-dim) !important;
}
[data-testid="stSidebar"] .stSlider > div > div > div > div {
  background: var(--hud-cyan) !important;
  box-shadow: var(--glow-cyan);
}

/* ── I: Tab bar — mission module selector ── */
[data-testid="stTabs"] [role="tablist"] {
  background: var(--hud-surface);
  border-bottom: 1px solid var(--hud-border);
  padding: 0.35rem 0.4rem;
  gap: 0.3rem;
}
[data-testid="stTabs"] button[role="tab"] {
  background: var(--hud-card);
  border: 1px solid var(--hud-dim) !important;
  border-radius: 2px !important;
  color: var(--hud-muted) !important;
  font-family: var(--mono) !important;
  font-size: 0.68rem !important;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  padding: 0.45rem 1.1rem !important;
  transition: all 0.18s ease;
  clip-path: polygon(0 0, calc(100% - 6px) 0, 100% 6px, 100% 100%, 0 100%);
}
[data-testid="stTabs"] button[role="tab"]:hover {
  background: rgba(0,212,255,0.06) !important;
  border-color: rgba(0,212,255,0.25) !important;
  color: rgba(0,212,255,0.7) !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
  background: rgba(0,212,255,0.1) !important;
  border-color: var(--hud-cyan) !important;
  color: var(--hud-cyan) !important;
  box-shadow: 0 0 14px rgba(0,212,255,0.25), inset 0 -2px 0 var(--hud-cyan);
}
/* Hide default Streamlit tab underline */
[data-testid="stTabs"] [role="tablist"] ~ div { border-top: none !important; }

/* ── A: Corner-bracket panels ── */
.hud-panel {
  position: relative;
  background: var(--hud-card);
  border: 1px solid var(--hud-border);
  padding: 1rem 1.1rem;
  margin-bottom: 0.6rem;
}
.hud-panel .c { position: absolute; width: var(--corner-size); height: var(--corner-size); }
.hud-panel .c.tl { top: -1px;  left: -1px;  border-top:    var(--corner-w) solid var(--hud-cyan); border-left:  var(--corner-w) solid var(--hud-cyan); }
.hud-panel .c.tr { top: -1px;  right: -1px; border-top:    var(--corner-w) solid var(--hud-cyan); border-right: var(--corner-w) solid var(--hud-cyan); }
.hud-panel .c.bl { bottom: -1px; left: -1px;  border-bottom: var(--corner-w) solid var(--hud-cyan); border-left:  var(--corner-w) solid var(--hud-cyan); }
.hud-panel .c.br { bottom: -1px; right: -1px; border-bottom: var(--corner-w) solid var(--hud-cyan); border-right: var(--corner-w) solid var(--hud-cyan); }

/* Corner variant colours */
.hud-panel.green .c { border-color: var(--hud-green); }
.hud-panel.orange .c { border-color: var(--hud-orange); }
.hud-panel.red .c  { border-color: var(--hud-red); }
.hud-panel.amber .c { border-color: var(--hud-amber); }

/* ── B: Monospace data values ── */
.hud-value {
  font-family: var(--mono);
  font-size: 1.7rem;
  color: var(--hud-text);
  line-height: 1.1;
  letter-spacing: 0.02em;
}
.hud-value.cyan   { color: var(--hud-cyan);   text-shadow: var(--glow-cyan); }
.hud-value.green  { color: var(--hud-green);  text-shadow: var(--glow-green); }
.hud-value.orange { color: var(--hud-orange); text-shadow: var(--glow-orange); }
.hud-value.red    { color: var(--hud-red);    text-shadow: var(--glow-red); }
.hud-value.amber  { color: var(--hud-amber); }

.hud-label {
  font-family: var(--mono);
  font-size: 0.6rem;
  color: var(--hud-muted);
  text-transform: uppercase;
  letter-spacing: 0.16em;
  margin-bottom: 0.25rem;
}
.hud-sub {
  font-family: var(--mono);
  font-size: 0.62rem;
  color: var(--hud-muted);
  margin-top: 0.3rem;
  letter-spacing: 0.06em;
}
.hud-sub.up   { color: var(--hud-red); }
.hud-sub.down { color: var(--hud-green); }

/* ── D: Radar-ping pulsing status indicator ── */
@keyframes radar-ping {
  0%   { box-shadow: 0 0 0 0px rgba(0,255,136,0.7); }
  60%  { box-shadow: 0 0 0 10px rgba(0,255,136,0); }
  100% { box-shadow: 0 0 0 0px rgba(0,255,136,0); }
}
@keyframes radar-ping-red {
  0%   { box-shadow: 0 0 0 0px rgba(255,26,60,0.8); }
  60%  { box-shadow: 0 0 0 14px rgba(255,26,60,0); }
  100% { box-shadow: 0 0 0 0px rgba(255,26,60,0); }
}
@keyframes radar-ping-amber {
  0%   { box-shadow: 0 0 0 0px rgba(255,170,0,0.7); }
  60%  { box-shadow: 0 0 0 12px rgba(255,170,0,0); }
  100% { box-shadow: 0 0 0 0px rgba(255,170,0,0); }
}
.ping-dot {
  display: inline-block;
  width: 8px; height: 8px;
  border-radius: 50%;
  margin-right: 6px;
  vertical-align: middle;
}
.ping-dot.green  { background: var(--hud-green);  animation: radar-ping       1.8s infinite; }
.ping-dot.red    { background: var(--hud-red);    animation: radar-ping-red   1.2s infinite; }
.ping-dot.amber  { background: var(--hud-amber);  animation: radar-ping-amber 1.5s infinite; }

/* ── G: Alert banner ── */
@keyframes alert-flash {
  0%,100% { opacity: 1; }
  50%      { opacity: 0.6; }
}
.alert-banner {
  width: 100%;
  padding: 0.6rem 1.2rem;
  font-family: var(--mono);
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  text-align: center;
  margin-bottom: 0.8rem;
  clip-path: polygon(6px 0%, calc(100% - 6px) 0%, 100% 6px, 100% 100%, calc(100% - 6px) 100%, 6px 100%, 0% calc(100% - 6px), 0% 6px);
}
.alert-abort {
  background: rgba(255,26,60,0.12);
  border: 1px solid var(--hud-red);
  color: var(--hud-red);
  box-shadow: 0 0 20px rgba(255,26,60,0.2), inset 0 0 30px rgba(255,26,60,0.05);
  animation: alert-flash 0.8s ease-in-out infinite;
}
.alert-monitor {
  background: rgba(255,170,0,0.10);
  border: 1px solid var(--hud-amber);
  color: var(--hud-amber);
  box-shadow: 0 0 16px rgba(255,170,0,0.15);
  animation: alert-flash 1.4s ease-in-out infinite;
}

/* ── Mission banner ── */
.mission-banner {
  background: linear-gradient(135deg, #060c18 0%, #080e1e 60%, #040a14 100%);
  border: 1px solid var(--hud-border);
  border-top: 2px solid var(--hud-cyan);
  padding: 1rem 1.4rem;
  margin-bottom: 0.8rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  clip-path: polygon(0 0, calc(100% - 12px) 0, 100% 12px, 100% 100%, 12px 100%, 0 calc(100% - 12px));
  position: relative;
}
.mission-banner::before {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(90deg,
    rgba(0,212,255,0.04) 0%,
    transparent 30%,
    transparent 70%,
    rgba(0,212,255,0.02) 100%
  );
  pointer-events: none;
}
.mission-title {
  font-family: var(--mono);
  font-size: 1.2rem;
  color: var(--hud-cyan);
  text-shadow: var(--glow-cyan);
  letter-spacing: 0.08em;
  margin: 0;
  text-transform: uppercase;
}
.mission-subtitle {
  font-family: var(--ui);
  font-size: 0.72rem;
  color: var(--hud-muted);
  margin: 0.25rem 0 0;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
.mission-ref {
  font-family: var(--mono);
  font-size: 0.6rem;
  color: var(--hud-dim);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  margin-top: 0.5rem;
}

/* ── Live indicator ── */
.live-badge {
  font-family: var(--mono);
  font-size: 0.62rem;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  padding: 0.3em 0.9em;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  clip-path: polygon(4px 0%, calc(100% - 4px) 0%, 100% 4px, 100% 100%, calc(100% - 4px) 100%, 4px 100%, 0% calc(100% - 4px), 0% 4px);
}
.live-badge.nominal {
  background: rgba(0,255,136,0.08);
  border: 1px solid rgba(0,255,136,0.35);
  color: var(--hud-green);
}
.live-badge.fallback {
  background: rgba(255,170,0,0.08);
  border: 1px solid rgba(255,170,0,0.35);
  color: var(--hud-amber);
}

/* ── Section headers ── */
.hud-section {
  font-family: var(--mono);
  font-size: 0.65rem;
  color: var(--hud-cyan);
  text-transform: uppercase;
  letter-spacing: 0.2em;
  margin: 1rem 0 0.6rem;
  padding-bottom: 0.3rem;
  border-bottom: 1px solid var(--hud-border);
  display: flex;
  align-items: center;
  gap: 8px;
}
.hud-section::before {
  content: '//';
  color: var(--hud-dim);
}

/* ── Metric grid ── */
.metric-grid { display: flex; gap: 0.6rem; margin-bottom: 0.8rem; flex-wrap: nowrap; }
.metric-card {
  flex: 1;
  background: var(--hud-card);
  border: 1px solid var(--hud-border);
  padding: 0.8rem 0.9rem 0.9rem;
  position: relative;
  min-width: 0;
  clip-path: polygon(0 0, calc(100% - 8px) 0, 100% 8px, 100% 100%, 8px 100%, 0 calc(100% - 8px));
}
/* corner brackets on metric cards */
.metric-card .c { position: absolute; width: 8px; height: 8px; }
.metric-card .c.tl { top: -1px;    left: -1px;   border-top:    1px solid; border-left:  1px solid; }
.metric-card .c.tr { top: -1px;    right: -1px;  border-top:    1px solid; border-right: 1px solid; }
.metric-card .c.bl { bottom: -1px; left: -1px;   border-bottom: 1px solid; border-left:  1px solid; }
.metric-card .c.br { bottom: -1px; right: -1px;  border-bottom: 1px solid; border-right: 1px solid; }
.metric-card.cyan .c   { color: var(--hud-cyan); }
.metric-card.green .c  { color: var(--hud-green); }
.metric-card.orange .c { color: var(--hud-orange); }
.metric-card.red .c    { color: var(--hud-red); }
.metric-card.amber .c  { color: var(--hud-amber); }
.metric-card.purple .c { color: var(--hud-purple); }
/* top accent bar */
.metric-card::after {
  content: '';
  position: absolute;
  top: 0; left: 12%; right: 12%;
  height: 1px;
}
.metric-card.cyan::after   { background: linear-gradient(90deg, transparent, var(--hud-cyan), transparent); }
.metric-card.green::after  { background: linear-gradient(90deg, transparent, var(--hud-green), transparent); }
.metric-card.orange::after { background: linear-gradient(90deg, transparent, var(--hud-orange), transparent); }
.metric-card.red::after    { background: linear-gradient(90deg, transparent, var(--hud-red), transparent); }
.metric-card.amber::after  { background: linear-gradient(90deg, transparent, var(--hud-amber), transparent); }
.metric-card.purple::after { background: linear-gradient(90deg, transparent, var(--hud-purple), transparent); }

/* ── Sidebar section titles ── */
.sb-group {
  font-family: var(--mono);
  font-size: 0.58rem;
  color: var(--hud-cyan);
  text-transform: uppercase;
  letter-spacing: 0.18em;
  padding: 0.3rem 0;
  margin-bottom: 0.2rem;
  border-bottom: 1px solid var(--hud-dim);
  display: flex; align-items: center; gap: 6px;
}
.sb-group::before { content: '▸'; color: var(--hud-muted); font-size: 0.5rem; }

/* ── Streamlit overrides ── */
[data-testid="stMetricValue"] { font-family: var(--mono) !important; font-size: 1.4rem !important; font-weight: 400 !important; color: var(--hud-text) !important; }
[data-testid="stMetricLabel"] { font-family: var(--mono) !important; font-size: 0.6rem !important; color: var(--hud-muted) !important; text-transform: uppercase !important; letter-spacing: 0.12em !important; }

[data-testid="stExpander"] {
  background: var(--hud-card) !important;
  border: 1px solid var(--hud-border) !important;
  border-radius: 0 !important;
}
[data-testid="stExpander"] summary {
  font-family: var(--mono);
  font-size: 0.68rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--hud-muted) !important;
}
[data-testid="stDataFrame"] {
  border: 1px solid var(--hud-border) !important;
  font-family: var(--mono) !important;
  font-size: 0.72rem !important;
}
[data-testid="stButton"] button {
  font-family: var(--mono) !important;
  font-size: 0.72rem !important;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  border-radius: 2px !important;
  clip-path: polygon(0 0, calc(100% - 6px) 0, 100% 6px, 100% 100%, 6px 100%, 0 calc(100% - 6px));
}
[data-testid="stButton"] button[kind="primary"] {
  background: rgba(0,212,255,0.12) !important;
  border: 1px solid var(--hud-cyan) !important;
  color: var(--hud-cyan) !important;
  box-shadow: 0 0 16px rgba(0,212,255,0.2);
}
[data-testid="stButton"] button[kind="primary"]:hover {
  background: rgba(0,212,255,0.2) !important;
  box-shadow: 0 0 24px rgba(0,212,255,0.35);
}
.stSuccess { background: rgba(0,255,136,0.06) !important; border: 1px solid rgba(0,255,136,0.25) !important; color: var(--hud-green) !important; font-family: var(--mono) !important; font-size: 0.72rem !important; border-radius: 0 !important; }
.stWarning { background: rgba(255,170,0,0.06) !important; border: 1px solid rgba(255,170,0,0.25) !important; color: var(--hud-amber) !important; font-family: var(--mono) !important; font-size: 0.72rem !important; border-radius: 0 !important; }
hr { border-color: var(--hud-border) !important; margin: 0.7rem 0 !important; }
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--hud-bg); }
::-webkit-scrollbar-thumb { background: var(--hud-dim); }
p, li { font-family: var(--ui); }
caption, .stCaption { font-family: var(--mono) !important; font-size: 0.62rem !important; color: var(--hud-muted) !important; letter-spacing: 0.06em; }
</style>
""", unsafe_allow_html=True)


# ── Cached helpers ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cached_biogears(eva_intensity: float, eva_duration_min: float, recovery_min: float, mode: str = "live"):
    return get_biogears_segment(eva_intensity, eva_duration_min, recovery_min, mode=mode)

@st.cache_data(show_spinner=False)
def _cached_patients():
    return load_patients()

ALL_PATIENTS = _cached_patients()
PATIENT_NAMES = list(ALL_PATIENTS.keys())

_TIER_COLOR = {"Elite": "#00ff88", "Good": "#00d4ff", "Normal": "#ffaa00", "Deconditioned": "#ff1a3c"}


# ── Sidebar — simplified to EVA + mission params only ─────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:0.8rem 0 0.6rem">
      <div style="font-family:var(--mono);font-size:0.58rem;color:var(--hud-muted);
                  letter-spacing:0.2em;text-transform:uppercase;margin-bottom:0.3rem">
        System Online
      </div>
      <div style="font-family:var(--mono);font-size:1.0rem;color:var(--hud-cyan);
                  text-shadow:0 0 8px rgba(0,212,255,0.6);letter-spacing:0.1em;text-transform:uppercase">
        <span style="color:var(--hud-muted)">&#9658;</span> ASTROGUARD
      </div>
      <div style="font-family:var(--mono);font-size:0.58rem;color:var(--hud-dim);
                  letter-spacing:0.12em;margin-top:0.1rem">
        MISSION CONTROL · v2.0
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="sb-group">BioGears Source</div>', unsafe_allow_html=True)
    bg_mode_label = st.radio(
        "Mode", options=["CSV (instant)", "Live BioGears"],
        index=0, horizontal=True,
        help="CSV: pre-computed instant load. Live: runs bg-scenario.exe (~4 min).",
    )
    bg_mode = "csv" if "CSV" in bg_mode_label else "live"
    st.divider()

    st.markdown('<div class="sb-group">EVA Parameters</div>', unsafe_allow_html=True)
    eva_intensity = st.slider("EVA Intensity", 0.10, 0.90, 0.50, 0.05,
        help="Workload fraction sent to BioGears (0 = rest, 1 = max exertion)")
    eva_duration_min = st.slider("Mission Duration (min)", 20, 240, 90, 10,
        help="Duration of the EVA exercise phase")
    recovery_min = st.slider("Recovery Time (min)", 10, 90, 30, 5,
        help="Post-EVA recovery phase duration")
    st.divider()

    st.markdown('<div class="sb-group">Mission Settings</div>', unsafe_allow_html=True)
    mission_hours = st.select_slider("Mission Length", options=[24, 48, 72], value=48)
    mission_day   = st.slider("Days in Space", 0, 180, 0, 1,
        help="Days in microgravity before this EVA — drives cardiovascular deconditioning")
    st.divider()

    st.markdown('<div class="sb-group">Monte Carlo</div>', unsafe_allow_html=True)
    n_sims = st.select_slider("Simulations", options=[10, 50, 100, 200, 500], value=100)
    st.divider()

    run_btn = st.button("EXECUTE SIMULATION", type="primary", use_container_width=True)

    # Fixed internals shown in info box
    n_evas    = 1
    threshold = 0.80
    mg        = microgravity_factors(mission_day)
    last_log  = st.session_state.get("last_log_path", "")
    last_log_name = last_log.split("\\")[-1] if last_log else "—"

    st.markdown(f"""
    <div style="margin-top:1rem;padding:0.7rem;background:var(--hud-surface);
                border:1px solid var(--hud-dim);font-family:var(--mono);
                font-size:0.55rem;color:var(--hud-muted);letter-spacing:0.1em;
                text-transform:uppercase;line-height:2">
      BioGears 8.2.0 &nbsp;|&nbsp; {"CSV" if bg_mode=="csv" else "LIVE"}<br>
      VO2max factor &nbsp;{mg.vo2max_factor:.3f}<br>
      HR offset &nbsp;+{mg.hr_offset:.1f} bpm<br>
      Muscle factor &nbsp;{mg.muscle_factor:.3f}<br>
      <span style="color:var(--hud-green)">LOG {last_log_name}</span>
    </div>
    """, unsafe_allow_html=True)


# ── Hero title ─────────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&display=swap" rel="stylesheet">
<div style="
  text-align:center;padding:2.2rem 0 1.4rem 0;letter-spacing:0.18em;
  font-family:'Orbitron',sans-serif;font-weight:900;
  font-size:clamp(2rem,5vw,3.6rem);color:#00d4ff;
  text-shadow:0 0 10px rgba(0,212,255,1),0 0 30px rgba(0,212,255,0.7),
              0 0 60px rgba(0,212,255,0.35),0 0 100px rgba(0,212,255,0.15);
  text-transform:uppercase;user-select:none;">
  ASTROGUARD
  <span style="display:block;font-size:clamp(0.65rem,1.6vw,1.0rem);font-weight:700;
               color:rgba(0,212,255,0.55);letter-spacing:0.55em;margin-top:0.35rem;
               text-shadow:0 0 8px rgba(0,212,255,0.4);">
    ASTRONAUT HEALTH DIGITAL TWIN
  </span>
</div>
""", unsafe_allow_html=True)


# ── Astronaut selection ────────────────────────────────────────────────────────
st.markdown('<div class="hud-section">Select Astronaut</div>', unsafe_allow_html=True)

if "selected_patient_name" not in st.session_state:
    st.session_state["selected_patient_name"] = "StandardMale"

# Patient grid — 4 per row
cols_per_row = 4
patient_name_list = PATIENT_NAMES
rows = [patient_name_list[i:i+cols_per_row] for i in range(0, len(patient_name_list), cols_per_row)]

for row in rows:
    grid_cols = st.columns(cols_per_row)
    for col, pname in zip(grid_cols, row):
        p = ALL_PATIENTS[pname]
        tier_color = _TIER_COLOR.get(p.fitness_tier, "#00d4ff")
        is_selected = (pname == st.session_state["selected_patient_name"])
        border_color = tier_color if is_selected else "#0c2040"
        bg_color     = "rgba(0,212,255,0.06)" if is_selected else "var(--hud-card)"
        with col:
            st.markdown(f"""
            <div style="background:{bg_color};border:1px solid {border_color};
                        padding:0.65rem 0.75rem;cursor:pointer;margin-bottom:0.3rem;
                        clip-path:polygon(0 0,calc(100% - 8px) 0,100% 8px,100% 100%,0 100%)">
              <div style="font-family:var(--mono);font-size:0.58rem;color:{tier_color};
                          text-transform:uppercase;letter-spacing:0.12em">
                {p.fitness_tier} {'&#9679;' if is_selected else '&#9675;'}
              </div>
              <div style="font-family:var(--mono);font-size:0.72rem;color:var(--hud-text);
                          margin:0.2rem 0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">
                {p.display_name}
              </div>
              <div style="font-family:var(--mono);font-size:0.55rem;color:var(--hud-muted)">
                {p.sex[0]} · {p.age_yr:.0f}yr · {p.bmi:.1f} BMI
              </div>
              <div style="font-family:var(--mono);font-size:0.55rem;color:var(--hud-muted)">
                HR {p.hr_baseline:.0f} · VO2 {p.vo2max_ml_kg_min:.0f}
              </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"SELECT", key=f"sel_{pname}", use_container_width=True):
                st.session_state["selected_patient_name"] = pname
                st.rerun()

# ── Selected astronaut profile card + nutrition editors ───────────────────────
sel_name = st.session_state["selected_patient_name"]
patient  = ALL_PATIENTS[sel_name]
mg_now   = microgravity_factors(mission_day)
adj      = mg_now.adjust_patient(patient)

tier_color = _TIER_COLOR.get(patient.fitness_tier, "#00d4ff")

st.markdown('<div class="hud-section">Astronaut Profile &amp; Mission Nutrition</div>',
            unsafe_allow_html=True)

card_col, config_col = st.columns([1, 1.6])

with card_col:
    st.markdown(f"""
    <div class="hud-panel" style="border-color:{tier_color}40">
      <span class="c tl" style="border-color:{tier_color}"></span>
      <span class="c tr" style="border-color:{tier_color}"></span>
      <span class="c bl" style="border-color:{tier_color}"></span>
      <span class="c br" style="border-color:{tier_color}"></span>

      <div style="font-family:var(--mono);font-size:0.58rem;color:{tier_color};
                  text-transform:uppercase;letter-spacing:0.18em;margin-bottom:0.5rem">
        {patient.fitness_tier} ASTRONAUT
      </div>
      <div style="font-family:var(--mono);font-size:1.1rem;color:var(--hud-text);
                  letter-spacing:0.06em;margin-bottom:0.8rem">
        {patient.display_name}
      </div>

      <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.4rem;
                  font-family:var(--mono);font-size:0.6rem">
        <div>
          <div style="color:var(--hud-muted);text-transform:uppercase;letter-spacing:0.1em">Sex / Age</div>
          <div style="color:var(--hud-text)">{patient.sex} / {patient.age_yr:.0f} yr</div>
        </div>
        <div>
          <div style="color:var(--hud-muted);text-transform:uppercase;letter-spacing:0.1em">Weight / Height</div>
          <div style="color:var(--hud-text)">{patient.weight_kg:.1f} kg / {patient.height_cm:.0f} cm</div>
        </div>
        <div>
          <div style="color:var(--hud-muted);text-transform:uppercase;letter-spacing:0.1em">BMI</div>
          <div style="color:var(--hud-text)">{patient.bmi:.1f}</div>
        </div>
        <div>
          <div style="color:var(--hud-muted);text-transform:uppercase;letter-spacing:0.1em">Body Fat</div>
          <div style="color:var(--hud-text)">{patient.body_fat_fraction*100:.0f}%</div>
        </div>
        <div>
          <div style="color:var(--hud-muted);text-transform:uppercase;letter-spacing:0.1em">Lean Mass</div>
          <div style="color:var(--hud-text)">{patient.lean_mass_kg:.1f} kg</div>
        </div>
        <div>
          <div style="color:var(--hud-muted);text-transform:uppercase;letter-spacing:0.1em">Glycogen Cap</div>
          <div style="color:var(--hud-text)">{patient.glycogen_capacity_g:.0f} g</div>
        </div>
        <div>
          <div style="color:var(--hud-muted);text-transform:uppercase;letter-spacing:0.1em">Resting HR</div>
          <div style="color:var(--hud-cyan)">{patient.hr_baseline:.0f} bpm
            <span style="color:var(--hud-muted)">+{mg_now.hr_offset:.1f} in space</span>
          </div>
        </div>
        <div>
          <div style="color:var(--hud-muted);text-transform:uppercase;letter-spacing:0.1em">BP Baseline</div>
          <div style="color:var(--hud-text)">{patient.systolic_bp:.0f}/{patient.diastolic_bp:.0f} mmHg</div>
        </div>
        <div>
          <div style="color:var(--hud-muted);text-transform:uppercase;letter-spacing:0.1em">VO2max (ground)</div>
          <div style="color:var(--hud-text)">{patient.vo2max_ml_kg_min:.1f} mL/kg/min</div>
        </div>
        <div>
          <div style="color:var(--hud-muted);text-transform:uppercase;letter-spacing:0.1em">VO2max (day {mission_day})</div>
          <div style="color:{'var(--hud-amber)' if mg_now.vo2max_factor < 0.90 else 'var(--hud-green)'}">{adj['vo2max_ml_kg_min']:.1f} mL/kg/min</div>
        </div>
        <div>
          <div style="color:var(--hud-muted);text-transform:uppercase;letter-spacing:0.1em">Glycogen (adj)</div>
          <div style="color:var(--hud-text)">{adj['glycogen_capacity_g']:.0f} g</div>
        </div>
        <div>
          <div style="color:var(--hud-muted);text-transform:uppercase;letter-spacing:0.1em">Est. daily kcal</div>
          <div style="color:var(--hud-text)">{patient.daily_kcal_estimate:.0f} kcal</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with config_col:
    st.markdown('<div style="font-family:var(--mono);font-size:0.6rem;color:var(--hud-cyan);'
                'text-transform:uppercase;letter-spacing:0.18em;margin-bottom:0.6rem">'
                '// Nutrition &amp; Recovery Configuration</div>', unsafe_allow_html=True)

    nc1, nc2 = st.columns(2)
    with nc1:
        carb_g_per_meal = st.slider(
            "Carbohydrates / meal (g)", 30, 300, 130, 5,
            help="BioGears Carbohydrate field. NASA ISS target: ~370g/day total.")
        protein_g_per_meal = st.slider(
            "Protein / meal (g)", 5, 80, 20, 5,
            help="BioGears Protein field. NASA EVA target: 1.6g/kg/day.")
        meals_per_day = st.select_slider(
            "Meals per day", options=[1, 2, 3, 4], value=3)
    with nc2:
        daily_water_L = st.slider(
            "Daily water (L)", 1.0, 4.0, 2.0, 0.1,
            help="BioGears Water field. NASA minimum: 2.0 L/day in space.")
        sodium_mg_per_day = st.slider(
            "Sodium / day (mg)", 500, 3000, 1500, 100,
            help="BioGears Sodium field. NASA limit: 2300 mg/day. "
                 "High sodium raises effective water need in microgravity.")
        sleep_hours = st.slider(
            "Sleep (hrs/night)", 4.0, 9.0, 8.0, 0.5,
            help="BioGears SleepAmount. ISS astronauts avg 6.5h; NASA target 8h.")

    # Live nutrition summary
    daily_carb   = carb_g_per_meal * meals_per_day
    daily_prot   = protein_g_per_meal * meals_per_day
    daily_kcal   = daily_carb * 4 + daily_prot * 4
    prot_target  = 1.6 * patient.weight_kg
    prot_pct     = min(100, daily_prot / prot_target * 100)
    prot_color   = "#00ff88" if prot_pct >= 90 else "#ffaa00" if prot_pct >= 60 else "#ff1a3c"

    st.markdown(f"""
    <div style="margin-top:0.8rem;padding:0.65rem 0.8rem;background:var(--hud-surface);
                border:1px solid var(--hud-dim);font-family:var(--mono);font-size:0.58rem;
                color:var(--hud-muted);letter-spacing:0.1em;text-transform:uppercase;
                display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.5rem">
      <div>
        <div>Daily carbs</div>
        <div style="color:var(--hud-cyan)">{daily_carb:.0f} g</div>
      </div>
      <div>
        <div>Protein vs target</div>
        <div style="color:{prot_color}">{daily_prot:.0f} / {prot_target:.0f} g ({prot_pct:.0f}%)</div>
      </div>
      <div>
        <div>Est. intake</div>
        <div style="color:var(--hud-text)">{daily_kcal:.0f} kcal/day</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()


# ── Run simulation (only on button click — no auto-run) ───────────────────────
if run_btn:

    with st.spinner("Loading BioGears EVA segment…"):
        t0 = time.time()
        biogears_df, used_real, bg_msg = cached_biogears(
            eva_intensity, eva_duration_min, recovery_min, bg_mode
        )
        bg_elapsed = time.time() - t0

    with st.spinner("Building mission timeline…"):
        events = sample_events(mission_hours=mission_hours, eva_intensity=eva_intensity,
                               n_evas=n_evas, seed=42)
        mission_df = build_mission_timeline(
            biogears_df=biogears_df, events=events, mission_hours=mission_hours,
            eva_duration_min=eva_duration_min, recovery_min=recovery_min,
            mission_day=mission_day, seed=0)
        # Legacy hydration/food arrays still passed to build_mission_timeline internals
        hydration_arr = build_hydration_timeline(
            events=events, mission_hours=mission_hours, eva_intensity=eva_intensity,
            water_intake_L_per_rest_hour=daily_water_L / 16.0, seed=0)
        food_arr = build_food_timeline(
            events=events, mission_hours=mission_hours, eva_intensity=eva_intensity,
            meals_per_day=meals_per_day, seed=0)

    with st.spinner("Computing fatigue model…"):
        mg_run = microgravity_factors(mission_day)
        fatigue, glycogen_fraction, risk_periods = compute_fatigue(
            hr_arr=mission_df["HeartRate"].values,
            events=events,
            threshold=threshold,
            patient=patient,
            mg_factors=mg_run,
            carb_g_per_meal=carb_g_per_meal,
            protein_g_per_meal=protein_g_per_meal,
            meals_per_day=meals_per_day,
            daily_water_L=daily_water_L,
            sodium_mg_per_day=sodium_mg_per_day,
            sleep_hours=sleep_hours,
            mission_day=mission_day,
        )
    bg_fatigue = normalise_biogears_fatigue(biogears_df, len(fatigue))
    analytics  = single_run_analytics(fatigue, threshold=threshold, mission_hours=mission_hours)
    status_label, status_color = mission_status(analytics, threshold)

    with st.spinner(f"Running {n_sims} Monte Carlo trajectories…"):
        mc = run_monte_carlo(
            biogears_df=biogears_df, mission_hours=mission_hours,
            eva_intensity=eva_intensity, eva_duration_min=eva_duration_min,
            recovery_min=recovery_min, n_sims=n_sims, threshold=threshold,
            n_evas=n_evas, mission_day=mission_day,
            water_intake=daily_water_L / 16.0, meals_per_day=meals_per_day,
            patient=patient,
            carb_g_per_meal=carb_g_per_meal, protein_g_per_meal=protein_g_per_meal,
            daily_water_L=daily_water_L, sodium_mg_per_day=sodium_mg_per_day,
            sleep_hours=sleep_hours,
        )
    mc_sum = mc_summary(mc)

    log_path = save_mission_log(
        eva_intensity=eva_intensity, eva_duration_min=eva_duration_min,
        recovery_min=recovery_min, mission_hours=mission_hours, n_evas=n_evas,
        threshold=threshold, bg_mode=bg_mode, n_sims=n_sims, mission_day=mission_day,
        water_intake=daily_water_L / 16.0, meals_per_day=meals_per_day,
        events=events, mission_df=mission_df, fatigue=fatigue,
        hydration_arr=hydration_arr, food_arr=food_arr,
        analytics=analytics, mc_sum=mc_sum,
        status_label=status_label, status_color=status_color, bg_msg=bg_msg,
    )

    st.session_state.update({
        "simulation_done":    True,
        "biogears_df":        biogears_df,
        "used_real":          used_real,
        "bg_msg":             bg_msg,
        "bg_elapsed":         bg_elapsed,
        "bg_mode":            bg_mode,
        "events":             events,
        "mission_df":         mission_df,
        "hydration_arr":      hydration_arr,
        "food_arr":           food_arr,
        "fatigue":            fatigue,
        "glycogen_fraction":  glycogen_fraction,
        "bg_fatigue":         bg_fatigue,
        "risk_periods":       risk_periods,
        "analytics":          analytics,
        "status_label":       status_label,
        "status_color":       status_color,
        "mc":                 mc,
        "mc_sum":             mc_sum,
        "mission_day":        mission_day,
        "last_log_path":      str(log_path),
    })
    st.rerun()


# ── Results — shown only after a simulation has run ───────────────────────────
if "simulation_done" not in st.session_state:
    st.markdown("""
    <div style="margin:3rem auto;max-width:480px;padding:2rem;text-align:center;
                background:var(--hud-card);border:1px solid var(--hud-border);
                font-family:var(--mono);
                clip-path:polygon(0 0,calc(100% - 16px) 0,100% 16px,100% 100%,16px 100%,0 calc(100% - 16px))">
      <div style="font-size:0.6rem;color:var(--hud-muted);letter-spacing:0.2em;
                  text-transform:uppercase;margin-bottom:1rem">
        Awaiting Mission Input
      </div>
      <div style="font-size:1rem;color:var(--hud-cyan);text-shadow:0 0 8px rgba(0,212,255,0.5);
                  letter-spacing:0.08em;margin-bottom:0.8rem">
        SELECT ASTRONAUT + CONFIGURE<br>THEN CLICK EXECUTE
      </div>
      <div style="font-size:0.58rem;color:var(--hud-dim);line-height:1.8">
        Pick a crew member above &nbsp;&bull;&nbsp;
        Set nutrition &amp; recovery &nbsp;&bull;&nbsp;
        Set EVA parameters in the sidebar
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Load results from session state ───────────────────────────────────────────
ss               = st.session_state
biogears_df      = ss["biogears_df"]
used_real        = ss["used_real"]
bg_msg           = ss["bg_msg"]
bg_mode_used     = ss.get("bg_mode", "csv")
events           = ss["events"]
mission_df       = ss["mission_df"]
fatigue          = ss["fatigue"]
bg_fatigue       = ss["bg_fatigue"]
risk_periods     = ss["risk_periods"]
analytics        = ss["analytics"]
status_label     = ss["status_label"]
status_color     = ss["status_color"]
mc               = ss["mc"]
mc_sum           = ss["mc_sum"]
mission_day_used = ss.get("mission_day", 0)

mission_min          = len(fatigue)
mission_hours_actual = mission_min // 60
peak_f               = analytics["peak_fatigue"]
peak_h               = analytics["peak_minute"] // 60
at_risk              = analytics["time_at_risk_pct"]
p_breach             = mc_sum["p_any_breach"] * 100


# ── Alert banner ───────────────────────────────────────────────────────────────
if status_label == "ABORT":
    st.markdown(
        '<div class="alert-banner alert-abort">'
        '&#9888; MASTER CAUTION &nbsp;&middot;&nbsp; FATIGUE THRESHOLD BREACHED '
        '&nbsp;&middot;&nbsp; MISSION ABORT RECOMMENDED &nbsp;&middot;&nbsp; &#9888;'
        '</div>', unsafe_allow_html=True)
elif status_label == "MONITOR":
    st.markdown(
        '<div class="alert-banner alert-monitor">'
        '&#9889; CAUTION &nbsp;&middot;&nbsp; ELEVATED FATIGUE DETECTED '
        '&nbsp;&middot;&nbsp; MONITOR ASTRONAUT STATUS &nbsp;&middot;&nbsp; &#9889;'
        '</div>', unsafe_allow_html=True)


# ── Mission banner ─────────────────────────────────────────────────────────────
if used_real and bg_mode_used == "live":
    ping_class, badge_class, bg_src_text = "green", "nominal",  "BIOGEARS LIVE"
elif used_real and bg_mode_used == "csv":
    ping_class, badge_class, bg_src_text = "green", "nominal",  "PRE-COMPUTED XML"
else:
    ping_class, badge_class, bg_src_text = "amber", "fallback", "SYNTH FALLBACK"

st.markdown(f"""
<div class="mission-banner">
  <div>
    <p class="mission-title hud-title-glow">&#9658; ASTROGUARD &middot; HEALTH DIGITAL TWIN</p>
    <p class="mission-subtitle">
      {patient.display_name} &nbsp;&middot;&nbsp;
      {mission_hours_actual}h Mission &nbsp;&middot;&nbsp; EVA {eva_intensity:.0%}
      &nbsp;&middot;&nbsp; Day {mission_day_used} in Space
    </p>
    <p class="mission-ref">
      RES-HSFC-2025-001 &nbsp;|&nbsp; MC-N {n_sims}
      &nbsp;|&nbsp; THRESHOLD {threshold:.0%}
    </p>
  </div>
  <div style="text-align:right">
    <div class="live-badge {badge_class}">
      <span class="ping-dot {ping_class}"></span>
      {bg_src_text}
    </div>
    <div style="font-family:var(--mono);font-size:0.55rem;color:var(--hud-dim);
                margin-top:0.5rem;letter-spacing:0.1em;text-transform:uppercase">
      {bg_msg[:60]}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

if used_real:
    st.success(f"BioGears OK — {bg_msg}", icon="✅")
else:
    st.warning(f"Fallback active — {bg_msg}", icon="⚠️")


# ── HUD Metric Cards ───────────────────────────────────────────────────────────
status_hud  = {"SAFE": "green", "MONITOR": "amber", "ABORT": "red"}.get(status_label, "cyan")
status_ping = {"SAFE": "green", "MONITOR": "amber", "ABORT": "red"}.get(status_label, "green")
trend_sym   = {"Rising": "^", "Falling": "v", "Stable": "="}.get(analytics["trend_label"], "-")
trend_cls   = {"Rising": "up", "Falling": "down", "Stable": ""}.get(analytics["trend_label"], "")
delta_cls   = "up" if peak_f > threshold else "down"
delta_val   = peak_f - threshold

st.markdown(f"""
<div class="metric-grid">
  <div class="metric-card {status_hud}">
    <span class="c tl"></span><span class="c tr"></span>
    <span class="c bl"></span><span class="c br"></span>
    <div class="hud-label">Mission Status</div>
    <div class="hud-value {status_hud}" style="font-size:1.3rem;display:flex;align-items:center;gap:6px">
      <span class="ping-dot {status_ping}"></span>{status_label}
    </div>
    <div class="hud-sub {trend_cls}">{trend_sym} {analytics["trend_label"]}</div>
  </div>
  <div class="metric-card orange">
    <span class="c tl"></span><span class="c tr"></span>
    <span class="c bl"></span><span class="c br"></span>
    <div class="hud-label">Peak Fatigue</div>
    <div class="hud-value orange">{peak_f:.3f}</div>
    <div class="hud-sub {delta_cls}">{delta_val:+.3f} vs threshold</div>
  </div>
  <div class="metric-card cyan">
    <span class="c tl"></span><span class="c tr"></span>
    <span class="c bl"></span><span class="c br"></span>
    <div class="hud-label">Peak Hour</div>
    <div class="hud-value cyan">H+{peak_h:02d}</div>
    <div class="hud-sub">of {mission_hours_actual}h mission</div>
  </div>
  <div class="metric-card purple">
    <span class="c tl"></span><span class="c tr"></span>
    <span class="c bl"></span><span class="c br"></span>
    <div class="hud-label">Time at Risk</div>
    <div class="hud-value" style="color:var(--hud-purple)">{at_risk:.1f}<span style="font-size:1rem">%</span></div>
    <div class="hud-sub">above threshold</div>
  </div>
  <div class="metric-card red">
    <span class="c tl"></span><span class="c tr"></span>
    <span class="c bl"></span><span class="c br"></span>
    <div class="hud-label">P(Breach)</div>
    <div class="hud-value red">{p_breach:.1f}<span style="font-size:1rem">%</span></div>
    <div class="hud-sub">{n_sims} MC trajectories</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "PHYSIO · OVERVIEW", "RISK · ANALYSIS", "DYNAMICS · PHASE", "MC · SIMULATION",
])

with tab1:
    st.markdown('<div class="hud-section">Physiological Time-Series</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_mission_overview(mission_df, events, fatigue, bg_fatigue, threshold),
                    use_container_width=True)
    col_sched, col_bg = st.columns(2)
    with col_sched:
        with st.expander("// MISSION EVENT SCHEDULE"):
            st.dataframe([{
                "Event": ev.event_type.value,
                "Start": f"H+{ev.start_min//60}h {ev.start_min%60:02d}m",
                "End":   f"H+{ev.end_min//60}h {ev.end_min%60:02d}m",
                "Dur":   f"{ev.duration_min} min",
                "Intensity": f"{ev.intensity:.2f}",
            } for ev in events], use_container_width=True)
    with col_bg:
        with st.expander("// BIOGEARS RAW EVA SEGMENT"):
            st.caption(f"Source: {'Live' if used_real else 'Synth'} · Rows: {len(biogears_df)}")
            st.plotly_chart(plot_biogears_raw(biogears_df), use_container_width=True)

with tab2:
    col_gauge, col_info = st.columns([1, 2])
    with col_gauge:
        st.plotly_chart(make_risk_gauge(peak_f, threshold), use_container_width=True)
    with col_info:
        st.markdown('<div class="hud-section">Fatigue Trend Analysis</div>', unsafe_allow_html=True)
        trend      = analytics["trend_label"]
        trend_icon = {"Rising": "Rising", "Falling": "Falling", "Stable": "Stable"}.get(trend, "")
        st.markdown(f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;margin-bottom:0.8rem">
  <div style="background:var(--hud-card);border:1px solid var(--hud-border);padding:0.7rem 0.9rem;
              clip-path:polygon(0 0,calc(100% - 6px) 0,100% 6px,100% 100%,6px 100%,0 calc(100% - 6px))">
    <div style="font-family:var(--mono);font-size:0.58rem;color:var(--hud-muted);
                text-transform:uppercase;letter-spacing:0.12em">End-of-mission trend</div>
    <div style="font-family:var(--mono);font-size:1.1rem;color:var(--hud-text);margin-top:0.2rem">
      {trend_icon}</div>
  </div>
  <div style="background:var(--hud-card);border:1px solid var(--hud-border);padding:0.7rem 0.9rem;
              clip-path:polygon(0 0,calc(100% - 6px) 0,100% 6px,100% 100%,6px 100%,0 calc(100% - 6px))">
    <div style="font-family:var(--mono);font-size:0.58rem;color:var(--hud-muted);
                text-transform:uppercase;letter-spacing:0.12em">Slope / min</div>
    <div style="font-family:var(--mono);font-size:1.1rem;color:var(--hud-text);margin-top:0.2rem">
      {analytics["trend_slope"]:.6f}</div>
  </div>
  <div style="background:var(--hud-card);border:1px solid var(--hud-border);padding:0.7rem 0.9rem;
              clip-path:polygon(0 0,calc(100% - 6px) 0,100% 6px,100% 100%,6px 100%,0 calc(100% - 6px))">
    <div style="font-family:var(--mono);font-size:0.58rem;color:var(--hud-muted);
                text-transform:uppercase;letter-spacing:0.12em">Peak fatigue</div>
    <div style="font-family:var(--mono);font-size:1.1rem;color:var(--hud-orange);
                text-shadow:var(--glow-orange);margin-top:0.2rem">
      {peak_f:.4f} H+{peak_h:02d}</div>
  </div>
  <div style="background:var(--hud-card);border:1px solid var(--hud-border);padding:0.7rem 0.9rem;
              clip-path:polygon(0 0,calc(100% - 6px) 0,100% 6px,100% 100%,6px 100%,0 calc(100% - 6px))">
    <div style="font-family:var(--mono);font-size:0.58rem;color:var(--hud-muted);
                text-transform:uppercase;letter-spacing:0.12em">MC worst-case</div>
    <div style="font-family:var(--mono);font-size:1.1rem;color:var(--hud-red);
                text-shadow:var(--glow-red);margin-top:0.2rem">
      {mc_sum["worst_case_fatigue"]:.4f}</div>
  </div>
</div>
""", unsafe_allow_html=True)
    st.markdown('<div class="hud-section">Risk Flag Log</div>', unsafe_allow_html=True)
    if risk_periods:
        st.dataframe([{
            "Flag": f"RF-{i:02d}",
            "Start": f"H+{s//60}h {s%60:02d}m",
            "End":   f"H+{e//60}h {e%60:02d}m",
            "Duration": f"{e-s} min",
            "Peak Fatigue": f"{fatigue[s:e].max():.4f}",
        } for i, (s, e) in enumerate(risk_periods, 1)], use_container_width=True)
    else:
        st.success("// NO RISK THRESHOLD BREACHES DETECTED", icon="✅")
    peak_idx = analytics["peak_minute"]
    st.markdown('<div class="hud-section">Physiological State at Peak Fatigue</div>',
                unsafe_allow_html=True)
    row      = mission_df.iloc[peak_idx]
    spo2_val = row["OxygenSaturation"] * 100 if row["OxygenSaturation"] <= 1.05 else row["OxygenSaturation"]
    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.metric("Heart Rate",   f"{row['HeartRate']:.1f} bpm")
    pc2.metric("SpO2",         f"{spo2_val:.2f}%")
    pc3.metric("Core Temp",    f"{row['CoreTemperature']:.2f} C")
    pc4.metric("Mission Time", f"H+{peak_idx//60}h {peak_idx%60:02d}m")

with tab3:
    st.markdown('<div class="hud-section">HR vs Fatigue Phase Space</div>', unsafe_allow_html=True)
    st.caption("// Each point = 1 mission minute · colour = time · red band = risk zone")
    st.plotly_chart(plot_phase_space(mission_df, fatigue, threshold), use_container_width=True)

with tab4:
    st.markdown(f'<div class="hud-section">Monte Carlo · {n_sims} Trajectories</div>',
                unsafe_allow_html=True)
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("P(any breach)",     f"{mc_sum['p_any_breach']*100:.1f}%")
    mc2.metric("Mean peak fatigue", f"{mc_sum['mean_peak_fatigue']:.4f}")
    mc3.metric("Worst-case",        f"{mc_sum['worst_case_fatigue']:.4f}")
    st.plotly_chart(plot_monte_carlo_envelope(mc, threshold), use_container_width=True)
    st.markdown('<div class="hud-section">Simulation Heatmap</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_risk_heatmap(mc), use_container_width=True)
    with st.expander("// PEAK FATIGUE DISTRIBUTION"):
        import plotly.graph_objects as _pgo
        hfig = _pgo.Figure()
        hfig.add_trace(_pgo.Histogram(
            x=mc["max_per_sim"], nbinsx=30,
            marker=dict(color="#ff6b00", opacity=0.75, line=dict(color="#0c2040", width=0.5)),
            hovertemplate="Fatigue %{x:.3f} · %{y} runs<extra></extra>",
        ))
        hfig.add_vline(x=threshold, line=dict(color="#ff1a3c", dash="dash", width=2),
                       annotation_text=f" Threshold {threshold:.2f}",
                       annotation_font=dict(color="#ff1a3c", size=10, family="Share Tech Mono"),
                       annotation_position="top right")
        hfig.update_layout(template="plotly_dark", paper_bgcolor="#060c18",
                           plot_bgcolor="#030608", height=300,
                           margin=dict(t=36, b=36, l=48, r=24),
                           title=dict(text="// PEAK FATIGUE DISTRIBUTION",
                                      font=dict(size=11, family="Share Tech Mono", color="#1e4060"), x=0.01),
                           xaxis=dict(gridcolor="rgba(0,212,255,0.04)", linecolor="#0c2040",
                                      tickfont=dict(family="Share Tech Mono", size=10)),
                           yaxis=dict(gridcolor="rgba(0,212,255,0.04)", linecolor="#0c2040",
                                      tickfont=dict(family="Share Tech Mono", size=10)),
                           bargap=0.06, font=dict(family="Share Tech Mono", color="#1e4060"))
        st.plotly_chart(hfig, use_container_width=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:1.5rem;padding:0.6rem 0;border-top:1px solid var(--hud-border);
            display:flex;justify-content:space-between;align-items:center">
  <span style="font-family:var(--mono);font-size:0.55rem;color:var(--hud-dim);
               letter-spacing:0.12em;text-transform:uppercase">
    AstroGuard &middot; BioGears 8.2.0 &middot; Musculoskeletal Fatigue &amp; Injury Risk
  </span>
  <span style="font-family:var(--mono);font-size:0.55rem;color:var(--hud-dim);
               letter-spacing:0.12em;text-transform:uppercase">
    RESPOND Basket 2025 &middot; RES-HSFC-2025-001
  </span>
</div>
""", unsafe_allow_html=True)
