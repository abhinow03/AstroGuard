"""
Astronaut Health Digital Twin — Streamlit Dashboard
====================================================
Run with:   streamlit run app.py
"""
from __future__ import annotations

import time

import numpy as np
import streamlit as st

from analytics.risk import mc_summary, mission_status, run_monte_carlo, single_run_analytics
from simulation.biogears import get_biogears_segment
from simulation.events import sample_events
from simulation.fatigue import compute_fatigue, normalise_biogears_fatigue
from simulation.health_vars import build_mission_timeline
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
    page_title="Astronaut Health Digital Twin",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #070b14;
    color: #cbd5e1;
}
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1e2d40;
}
[data-testid="stSidebar"] .stSlider > div > div > div { background: #1e2d40; }
[data-testid="stSidebar"] .stSlider > div > div > div > div { background: #f97316; }

/* ── Mission banner ── */
.mission-banner {
    background: linear-gradient(135deg, #0d1117 0%, #111827 50%, #0a1628 100%);
    border: 1px solid #1e2d40;
    border-top: 3px solid #f97316;
    border-radius: 8px;
    padding: 1.1rem 1.5rem;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.mission-title {
    font-size: 1.45rem;
    font-weight: 800;
    letter-spacing: 0.04em;
    color: #f1f5f9;
    margin: 0;
}
.mission-subtitle {
    font-size: 0.78rem;
    color: #64748b;
    margin: 0.15rem 0 0 0;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.live-badge {
    background: rgba(34,197,94,0.15);
    border: 1px solid rgba(34,197,94,0.4);
    color: #22c55e;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    padding: 0.25em 0.9em;
    border-radius: 2em;
    text-transform: uppercase;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%,100% { opacity: 1; }
    50%      { opacity: 0.5; }
}
.ref-tag {
    font-size: 0.68rem;
    color: #475569;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 0.4rem;
}

/* ── Metric cards ── */
.metric-grid { display: flex; gap: 0.8rem; margin-bottom: 1rem; }
.metric-card {
    flex: 1;
    background: #111827;
    border: 1px solid #1e2d40;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: "";
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    border-radius: 8px 0 0 8px;
}
.metric-card.orange::before { background: #f97316; }
.metric-card.blue::before   { background: #3b82f6; }
.metric-card.purple::before { background: #a855f7; }
.metric-card.green::before  { background: #22c55e; }
.metric-card.red::before    { background: #ef4444; }
.metric-card.yellow::before { background: #eab308; }
.metric-label {
    font-size: 0.7rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-size: 1.55rem;
    font-weight: 800;
    color: #f1f5f9;
    line-height: 1.1;
}
.metric-delta {
    font-size: 0.72rem;
    margin-top: 0.3rem;
    color: #64748b;
}
.metric-delta.up   { color: #ef4444; }
.metric-delta.down { color: #22c55e; }

/* ── Status badge ── */
.status-badge {
    display: inline-block;
    padding: 0.3em 1.1em;
    border-radius: 1.5em;
    font-size: 0.9rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* ── Section headers ── */
.section-header {
    border-left: 3px solid #f97316;
    padding-left: 0.6rem;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94a3b8;
}

/* ── Sidebar group boxes ── */
.sidebar-group {
    background: #0a0f1a;
    border: 1px solid #1e2d40;
    border-radius: 6px;
    padding: 0.7rem 0.8rem;
    margin-bottom: 0.8rem;
}
.sidebar-group-title {
    font-size: 0.68rem;
    color: #f97316;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

/* ── Tabs ── */
[data-testid="stTabs"] button {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}

/* ── Streamlit metric overrides ── */
[data-testid="stMetricValue"]   { font-size: 1.6rem; font-weight: 700; color: #f1f5f9; }
[data-testid="stMetricLabel"]   { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; }

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: #0d1117;
    border: 1px solid #1e2d40;
    border-radius: 6px;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid #1e2d40; border-radius: 6px; }

/* ── Divider ── */
hr { border-color: #1e2d40 !important; margin: 0.8rem 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #070b14; }
::-webkit-scrollbar-thumb { background: #1e2d40; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── BioGears cached call ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cached_biogears(eva_intensity: float, eva_duration_min: float, recovery_min: float):
    return get_biogears_segment(eva_intensity, eva_duration_min, recovery_min)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.5rem">
      <div style="font-size:1.4rem">🚀</div>
      <div>
        <div style="font-weight:800;font-size:0.95rem;color:#f1f5f9">Mission Control</div>
        <div style="font-size:0.65rem;color:#475569;text-transform:uppercase;letter-spacing:0.1em">
          Fatigue Digital Twin v1.0
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # ── BioGears group ──
    st.markdown('<div class="sidebar-group-title">⚡ BioGears EVA Scenario</div>', unsafe_allow_html=True)
    eva_intensity = st.slider(
        "EVA Intensity", 0.10, 0.90, 0.50, 0.05,
        help="Exercise intensity sent to BioGears (0=rest, 1=max exertion)",
    )
    eva_duration_min = st.slider(
        "EVA Duration (min)", 20, 240, 90, 10,
        help="Duration of the EVA exercise phase",
    )
    recovery_min = st.slider(
        "Recovery Duration (min)", 10, 90, 30, 5,
        help="Post-EVA recovery phase duration",
    )
    st.divider()

    # ── Mission group ──
    st.markdown('<div class="sidebar-group-title">🛰 Mission Parameters</div>', unsafe_allow_html=True)
    mission_hours = st.select_slider(
        "Mission Duration", options=[24, 48, 72], value=48,
        help="Total simulated mission length",
    )
    n_evas = st.slider(
        "Number of EVAs", 1, 3, 1,
        help="How many EVA events to schedule in the mission",
    )
    threshold = st.slider(
        "Risk Threshold", 0.50, 0.95, 0.80, 0.05,
        help="Fatigue level above which the astronaut is at risk",
    )
    st.divider()

    # ── Monte Carlo group ──
    st.markdown('<div class="sidebar-group-title">🎲 Monte Carlo</div>', unsafe_allow_html=True)
    n_sims = st.select_slider(
        "Simulations (N)", options=[10, 50, 100, 200, 500], value=100,
    )
    st.divider()

    run_btn = st.button(
        "▶  Run BioGears + Simulate",
        type="primary",
        use_container_width=True,
    )

    st.markdown("""
    <div style="margin-top:1rem;padding:0.7rem;background:#0a0f1a;border:1px solid #1e2d40;
                border-radius:6px;font-size:0.65rem;color:#475569;text-transform:uppercase;
                letter-spacing:0.08em;line-height:1.7">
      BioGears 8.2.0<br>
      BG_EVA_DUR = 10 min<br>
      Fallback: synthesised signals<br>
      MC seed = 42
    </div>
    """, unsafe_allow_html=True)


# ── Run simulation ─────────────────────────────────────────────────────────────
if run_btn or "simulation_done" not in st.session_state:

    with st.spinner("Running BioGears EVA simulation…"):
        t0 = time.time()
        biogears_df, used_real, bg_msg = cached_biogears(
            eva_intensity, eva_duration_min, recovery_min
        )
        bg_elapsed = time.time() - t0

    with st.spinner("Building mission timeline…"):
        events = sample_events(
            mission_hours=mission_hours,
            eva_intensity=eva_intensity,
            n_evas=n_evas,
            seed=42,
        )
        mission_df = build_mission_timeline(
            biogears_df=biogears_df,
            events=events,
            mission_hours=mission_hours,
            eva_duration_min=eva_duration_min,
            recovery_min=recovery_min,
            seed=0,
        )

    fatigue, risk_periods = compute_fatigue(
        hr_arr=mission_df["HeartRate"].values,
        events=events,
        threshold=threshold,
    )
    bg_fatigue = normalise_biogears_fatigue(biogears_df, len(fatigue))

    analytics = single_run_analytics(fatigue, threshold=threshold, mission_hours=mission_hours)
    status_label, status_color = mission_status(analytics, threshold)

    with st.spinner(f"Running {n_sims} Monte Carlo simulations…"):
        mc = run_monte_carlo(
            biogears_df=biogears_df,
            mission_hours=mission_hours,
            eva_intensity=eva_intensity,
            eva_duration_min=eva_duration_min,
            recovery_min=recovery_min,
            n_sims=n_sims,
            threshold=threshold,
            n_evas=n_evas,
        )
    mc_sum = mc_summary(mc)

    st.session_state.update({
        "simulation_done": True,
        "biogears_df":     biogears_df,
        "used_real":       used_real,
        "bg_msg":          bg_msg,
        "bg_elapsed":      bg_elapsed,
        "events":          events,
        "mission_df":      mission_df,
        "fatigue":         fatigue,
        "bg_fatigue":      bg_fatigue,
        "risk_periods":    risk_periods,
        "analytics":       analytics,
        "status_label":    status_label,
        "status_color":    status_color,
        "mc":              mc,
        "mc_sum":          mc_sum,
    })

# ── Load from session state ────────────────────────────────────────────────────
ss            = st.session_state
biogears_df   = ss["biogears_df"]
used_real     = ss["used_real"]
bg_msg        = ss["bg_msg"]
events        = ss["events"]
mission_df    = ss["mission_df"]
fatigue       = ss["fatigue"]
bg_fatigue    = ss["bg_fatigue"]
risk_periods  = ss["risk_periods"]
analytics     = ss["analytics"]
status_label  = ss["status_label"]
status_color  = ss["status_color"]
mc            = ss["mc"]
mc_sum        = ss["mc_sum"]

mission_min          = len(fatigue)
mission_hours_actual = mission_min // 60
peak_f               = analytics["peak_fatigue"]
peak_h               = analytics["peak_minute"] // 60
at_risk              = analytics["time_at_risk_pct"]
p_breach             = mc_sum["p_any_breach"] * 100


# ── Mission banner ─────────────────────────────────────────────────────────────
bg_source = "LIVE BIOGEARS" if used_real else "SYNTHESISED FALLBACK"
bg_badge_color = "#22c55e" if used_real else "#eab308"
bg_badge_bg    = "rgba(34,197,94,0.15)" if used_real else "rgba(234,179,8,0.15)"
bg_badge_border = "rgba(34,197,94,0.4)" if used_real else "rgba(234,179,8,0.4)"

st.markdown(f"""
<div class="mission-banner">
  <div>
    <p class="mission-title">🚀 Astronaut Health Digital Twin</p>
    <p class="mission-subtitle">Musculoskeletal Fatigue &amp; Injury Risk Monitor &nbsp;·&nbsp;
       BioGears Cardiovascular Physiology &nbsp;·&nbsp; Monte Carlo Risk Analysis</p>
    <p class="ref-tag">RES-HSFC-2025-001 &nbsp;·&nbsp; {mission_hours_actual}h Mission &nbsp;·&nbsp;
       EVA intensity {eva_intensity:.0%} &nbsp;·&nbsp; Threshold {threshold:.0%}</p>
  </div>
  <div style="text-align:right">
    <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;
                color:{bg_badge_color};background:{bg_badge_bg};border:1px solid {bg_badge_border};
                border-radius:2em;padding:0.25em 1em;display:inline-block;margin-bottom:0.4rem">
      ● LIVE
    </div>
    <div style="font-size:0.65rem;color:#475569;margin-top:0.3rem">{bg_source}</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── BioGears status strip ──────────────────────────────────────────────────────
if used_real:
    st.success(f"**BioGears** (live run) — {bg_msg}", icon="🔬")
else:
    st.warning(f"**BioGears** (fallback) — {bg_msg}", icon="⚠️")


# ── Metric cards ───────────────────────────────────────────────────────────────
status_accent = {"SAFE": "green", "MONITOR": "yellow", "ABORT": "red"}.get(status_label, "blue")
trend_icon = {"Rising": "↑", "Falling": "↓", "Stable": "→"}.get(analytics["trend_label"], "")
trend_class = {"Rising": "up", "Falling": "down", "Stable": ""}.get(analytics["trend_label"], "")
delta_sign  = "up" if peak_f > threshold else "down"

st.markdown(f"""
<div class="metric-grid">

  <div class="metric-card {status_accent}">
    <div class="metric-label">Mission Status</div>
    <div class="metric-value"
         style="font-size:1.25rem;color:{status_color}">{status_label}</div>
    <div class="metric-delta">Trend: {trend_icon} {analytics["trend_label"]}</div>
  </div>

  <div class="metric-card orange">
    <div class="metric-label">Peak Fatigue</div>
    <div class="metric-value">{peak_f:.3f}</div>
    <div class="metric-delta {delta_sign}">{peak_f - threshold:+.3f} vs threshold</div>
  </div>

  <div class="metric-card blue">
    <div class="metric-label">Peak at Hour</div>
    <div class="metric-value">H+{peak_h}</div>
    <div class="metric-delta">of {mission_hours_actual}h mission</div>
  </div>

  <div class="metric-card purple">
    <div class="metric-label">Time at Risk</div>
    <div class="metric-value">{at_risk:.1f}%</div>
    <div class="metric-delta">minutes above threshold</div>
  </div>

  <div class="metric-card red">
    <div class="metric-label">P(Breach) MC</div>
    <div class="metric-value">{p_breach:.1f}%</div>
    <div class="metric-delta">{n_sims} simulations</div>
  </div>

</div>
""", unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈  Mission Overview",
    "⚠️  Risk Analysis",
    "🔵  Phase Space",
    "🎲  Monte Carlo",
])


# ── Tab 1: Mission Overview ────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">Physiological Time-Series</div>',
                unsafe_allow_html=True)
    st.plotly_chart(
        plot_mission_overview(mission_df, events, fatigue, bg_fatigue, threshold),
        use_container_width=True,
    )

    col_sched, col_bg = st.columns(2)

    with col_sched:
        with st.expander("📅  Mission Event Schedule"):
            rows = []
            for ev in events:
                rows.append({
                    "Event":     ev.event_type.value,
                    "Start":     f"H+{ev.start_min // 60}h {ev.start_min % 60:02d}m",
                    "End":       f"H+{ev.end_min   // 60}h {ev.end_min   % 60:02d}m",
                    "Duration":  f"{ev.duration_min} min",
                    "Intensity": f"{ev.intensity:.2f}",
                })
            st.dataframe(rows, use_container_width=True)

    with col_bg:
        with st.expander("🔬  BioGears Raw EVA Segment"):
            src = "Live BioGears" if used_real else "Synthesised fallback"
            st.markdown(f"**Source:** {src} &nbsp;·&nbsp; {len(biogears_df)} rows")
            st.plotly_chart(plot_biogears_raw(biogears_df), use_container_width=True)


# ── Tab 2: Risk Analysis ───────────────────────────────────────────────────────
with tab2:
    col_gauge, col_info = st.columns([1, 2])

    with col_gauge:
        st.plotly_chart(make_risk_gauge(peak_f, threshold), use_container_width=True)

    with col_info:
        st.markdown('<div class="section-header">Fatigue Trend Analysis</div>',
                    unsafe_allow_html=True)

        # Info grid
        trend      = analytics["trend_label"]
        trend_icon = {"Rising": "📈", "Falling": "📉", "Stable": "➡️"}.get(trend, "")
        st.markdown(f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.6rem;margin-bottom:1rem">
  <div style="background:#111827;border:1px solid #1e2d40;border-radius:6px;padding:0.7rem 1rem">
    <div style="font-size:0.65rem;color:#64748b;text-transform:uppercase;letter-spacing:0.08em">
      End-of-mission trend</div>
    <div style="font-size:1.15rem;font-weight:700;color:#f1f5f9;margin-top:0.2rem">
      {trend_icon} {trend}</div>
  </div>
  <div style="background:#111827;border:1px solid #1e2d40;border-radius:6px;padding:0.7rem 1rem">
    <div style="font-size:0.65rem;color:#64748b;text-transform:uppercase;letter-spacing:0.08em">
      Slope (per minute)</div>
    <div style="font-size:1.15rem;font-weight:700;color:#f1f5f9;margin-top:0.2rem">
      {analytics["trend_slope"]:.6f}</div>
  </div>
  <div style="background:#111827;border:1px solid #1e2d40;border-radius:6px;padding:0.7rem 1rem">
    <div style="font-size:0.65rem;color:#64748b;text-transform:uppercase;letter-spacing:0.08em">
      Peak fatigue</div>
    <div style="font-size:1.15rem;font-weight:700;color:#f97316;margin-top:0.2rem">
      {peak_f:.4f} @ H+{peak_h}</div>
  </div>
  <div style="background:#111827;border:1px solid #1e2d40;border-radius:6px;padding:0.7rem 1rem">
    <div style="font-size:0.65rem;color:#64748b;text-transform:uppercase;letter-spacing:0.08em">
      MC worst-case</div>
    <div style="font-size:1.15rem;font-weight:700;color:#ef4444;margin-top:0.2rem">
      {mc_sum["worst_case_fatigue"]:.4f}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Risk flag log
    st.markdown('<div class="section-header">Risk Flag Log</div>', unsafe_allow_html=True)
    if risk_periods:
        flag_rows = [
            {
                "Flag #":       i,
                "Start":        f"H+{s // 60}h {s % 60:02d}m",
                "End":          f"H+{e // 60}h {e % 60:02d}m",
                "Duration":     f"{e - s} min",
                "Peak fatigue": f"{fatigue[s:e].max():.4f}",
            }
            for i, (s, e) in enumerate(risk_periods, 1)
        ]
        st.dataframe(flag_rows, use_container_width=True)
    else:
        st.success("No risk threshold breaches in this simulation run.", icon="✅")

    # Physiology at peak
    peak_idx = analytics["peak_minute"]
    st.markdown('<div class="section-header">Physiological State at Peak Fatigue</div>',
                unsafe_allow_html=True)
    row      = mission_df.iloc[peak_idx]
    spo2_val = row["OxygenSaturation"]
    if spo2_val <= 1.05:
        spo2_val *= 100

    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.metric("Heart Rate",   f"{row['HeartRate']:.1f} bpm")
    pc2.metric("SpO₂",         f"{spo2_val:.2f}%")
    pc3.metric("Core Temp",    f"{row['CoreTemperature']:.2f}°C")
    pc4.metric("Mission Time", f"H+{peak_idx // 60}h {peak_idx % 60:02d}m")


# ── Tab 3: Phase Space ─────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">HR vs Fatigue Phase Space</div>',
                unsafe_allow_html=True)
    st.caption(
        "Each point is one mission minute. Colour = mission time (dark→early, bright→late). "
        "The red band marks the risk zone. Trajectory shape reveals how workload and fatigue co-evolve."
    )
    st.plotly_chart(
        plot_phase_space(mission_df, fatigue, threshold),
        use_container_width=True,
    )


# ── Tab 4: Monte Carlo ─────────────────────────────────────────────────────────
with tab4:
    st.markdown(
        f'<div class="section-header">Monte Carlo Results — {n_sims} simulations</div>',
        unsafe_allow_html=True,
    )

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("P(any breach)",     f"{mc_sum['p_any_breach']*100:.1f}%",
               help="Fraction of runs that crossed the threshold at any point")
    mc2.metric("Mean peak fatigue", f"{mc_sum['mean_peak_fatigue']:.4f}")
    mc3.metric("Worst-case",        f"{mc_sum['worst_case_fatigue']:.4f}")

    st.plotly_chart(plot_monte_carlo_envelope(mc, threshold), use_container_width=True)

    st.markdown('<div class="section-header">Simulation Heatmap</div>',
                unsafe_allow_html=True)
    st.caption("Each row is one simulation. Red = high fatigue, green = low.")
    st.plotly_chart(plot_risk_heatmap(mc), use_container_width=True)

    with st.expander("📊  Peak Fatigue Distribution"):
        import plotly.graph_objects as _pgo

        hist_data = mc["max_per_sim"]
        hist_fig  = _pgo.Figure()
        hist_fig.add_trace(_pgo.Histogram(
            x=hist_data,
            nbinsx=30,
            marker_color="#f97316",
            opacity=0.8,
            name="Peak fatigue",
        ))
        hist_fig.add_vline(
            x=threshold, line_dash="dash", line_color="#ef4444", line_width=2,
            annotation_text=f"Threshold {threshold:.2f}",
            annotation_font_color="#ef4444",
        )
        hist_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#070b14",
            height=320,
            margin=dict(t=40, b=40, l=50, r=30),
            title=dict(text="Distribution of Peak Fatigue across MC Runs",
                       font=dict(size=13), x=0.01),
            xaxis_title="Peak Fatigue per Simulation",
            yaxis_title="Count",
            bargap=0.05,
        )
        st.plotly_chart(hist_fig, use_container_width=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="margin:1.5rem 0 0.6rem">
<div style="display:flex;justify-content:space-between;align-items:center;
            font-size:0.65rem;color:#334155;letter-spacing:0.06em;text-transform:uppercase">
  <span>Astronaut Health Digital Twin &nbsp;·&nbsp; BioGears 8.2.0 &nbsp;·&nbsp;
        Musculoskeletal Fatigue &amp; Injury Risk</span>
  <span>RESPOND Basket 2025 &nbsp;·&nbsp; RES-HSFC-2025-001</span>
</div>
""", unsafe_allow_html=True)
