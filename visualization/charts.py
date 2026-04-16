"""
All Plotly chart functions for the Astronaut Fatigue Digital Twin dashboard.
Consistent space-mission dark theme throughout.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from simulation.events import EventType, MissionEvent, EVENT_COLORS

# ── Shared design tokens ───────────────────────────────────────────────────────
C = {
    "bg":           "#070b14",
    "paper":        "#0d1117",
    "card":         "#111827",
    "border":       "#1e2d40",
    "grid":         "rgba(255,255,255,0.04)",
    "font":         "#cbd5e1",
    "muted":        "#475569",
    "hr":           "#f97316",   # EVA orange — heart rate
    "spo2":         "#3b82f6",   # Blue — SpO2
    "temp":         "#a855f7",   # Purple — temperature
    "rr":           "#06b6d4",   # Cyan — respiration rate
    "fatigue":      "#22c55e",   # Green — our fatigue model
    "bg_fatigue":   "#10b981",   # Teal — BioGears FatigueLevel
    "threshold":    "#ef4444",   # Red — risk threshold
    "mc_fill":      "rgba(34,197,94,0.12)",
    "risk_fill":    "rgba(239,68,68,0.10)",
    "eva_fill":     "rgba(249,115,22,0.18)",
    "sleep_fill":   "rgba(59,130,246,0.14)",
    "deh_fill":     "rgba(239,68,68,0.12)",
}

_LAYOUT_BASE = dict(
    template="plotly_dark",
    paper_bgcolor=C["paper"],
    plot_bgcolor=C["bg"],
    font=dict(family="'Inter', 'Segoe UI', sans-serif", color=C["font"], size=12),
    margin=dict(l=60, r=40, t=50, b=50),
    hoverlabel=dict(
        bgcolor=C["card"],
        bordercolor=C["border"],
        font=dict(color=C["font"], size=12),
    ),
)

_LEGEND_BASE = dict(
    bgcolor="rgba(13,17,23,0.85)",
    bordercolor=C["border"],
    borderwidth=1,
    font=dict(size=11),
)

_AXIS_STYLE = dict(
    gridcolor=C["grid"],
    zeroline=False,
    linecolor=C["border"],
    tickcolor=C["border"],
    tickfont=dict(size=11, color=C["muted"]),
    title_font=dict(size=12, color=C["font"]),
)


def _axis(**kw):
    d = dict(_AXIS_STYLE)
    d.update(kw)
    return d


def _hour_ticks(mission_min: int) -> tuple:
    step = 240 if mission_min > 2000 else 120
    tv = list(range(0, mission_min + 1, step))
    tt = [f"H+{v // 60}" for v in tv]
    return tv, tt


def _event_bands(fig, events, n_rows=1):
    """Add translucent event bands to every row."""
    fill_map = {
        EventType.EVA:         C["eva_fill"],
        EventType.SLEEP:       C["sleep_fill"],
        EventType.DEHYDRATION: C["deh_fill"],
    }
    for ev in events:
        color = fill_map.get(ev.event_type, "rgba(128,128,128,0.10)")
        for row in range(1, n_rows + 1):
            fig.add_vrect(
                x0=ev.start_min, x1=ev.end_min,
                fillcolor=color, line_width=0,
                row=row, col=1,
            )


def _eva_annotations(fig, events, row=4):
    """Add labelled vertical lines at EVA start/end."""
    for ev in events:
        if ev.event_type != EventType.EVA:
            continue
        for x, label in [(ev.start_min, "EVA ▶"), (ev.end_min, "◀ REC")]:
            fig.add_vline(
                x=x, line=dict(color=C["hr"], width=1, dash="dot"),
                row=row, col=1,
            )
            fig.add_annotation(
                x=x, y=1.06, xref="x" if row == 1 else f"x{row}",
                yref="paper", text=label,
                font=dict(size=9, color=C["hr"]),
                showarrow=False, bgcolor=C["paper"],
                bordercolor=C["hr"], borderwidth=1,
            )


# ── Chart 1: Mission Overview ──────────────────────────────────────────────────

def plot_mission_overview(
    mission_df: pd.DataFrame,
    events: List[MissionEvent],
    fatigue: np.ndarray,
    bg_fatigue: Optional[np.ndarray],
    threshold: float = 0.80,
) -> go.Figure:
    minutes     = mission_df["minute"].values
    mission_min = len(minutes)
    tv, tt      = _hour_ticks(mission_min)

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.22, 0.22, 0.22, 0.34],
        subplot_titles=["", "", "", ""],
    )

    # ── Row 1: Heart Rate ──
    fig.add_trace(go.Scatter(
        x=minutes, y=mission_df["HeartRate"].values,
        name="Heart Rate", fill="tozeroy",
        fillcolor="rgba(249,115,22,0.07)",
        line=dict(color=C["hr"], width=1.8),
        hovertemplate="<b>HR</b> %{y:.1f} bpm  @min %{x}<extra></extra>",
    ), row=1, col=1)
    fig.update_yaxes(title_text="HR (bpm)", row=1, col=1, **_AXIS_STYLE)

    # ── Row 2: SpO2 ──
    spo2 = mission_df["OxygenSaturation"].values
    if spo2.max() <= 1.05:
        spo2 = spo2 * 100
    fig.add_trace(go.Scatter(
        x=minutes, y=spo2,
        name="SpO₂", fill="tozeroy",
        fillcolor="rgba(59,130,246,0.07)",
        line=dict(color=C["spo2"], width=1.8),
        hovertemplate="<b>SpO₂</b> %{y:.2f}%  @min %{x}<extra></extra>",
    ), row=2, col=1)
    y0_spo2 = max(85, spo2.min() - 1)
    fig.update_yaxes(title_text="SpO₂ (%)", row=2, col=1,
                     range=[y0_spo2, 101], **_AXIS_STYLE)

    # ── Row 3: Core Temperature ──
    fig.add_trace(go.Scatter(
        x=minutes, y=mission_df["CoreTemperature"].values,
        name="Core Temp", fill="tozeroy",
        fillcolor="rgba(168,85,247,0.07)",
        line=dict(color=C["temp"], width=1.8),
        hovertemplate="<b>Temp</b> %{y:.2f}°C  @min %{x}<extra></extra>",
    ), row=3, col=1)
    fig.update_yaxes(title_text="Temp (°C)", row=3, col=1, **_AXIS_STYLE)

    # ── Row 4: Fatigue ──
    fig.add_trace(go.Scatter(
        x=minutes, y=fatigue,
        name="Fatigue (model)", fill="tozeroy",
        fillcolor="rgba(34,197,94,0.08)",
        line=dict(color=C["fatigue"], width=2.2),
        hovertemplate="<b>Fatigue</b> %{y:.4f}  @min %{x}<extra></extra>",
    ), row=4, col=1)

    if bg_fatigue is not None and np.any(bg_fatigue > 0):
        fig.add_trace(go.Scatter(
            x=minutes, y=bg_fatigue,
            name="BioGears FatigueLevel",
            line=dict(color=C["bg_fatigue"], width=1.5, dash="dot"),
            opacity=0.80,
            hovertemplate="<b>BG Fatigue</b> %{y:.4f}  @min %{x}<extra></extra>",
        ), row=4, col=1)

    fig.add_hline(
        y=threshold,
        line=dict(color=C["threshold"], dash="dash", width=1.5),
        row=4, col=1,
        annotation_text=f" Risk {threshold:.2f}",
        annotation_font=dict(color=C["threshold"], size=10),
        annotation_position="bottom right",
    )
    fig.update_yaxes(title_text="Fatigue", row=4, col=1,
                     range=[-0.02, 1.05], **_AXIS_STYLE)

    # ── Event bands on all rows ──
    _event_bands(fig, events, n_rows=4)

    # ── EVA annotations on row 4 ──
    for ev in events:
        if ev.event_type != EventType.EVA:
            continue
        fig.add_annotation(
            x=ev.start_min + (ev.end_min - ev.start_min) / 2,
            y=1.02, xref="x4", yref="paper",
            text="⚡ EVA",
            font=dict(size=10, color=C["hr"]),
            showarrow=False,
            bgcolor="rgba(249,115,22,0.15)",
            bordercolor=C["hr"], borderwidth=1,
            borderpad=3,
        )

    # ── Legend colour patches for events ──
    for label, color in [("EVA", C["eva_fill"].replace("0.18", "0.7")),
                          ("Sleep", C["sleep_fill"].replace("0.14", "0.7")),
                          ("Dehydration", C["deh_fill"].replace("0.12", "0.7"))]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=color, symbol="square"),
            name=label, showlegend=True,
        ))

    # ── Shared x-axis ──
    fig.update_xaxes(
        tickvals=tv, ticktext=tt,
        title_text="Mission Time", row=4, col=1,
        **_AXIS_STYLE,
    )
    for row in range(1, 4):
        fig.update_xaxes(showticklabels=False, row=row, col=1, **_AXIS_STYLE)

    # ── Row labels (simulated annotations) ──
    for row, label in enumerate(["HEART RATE", "OXYGEN SAT", "CORE TEMP", "FATIGUE"], 1):
        fig.add_annotation(
            x=0, y=1, xref="paper", yref=f"y{row} domain" if row > 1 else "y domain",
            text=f"<b>{label}</b>",
            font=dict(size=9, color=C["muted"]),
            showarrow=False, xanchor="left",
        )

    fig.update_layout(
        **_LAYOUT_BASE,
        height=820,
        legend=dict(**_LEGEND_BASE, orientation="h", y=-0.06),
        title=dict(
            text="Mission Physiology Overview",
            font=dict(size=14, color=C["font"]),
            x=0.01,
        ),
    )
    return fig


# ── Chart 2: Monte Carlo envelope ─────────────────────────────────────────────

def plot_monte_carlo_envelope(mc: Dict, threshold: float) -> go.Figure:
    t      = mc["time_axis"]
    tv, tt = _hour_ticks(len(t))

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # CI ribbon
    fig.add_trace(go.Scatter(
        x=np.concatenate([t, t[::-1]]),
        y=np.concatenate([mc["p95_fatigue"], mc["p5_fatigue"][::-1]]),
        fill="toself",
        fillcolor=C["mc_fill"],
        line=dict(color="rgba(0,0,0,0)"),
        name="5–95th percentile",
        hoverinfo="skip",
    ), secondary_y=False)

    # Mean
    fig.add_trace(go.Scatter(
        x=t, y=mc["mean_fatigue"],
        name="Mean fatigue",
        line=dict(color=C["fatigue"], width=2.5),
        hovertemplate="<b>Mean fatigue</b> %{y:.4f}  @min %{x}<extra></extra>",
    ), secondary_y=False)

    # Threshold
    fig.add_hline(
        y=threshold,
        line=dict(color=C["threshold"], dash="dash", width=1.5),
        annotation_text=f" Threshold {threshold:.2f}",
        annotation_font=dict(color=C["threshold"], size=10),
        annotation_position="top left",
        secondary_y=False,
    )

    # P(risk)
    fig.add_trace(go.Scatter(
        x=t, y=mc["p_risk"] * 100,
        name="P(risk) %",
        line=dict(color=C["threshold"], width=1.5, dash="dot"),
        opacity=0.85,
        hovertemplate="<b>P(risk)</b> %{y:.1f}%  @min %{x}<extra></extra>",
    ), secondary_y=True)

    fig.update_yaxes(title_text="Fatigue Index", range=[0, 1.08],
                     secondary_y=False, **_AXIS_STYLE)
    fig.update_yaxes(title_text="P(fatigue > threshold) %", range=[0, 108],
                     secondary_y=True, **_AXIS_STYLE)
    fig.update_xaxes(tickvals=tv, ticktext=tt, title_text="Mission Time", **_AXIS_STYLE)
    fig.update_layout(
        **_LAYOUT_BASE,
        height=440,
        title=dict(
            text=f"Monte Carlo Fatigue Envelope  ·  n={mc['n_sims']} simulations",
            font=dict(size=13, color=C["font"]), x=0.01,
        ),
        legend=dict(**_LEGEND_BASE, orientation="h", y=-0.18),
    )
    return fig


# ── Chart 3: Phase space ───────────────────────────────────────────────────────

def plot_phase_space(
    mission_df: pd.DataFrame,
    fatigue: np.ndarray,
    threshold: float = 0.80,
) -> go.Figure:
    hr      = mission_df["HeartRate"].values
    minutes = mission_df["minute"].values
    step    = max(1, len(minutes) // 2000)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hr[::step], y=fatigue[::step],
        mode="markers",
        marker=dict(
            color=minutes[::step],
            colorscale=[
                [0,   "#1e3a5f"],
                [0.3, "#1a73e8"],
                [0.6, "#22c55e"],
                [0.85, "#f97316"],
                [1,   "#ef4444"],
            ],
            size=4,
            opacity=0.75,
            colorbar=dict(
                title=dict(text="Min", font=dict(size=11, color=C["muted"])),
                tickfont=dict(size=10, color=C["muted"]),
                outlinecolor=C["border"],
                outlinewidth=1,
                bgcolor=C["card"],
            ),
            showscale=True,
        ),
        name="Trajectory",
        hovertemplate="<b>HR</b> %{x:.1f} bpm<br><b>Fatigue</b> %{y:.4f}<extra></extra>",
    ))

    # Risk zone
    fig.add_hrect(
        y0=threshold, y1=1.08,
        fillcolor=C["risk_fill"], line_width=0,
        annotation_text="⚠ Risk Zone",
        annotation_font=dict(color=C["threshold"], size=10),
        annotation_position="top left",
    )

    # Threshold line
    fig.add_hline(
        y=threshold,
        line=dict(color=C["threshold"], dash="dash", width=1.2),
    )

    fig.update_layout(
        **_LAYOUT_BASE,
        height=460,
        xaxis=dict(title_text="Heart Rate (bpm)", **_AXIS_STYLE),
        yaxis=dict(title_text="Fatigue Index", range=[-0.02, 1.08], **_AXIS_STYLE),
        title=dict(
            text="Phase Space  ·  Heart Rate vs Fatigue  ·  coloured by mission time",
            font=dict(size=13, color=C["font"]), x=0.01,
        ),
    )
    return fig


# ── Chart 4: Risk heatmap ──────────────────────────────────────────────────────

def plot_risk_heatmap(mc: Dict) -> go.Figure:
    mat    = mc["fatigue_matrix"]
    t      = mc["time_axis"]
    tv, tt = _hour_ticks(len(t))

    step   = max(1, mat.shape[1] // 500)
    mat_ds = mat[:, ::step]
    t_ds   = t[::step]

    fig = go.Figure(go.Heatmap(
        z=mat_ds,
        x=t_ds,
        y=list(range(mc["n_sims"])),
        colorscale=[
            [0,    "#052e16"],
            [0.4,  "#166534"],
            [0.7,  "#ca8a04"],
            [0.85, "#ea580c"],
            [1,    "#dc2626"],
        ],
        zmin=0, zmax=1,
        colorbar=dict(
            title=dict(text="Fatigue", font=dict(size=11, color=C["muted"])),
            tickfont=dict(size=10, color=C["muted"]),
            outlinecolor=C["border"], outlinewidth=1,
            bgcolor=C["card"],
        ),
        hovertemplate="<b>Sim %{y}</b>  @min %{x}<br>Fatigue %{z:.3f}<extra></extra>",
    ))

    fig.update_xaxes(tickvals=tv, ticktext=tt, title_text="Mission Time", **_AXIS_STYLE)
    fig.update_yaxes(title_text="Simulation #", **_AXIS_STYLE)
    fig.update_layout(
        **_LAYOUT_BASE,
        height=420,
        title=dict(
            text=f"Fatigue Heatmap  ·  {mc['n_sims']} Monte Carlo Runs",
            font=dict(size=13, color=C["font"]), x=0.01,
        ),
    )
    return fig


# ── Chart 5: Risk gauge ────────────────────────────────────────────────────────

def make_risk_gauge(fatigue_value: float, threshold: float = 0.80) -> go.Figure:
    if fatigue_value < threshold * 0.70:
        bar_color, glow = "#22c55e", "rgba(34,197,94,0.3)"
    elif fatigue_value < threshold:
        bar_color, glow = "#f97316", "rgba(249,115,22,0.3)"
    else:
        bar_color, glow = "#ef4444", "rgba(239,68,68,0.3)"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=fatigue_value,
        delta={"reference": threshold, "valueformat": ".3f",
               "increasing": {"color": "#ef4444"},
               "decreasing": {"color": "#22c55e"}},
        number={"valueformat": ".3f",
                "font": {"size": 40, "color": bar_color}},
        gauge={
            "axis":      {"range": [0, 1], "tickwidth": 1,
                          "tickcolor": C["muted"], "tickfont": {"size": 10}},
            "bar":       {"color": bar_color, "thickness": 0.28},
            "bgcolor":   C["card"],
            "borderwidth": 1,
            "bordercolor": C["border"],
            "steps": [
                {"range": [0, threshold * 0.70], "color": "rgba(34,197,94,0.10)"},
                {"range": [threshold * 0.70, threshold], "color": "rgba(249,115,22,0.10)"},
                {"range": [threshold, 1.0],     "color": "rgba(239,68,68,0.10)"},
            ],
            "threshold": {
                "line":  {"color": C["threshold"], "width": 3},
                "value": threshold,
            },
        },
        title={"text": "Peak Fatigue Index",
               "font": {"size": 13, "color": C["muted"]}},
    ))

    fig.update_layout(
        height=270,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor=C["paper"],
        font=dict(color=C["font"]),
    )
    return fig


# ── Chart 6: BioGears raw signals ─────────────────────────────────────────────

def plot_biogears_raw(biogears_df: pd.DataFrame) -> go.Figure:
    t  = biogears_df["Time_s"].values
    hr = biogears_df["HeartRate"].values

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=["Heart Rate  (bpm)", "BioGears FatigueLevel"],
        vertical_spacing=0.14,
    )

    fig.add_trace(go.Scatter(
        x=t, y=hr, name="Heart Rate",
        fill="tozeroy", fillcolor="rgba(249,115,22,0.08)",
        line=dict(color=C["hr"], width=1.8)),
        row=1, col=1,
    )

    if "FatigueLevel" in biogears_df.columns:
        fig.add_trace(go.Scatter(
            x=t, y=biogears_df["FatigueLevel"].values,
            name="FatigueLevel",
            fill="tozeroy", fillcolor="rgba(34,197,94,0.08)",
            line=dict(color=C["fatigue"], width=1.8)),
            row=2, col=1,
        )
    else:
        fig.add_trace(go.Scatter(
            x=t, y=np.zeros_like(t), name="FatigueLevel (N/A)",
            line=dict(color=C["muted"], dash="dot")),
            row=2, col=1,
        )

    fig.update_xaxes(title_text="BioGears Time (s)", row=2, col=1, **_AXIS_STYLE)
    fig.update_xaxes(showticklabels=False, row=1, col=1, **_AXIS_STYLE)
    fig.update_yaxes(**_AXIS_STYLE)
    fig.update_layout(
        **_LAYOUT_BASE,
        height=340,
        title=dict(
            text="Raw BioGears EVA Segment Output",
            font=dict(size=13, color=C["font"]), x=0.01,
        ),
        showlegend=False,
    )
    return fig


# ── Chart 7: Peak fatigue histogram ───────────────────────────────────────────

def plot_peak_histogram(max_per_sim: np.ndarray, threshold: float) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=max_per_sim,
        nbinsx=30,
        marker=dict(
            color=C["fatigue"],
            opacity=0.75,
            line=dict(color=C["border"], width=0.5),
        ),
        name="Peak Fatigue",
        hovertemplate="Fatigue %{x:.3f}  ·  %{y} simulations<extra></extra>",
    ))

    fig.add_vline(
        x=threshold,
        line=dict(color=C["threshold"], dash="dash", width=2),
        annotation_text=f" Threshold {threshold:.2f}",
        annotation_font=dict(color=C["threshold"], size=10),
        annotation_position="top right",
    )

    fig.update_layout(
        **_LAYOUT_BASE,
        height=300,
        xaxis=dict(title_text="Peak Fatigue per Simulation", **_AXIS_STYLE),
        yaxis=dict(title_text="Count", **_AXIS_STYLE),
        title=dict(
            text="Peak Fatigue Distribution across Monte Carlo Runs",
            font=dict(size=13, color=C["font"]), x=0.01,
        ),
        showlegend=False,
    )
    return fig
