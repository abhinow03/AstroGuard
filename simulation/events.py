"""
Discrete-event engine for the astronaut mission timeline.
Defines event types (EVA, Sleep, Dehydration) and stochastic event sampling.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np


class EventType(Enum):
    EVA = "EVA"
    SLEEP = "Sleep"
    DEHYDRATION = "Dehydration"


# Colour coding used throughout charts
EVENT_COLORS = {
    EventType.EVA: "rgba(255,140,0,0.25)",
    EventType.SLEEP: "rgba(70,130,180,0.20)",
    EventType.DEHYDRATION: "rgba(220,50,50,0.18)",
}


@dataclass
class MissionEvent:
    event_type: EventType
    start_min: int   # minutes from mission start
    end_min: int
    intensity: float  # 0–1, workload fraction (meaningful for EVA / Dehydration)

    @property
    def duration_min(self) -> int:
        return max(0, self.end_min - self.start_min)

    @property
    def label(self) -> str:
        return self.event_type.value


def sample_events(
    mission_hours: int,
    eva_intensity: float,
    n_evas: int = 1,
    seed: int = 42,
) -> List[MissionEvent]:
    """
    Stochastically sample a mission event schedule.

    EVA start time  ~ Uniform(6, 14) hours into each mission day
    EVA duration    ~ Normal(4, 0.5) hours, clipped [2, 6]
    Sleep start     ~ fixed ~22 h ± Uniform(-1, 1) h per day
    Dehydration     ~ Bernoulli(0.3) of occurring; start Uniform(12, mission-4) h
    """
    rng = np.random.default_rng(seed)
    events: List[MissionEvent] = []
    mission_min = mission_hours * 60

    # ── EVA events ────────────────────────────────────────────────────────────
    spacing_h = mission_hours / max(n_evas, 1)
    for i in range(n_evas):
        day_offset_h = i * spacing_h
        eva_start_h = day_offset_h + float(rng.uniform(6, min(14, spacing_h - 6)))
        eva_dur_h = float(np.clip(rng.normal(4.0, 0.5), 2.0, 6.0))

        start_min = int(eva_start_h * 60)
        end_min = int((eva_start_h + eva_dur_h) * 60)

        if start_min < mission_min and end_min > start_min:
            events.append(MissionEvent(
                event_type=EventType.EVA,
                start_min=start_min,
                end_min=min(end_min, mission_min),
                intensity=float(eva_intensity),
            ))

    # ── Sleep events (one per 24-h block) ────────────────────────────────────
    for day in range(mission_hours // 24 + 1):
        jitter = float(rng.uniform(-1.0, 1.0))
        sleep_start_h = day * 24 + 22 + jitter
        sleep_dur_h = float(rng.uniform(7.0, 9.0))
        sleep_end_h = sleep_start_h + sleep_dur_h

        start_min = int(sleep_start_h * 60)
        end_min = int(sleep_end_h * 60)

        if start_min < mission_min and end_min > start_min:
            events.append(MissionEvent(
                event_type=EventType.SLEEP,
                start_min=max(0, start_min),
                end_min=min(end_min, mission_min),
                intensity=0.0,
            ))

    # ── Dehydration (probabilistic) ───────────────────────────────────────────
    if rng.random() < 0.30 and mission_hours >= 16:
        deh_start_h = float(rng.uniform(12, max(13, mission_hours - 4)))
        deh_dur_h = float(rng.uniform(2.0, 5.0))
        deh_intensity = float(rng.uniform(0.3, 0.8))

        start_min = int(deh_start_h * 60)
        end_min = int((deh_start_h + deh_dur_h) * 60)

        if start_min < mission_min and end_min > start_min:
            events.append(MissionEvent(
                event_type=EventType.DEHYDRATION,
                start_min=start_min,
                end_min=min(end_min, mission_min),
                intensity=deh_intensity,
            ))

    return events


def get_event_at_minute(events: List[MissionEvent], minute: int) -> Optional[MissionEvent]:
    """Return the first active event at the given mission minute, or None."""
    for ev in events:
        if ev.start_min <= minute < ev.end_min:
            return ev
    return None


def build_event_mask(events: List[MissionEvent], mission_min: int) -> np.ndarray:
    """
    Return an integer array of length `mission_min` where each element is
    0 = rest, 1 = EVA, 2 = Sleep, 3 = Dehydration.
    """
    mask = np.zeros(mission_min, dtype=np.int8)
    type_map = {EventType.EVA: 1, EventType.SLEEP: 2, EventType.DEHYDRATION: 3}
    for ev in events:
        s = max(0, ev.start_min)
        e = min(ev.end_min, mission_min)
        mask[s:e] = type_map.get(ev.event_type, 0)
    return mask
