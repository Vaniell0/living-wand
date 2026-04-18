"""Event types and synthetic biography generator."""

import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import NamedTuple


class EventType(IntEnum):
    OBSERVATION = 0
    ACTION = 1
    EMOTION = 2
    INTERACTION = 3
    LEARNING = 4
    LOSS = 5
    REVELATION = 6


class Pole(IntEnum):
    """Six poles of the Affinity Hexagon."""
    ELEMENT = 0      # Стихия
    TRANSFORM = 1    # Трансформация
    BIOLOGY = 2      # Биология
    MENTAL = 3       # Ментал
    SPACE = 4        # Пространство
    TIME = 5         # Время

NUM_EVENT_TYPES = len(EventType)
NUM_POLES = len(Pole)


class Emotion(NamedTuple):
    valence: float    # [-1, 1] bad/good
    intensity: float  # [0, 1]
    arousal: float    # [0, 1]


@dataclass
class Event:
    event_type: EventType
    pole_affinity: list[float]  # 6-dim, which poles this event relates to
    emotion: Emotion
    weight: float               # learning weight

    def to_tensor_components(self) -> list[float]:
        """Flatten event into a feature vector (19 dims)."""
        type_onehot = [0.0] * NUM_EVENT_TYPES
        type_onehot[self.event_type] = 1.0
        return (
            type_onehot                           # 7
            + self.pole_affinity                   # 6
            + [self.emotion.valence,               # 3
               self.emotion.intensity,
               self.emotion.arousal]
            + [self.weight]                        # 1
            + [self.weight * self.emotion.intensity]  # 1 (derived)
            + [self.weight * abs(self.emotion.valence)]  # 1 (derived)
        )

    @staticmethod
    def feature_dim() -> int:
        return 19


# --- Biography presets ---

def _rand_emotion(valence_range: tuple, intensity_range: tuple, arousal_range: tuple) -> Emotion:
    return Emotion(
        valence=random.uniform(*valence_range),
        intensity=random.uniform(*intensity_range),
        arousal=random.uniform(*arousal_range),
    )


def _pole_vec(primary: Pole, secondary: Pole | None = None, strength: float = 0.8) -> list[float]:
    vec = [0.0] * NUM_POLES
    vec[primary] = strength
    if secondary is not None:
        vec[secondary] = strength * 0.6
    return vec


# Biography archetypes — each returns a list of events
def biography_healer(n_events: int = 200) -> list[Event]:
    """A healer witch: lots of biology/mental, caring, calm."""
    events = []
    for _ in range(n_events):
        r = random.random()
        if r < 0.35:  # healing acts
            events.append(Event(
                EventType.ACTION,
                _pole_vec(Pole.BIOLOGY, Pole.MENTAL),
                _rand_emotion((0.3, 0.9), (0.3, 0.7), (0.1, 0.4)),
                weight=random.uniform(0.1, 0.3),
            ))
        elif r < 0.55:  # caring interactions
            events.append(Event(
                EventType.INTERACTION,
                _pole_vec(Pole.BIOLOGY, Pole.MENTAL, 0.6),
                _rand_emotion((0.2, 0.8), (0.3, 0.6), (0.1, 0.3)),
                weight=random.uniform(0.2, 0.5),
            ))
        elif r < 0.70:  # study herbs/medicine
            events.append(Event(
                EventType.LEARNING,
                _pole_vec(Pole.BIOLOGY),
                _rand_emotion((0.1, 0.5), (0.2, 0.5), (0.1, 0.3)),
                weight=random.uniform(0.05, 0.15),
            ))
        elif r < 0.85:  # calm observations
            events.append(Event(
                EventType.OBSERVATION,
                _pole_vec(Pole.BIOLOGY, Pole.SPACE, 0.4),
                _rand_emotion((0.0, 0.4), (0.1, 0.3), (0.0, 0.2)),
                weight=random.uniform(0.01, 0.05),
            ))
        elif r < 0.95:  # positive emotions
            events.append(Event(
                EventType.EMOTION,
                _pole_vec(Pole.MENTAL, Pole.BIOLOGY, 0.5),
                _rand_emotion((0.3, 0.9), (0.4, 0.8), (0.1, 0.5)),
                weight=random.uniform(0.1, 0.3),
            ))
        else:  # rare loss (patient dies)
            events.append(Event(
                EventType.LOSS,
                _pole_vec(Pole.BIOLOGY, Pole.TIME, 0.4),
                _rand_emotion((-0.9, -0.5), (0.7, 1.0), (0.5, 0.9)),
                weight=random.uniform(0.5, 1.5),
            ))
    return events


def biography_warrior(n_events: int = 200) -> list[Event]:
    """A battle witch: element/transform heavy, intense, aggressive."""
    events = []
    for _ in range(n_events):
        r = random.random()
        if r < 0.35:  # combat
            events.append(Event(
                EventType.ACTION,
                _pole_vec(Pole.ELEMENT, Pole.TRANSFORM),
                _rand_emotion((-0.3, 0.3), (0.5, 1.0), (0.6, 1.0)),
                weight=random.uniform(0.1, 0.4),
            ))
        elif r < 0.55:  # rivalry/confrontation
            events.append(Event(
                EventType.INTERACTION,
                _pole_vec(Pole.ELEMENT, Pole.MENTAL, 0.3),
                _rand_emotion((-0.5, 0.1), (0.4, 0.8), (0.5, 0.9)),
                weight=random.uniform(0.2, 0.5),
            ))
        elif r < 0.70:  # weapon/spell training
            events.append(Event(
                EventType.LEARNING,
                _pole_vec(Pole.TRANSFORM, Pole.ELEMENT),
                _rand_emotion((0.0, 0.4), (0.3, 0.6), (0.3, 0.6)),
                weight=random.uniform(0.05, 0.2),
            ))
        elif r < 0.80:  # adrenaline emotions
            events.append(Event(
                EventType.EMOTION,
                _pole_vec(Pole.ELEMENT),
                _rand_emotion((-0.2, 0.5), (0.5, 1.0), (0.7, 1.0)),
                weight=random.uniform(0.1, 0.3),
            ))
        elif r < 0.90:  # watching destruction
            events.append(Event(
                EventType.OBSERVATION,
                _pole_vec(Pole.ELEMENT, Pole.TRANSFORM, 0.5),
                _rand_emotion((-0.3, 0.2), (0.3, 0.6), (0.3, 0.7)),
                weight=random.uniform(0.02, 0.08),
            ))
        else:  # trauma/loss in battle
            events.append(Event(
                EventType.LOSS,
                _pole_vec(Pole.ELEMENT, Pole.TIME, 0.3),
                _rand_emotion((-1.0, -0.5), (0.8, 1.0), (0.7, 1.0)),
                weight=random.uniform(0.8, 2.0),
            ))
    return events


def biography_scholar(n_events: int = 200) -> list[Event]:
    """A scholar witch: mental/space dominant, calm, analytical."""
    events = []
    for _ in range(n_events):
        r = random.random()
        if r < 0.30:  # study/research
            events.append(Event(
                EventType.LEARNING,
                _pole_vec(Pole.MENTAL, Pole.SPACE),
                _rand_emotion((0.1, 0.6), (0.2, 0.5), (0.1, 0.3)),
                weight=random.uniform(0.05, 0.2),
            ))
        elif r < 0.50:  # observation/analysis
            events.append(Event(
                EventType.OBSERVATION,
                _pole_vec(Pole.MENTAL, Pole.TIME, 0.4),
                _rand_emotion((0.0, 0.3), (0.1, 0.4), (0.0, 0.2)),
                weight=random.uniform(0.02, 0.08),
            ))
        elif r < 0.65:  # intellectual discussion
            events.append(Event(
                EventType.INTERACTION,
                _pole_vec(Pole.MENTAL),
                _rand_emotion((0.0, 0.5), (0.2, 0.5), (0.1, 0.4)),
                weight=random.uniform(0.1, 0.3),
            ))
        elif r < 0.75:  # space experiments
            events.append(Event(
                EventType.ACTION,
                _pole_vec(Pole.SPACE, Pole.TIME, 0.5),
                _rand_emotion((0.0, 0.4), (0.3, 0.6), (0.2, 0.5)),
                weight=random.uniform(0.1, 0.3),
            ))
        elif r < 0.90:  # eureka moments
            events.append(Event(
                EventType.EMOTION,
                _pole_vec(Pole.MENTAL, Pole.SPACE, 0.4),
                _rand_emotion((0.3, 0.9), (0.5, 0.9), (0.3, 0.7)),
                weight=random.uniform(0.2, 0.5),
            ))
        else:  # revelation
            events.append(Event(
                EventType.REVELATION,
                _pole_vec(Pole.MENTAL, Pole.TIME, 0.6),
                _rand_emotion((0.5, 1.0), (0.8, 1.0), (0.5, 0.9)),
                weight=random.uniform(1.0, 3.0),
            ))
    return events


BIOGRAPHIES = {
    "healer": biography_healer,
    "warrior": biography_warrior,
    "scholar": biography_scholar,
}
