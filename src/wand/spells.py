"""Spell primitives, operators, and encoding."""

from dataclasses import dataclass, field
from enum import IntEnum
from .events import Pole, NUM_POLES

import torch


# --- Primitives ---

class PrimCategory(IntEnum):
    ELEMENT = 0
    DOMAIN = 1
    DIRECTION = 2
    QUANTITY = 3
    MODIFIER = 4
    META = 5

NUM_CATEGORIES = len(PrimCategory)


@dataclass(frozen=True)
class Primitive:
    name: str
    category: PrimCategory
    pole: Pole | None     # primary pole affinity, None if neutral
    index: int            # global index for encoding


# Starter set — 30 primitives
_PRIMITIVES: list[Primitive] = []

def _add(name: str, cat: PrimCategory, pole: Pole | None = None) -> Primitive:
    p = Primitive(name, cat, pole, len(_PRIMITIVES))
    _PRIMITIVES.append(p)
    return p

# Elements (6)
FIRE   = _add("fire",   PrimCategory.ELEMENT, Pole.ELEMENT)
WATER  = _add("water",  PrimCategory.ELEMENT, Pole.ELEMENT)
AIR    = _add("air",    PrimCategory.ELEMENT, Pole.ELEMENT)
EARTH  = _add("earth",  PrimCategory.ELEMENT, Pole.ELEMENT)
LIGHT  = _add("light",  PrimCategory.ELEMENT, Pole.ELEMENT)
DARK   = _add("dark",   PrimCategory.ELEMENT, Pole.ELEMENT)

# Domains (6)
BODY   = _add("body",   PrimCategory.DOMAIN, Pole.BIOLOGY)
MIND   = _add("mind",   PrimCategory.DOMAIN, Pole.MENTAL)
MATTER = _add("matter", PrimCategory.DOMAIN, Pole.TRANSFORM)
SPACE  = _add("space",  PrimCategory.DOMAIN, Pole.SPACE)
TIME   = _add("time",   PrimCategory.DOMAIN, Pole.TIME)
LIFE   = _add("life",   PrimCategory.DOMAIN, Pole.BIOLOGY)

# Directions (6)
INWARD  = _add("inward",  PrimCategory.DIRECTION)
OUTWARD = _add("outward", PrimCategory.DIRECTION)
UP      = _add("up",      PrimCategory.DIRECTION)
DOWN    = _add("down",    PrimCategory.DIRECTION)
THROUGH = _add("through", PrimCategory.DIRECTION)
REVERSE = _add("reverse", PrimCategory.DIRECTION, Pole.TIME)

# Quantity (4)
POINT  = _add("point",  PrimCategory.QUANTITY)
FLOW   = _add("flow",   PrimCategory.QUANTITY)
WAVE   = _add("wave",   PrimCategory.QUANTITY)
ALL    = _add("all",    PrimCategory.QUANTITY)

# Modifiers (4)
AMPLIFY  = _add("amplify",  PrimCategory.MODIFIER)
WEAKEN   = _add("weaken",   PrimCategory.MODIFIER)
HOLD     = _add("hold",     PrimCategory.MODIFIER)
DISPEL   = _add("dispel",   PrimCategory.MODIFIER)

# Meta (4)
HERE   = _add("here",   PrimCategory.META)
THERE  = _add("there",  PrimCategory.META, Pole.SPACE)
SELF   = _add("self",   PrimCategory.META)
TARGET = _add("target", PrimCategory.META)

NUM_PRIMITIVES = len(_PRIMITIVES)
PRIMITIVE_BY_NAME = {p.name: p for p in _PRIMITIVES}


def get_primitive(name: str) -> Primitive:
    return PRIMITIVE_BY_NAME[name]


# --- Operators ---

class Operator(IntEnum):
    DIRECT = 0       # A → B
    REVERSE = 1      # B → A
    MUTUAL = 2       # A ↔ B
    SEQUENCE = 3     # A then B
    CONDITION = 4    # if A then B
    NEGATE = 5       # not A
    ITERATE = 6      # repeat A
    LIMIT = 7        # bound A

NUM_OPERATORS = len(Operator)


# --- Connections ---

@dataclass
class Connection:
    from_idx: int     # index into spell's primitive list
    to_idx: int       # index into spell's primitive list
    operator: Operator


# --- Spell descriptor ---

@dataclass
class Spell:
    """A complete spell: circle + primitives + connections."""
    primitives: list[Primitive]
    connections: list[Connection]
    circle_closed: bool = True
    circle_radius: float = 1.0    # relative size, affects effect area

    def pole_signature(self) -> list[float]:
        """Compute which poles this spell relates to, based on its primitives."""
        sig = [0.0] * NUM_POLES
        for p in self.primitives:
            if p.pole is not None:
                sig[p.pole] += 1.0
        total = sum(sig)
        if total > 0:
            sig = [s / total for s in sig]
        return sig

    def encode(self) -> torch.Tensor:
        """Encode spell into a fixed-size tensor for wand input.

        Layout (total = SPELL_DIM):
          [0:NUM_PRIMITIVES]     primitive presence (multi-hot)
          [NP:NP+NUM_CATEGORIES] category counts (normalized)
          [NP+NC:NP+NC+NUM_POLES] pole signature
          [NP+NC+NP2:...]        operator histogram + circle features
        """
        vec = [0.0] * SPELL_DIM

        # Multi-hot primitive presence
        for p in self.primitives:
            vec[p.index] = 1.0

        # Category counts
        cat_counts = [0.0] * NUM_CATEGORIES
        for p in self.primitives:
            cat_counts[p.category] += 1.0
        n = max(len(self.primitives), 1)
        offset = NUM_PRIMITIVES
        for i in range(NUM_CATEGORIES):
            vec[offset + i] = cat_counts[i] / n

        # Pole signature
        sig = self.pole_signature()
        offset += NUM_CATEGORIES
        for i in range(NUM_POLES):
            vec[offset + i] = sig[i]

        # Operator histogram
        op_hist = [0.0] * NUM_OPERATORS
        for c in self.connections:
            op_hist[c.operator] += 1.0
        n_conn = max(len(self.connections), 1)
        offset += NUM_POLES
        for i in range(NUM_OPERATORS):
            vec[offset + i] = op_hist[i] / n_conn

        # Circle features
        offset += NUM_OPERATORS
        vec[offset] = 1.0 if self.circle_closed else 0.0
        vec[offset + 1] = min(self.circle_radius, 3.0) / 3.0  # normalized

        return torch.tensor(vec, dtype=torch.float32)


# Total spell encoding dimension
SPELL_DIM = NUM_PRIMITIVES + NUM_CATEGORIES + NUM_POLES + NUM_OPERATORS + 2  # 30+6+6+8+2 = 52


# --- Preset spells ---

def spell_fireball() -> Spell:
    """Fire + outward + target — basic element attack."""
    prims = [FIRE, OUTWARD, TARGET]
    conns = [
        Connection(0, 1, Operator.DIRECT),   # fire → outward
        Connection(1, 2, Operator.DIRECT),   # outward → target
    ]
    return Spell(prims, conns)


def spell_heal() -> Spell:
    """Life + body + amplify — basic biology heal."""
    prims = [LIFE, BODY, AMPLIFY]
    conns = [
        Connection(0, 1, Operator.MUTUAL),   # life ↔ body
        Connection(2, 0, Operator.DIRECT),   # amplify → life
    ]
    return Spell(prims, conns)


def spell_shield() -> Spell:
    """Earth + hold + here — basic element defense."""
    prims = [EARTH, HOLD, HERE]
    conns = [
        Connection(0, 2, Operator.DIRECT),   # earth → here
        Connection(1, 0, Operator.DIRECT),   # hold → earth
    ]
    return Spell(prims, conns)


def spell_teleport() -> Spell:
    """Space + here + there + self + through — space movement."""
    prims = [SPACE, HERE, THERE, SELF, THROUGH]
    conns = [
        Connection(0, 1, Operator.DIRECT),   # space → here
        Connection(0, 2, Operator.DIRECT),   # space → there
        Connection(1, 2, Operator.MUTUAL),   # here ↔ there
        Connection(3, 4, Operator.DIRECT),   # self → through
        Connection(4, 0, Operator.DIRECT),   # through → space
    ]
    return Spell(prims, conns)


def spell_illusion() -> Spell:
    """Mind + light + there + target — mental illusion."""
    prims = [MIND, LIGHT, THERE, TARGET]
    conns = [
        Connection(0, 1, Operator.DIRECT),   # mind → light
        Connection(1, 2, Operator.DIRECT),   # light → there
        Connection(0, 3, Operator.DIRECT),   # mind → target
    ]
    return Spell(prims, conns)


def spell_time_slow() -> Spell:
    """Time + weaken + flow + there — slow down area."""
    prims = [TIME, WEAKEN, FLOW, THERE]
    conns = [
        Connection(1, 2, Operator.DIRECT),   # weaken → flow
        Connection(0, 2, Operator.DIRECT),   # time → flow
        Connection(2, 3, Operator.DIRECT),   # flow → there
    ]
    return Spell(prims, conns)


def spell_transform_matter() -> Spell:
    """Matter + target + amplify — transform object properties."""
    prims = [MATTER, TARGET, AMPLIFY]
    conns = [
        Connection(0, 1, Operator.DIRECT),   # matter → target
        Connection(2, 0, Operator.DIRECT),   # amplify → matter
    ]
    return Spell(prims, conns)


PRESET_SPELLS = {
    "fireball": spell_fireball,
    "heal": spell_heal,
    "shield": spell_shield,
    "teleport": spell_teleport,
    "illusion": spell_illusion,
    "time_slow": spell_time_slow,
    "transform": spell_transform_matter,
}
