"""Spell algebra — compositional semantics for magic.

Instead of encoding spells as flat vectors, this module defines
an ALGEBRA where symbols have types, operators have semantics,
and combinations produce emergent effects through well-defined rules.

Inspired by:
- Magicka: element cancellation + reinforcement
- Noita: nesting/wrapping/triggering
- Type theory: typed composition, only compatible types combine
- Graph rewriting: spell = graph, casting = evaluation

Core idea: every primitive produces a typed SIGNAL. Operators
transform, combine, route, or condition signals. The spell graph
is EVALUATED (not just encoded) to produce the final effect.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import NamedTuple
import math


# ═══════════════════════════════════════════
# 1. Signal Types — what flows through a spell
# ═══════════════════════════════════════════

class SignalKind(IntEnum):
    """What kind of magical signal a node produces."""
    FORCE = 0       # raw energy (fire, water, earth, air, light, dark)
    DOMAIN = 1      # area of influence (body, mind, matter, space, time, life)
    VECTOR = 2      # direction/targeting (inward, outward, up, down, through, reverse)
    SCALAR = 3      # quantity/intensity (point, flow, wave, all)
    MODIFIER = 4    # transforms another signal (amplify, weaken, hold, dispel)
    REFERENCE = 5   # pointer to target (here, there, self, target)


class Element(IntEnum):
    """Sub-types within FORCE signals."""
    FIRE = 0
    WATER = 1
    AIR = 2
    EARTH = 3
    LIGHT = 4
    DARK = 5


# Cancellation pairs — when connected, they partially annihilate
CANCEL_PAIRS = {
    frozenset({Element.FIRE, Element.WATER}): 0.3,   # mostly cancel
    frozenset({Element.LIGHT, Element.DARK}): 0.2,   # strongly cancel
    frozenset({Element.AIR, Element.EARTH}): 0.5,    # partially cancel
}

# Synergy pairs — when combined, they produce emergent effects
SYNERGY_PAIRS = {
    frozenset({Element.FIRE, Element.AIR}): ("storm", 1.5),      # fire storm
    frozenset({Element.WATER, Element.EARTH}): ("solid", 1.3),   # mud/clay
    frozenset({Element.FIRE, Element.EARTH}): ("forge", 1.4),    # smelting
    frozenset({Element.LIGHT, Element.AIR}): ("flash", 1.3),     # blinding flash
    frozenset({Element.DARK, Element.WATER}): ("abyss", 1.2),    # deep water
    frozenset({Element.DARK, Element.EARTH}): ("cave", 1.2),     # underground
}


# ═══════════════════════════════════════════
# 2. Signals — typed values flowing in spell graph
# ═══════════════════════════════════════════

@dataclass
class Signal:
    """A typed magical signal flowing through the spell graph."""
    kind: SignalKind
    element: Element | None = None    # only for FORCE signals
    magnitude: float = 1.0            # strength
    tags: set[str] = field(default_factory=set)  # emergent properties

    def __repr__(self):
        parts = [self.kind.name]
        if self.element is not None:
            parts.append(self.element.name)
        parts.append(f"×{self.magnitude:.2f}")
        if self.tags:
            parts.append(f"[{','.join(self.tags)}]")
        return f"Signal({' '.join(parts)})"


# ═══════════════════════════════════════════
# 3. Nodes — primitives that produce/transform signals
# ═══════════════════════════════════════════

class NodeType(IntEnum):
    SOURCE = 0      # produces a signal (primitives)
    OPERATOR = 1    # transforms signals (operators)
    OUTPUT = 2      # collects final effect


@dataclass
class SpellNode:
    """A node in the spell graph."""
    id: int
    name: str
    node_type: NodeType
    signal_kind: SignalKind
    element: Element | None = None

    def produce(self) -> Signal:
        """Produce this node's signal (for SOURCE nodes)."""
        return Signal(
            kind=self.signal_kind,
            element=self.element,
            magnitude=1.0,
        )


# ═══════════════════════════════════════════
# 4. Edges — operators that connect nodes
# ═══════════════════════════════════════════

class EdgeType(IntEnum):
    DIRECT = 0      # A flows into B: A modifies B, or A targets B
    MUTUAL = 1      # A and B mix: combination, synthesis
    SEQUENCE = 2    # first A, then B: temporal ordering
    CONDITION = 3   # if A succeeds, then B triggers
    NEGATE = 4      # NOT A: inversion of signal
    ITERATE = 5     # repeat A: loop/sustain
    LIMIT = 6       # A bounded: reduces range/duration


@dataclass
class SpellEdge:
    from_id: int
    to_id: int
    edge_type: EdgeType


# ═══════════════════════════════════════════
# 5. Spell Graph — the complete spell
# ═══════════════════════════════════════════

@dataclass
class SpellGraph:
    """A spell as a directed graph with typed nodes and edges."""
    nodes: dict[int, SpellNode] = field(default_factory=dict)
    edges: list[SpellEdge] = field(default_factory=list)
    circle_closed: bool = True
    circle_radius: float = 1.0

    def add_node(self, name: str, signal_kind: SignalKind,
                 element: Element | None = None) -> int:
        nid = len(self.nodes)
        self.nodes[nid] = SpellNode(nid, name, NodeType.SOURCE, signal_kind, element)
        return nid

    def connect(self, from_id: int, to_id: int, edge_type: EdgeType = EdgeType.DIRECT):
        self.edges.append(SpellEdge(from_id, to_id, edge_type))

    def adjacency(self) -> dict[int, list[tuple[int, EdgeType]]]:
        """Build adjacency list (outgoing edges per node)."""
        adj: dict[int, list[tuple[int, EdgeType]]] = {nid: [] for nid in self.nodes}
        for e in self.edges:
            adj[e.from_id].append((e.to_id, e.edge_type))
            if e.edge_type == EdgeType.MUTUAL:
                adj[e.to_id].append((e.from_id, e.edge_type))
        return adj


# ═══════════════════════════════════════════
# 6. Evaluator — resolves spell graph into effect
# ═══════════════════════════════════════════

@dataclass
class SpellEffect:
    """The resolved effect of a spell."""
    # Per-pole power distribution
    force_power: dict[Element, float] = field(default_factory=dict)
    domain_power: dict[str, float] = field(default_factory=dict)

    # Aggregate
    total_power: float = 0.0
    stability: float = 1.0
    area: float = 1.0
    duration: float = 1.0

    # Emergent properties
    tags: set[str] = field(default_factory=set)
    synergies: list[str] = field(default_factory=list)
    cancellations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> str:
        """Human-readable description of the effect."""
        parts = []
        if self.synergies:
            parts.append(f"Synergies: {', '.join(self.synergies)}")
        if self.cancellations:
            parts.append(f"Cancellations: {', '.join(self.cancellations)}")

        # Dominant force
        if self.force_power:
            dominant = max(self.force_power, key=self.force_power.get)
            parts.append(f"Dominant: {dominant.name} ({self.force_power[dominant]:.2f})")

        parts.append(f"Power={self.total_power:.2f} Stability={self.stability:.2f} "
                     f"Area={self.area:.2f} Duration={self.duration:.2f}")

        if self.tags:
            parts.append(f"Tags: {', '.join(sorted(self.tags))}")
        if self.warnings:
            parts.append(f"⚠ {'; '.join(self.warnings)}")

        return " | ".join(parts)


def evaluate_spell(graph: SpellGraph) -> SpellEffect:
    """Evaluate a spell graph into a concrete effect.

    This is the core algorithm:
    1. Each SOURCE node produces its typed signal
    2. Edges define how signals interact:
       - DIRECT: source modifies target (modifier→force = amplified force)
       - MUTUAL: signals combine (force+force = synergy/cancel check)
       - SEQUENCE: ordered application
       - CONDITION: conditional trigger
       - NEGATE: invert signal
       - ITERATE: multiply magnitude
       - LIMIT: reduce area/duration
    3. Incompatible type connections produce warnings and instability
    4. Final effect = aggregation of all resolved signals
    """
    effect = SpellEffect()

    # Step 1: produce signals from all source nodes
    signals: dict[int, Signal] = {}
    for nid, node in graph.nodes.items():
        signals[nid] = node.produce()

    adj = graph.adjacency()

    # Step 2: resolve edges (propagate modifications)
    # Process in topological-ish order (simple: iterate edges)
    processed_edges: set[tuple[int, int]] = set()

    for edge in graph.edges:
        key = (edge.from_id, edge.to_id)
        if key in processed_edges:
            continue
        processed_edges.add(key)

        sig_from = signals.get(edge.from_id)
        sig_to = signals.get(edge.to_id)
        if sig_from is None or sig_to is None:
            continue

        if edge.edge_type == EdgeType.DIRECT:
            _resolve_direct(sig_from, sig_to, effect)

        elif edge.edge_type == EdgeType.MUTUAL:
            _resolve_mutual(sig_from, sig_to, effect)

        elif edge.edge_type == EdgeType.SEQUENCE:
            # Sequence: first signal's tags carry over
            sig_to.tags.update(sig_from.tags)
            sig_to.tags.add("sequenced")

        elif edge.edge_type == EdgeType.CONDITION:
            # Conditional: to-signal only activates if from-signal succeeds
            sig_to.magnitude *= 0.8  # slight penalty for conditionality
            sig_to.tags.add("conditional")

        elif edge.edge_type == EdgeType.NEGATE:
            sig_to.magnitude *= -0.7
            sig_to.tags.add("negated")

        elif edge.edge_type == EdgeType.ITERATE:
            sig_to.magnitude *= 1.5
            sig_to.tags.add("sustained")
            effect.duration *= 1.5

        elif edge.edge_type == EdgeType.LIMIT:
            sig_to.magnitude *= 0.8
            effect.area *= 0.5
            sig_to.tags.add("focused")

    # Step 3: aggregate signals into effect
    for nid, sig in signals.items():
        if sig.kind == SignalKind.FORCE and sig.element is not None:
            current = effect.force_power.get(sig.element, 0.0)
            effect.force_power[sig.element] = current + sig.magnitude

        elif sig.kind == SignalKind.DOMAIN:
            node = graph.nodes[nid]
            current = effect.domain_power.get(node.name, 0.0)
            effect.domain_power[node.name] = current + sig.magnitude

        elif sig.kind == SignalKind.SCALAR:
            node = graph.nodes[nid]
            if node.name == "flow":
                effect.duration *= 1.3
            elif node.name == "wave":
                effect.area *= 1.5
            elif node.name == "all":
                effect.area *= 3.0
                effect.stability *= 0.7
            elif node.name == "point":
                effect.area *= 0.3
                effect.stability *= 1.2

        elif sig.kind == SignalKind.REFERENCE:
            node = graph.nodes[nid]
            if node.name == "there":
                effect.area *= 1.2
                effect.tags.add("ranged")
            elif node.name == "here":
                effect.tags.add("local")
            elif node.name == "self":
                effect.tags.add("self-cast")

        effect.tags.update(sig.tags)

    # Step 4: circle effects
    if not graph.circle_closed:
        effect.stability *= 0.5
        effect.total_power *= 0.7
        effect.warnings.append("Open circle — energy leaks")

    effect.area *= graph.circle_radius

    # Total power = sum of all force + domain magnitudes
    force_total = sum(max(0, v) for v in effect.force_power.values())
    domain_total = sum(max(0, v) for v in effect.domain_power.values())
    effect.total_power = (force_total + domain_total) * effect.stability

    return effect


def _resolve_direct(source: Signal, target: Signal, effect: SpellEffect):
    """DIRECT edge: source modifies/targets the target signal."""

    # Modifier → anything: amplify/weaken/hold/dispel
    if source.kind == SignalKind.MODIFIER:
        if "amplify" in source.tags or source.element is None:
            # Generic modifier behavior based on name would need node ref
            # For now: positive modifier = boost, negative = reduce
            if source.magnitude > 0:
                target.magnitude *= 1.5
                target.tags.add("amplified")
            else:
                target.magnitude *= 0.5
                target.tags.add("weakened")

    # Vector → anything: adds directional tag
    elif source.kind == SignalKind.VECTOR:
        target.tags.add(f"directed")

    # Force → Force (different elements via DIRECT = one attacks the other)
    elif source.kind == SignalKind.FORCE and target.kind == SignalKind.FORCE:
        if source.element is not None and target.element is not None:
            pair = frozenset({source.element, target.element})
            if pair in CANCEL_PAIRS:
                factor = CANCEL_PAIRS[pair]
                source.magnitude *= factor
                target.magnitude *= factor
                effect.cancellations.append(
                    f"{source.element.name}→{target.element.name} (×{factor:.1f})")
                effect.stability *= 0.8

    # Force → Domain: elemental influence on domain
    elif source.kind == SignalKind.FORCE and target.kind == SignalKind.DOMAIN:
        target.magnitude *= 1.2
        target.tags.add(f"elemental-{source.element.name}" if source.element else "elemental")

    # Domain → Force: domain focuses the element
    elif source.kind == SignalKind.DOMAIN and target.kind == SignalKind.FORCE:
        target.tags.add("domain-focused")
        effect.stability *= 1.1


def _resolve_mutual(sig_a: Signal, sig_b: Signal, effect: SpellEffect):
    """MUTUAL edge: two signals combine/synthesize."""

    # Force + Force: check synergy/cancellation
    if sig_a.kind == SignalKind.FORCE and sig_b.kind == SignalKind.FORCE:
        if sig_a.element is not None and sig_b.element is not None:
            pair = frozenset({sig_a.element, sig_b.element})

            # Same element: reinforcement
            if sig_a.element == sig_b.element:
                sig_a.magnitude *= 1.8
                sig_b.magnitude *= 0.2  # absorbed into A
                sig_a.tags.add("reinforced")
                return

            # Synergy check
            if pair in SYNERGY_PAIRS:
                name, mult = SYNERGY_PAIRS[pair]
                sig_a.magnitude *= mult
                sig_b.magnitude *= mult
                sig_a.tags.add(name)
                sig_b.tags.add(name)
                effect.synergies.append(
                    f"{sig_a.element.name}+{sig_b.element.name} → {name} (×{mult})")
                return

            # Cancellation check
            if pair in CANCEL_PAIRS:
                factor = CANCEL_PAIRS[pair]
                sig_a.magnitude *= factor
                sig_b.magnitude *= factor
                effect.cancellations.append(
                    f"{sig_a.element.name}↔{sig_b.element.name} (×{factor:.1f})")
                effect.stability *= 0.6
                return

    # Domain + Domain: broader effect
    if sig_a.kind == SignalKind.DOMAIN and sig_b.kind == SignalKind.DOMAIN:
        sig_a.tags.add("multi-domain")
        sig_b.tags.add("multi-domain")
        effect.stability *= 0.9  # harder to control

    # Force + Domain: infuse domain with element
    kinds = {sig_a.kind, sig_b.kind}
    if kinds == {SignalKind.FORCE, SignalKind.DOMAIN}:
        force_sig = sig_a if sig_a.kind == SignalKind.FORCE else sig_b
        domain_sig = sig_a if sig_a.kind == SignalKind.DOMAIN else sig_b
        domain_sig.magnitude *= 1.3
        domain_sig.tags.add(f"infused-{force_sig.element.name}" if force_sig.element else "infused")
        effect.tags.add("infused")


# ═══════════════════════════════════════════
# 7. Preset spell builders (using new algebra)
# ═══════════════════════════════════════════

def build_fireball() -> SpellGraph:
    """Fire → outward → target: basic ranged fire attack."""
    g = SpellGraph()
    fire = g.add_node("fire", SignalKind.FORCE, Element.FIRE)
    out = g.add_node("outward", SignalKind.VECTOR)
    tgt = g.add_node("target", SignalKind.REFERENCE)
    g.connect(fire, out, EdgeType.DIRECT)
    g.connect(out, tgt, EdgeType.DIRECT)
    return g


def build_heal() -> SpellGraph:
    """Life ═══ body, amplify → life: healing spell."""
    g = SpellGraph()
    life = g.add_node("life", SignalKind.DOMAIN)
    body = g.add_node("body", SignalKind.DOMAIN)
    amp = g.add_node("amplify", SignalKind.MODIFIER)
    g.connect(life, body, EdgeType.MUTUAL)
    g.connect(amp, life, EdgeType.DIRECT)
    return g


def build_fire_storm() -> SpellGraph:
    """Fire ═══ air, wave, outward: AOE fire+air synergy."""
    g = SpellGraph()
    fire = g.add_node("fire", SignalKind.FORCE, Element.FIRE)
    air = g.add_node("air", SignalKind.FORCE, Element.AIR)
    wave = g.add_node("wave", SignalKind.SCALAR)
    out = g.add_node("outward", SignalKind.VECTOR)
    g.connect(fire, air, EdgeType.MUTUAL)
    g.connect(fire, out, EdgeType.DIRECT)
    g.connect(wave, fire, EdgeType.DIRECT)  # wave modifies fire's area
    return g


def build_ice_wall() -> SpellGraph:
    """Water → earth (synergy: solid), hold, here: defensive wall."""
    g = SpellGraph()
    water = g.add_node("water", SignalKind.FORCE, Element.WATER)
    earth = g.add_node("earth", SignalKind.FORCE, Element.EARTH)
    hold = g.add_node("hold", SignalKind.MODIFIER)
    here = g.add_node("here", SignalKind.REFERENCE)
    g.connect(water, earth, EdgeType.MUTUAL)
    g.connect(hold, earth, EdgeType.DIRECT)
    g.connect(earth, here, EdgeType.DIRECT)
    return g


def build_fire_water_cancel() -> SpellGraph:
    """Fire ═══ water: antagonism, mostly cancels."""
    g = SpellGraph()
    fire = g.add_node("fire", SignalKind.FORCE, Element.FIRE)
    water = g.add_node("water", SignalKind.FORCE, Element.WATER)
    g.connect(fire, water, EdgeType.MUTUAL)
    return g


def build_necromancy() -> SpellGraph:
    """Dark ═══ life, amplify → dark: forbidden magic."""
    g = SpellGraph()
    dark = g.add_node("dark", SignalKind.FORCE, Element.DARK)
    life = g.add_node("life", SignalKind.DOMAIN)
    amp = g.add_node("amplify", SignalKind.MODIFIER)
    tgt = g.add_node("target", SignalKind.REFERENCE)
    g.connect(dark, life, EdgeType.MUTUAL)
    g.connect(amp, dark, EdgeType.DIRECT)
    g.connect(dark, tgt, EdgeType.DIRECT)
    return g


def build_time_reverse() -> SpellGraph:
    """Time + reverse + target + hold: rewind time on target."""
    g = SpellGraph()
    time_node = g.add_node("time", SignalKind.DOMAIN)
    reverse = g.add_node("reverse", SignalKind.VECTOR)
    tgt = g.add_node("target", SignalKind.REFERENCE)
    hold = g.add_node("hold", SignalKind.MODIFIER)
    g.connect(reverse, time_node, EdgeType.DIRECT)
    g.connect(time_node, tgt, EdgeType.DIRECT)
    g.connect(hold, time_node, EdgeType.DIRECT)
    return g


def build_sustained_fire() -> SpellGraph:
    """Fire with iterate: sustained flame (DoT)."""
    g = SpellGraph()
    fire = g.add_node("fire", SignalKind.FORCE, Element.FIRE)
    flow = g.add_node("flow", SignalKind.SCALAR)
    tgt = g.add_node("target", SignalKind.REFERENCE)
    g.connect(fire, tgt, EdgeType.DIRECT)
    g.connect(fire, fire, EdgeType.ITERATE)  # self-loop = sustain
    g.connect(flow, fire, EdgeType.DIRECT)
    return g


def build_conditional_heal() -> SpellGraph:
    """Fire → [condition] → heal: if fire hits, then heal self."""
    g = SpellGraph()
    fire = g.add_node("fire", SignalKind.FORCE, Element.FIRE)
    life = g.add_node("life", SignalKind.DOMAIN)
    body = g.add_node("body", SignalKind.DOMAIN)
    self_ref = g.add_node("self", SignalKind.REFERENCE)
    tgt = g.add_node("target", SignalKind.REFERENCE)

    g.connect(fire, tgt, EdgeType.DIRECT)           # fire at target
    g.connect(fire, life, EdgeType.CONDITION)        # if fire succeeds → heal
    g.connect(life, body, EdgeType.MUTUAL)
    g.connect(body, self_ref, EdgeType.DIRECT)       # heal self
    return g


ALGEBRA_PRESETS = {
    "fireball": build_fireball,
    "heal": build_heal,
    "fire_storm": build_fire_storm,
    "ice_wall": build_ice_wall,
    "fire+water": build_fire_water_cancel,
    "necromancy": build_necromancy,
    "time_reverse": build_time_reverse,
    "sustained_fire": build_sustained_fire,
    "conditional_heal": build_conditional_heal,
}
