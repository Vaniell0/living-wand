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


# Single source of truth for the rune alphabet: name → (SignalKind, Element|None).
# Order matches tokenizer.RUNE_NAMES and spells.py primitives.
RUNES: dict[str, tuple[SignalKind, Element | None]] = {
    "fire": (SignalKind.FORCE, Element.FIRE),
    "water": (SignalKind.FORCE, Element.WATER),
    "air": (SignalKind.FORCE, Element.AIR),
    "earth": (SignalKind.FORCE, Element.EARTH),
    "light": (SignalKind.FORCE, Element.LIGHT),
    "dark": (SignalKind.FORCE, Element.DARK),
    "body": (SignalKind.DOMAIN, None),
    "mind": (SignalKind.DOMAIN, None),
    "matter": (SignalKind.DOMAIN, None),
    "space": (SignalKind.DOMAIN, None),
    "time": (SignalKind.DOMAIN, None),
    "life": (SignalKind.DOMAIN, None),
    "inward": (SignalKind.VECTOR, None),
    "outward": (SignalKind.VECTOR, None),
    "up": (SignalKind.VECTOR, None),
    "down": (SignalKind.VECTOR, None),
    "through": (SignalKind.VECTOR, None),
    "reverse": (SignalKind.VECTOR, None),
    "point": (SignalKind.SCALAR, None),
    "flow": (SignalKind.SCALAR, None),
    "wave": (SignalKind.SCALAR, None),
    "all": (SignalKind.SCALAR, None),
    "amplify": (SignalKind.MODIFIER, None),
    "weaken": (SignalKind.MODIFIER, None),
    "hold": (SignalKind.MODIFIER, None),
    "dispel": (SignalKind.MODIFIER, None),
    "here": (SignalKind.REFERENCE, None),
    "there": (SignalKind.REFERENCE, None),
    "self": (SignalKind.REFERENCE, None),
    "target": (SignalKind.REFERENCE, None),
}


# Data-driven combos — loaded from JSON via combo_data.py
# Lazy import to avoid circular dependency (combo_data imports Element from here)
_combo_data = None

def _combos():
    global _combo_data
    if _combo_data is None:
        from . import combo_data as _cd
        _combo_data = _cd
    return _combo_data

# Backward-compatible module-level names (populated on first access)
# game.py imports these but never iterates them at module level,
# so lazy property via __getattr__ works fine.
def __getattr__(name):
    if name == "SYNERGY_PAIRS":
        return _combos().SYNERGY_PAIRS
    if name == "CANCEL_PAIRS":
        return _combos().CANCEL_PAIRS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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

    def add_rune(self, name: str) -> int:
        """Add a node by rune name, looking up its kind/element in RUNES."""
        kind, element = RUNES[name]
        return self.add_node(name, kind, element)

    def connect(self, from_id: int, to_id: int, edge_type: EdgeType = EdgeType.DIRECT):
        self.edges.append(SpellEdge(from_id, to_id, edge_type))

    def to_spec(self) -> dict:
        """JSON-friendly description (inverse of graph_from_spec)."""
        return {
            "nodes": [{"id": str(nid), "name": n.name} for nid, n in self.nodes.items()],
            "edges": [[str(e.from_id), str(e.to_id), e.edge_type.name] for e in self.edges],
            "circle_closed": self.circle_closed,
            "circle_radius": self.circle_radius,
        }

    def adjacency(self) -> dict[int, list[tuple[int, EdgeType]]]:
        """Build adjacency list (outgoing edges per node)."""
        adj: dict[int, list[tuple[int, EdgeType]]] = {nid: [] for nid in self.nodes}
        for e in self.edges:
            adj[e.from_id].append((e.to_id, e.edge_type))
            if e.edge_type == EdgeType.MUTUAL:
                adj[e.to_id].append((e.from_id, e.edge_type))
        return adj


def graph_from_spec(spec: dict) -> SpellGraph:
    """Build a SpellGraph from a JSON spec (recipes, engine API, tests).

    spec = {
      "nodes": [{"name": "fire"}, {"id": "hold2", "name": "hold"}, ...],
      "edges": [["hold2", "fire", "DIRECT"], ...],   # by id (defaults to name)
      "circle_closed": true, "circle_radius": 1.0
    }
    Unknown rune names raise KeyError — callers decide how to report it.
    """
    graph = SpellGraph(
        circle_closed=bool(spec.get("circle_closed", True)),
        circle_radius=float(spec.get("circle_radius", 1.0)),
    )
    ref_to_id: dict[str, int] = {}
    for node in spec.get("nodes", []):
        name = node["name"] if isinstance(node, dict) else str(node)
        ref = node.get("id", name) if isinstance(node, dict) else name
        nid = graph.add_rune(name)
        ref_to_id.setdefault(str(ref), nid)
    for edge in spec.get("edges", []):
        src, dst, etype = edge[0], edge[1], edge[2] if len(edge) > 2 else "DIRECT"
        graph.connect(ref_to_id[str(src)], ref_to_id[str(dst)], EdgeType[str(etype).upper()])
    return graph


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


# Operator precedence for evaluation: negation binds first, sequencing last.
EDGE_PRECEDENCE = {
    EdgeType.NEGATE: 0,
    EdgeType.LIMIT: 1,
    EdgeType.DIRECT: 2,
    EdgeType.MUTUAL: 3,
    EdgeType.ITERATE: 4,
    EdgeType.CONDITION: 5,
    EdgeType.SEQUENCE: 6,
}


def ordered_edges(graph: SpellGraph) -> list[SpellEdge]:
    """Edges in evaluation order, independent of insertion order.

    Within one precedence level, modifier sources go first (so `weaken→fire`
    applies before `fire→target`), then edges are ordered by rune names.
    """
    def key(e: SpellEdge):
        src, dst = graph.nodes[e.from_id], graph.nodes[e.to_id]
        names = (src.name, dst.name)
        if e.edge_type == EdgeType.MUTUAL:
            names = tuple(sorted(names))
        src_is_modifier = src.signal_kind == SignalKind.MODIFIER
        return (EDGE_PRECEDENCE[e.edge_type], not src_is_modifier, names)
    return sorted(graph.edges, key=key)


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
    # Edges are applied in a fixed operator precedence, not in drawing order,
    # so the same spell drawn in a different order evaluates identically.
    processed_edges: set[tuple[int, int]] = set()

    for edge in ordered_edges(graph):
        key = (edge.from_id, edge.to_id)
        if key in processed_edges:
            continue
        processed_edges.add(key)

        sig_from = signals.get(edge.from_id)
        sig_to = signals.get(edge.to_id)
        if sig_from is None or sig_to is None:
            continue

        if edge.edge_type == EdgeType.DIRECT:
            _resolve_direct(sig_from, sig_to, effect, graph, edge.from_id, edge.to_id)

        elif edge.edge_type == EdgeType.MUTUAL:
            _resolve_mutual(sig_from, sig_to, effect, graph, edge.from_id, edge.to_id)

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

    # Step 2b: detect triple synergies (3 nodes all MUTUAL-connected)
    _detect_triples(graph, signals, effect)

    # Step 2c: detect modifier combos (two modifiers targeting same node)
    _detect_modifier_combos(graph, signals, effect)

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
    leak = 1.0
    if not graph.circle_closed:
        effect.stability *= 0.5
        leak = 0.7  # applied below; total_power is not computed yet here
        effect.warnings.append("Open circle — energy leaks")

    effect.area *= graph.circle_radius

    # Total power = sum of all force + domain magnitudes
    force_total = sum(max(0, v) for v in effect.force_power.values())
    domain_total = sum(max(0, v) for v in effect.domain_power.values())
    effect.total_power = (force_total + domain_total) * effect.stability * leak

    return effect


def _resolve_direct(source: Signal, target: Signal, effect: SpellEffect,
                    graph: SpellGraph, from_id: int, to_id: int):
    """DIRECT edge: source modifies/targets the target signal."""
    cd = _combos()

    # Modifier → anything: each modifier rune has its own semantics.
    # A negated modifier (NEGATE edge into it earlier) inverts its meaning.
    if source.kind == SignalKind.MODIFIER:
        _apply_modifier(graph.nodes[from_id].name, source, target, effect)

    # Vector → anything: adds directional tag
    elif source.kind == SignalKind.VECTOR:
        target.tags.add("directed")

    # Force → Force (different elements via DIRECT = one attacks the other)
    elif source.kind == SignalKind.FORCE and target.kind == SignalKind.FORCE:
        if source.element is not None and target.element is not None:
            pair = frozenset({source.element, target.element})
            combo = cd.ELEMENT_PAIRS.get(pair)
            if combo and combo.cancel_factor is not None:
                factor = combo.cancel_factor
                source.magnitude *= factor
                target.magnitude *= factor
                effect.cancellations.append(
                    f"{source.element.name}→{target.element.name} (×{factor:.1f})")
                effect.stability *= 0.8

    # Force → Domain: elemental influence on domain (data-driven)
    elif source.kind == SignalKind.FORCE and target.kind == SignalKind.DOMAIN:
        if source.element is not None:
            domain_name = graph.nodes[to_id].name
            key = (source.element, domain_name)
            combo = cd.ELEMENT_DOMAIN.get(key)
            if combo:
                target.magnitude *= combo.multiplier
                for tag in combo.tags:
                    target.tags.add(tag)
            else:
                target.magnitude *= 1.2
            target.tags.add(f"elemental-{source.element.name}")
        else:
            target.magnitude *= 1.2
            target.tags.add("elemental")

    # Domain → Force: domain focuses the element
    elif source.kind == SignalKind.DOMAIN and target.kind == SignalKind.FORCE:
        target.tags.add("domain-focused")
        effect.stability *= 1.1


# name → (magnitude mult, duration mult, stability mult, tag)
MODIFIER_RULES: dict[str, tuple[float, float, float, str]] = {
    "amplify": (1.5, 1.0, 0.95, "amplified"),
    "weaken":  (0.5, 1.0, 1.10, "weakened"),
    "hold":    (0.9, 1.6, 1.05, "sustained"),
    "dispel":  (0.25, 0.7, 1.0, "dispelled"),
}


def _apply_modifier(name: str, source: Signal, target: Signal, effect: SpellEffect):
    """Apply a modifier rune to its DIRECT target.

    Before this table existed every modifier fell into the `magnitude > 0`
    branch, so weaken/hold/dispel all behaved like amplify.
    """
    mag, dur, stab, tag = MODIFIER_RULES.get(name, (1.5, 1.0, 1.0, "amplified"))
    if source.magnitude < 0:
        # negated modifier: invert the magnitude effect (anti-amplify = weaken)
        mag = 1.0 / mag
        tag = f"inverted-{tag}"
    target.magnitude *= mag
    target.tags.add(tag)
    effect.duration *= dur
    effect.stability *= stab
    if name == "dispel":
        effect.tags.add("dispel")


def _resolve_mutual(sig_a: Signal, sig_b: Signal, effect: SpellEffect,
                    graph: SpellGraph, nid_a: int, nid_b: int):
    """MUTUAL edge: two signals combine/synthesize."""
    cd = _combos()

    # Force + Force: data-driven element pairs
    if sig_a.kind == SignalKind.FORCE and sig_b.kind == SignalKind.FORCE:
        if sig_a.element is not None and sig_b.element is not None:
            # Same element: reinforcement
            if sig_a.element == sig_b.element:
                sig_a.magnitude *= 1.8
                sig_b.magnitude *= 0.2
                sig_a.tags.add("reinforced")
                return

            pair = frozenset({sig_a.element, sig_b.element})
            combo = cd.ELEMENT_PAIRS.get(pair)
            if combo:
                if combo.type == "synergy":
                    sig_a.magnitude *= combo.multiplier
                    sig_b.magnitude *= combo.multiplier
                    for tag in combo.tags:
                        sig_a.tags.add(tag)
                        sig_b.tags.add(tag)
                    name = combo.products[0] if combo.products else "synergy"
                    effect.synergies.append(
                        f"{sig_a.element.name}+{sig_b.element.name} → {name} (×{combo.multiplier})")
                elif combo.type == "transform":
                    # Partial cancel + emergent product
                    factor = combo.cancel_factor or 0.5
                    sig_a.magnitude *= combo.multiplier
                    sig_b.magnitude *= combo.multiplier
                    for tag in combo.tags:
                        sig_a.tags.add(tag)
                    name = combo.products[0] if combo.products else "transform"
                    effect.synergies.append(
                        f"{sig_a.element.name}+{sig_b.element.name} → {name} (×{combo.multiplier})")
                    effect.stability *= 0.8
                elif combo.type == "cancel":
                    factor = combo.cancel_factor or 0.3
                    sig_a.magnitude *= factor
                    sig_b.magnitude *= factor
                    effect.cancellations.append(
                        f"{sig_a.element.name}↔{sig_b.element.name} (×{factor:.1f})")
                    effect.stability *= 0.6
            return

    # Domain + Domain: data-driven
    if sig_a.kind == SignalKind.DOMAIN and sig_b.kind == SignalKind.DOMAIN:
        name_a = graph.nodes[nid_a].name
        name_b = graph.nodes[nid_b].name
        pair_key = frozenset({name_a, name_b})
        combo = cd.DOMAIN_PAIRS.get(pair_key)
        if combo:
            sig_a.magnitude *= combo.multiplier
            for tag in combo.tags:
                sig_a.tags.add(tag)
                sig_b.tags.add(tag)
            effect.stability *= combo.stability_penalty
        else:
            sig_a.tags.add("multi-domain")
            sig_b.tags.add("multi-domain")
            effect.stability *= 0.9
        return

    # Force + Domain: data-driven infusion
    kinds = {sig_a.kind, sig_b.kind}
    if kinds == {SignalKind.FORCE, SignalKind.DOMAIN}:
        force_sig = sig_a if sig_a.kind == SignalKind.FORCE else sig_b
        domain_sig = sig_a if sig_a.kind == SignalKind.DOMAIN else sig_b
        force_nid = nid_a if sig_a.kind == SignalKind.FORCE else nid_b
        domain_nid = nid_a if sig_a.kind == SignalKind.DOMAIN else nid_b

        if force_sig.element is not None:
            domain_name = graph.nodes[domain_nid].name
            key = (force_sig.element, domain_name)
            combo = _combos().ELEMENT_DOMAIN.get(key)
            if combo:
                domain_sig.magnitude *= combo.multiplier
                for tag in combo.tags:
                    domain_sig.tags.add(tag)
                    force_sig.tags.add(tag)
            else:
                domain_sig.magnitude *= 1.3
            domain_sig.tags.add(f"infused-{force_sig.element.name}")
        else:
            domain_sig.magnitude *= 1.3
            domain_sig.tags.add("infused")
        effect.tags.add("infused")


def _detect_triples(graph: SpellGraph, signals: dict[int, Signal], effect: SpellEffect):
    """Detect triple synergies: 3 nodes all MUTUAL-connected."""
    cd = _combos()
    if not cd.TRIPLE_COMBOS:
        return

    # Build set of MUTUAL pairs
    mutual_pairs: set[frozenset[int]] = set()
    for edge in graph.edges:
        if edge.edge_type == EdgeType.MUTUAL:
            mutual_pairs.add(frozenset({edge.from_id, edge.to_id}))

    if len(mutual_pairs) < 3:
        return

    # Find cliques of 3
    node_ids = list(graph.nodes.keys())
    applied = set()
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            for k in range(j + 1, len(node_ids)):
                a, b, c = node_ids[i], node_ids[j], node_ids[k]
                if (frozenset({a, b}) in mutual_pairs and
                    frozenset({a, c}) in mutual_pairs and
                    frozenset({b, c}) in mutual_pairs):
                    # Check if this triple has a combo
                    names = frozenset({
                        graph.nodes[a].name,
                        graph.nodes[b].name,
                        graph.nodes[c].name,
                    })
                    if names in applied:
                        continue
                    combo = cd.TRIPLE_COMBOS.get(names)
                    if combo:
                        applied.add(names)
                        # Apply triple bonus on top of pairwise
                        boost = combo.multiplier / 1.3  # relative to avg pairwise
                        for nid in (a, b, c):
                            if nid in signals:
                                signals[nid].magnitude *= boost
                                for tag in combo.tags:
                                    signals[nid].tags.add(tag)
                        effect.synergies.append(
                            f"{'∆'.join(sorted(names))} (×{combo.multiplier})")


def _detect_modifier_combos(graph: SpellGraph, signals: dict[int, Signal],
                            effect: SpellEffect):
    """Detect modifier combos: two modifiers both DIRECT into the same target."""
    cd = _combos()

    # Build: target_nid -> list of modifier node names
    modifier_targets: dict[int, list[tuple[int, str]]] = {}
    for edge in graph.edges:
        if edge.edge_type == EdgeType.DIRECT:
            from_node = graph.nodes.get(edge.from_id)
            if from_node and from_node.signal_kind == SignalKind.MODIFIER:
                modifier_targets.setdefault(edge.to_id, []).append(
                    (edge.from_id, from_node.name))

    for target_nid, modifiers in modifier_targets.items():
        if len(modifiers) < 2:
            continue
        # Check all pairs
        for i in range(len(modifiers)):
            for j in range(i + 1, len(modifiers)):
                nid_a, name_a = modifiers[i]
                nid_b, name_b = modifiers[j]

                # Same-name combo (amplify+amplify, hold+hold, etc.)
                if name_a == name_b:
                    combo = cd.SAME_MODIFIER_COMBOS.get(name_a)
                else:
                    pair_key = frozenset({name_a, name_b})
                    combo = cd.MODIFIER_COMBOS.get(pair_key)

                if combo:
                    target_sig = signals.get(target_nid)
                    if target_sig:
                        target_sig.magnitude *= combo.multiplier
                        for tag in combo.tags:
                            target_sig.tags.add(tag)
                    if combo.stability_penalty is not None:
                        effect.stability *= combo.stability_penalty


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
