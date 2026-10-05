"""SpellProgram — the engine-facing action IR.

The magic pipeline used to end in particles inside the pygame prototype.
This module compiles a cast into a plain, versioned, JSON-serialisable
program that ANY engine can execute (Godot, Unity, Unreal, Bevy, a web
canvas, a headless simulator for RL):

    SpellProgram
      ├─ target    who/where      self · target · point · caster_area
      ├─ delivery  how it travels instant · projectile · beam · zone · aura · touch
      ├─ shape     footprint      single · sphere · cone · line · ring · wall
      ├─ ops[]     what happens   generic verbs (data/verbs.json): damage, heal,
      │                           shield, status, push, pull, move, timescale,
      │                           light, obscure, transform, grow, temperature…
      │                           + trigger (cast · on_hit · after) for
      │                           CONDITION / SEQUENCE edges
      ├─ phases[]  timeline       charge → release → travel → impact → linger
      ├─ visual    look           palette, motions, signatures, turbulence —
      │                           a renderer-agnostic VFX description
      ├─ risk      instability    backfire chance and what backfires
      └─ provenance               algebra vs model numbers and their agreement

Design rule for stability: discrete structure (who, how, which verbs) comes
from the spell graph and the deterministic algebra; the learned model and
the personal wand modulate continuous values (power, area, duration, colour).
A small wobble in the network can make a fireball hotter — never turn it
into a heal. `authority` switches this ("algebra" | "blend" | "model").

Amounts are normalised (≈0..1). The engine owns balance: it multiplies by
its own constants (damage per unit, metres per unit, seconds per unit).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .algebra import EdgeType, Element, SignalKind, SpellEffect, SpellGraph

SCHEMA_VERSION = "1.0"

_DATA = Path(__file__).parent.parent.parent / "data"
_VERBS_PATH = _DATA / "verbs.json"
_VERBS_CACHE: dict | None = None
_JSON_CACHE: dict[str, dict] = {}


def _data(name: str) -> dict:
    if name not in _JSON_CACHE:
        with open(_DATA / name) as f:
            _JSON_CACHE[name] = json.load(f)
    return _JSON_CACHE[name]


def load_verbs(path: str | Path | None = None) -> dict:
    """Verb table (data/verbs.json). Swap the file to retarget a domain."""
    global _VERBS_CACHE
    if path is None and _VERBS_CACHE is not None:
        return _VERBS_CACHE
    with open(path or _VERBS_PATH) as f:
        data = json.load(f)
    for p in data["patterns"]:
        p["_rx"] = re.compile(p["re"])
    if path is None:
        _VERBS_CACHE = data
    return data


# ═══════════════════════════════════════════
# IR dataclasses
# ═══════════════════════════════════════════

@dataclass
class Target:
    selector: str          # "self" | "target" | "point" | "caster_area"
    range: float           # world units (engine scales); 0 = on caster


@dataclass
class Delivery:
    kind: str              # "instant" | "projectile" | "beam" | "zone" | "aura" | "touch"
                           # crafts: "consume" (potion drunk), "splash" (potion thrown),
                           #         "imbue" (bound to an item, fires on its trigger)
    speed: float = 0.0     # units/s for projectiles
    piercing: bool = False
    direction: str = "forward"  # forward | up | down | inward


@dataclass
class Shape:
    kind: str              # "single" | "sphere" | "cone" | "line" | "ring" | "wall"
    radius: float = 0.5
    length: float = 0.0
    angle: float = 0.0     # degrees (cone)


@dataclass
class Op:
    verb: str
    channel: str           # element/domain name or "arcane"
    amount: float          # normalised magnitude
    duration: float = 0.0  # seconds; 0 = instant
    period: float = 0.0    # tick interval for over-time effects
    trigger: str = "cast"  # "cast" | "on_hit" | "after:<seconds>"
    params: dict[str, Any] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)  # tags/channels that voted for it


@dataclass
class Field:
    """Physics layer: a change of a world property inside the spell's shape.

    Engines with materials consume these and let consequences emerge
    (heat ignites wood, moisture puts fire out, solidity builds walls).
    """
    property: str          # heat | moisture | motion | solidity | light | vitality | will | position | timeflow
    delta: float           # signed, normalised
    duration: float = 0.0  # seconds; 0 = pulse
    falloff: str = "linear"   # linear | none (edge of shape keeps full strength)
    trigger: str = "cast"


@dataclass
class Phase:
    name: str              # charge | release | travel | impact | linger
    start: float
    duration: float


@dataclass
class Visual:
    palette: list[tuple[tuple[int, int, int], float]]
    core_color: tuple[int, int, int]
    glow_color: tuple[int, int, int]
    motions: list[str]          # rise, fall, orbit, pulse, scatter, implode, burst, drift
    signatures: list[str]       # named flourishes: steam, storm, lava, crystal …
    intensity: float            # 0..1 (power)
    turbulence: float           # 0..1 (1 - stability)
    scale: float                # relative footprint (area)
    lifetime: float             # seconds the effect stays visible


@dataclass
class Risk:
    backfire_chance: float
    backfire: list[Op] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


@dataclass
class SpellProgram:
    schema: str
    id: str
    title: str
    description: str
    runes: list[str]
    edges: list[tuple[str, str, str]]
    circle_closed: bool
    power: float
    stability: float
    area: float
    duration: float
    cost: float
    cast_time: float
    cooldown: float
    target: Target
    delivery: Delivery
    shape: Shape
    ops: list[Op]
    fields: list[Field]
    phases: list[Phase]
    visual: Visual
    risk: Risk
    tags: list[str]
    provenance: dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kw)

    def roll_backfire(self, rng: random.Random | None = None) -> bool:
        """Execution is stochastic, the program is not: the engine rolls."""
        return (rng or random).random() < self.risk.backfire_chance

    @property
    def primary_verb(self) -> str:
        return self.ops[0].verb if self.ops else "none"


# ═══════════════════════════════════════════
# Compiler
# ═══════════════════════════════════════════

_ELEMENT_NAMES = [e.name.lower() for e in Element]
_DOMAIN_NAMES = ["body", "mind", "matter", "space", "time", "life"]

_SIGNATURE_TAGS = {
    "steam", "storm", "lava", "forge", "radiance", "hellfire", "mist", "rain", "mud",
    "clay", "rainbow", "prism", "abyss", "dust", "erosion", "flash", "void_wind",
    "crystal", "gem", "cave", "twilight", "eclipse", "reinforced", "overcharged",
    "sustained", "dispel", "phoenix", "plague", "necromancy", "golem", "flood",
}


def _norm_power(p: float) -> float:
    """Same squashing datagen uses for the model target (vec[6])."""
    p = max(p, 0.0)
    return min(p / (p + 3.0), 1.0)


def _canonical_id(graph: SpellGraph) -> str:
    nodes = sorted(n.name for n in graph.nodes.values())
    edges = []
    for e in graph.edges:
        a, b = graph.nodes[e.from_id].name, graph.nodes[e.to_id].name
        if e.edge_type == EdgeType.MUTUAL:
            a, b = sorted((a, b))
        edges.append(f"{a}>{e.edge_type.name}>{b}")
    raw = "|".join(nodes) + "#" + "|".join(sorted(edges)) + f"#{int(graph.circle_closed)}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def _names(graph: SpellGraph, kind: SignalKind) -> list[str]:
    return [n.name for n in graph.nodes.values() if n.signal_kind == kind]


def _blend(a: float, b: float | None, authority: str, w_model: float = 0.5) -> float:
    if b is None or authority == "algebra":
        return a
    if authority == "model":
        return b
    return (1 - w_model) * a + w_model * b


def _explicit_target(graph: SpellGraph, vectors: list[str]) -> tuple[Target | None, str]:
    """Target and intent stated by the runes themselves, if any."""
    refs = _names(graph, SignalKind.REFERENCE)
    for name, sel, rng, intent in (("target", "target", 8.0, "hostile"),
                                   ("there", "point", 10.0, "hostile"),
                                   ("self", "self", 0.0, "support"),
                                   ("here", "caster_area", 0.0, "environment")):
        if name in refs:
            return Target(sel, rng), intent
    if "outward" in vectors or "through" in vectors:
        return Target("target", 8.0), "hostile"
    if "inward" in vectors:
        return Target("self", 0.0), "support"
    return None, "neutral"


def _verb_class(verb: str, verbs_cfg: dict) -> str:
    for cls, members in verbs_cfg["intent"]["classes"].items():
        if verb in members:
            return cls
    return "utility"


def _select_delivery(vectors: list[str], scalars: list[str], tags: set[str],
                     target: Target, area: float) -> Delivery:
    if "flow" in scalars and target.selector in ("target", "point"):
        return Delivery("beam", speed=0.0, direction="forward")
    if "through" in vectors:
        return Delivery("projectile", speed=14.0, piercing=True)
    if "outward" in vectors and target.selector != "self":
        return Delivery("projectile", speed=10.0)
    if "up" in vectors:
        return Delivery("zone", direction="up")      # falls from above (rain, meteor)
    if "down" in vectors:
        return Delivery("zone", direction="down")    # erupts from below
    if "inward" in vectors:
        return Delivery("aura", direction="inward")
    sustained = bool(tags & {"sustained", "sustained_power", "permanent", "eternal"})
    if target.selector == "self":
        return Delivery("aura" if sustained else "touch")
    if sustained:
        return Delivery("zone")
    return Delivery("instant")


def _select_shape(scalars: list[str], tags: set[str], delivery: Delivery, area: float) -> Shape:
    radius = round(min(0.5 + 0.9 * area, 9.0), 3)
    if "point" in scalars or "focused" in tags:
        return Shape("single", radius=0.4)
    if "all" in scalars:
        return Shape("sphere", radius=round(min(radius * 2.0, 12.0), 3))
    if "wave" in scalars:
        if delivery.kind in ("projectile", "beam"):
            return Shape("cone", radius=radius, length=radius * 2, angle=60.0)
        return Shape("ring", radius=radius)
    if tags & {"wall", "barrier", "firewall", "fortress"}:
        return Shape("wall", radius=radius, length=radius * 2.5)
    if delivery.kind == "beam":
        return Shape("line", radius=0.3, length=8.0)
    return Shape("sphere", radius=radius)


def _node_triggers(graph: SpellGraph) -> dict[str, str]:
    """Rune name → trigger, from CONDITION / SEQUENCE edges."""
    trig: dict[str, str] = {}
    for e in graph.edges:
        dst = graph.nodes[e.to_id].name
        if e.edge_type == EdgeType.CONDITION:
            trig[dst] = "on_hit"
        elif e.edge_type == EdgeType.SEQUENCE and dst not in trig:
            trig[dst] = "after:0.6"
    return trig


def _score_verbs(channels: dict[str, float], tag_weights: dict[str, float],
                 verbs_cfg: dict) -> dict[str, dict]:
    """verb → {score, channels{name: w}, params{}, sources[]}."""
    acc: dict[str, dict] = {}

    def vote(verb: str, w: float, channel: str | None, params: dict, source: str):
        slot = acc.setdefault(verb, {"score": 0.0, "channels": {}, "params": {}, "sources": []})
        slot["score"] += w
        if channel:
            slot["channels"][channel] = slot["channels"].get(channel, 0.0) + abs(w)
        for k, v in params.items():
            slot["params"].setdefault(k, v)
        if source not in slot["sources"]:
            slot["sources"].append(source)

    for ch, p in channels.items():
        defaults = verbs_cfg["channel_defaults"].get(ch)
        if not defaults or abs(p) < 1e-6:
            continue
        for d in defaults:
            verb = d["verb"]
            if p < 0:  # negated channel inverts its meaning
                verb = verbs_cfg["invert"].get(verb, verb)
            params = {k: v for k, v in d.items() if k not in ("verb", "w")}
            vote(verb, abs(p) * d.get("w", 1.0), ch, params, ch)

    for tag, w in tag_weights.items():
        for pat in verbs_cfg["patterns"]:
            if pat["_rx"].search(tag):
                params = {k: v for k, v in pat.items() if k not in ("verb", "weight", "re", "_rx")}
                vote(pat["verb"], pat["weight"] * w, None, params, tag)
    return acc


def compile_program(
    graph: SpellGraph,
    algebra: SpellEffect,
    model: Any | None = None,          # effect.EffectDescription (optional)
    *,
    check: Any | None = None,          # checker.CheckResult (optional)
    clarity: float = 1.0,
    authority: str = "blend",
    stability_bonus: float = 0.0,
    context: dict | None = None,
    affinity: dict | None = None,
    vessel_capacity: float | None = None,
    verbs: dict | None = None,
    max_ops: int = 3,
) -> SpellProgram:
    """Compile one cast into a SpellProgram.

    `algebra` is evaluate_spell(graph); `model` is the SpellFormer+wand
    interpretation (EffectDescription) when a trained model is available.
    `clarity` is the OCR confidence of the drawing; `stability_bonus` comes
    from the caster's mastery. `context` describes the world at the cast
    point, e.g. {"weather": "rain", "time": "night", "place": "forest"}
    (see data/environment.json). `affinity` multiplies channels like the
    context does ({"light": 1.4, "_all": 0.9, "_stability": 1.1}) — used by
    enchanting (item material). `vessel_capacity` replaces the drawn circle
    as the vessel (an alchemist's flask, an enchanted item).
    """
    verbs_cfg = verbs or load_verbs()
    has_model = model is not None
    if not has_model:
        authority = "algebra"

    # ── continuous values: algebra baseline, model modulation ──
    a_power = _norm_power(algebra.total_power)
    a_stab = max(0.0, min(1.0, algebra.stability))
    a_area = max(0.0, min(10.0, algebra.area))
    a_dur = max(0.0, min(10.0, algebra.duration))
    m = model
    power = _blend(a_power, m.power if has_model else None, authority)
    stability = _blend(a_stab, m.stability if has_model else None, authority)
    area = _blend(a_area, m.area if has_model else None, authority)
    duration = _blend(a_dur, m.duration if has_model else None, authority)

    if check is not None:
        stability *= check.stability_penalty
    power *= clarity
    stability *= 0.5 + 0.5 * clarity
    stability = max(0.0, min(1.0, stability + stability_bonus))

    # ── the circle is a vessel: its capacity bounds how much spell it holds ──
    # Without this, "connect everything to everything" was the optimal spell
    # (every edge is another multiplier and edges were free) — found by
    # explore.map_elites.
    load = len(graph.nodes) + 0.5 * len(graph.edges)
    if vessel_capacity is not None:
        capacity = float(vessel_capacity)
    else:
        capacity = 7.0 * max(graph.circle_radius, 0.3) * (0.6 + 0.4 * clarity)
    overload = load / capacity
    if overload > 1.0:
        stability *= overload ** -1.5
        power *= overload ** -0.5

    # ── discrete structure from the graph ──
    vectors = _names(graph, SignalKind.VECTOR)
    scalars = _names(graph, SignalKind.SCALAR)
    tags = set(algebra.tags)
    target, intent = _explicit_target(graph, vectors)

    # ── verbs ──
    channels: dict[str, float] = {}
    for elem, p in algebra.force_power.items():
        channels[elem.name.lower()] = channels.get(elem.name.lower(), 0.0) + p
    for dom, p in algebra.domain_power.items():
        channels[dom] = channels.get(dom, 0.0) + p

    context_effects: dict[str, float] = {}
    entries = []
    if context:
        env = _data("environment.json")
        entries += [env.get(axis, {}).get(context.get(axis, ""), {})
                    for axis in ("weather", "time", "place")]
    if affinity:
        entries.append(affinity)
    if entries:
        mult: dict[str, float] = {}
        stab_mult = 1.0
        for entry in entries:
            for key, v in entry.items():
                if key == "_stability":
                    stab_mult *= v
                elif key == "_all":
                    for ch in channels:
                        mult[ch] = mult.get(ch, 1.0) * v
                elif not key.startswith("_"):
                    mult[key] = mult.get(key, 1.0) * v
        before = sum(abs(v) for v in channels.values()) or 1.0
        for ch in channels:
            if ch in mult:
                channels[ch] *= mult[ch]
                context_effects[ch] = round(mult[ch], 3)
        after = sum(abs(v) for v in channels.values())
        power = min(1.0, power * after / before)
        stability = max(0.0, min(1.0, stability * stab_mult))
        if stab_mult != 1.0:
            context_effects["_stability"] = round(stab_mult, 3)

    tag_weights = {t: 1.0 for t in tags}
    for syn in algebra.synergies:  # "FIRE+AIR → storm (×1.5)"
        mm = re.search(r"→\s*(\w+)", syn)
        if mm:
            tag_weights[mm.group(1)] = tag_weights.get(mm.group(1), 0.0) + 1.0
    if has_model and authority != "algebra":
        w_model = 1.0 if authority == "model" else 0.5
        for name, sim in getattr(m, "tags", [])[:4]:
            if sim > 0.3:
                tag_weights[name] = tag_weights.get(name, 0.0) + w_model * float(sim)

    scored = _score_verbs(channels, tag_weights, verbs_cfg)
    intent_w = verbs_cfg["intent"]["weights"][intent]
    for verb, info in scored.items():
        info["score"] *= intent_w[_verb_class(verb, verbs_cfg)]
    ranked = sorted(scored.items(), key=lambda kv: (-kv[1]["score"], kv[0]))
    total = sum(max(v["score"], 0.0) for _, v in ranked) or 1.0
    dominant_channel = max(channels, key=lambda c: abs(channels[c])) if channels else "arcane"
    triggers = _node_triggers(graph)
    sustained = bool(tags & {"sustained", "sustained_power", "permanent", "eternal"})

    ops: list[Op] = []
    for verb, info in ranked:
        share = info["score"] / total
        if ops and share < 0.15:
            break
        channel = (max(info["channels"], key=info["channels"].get)
                   if info["channels"] else dominant_channel)
        params = dict(info["params"])
        op_duration = 0.0
        period = 0.0
        if verb in ("status", "shield", "timescale", "obscure", "light", "grow") or sustained:
            op_duration = round(1.5 + 2.5 * duration, 2)
        if verb in ("damage", "heal") and (sustained or "status" in params):
            op_duration = round(1.5 + 2.5 * duration, 2)
            period = 0.5
        ops.append(Op(
            verb=verb,
            channel=channel,
            amount=round(power * share, 4),
            duration=op_duration,
            period=period,
            trigger=triggers.get(channel, "cast"),
            params=params,
            sources=info["sources"][:6],
        ))
        if len(ops) >= max_ops:
            break

    # No reference rune: aim by what the spell mostly does.
    primary_class = _verb_class(ops[0].verb, verbs_cfg) if ops else "utility"
    if target is None:
        target = {"harmful": Target("target", 6.0),
                  "supportive": Target("self", 0.0)}.get(primary_class, Target("caster_area", 0.0))
    # damaging an area "on yourself" means a nova around the caster
    if target.selector == "self" and primary_class == "harmful":
        target = Target("caster_area", 0.0)
    delivery = _select_delivery(vectors, scalars, tags, target, area)
    shape = _select_shape(scalars, tags, delivery, area)
    fields = _fields(channels, power, duration, sustained, triggers, vectors)

    # ── cost & timing ──
    n_runes = len(graph.nodes)
    cast_time = round(0.35 + 0.12 * n_runes, 3)
    cost = round(4.0 + 22.0 * power * (1 + 0.12 * n_runes) / (0.5 + 0.5 * stability), 2)
    cooldown = round(cast_time * (1.0 + 2.0 * power), 3)

    phases = [Phase("charge", 0.0, cast_time), Phase("release", cast_time, 0.15)]
    t = cast_time + 0.15
    if delivery.kind == "projectile" and delivery.speed > 0:
        travel = round(target.range / delivery.speed, 3)
        phases.append(Phase("travel", t, travel))
        t += travel
    phases.append(Phase("impact", t, 0.3))
    t += 0.3
    linger = max((op.duration for op in ops), default=0.0)
    if delivery.kind in ("zone", "aura", "beam") or linger > 0:
        phases.append(Phase("linger", t, round(max(linger, 1.0), 3)))

    # ── risk ──
    reasons = []
    chance = 0.0
    if stability < 0.55:
        chance += 0.6 * ((0.55 - stability) / 0.55) ** 1.5
        reasons.append(f"low stability {stability:.2f}")
    if not graph.circle_closed:
        chance += 0.1
        reasons.append("open circle")
    if check is not None and getattr(check, "has_collapse_risk", False):
        chance += 0.3
        reasons.append("collapse risk")
    if "overcharged" in tags or "explosion_risk" in tags:
        chance += 0.15
        reasons.append("overcharged")
    if overload > 1.0:
        chance += min(0.4, 0.3 * (overload - 1.0))
        reasons.append(f"vessel overloaded ×{overload:.2f}")
    chance = round(min(chance, 0.95), 4)
    backfire = []
    if chance > 0:
        backfire = [Op("damage", dominant_channel, round(0.5 * power, 4), trigger="cast",
                       params={"target_override": "self"}, sources=["backfire"])]
        if ops:
            first = ops[0]
            backfire.append(Op(first.verb, first.channel, round(first.amount * 0.5, 4),
                               first.duration, first.period, "cast",
                               {**first.params, "target_override": "self"}, ["backfire"]))

    visual = _visual(algebra, model if has_model else None, channels, tags, power, stability,
                     area, phases)
    if overload > 1.0:
        visual.signatures.append("overload")
    if not graph.circle_closed:
        visual.signatures.append("leak")
    title, description = _texts(graph, algebra, model if has_model else None, ops)

    agreement = None
    if has_model:
        diffs = [abs(a_power - m.power), abs(a_stab - m.stability),
                 min(abs(a_area - m.area) / 5.0, 1.0), min(abs(a_dur - m.duration) / 5.0, 1.0)]
        agreement = round(1.0 - sum(diffs) / len(diffs), 4)

    return SpellProgram(
        schema=SCHEMA_VERSION,
        id=_canonical_id(graph),
        title=title,
        description=description,
        runes=sorted(n.name for n in graph.nodes.values()),
        edges=[(graph.nodes[e.from_id].name, e.edge_type.name, graph.nodes[e.to_id].name)
               for e in graph.edges],
        circle_closed=graph.circle_closed,
        power=round(power, 4),
        stability=round(stability, 4),
        area=round(area, 4),
        duration=round(duration, 4),
        cost=cost,
        cast_time=cast_time,
        cooldown=cooldown,
        target=target,
        delivery=delivery,
        shape=shape,
        ops=ops,
        fields=fields,
        phases=phases,
        visual=visual,
        risk=Risk(chance, backfire, reasons),
        tags=sorted(tags),
        provenance={
            "authority": authority,
            "agreement": agreement,
            "algebra": {"power": round(a_power, 4), "stability": round(a_stab, 4),
                        "area": round(a_area, 4), "duration": round(a_dur, 4),
                        "synergies": algebra.synergies, "cancellations": algebra.cancellations},
            "model": None if not has_model else {
                "power": round(m.power, 4), "stability": round(m.stability, 4),
                "area": round(m.area, 4), "duration": round(m.duration, 4),
                "title": m.title, "tags": [t for t, _ in m.tags[:4]]},
            "clarity": round(clarity, 4),
            "vessel": {"load": round(load, 2), "capacity": round(capacity, 2),
                       "overload": round(max(overload, 1.0), 3)},
            "context": dict(context or {}),
            "context_effects": context_effects,
        },
    )


def _fields(channels: dict[str, float], power: float, duration: float, sustained: bool,
            triggers: dict[str, str], vectors: list[str]) -> list[Field]:
    """Sum per-channel property deltas (data/fields.json) into a few fields."""
    table = _data("fields.json")["channels"]
    total = sum(abs(v) for v in channels.values()) or 1.0
    acc: dict[str, float] = {}
    trig: dict[str, str] = {}
    for ch, p in channels.items():
        share = p / total  # signed: negated channels flip their properties
        for prop, k in table.get(ch, {}).items():
            acc[prop] = acc.get(prop, 0.0) + k * share
            if ch in triggers and prop not in trig:
                trig[prop] = triggers[ch]
    if "reverse" in vectors and "timeflow" in acc:
        acc["timeflow"] = -abs(acc["timeflow"]) * 2.0   # rewind, not just slow
    span = round(1.5 + 2.5 * duration, 2) if sustained else 0.0
    out = [Field(prop, round(power * d, 4), span, "linear", trig.get(prop, "cast"))
           for prop, d in acc.items() if abs(d) * power >= 0.02]
    out.sort(key=lambda f: -abs(f.delta))
    return out[:5]


def _visual(algebra: SpellEffect, model, channels: dict[str, float], tags: set[str],
            power: float, stability: float, area: float, phases: list[Phase]) -> Visual:
    from .effect import (DOMAIN_BEHAVIORS, DOMAIN_COLORS, ELEMENT_BEHAVIORS, ELEMENT_COLORS)

    if model is not None and model.palette:
        palette = [(tuple(int(v) for v in c), round(float(w), 4)) for c, w in model.palette]
        core = tuple(model.dominant_color)
        glow = tuple(model.aura_color)
    else:
        palette = []
        for ch, p in sorted(channels.items(), key=lambda kv: -abs(kv[1])):
            color = ELEMENT_COLORS.get(ch) or DOMAIN_COLORS.get(ch)
            if color and abs(p) > 0:
                w = abs(p) * (1.0 if ch in ELEMENT_COLORS else 0.7)
                palette.append((color, round(w, 4)))
        if palette:
            tw = sum(w for _, w in palette)
            core = tuple(int(sum(c[i] * w for c, w in palette) / tw) for i in range(3))
            glow = palette[0][0]
        else:
            core = glow = (180, 180, 200)

    motions = []
    for ch, p in sorted(channels.items(), key=lambda kv: -abs(kv[1])):
        mv = ELEMENT_BEHAVIORS.get(ch) or DOMAIN_BEHAVIORS.get(ch)
        if mv and mv != "default" and mv not in motions and abs(p) > 0:
            motions.append(mv)
    if not motions:
        motions = ["pulse"]

    signatures = sorted(t for t in tags if t in _SIGNATURE_TAGS)
    for syn in algebra.synergies:
        mm = re.search(r"→\s*(\w+)", syn)
        if mm and mm.group(1) not in signatures:
            signatures.append(mm.group(1))
    if algebra.cancellations:
        signatures.append("fizzle")

    end = max((p.start + p.duration for p in phases), default=1.0)
    return Visual(
        palette=palette[:6],
        core_color=core,
        glow_color=glow,
        motions=motions[:4],
        signatures=signatures[:6],
        intensity=round(power, 4),
        turbulence=round(1.0 - stability, 4),
        scale=round(min(0.5 + area / 3.0, 3.0), 4),
        lifetime=round(end, 3),
    )


def _texts(graph: SpellGraph, algebra: SpellEffect, model, ops: list[Op]) -> tuple[str, str]:
    if model is not None and model.title:
        return model.title, model.description
    from .combo_data import match_cast

    names = sorted(n.name for n in graph.nodes.values())
    # most specific known combination covered by these runes
    known = sorted(match_cast(names),
                   key=lambda d: (not d.category.startswith("recipe"), -len(d.runes), d.key))
    if known:
        title = known[0].name_ru
    elif algebra.synergies:
        mm = re.search(r"→\s*(\w+)", algebra.synergies[0])
        title = mm.group(1) if mm else "+".join(names[:3])
    else:
        title = "+".join(names[:3]) or "empty"
    verbs = ", ".join(f"{o.verb}({o.channel})" for o in ops)
    return title, f"{title}: {verbs}" if verbs else title


# ═══════════════════════════════════════════
# Outcome feedback (engine → wand experience)
# ═══════════════════════════════════════════

@dataclass
class Outcome:
    """What actually happened when the engine executed a program.

    The engine knows things the wand cannot: whether the bolt hit, how much
    it healed, whether it saved an ally. That is the real experience signal.
    """
    program_id: str
    hit: bool = True
    backfired: bool = False
    targets_affected: int = 1
    totals: dict[str, float] = field(default_factory=dict)  # verb → engine units
    context: dict[str, Any] = field(default_factory=dict)   # e.g. {"in_combat": true}

    @classmethod
    def from_dict(cls, d: dict) -> "Outcome":
        return cls(
            program_id=str(d["program_id"]),
            hit=bool(d.get("hit", True)),
            backfired=bool(d.get("backfired", False)),
            targets_affected=int(d.get("targets_affected", 1)),
            totals={k: float(v) for k, v in d.get("totals", {}).items()},
            context=dict(d.get("context", {})),
        )


_CHANNEL_POLE = {
    **{e: 0 for e in _ELEMENT_NAMES},
    "matter": 1, "body": 2, "life": 2, "mind": 3, "space": 4, "time": 5,
}


def outcome_to_event(program: SpellProgram, outcome: Outcome, repeat_count: int = 0):
    """Turn an engine outcome into a wand Event.

    - poles come from the channels the program actually used, weighted by op amount;
    - impact (hit, targets, engine totals) scales the learning weight;
    - repeating the same program teaches less and less (diminishing returns),
      so experience rewards exploring, not grinding one spell.
    """
    from .events import Emotion, Event, EventType

    poles = [0.0] * 6
    for op in program.ops:
        poles[_CHANNEL_POLE.get(op.channel, 0)] += op.amount
    s = sum(poles)
    poles = [p / s for p in poles] if s > 0 else [1 / 6] * 6

    magnitude = sum(abs(v) for v in outcome.totals.values())
    impact = (0.3 + 0.7 * float(outcome.hit)) * (1.0 + math.log1p(magnitude) / 3.0)
    impact *= 1.0 + 0.15 * max(outcome.targets_affected - 1, 0)
    novelty = 1.0 / math.sqrt(1.0 + repeat_count / 5.0)
    weight = round(0.1 * impact * novelty * (0.5 if outcome.backfired else 1.0), 5)

    valence = -0.6 if outcome.backfired else (0.5 if outcome.hit else -0.1)
    arousal = 0.8 if outcome.context.get("in_combat") else 0.35
    etype = EventType.LOSS if outcome.backfired else EventType.ACTION
    return Event(etype, poles, Emotion(valence, min(impact / 2.0, 1.0), arousal), weight)


# ═══════════════════════════════════════════
# JSON Schema (for engine developers)
# ═══════════════════════════════════════════

def json_schema() -> dict:
    """Minimal JSON Schema of SpellProgram — enough for codegen / validation."""
    num = {"type": "number"}
    s = {"type": "string"}
    rgb = {"type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3}
    op = {
        "type": "object",
        "required": ["verb", "channel", "amount", "duration", "period", "trigger", "params"],
        "properties": {
            "verb": {"enum": sorted(load_verbs()["verbs"].keys())},
            "channel": s, "amount": num, "duration": num, "period": num,
            "trigger": {"type": "string", "pattern": "^(cast|on_hit|after:[0-9.]+)$"},
            "params": {"type": "object"},
            "sources": {"type": "array", "items": s},
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": f"SpellProgram v{SCHEMA_VERSION}",
        "type": "object",
        "required": ["schema", "id", "power", "stability", "target", "delivery", "shape",
                     "ops", "fields", "phases", "visual", "risk"],
        "properties": {
            "schema": {"const": SCHEMA_VERSION},
            "id": s, "title": s, "description": s,
            "runes": {"type": "array", "items": s},
            "edges": {"type": "array", "items": {"type": "array", "items": s}},
            "circle_closed": {"type": "boolean"},
            "power": num, "stability": num, "area": num, "duration": num,
            "cost": num, "cast_time": num, "cooldown": num,
            "target": {"type": "object", "properties": {
                "selector": {"enum": ["self", "target", "point", "caster_area"]}, "range": num}},
            "delivery": {"type": "object", "properties": {
                "kind": {"enum": ["instant", "projectile", "beam", "zone", "aura", "touch",
                                  "consume", "splash", "imbue"]},
                "speed": num, "piercing": {"type": "boolean"}, "direction": s}},
            "shape": {"type": "object", "properties": {
                "kind": {"enum": ["single", "sphere", "cone", "line", "ring", "wall"]},
                "radius": num, "length": num, "angle": num}},
            "ops": {"type": "array", "items": op},
            "fields": {"type": "array", "items": {"type": "object", "properties": {
                "property": {"enum": sorted(_data("fields.json")["properties"])},
                "delta": num, "duration": num, "falloff": {"enum": ["linear", "none"]},
                "trigger": s}}},
            "phases": {"type": "array", "items": {"type": "object", "properties": {
                "name": {"enum": ["charge", "release", "travel", "impact", "linger"]},
                "start": num, "duration": num}}},
            "visual": {"type": "object", "properties": {
                "palette": {"type": "array"}, "core_color": rgb, "glow_color": rgb,
                "motions": {"type": "array", "items": s},
                "signatures": {"type": "array", "items": s},
                "intensity": num, "turbulence": num, "scale": num, "lifetime": num}},
            "risk": {"type": "object", "properties": {
                "backfire_chance": num, "backfire": {"type": "array", "items": op},
                "reasons": {"type": "array", "items": s}}},
            "tags": {"type": "array", "items": s},
            "provenance": {"type": "object"},
        },
    }


# ═══════════════════════════════════════════
# Wand signatures (experience → visible spell changes)
# ═══════════════════════════════════════════

def apply_signatures(program: SpellProgram, pole_levels: dict[str, int],
                     table: dict | None = None) -> SpellProgram:
    """Apply unlocked signatures (data/signatures.json) to a compiled program.

    pole_levels: {"ELEMENT": 3, "BIOLOGY": 0, ...}. Mutates and returns the
    program; the applied signature ids land in provenance["signatures"].
    """
    rules = (table or _data("signatures.json"))["signatures"]
    applied = []
    for rule in rules:
        if pole_levels.get(rule["pole"], 0) < rule["level"]:
            continue
        when = rule.get("when", {})
        if "delivery" in when and program.delivery.kind != when["delivery"]:
            continue
        targets = [op for op in program.ops if "verb" not in when or op.verb == when["verb"]]
        if "verb" in when and not targets:
            continue
        for op in targets:
            op.amount = round(op.amount * rule.get("amount_mul", 1.0), 4)
            if op.duration:
                op.duration = round(op.duration * rule.get("duration_mul", 1.0), 3)
            op.params.update(rule.get("add_params", {}))
        if "verb" not in when and "duration_mul" in rule:
            for f in program.fields:
                f.duration = round(f.duration * rule["duration_mul"], 3)
        if "add_op" in rule:
            a = rule["add_op"]
            program.ops.append(Op(a["verb"], a.get("channel", "arcane"),
                                  round(program.power * a.get("amount_frac", 0.3), 4),
                                  trigger=a.get("trigger", "cast"), sources=[rule["id"]]))
        program.shape.radius = round(program.shape.radius * rule.get("radius_mul", 1.0), 3)
        program.target.range = round(program.target.range * rule.get("range_mul", 1.0), 3)
        program.delivery.speed = round(program.delivery.speed * rule.get("speed_mul", 1.0), 3)
        program.cost = round(program.cost * rule.get("cost_mul", 1.0), 2)
        program.cast_time = round(program.cast_time * rule.get("cast_time_mul", 1.0), 3)
        program.stability = round(min(1.0, program.stability + rule.get("stability_add", 0.0)), 4)
        applied.append(rule["id"])
        if rule["id"] not in program.visual.signatures:
            program.visual.signatures.append(rule["id"])
    program.provenance["signatures"] = applied
    return program


def signature_catalog() -> list[dict]:
    """For UIs: everything a wand can grow into."""
    return [{k: v for k, v in r.items()} for r in _data("signatures.json")["signatures"]]
