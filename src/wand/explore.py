"""Exploring the space of spells — tools for designers, agents and RL.

The spell space is a discrete, combinatorial, multi-dimensional space: an
agent with a search tree will map it far faster than a player. That is a
design risk (the meta gets "solved") and a design tool (you can see it
coming). This module gives you both sides:

    SpellGenome / mutate()      — editable spell + local edit operators
    evaluate()                  — spell → SpellProgram + feature vector
    map_elites()                — quality-diversity atlas: the best spell in
                                  every niche (primary verb × delivery × size)
    beam_search()               — inverse design / tree search: "find runes for
                                  a piercing fire projectile with stability>0.8"
    exploit_report()            — dominant runes, dead verbs, scaling exploits,
                                  power-per-rune outliers, coverage of the catalog
    SpellEnv                    — gym-style env (reset/step) for RL agents,
                                  no gym dependency; observation = token ids
    atlas_points()              — 2-D projection of explored spells (PCA)

Everything runs on the deterministic algebra by default (~0.25 ms/spell);
pass `session=` to search through the trained model + a personal wand.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

from .actions import SpellProgram, compile_program, load_verbs
from .algebra import RUNES, EdgeType, SpellGraph, evaluate_spell
from .checker import check_graph

RUNE_NAMES = list(RUNES)
EDGE_NAMES = [e.name for e in EdgeType]
VERB_NAMES = sorted(load_verbs()["verbs"])
DELIVERY_KINDS = ["instant", "projectile", "beam", "zone", "aura", "touch", "consume", "splash", "imbue"]


# ═══════════════════════════════════════════
# Genome
# ═══════════════════════════════════════════

@dataclass(frozen=True)
class SpellGenome:
    """Immutable, hashable spell description: runes + edges by index."""
    runes: tuple[str, ...]
    edges: tuple[tuple[int, int, str], ...] = ()
    circle_closed: bool = True

    def graph(self) -> SpellGraph:
        g = SpellGraph(circle_closed=self.circle_closed)
        ids = [g.add_rune(r) for r in self.runes]
        for a, b, t in self.edges:
            g.connect(ids[a], ids[b], EdgeType[t])
        return g

    def key(self) -> tuple:
        """Order-independent identity (same spell drawn differently)."""
        edges = []
        for a, b, t in self.edges:
            ra, rb = self.runes[a], self.runes[b]
            if t == "MUTUAL":
                ra, rb = sorted((ra, rb))
            edges.append((ra, t, rb))
        return (tuple(sorted(self.runes)), tuple(sorted(edges)), self.circle_closed)

    def describe(self) -> str:
        parts = [f"{self.runes[a]}-{t[:3].lower()}->{self.runes[b]}" for a, b, t in self.edges]
        loose = [r for i, r in enumerate(self.runes)
                 if not any(i in (a, b) for a, b, _ in self.edges)]
        circ = "" if self.circle_closed else " (open)"
        return " ".join(parts + loose) + circ

    def to_spec(self) -> dict:
        return self.graph().to_spec()

    @classmethod
    def from_graph(cls, g: SpellGraph) -> "SpellGenome":
        order = sorted(g.nodes)
        idx = {nid: i for i, nid in enumerate(order)}
        return cls(tuple(g.nodes[n].name for n in order),
                   tuple((idx[e.from_id], idx[e.to_id], e.edge_type.name) for e in g.edges),
                   g.circle_closed)


def random_genome(rng: random.Random, min_runes: int = 2, max_runes: int = 6) -> SpellGenome:
    n = rng.randint(min_runes, max_runes)
    runes = tuple(rng.choice(RUNE_NAMES) for _ in range(n))
    order = list(range(n))
    rng.shuffle(order)
    edges = []
    for i in range(1, n):  # spanning tree keeps it connected
        a, b = order[rng.randrange(i)], order[i]
        if rng.random() < 0.5:
            a, b = b, a
        edges.append((a, b, rng.choice(["DIRECT", "DIRECT", "MUTUAL", "MUTUAL"] + EDGE_NAMES)))
    return SpellGenome(runes, tuple(edges), rng.random() > 0.05)


def mutate(gen: SpellGenome, rng: random.Random, max_runes: int = 8) -> SpellGenome:
    """One local edit — the 'moves' an agent can make."""
    runes, edges, circle = list(gen.runes), list(gen.edges), gen.circle_closed
    op = rng.choice(["add_rune", "drop_rune", "swap_rune", "add_edge", "drop_edge",
                     "retype_edge", "flip_edge", "circle"])
    if op == "add_rune" and len(runes) < max_runes:
        runes.append(rng.choice(RUNE_NAMES))
        if len(runes) > 1:
            other = rng.randrange(len(runes) - 1)
            a, b = (len(runes) - 1, other) if rng.random() < 0.5 else (other, len(runes) - 1)
            edges.append((a, b, rng.choice(EDGE_NAMES)))
    elif op == "drop_rune" and len(runes) > 1:
        k = rng.randrange(len(runes))
        runes.pop(k)
        edges = [(a - (a > k), b - (b > k), t) for a, b, t in edges if k not in (a, b)]
    elif op == "swap_rune" and runes:
        runes[rng.randrange(len(runes))] = rng.choice(RUNE_NAMES)
    elif op == "add_edge" and len(runes) > 1:
        a, b = rng.sample(range(len(runes)), 2)
        edges.append((a, b, rng.choice(EDGE_NAMES)))
    elif op == "drop_edge" and edges:
        edges.pop(rng.randrange(len(edges)))
    elif op == "retype_edge" and edges:
        k = rng.randrange(len(edges))
        a, b, _ = edges[k]
        edges[k] = (a, b, rng.choice(EDGE_NAMES))
    elif op == "flip_edge" and edges:
        k = rng.randrange(len(edges))
        a, b, t = edges[k]
        edges[k] = (b, a, t)
    elif op == "circle":
        circle = not circle
    return SpellGenome(tuple(runes), tuple(edges), circle)


# ═══════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════

@dataclass
class Evaluated:
    genome: SpellGenome
    program: SpellProgram
    features: np.ndarray

    @property
    def niche(self) -> tuple[str, str, int]:
        p = self.program
        size = 0 if p.shape.kind == "single" else (1 if p.shape.radius < 2.0 else 2)
        return (p.primary_verb, p.delivery.kind, size)


def features(p: SpellProgram) -> np.ndarray:
    """Fixed-length numeric view of a program (for atlases / novelty)."""
    verb = np.zeros(len(VERB_NAMES), dtype=np.float32)
    for op in p.ops:
        verb[VERB_NAMES.index(op.verb)] += op.amount
    deliv = np.zeros(len(DELIVERY_KINDS), dtype=np.float32)
    deliv[DELIVERY_KINDS.index(p.delivery.kind)] = 1.0
    scal = np.array([p.power, p.stability, min(p.area / 5, 2), min(p.duration / 5, 2),
                     p.risk.backfire_chance, p.shape.radius / 10], dtype=np.float32)
    return np.concatenate([verb, deliv, scal])


def evaluate(genome: SpellGenome, session=None) -> Evaluated:
    graph = genome.graph()
    if session is not None:
        program = session.cast(graph, record=False).program
    else:
        program = compile_program(graph, evaluate_spell(graph), check=check_graph(graph))
    return Evaluated(genome, program, features(program))


# Built-in objectives. Each maps a program to a scalar to MAXIMISE.
def efficiency(p: SpellProgram) -> float:
    """What a min-maxing player wants: reliable power per mana."""
    return p.power * (1 - p.risk.backfire_chance) * p.stability / (1.0 + p.cost / 20.0)


def raw_power(p: SpellProgram) -> float:
    return p.power * (1 - p.risk.backfire_chance)


def power_per_rune(p: SpellProgram) -> float:
    return raw_power(p) / max(len(p.runes), 1)


OBJECTIVES: dict[str, Callable[[SpellProgram], float]] = {
    "efficiency": efficiency,
    "power": raw_power,
    "power_per_rune": power_per_rune,
}


def model_disagreement(p: SpellProgram) -> float:
    """Where the network departs from the algebra (needs session with a model).

    High values are training-data holes — or candidates for 'wild magic'.
    """
    a = p.provenance.get("agreement")
    return 0.0 if a is None else 1.0 - a


# ═══════════════════════════════════════════
# MAP-Elites
# ═══════════════════════════════════════════

@dataclass
class Archive:
    elites: dict[tuple, Evaluated] = field(default_factory=dict)
    scores: dict[tuple, float] = field(default_factory=dict)
    seen: list[Evaluated] = field(default_factory=list)
    evaluations: int = 0
    history: list[tuple[int, int, float]] = field(default_factory=list)  # (evals, cells, qd)

    @property
    def qd_score(self) -> float:
        return float(sum(self.scores.values()))

    def top(self, k: int = 10) -> list[tuple[float, Evaluated]]:
        return sorted(((self.scores[c], e) for c, e in self.elites.items()),
                      key=lambda t: -t[0])[:k]


def map_elites(
    iterations: int = 5000,
    objective: Callable[[SpellProgram], float] = efficiency,
    seed: int = 0,
    init: int = 300,
    session=None,
    keep_seen: int = 4000,
    seeds: Iterable[SpellGenome] = (),
) -> Archive:
    """Quality-diversity search: best spell for every (verb, delivery, size) niche."""
    rng = random.Random(seed)
    arch = Archive()
    seen_keys: set = set()

    def consider(gen: SpellGenome):
        k = gen.key()
        if k in seen_keys or not gen.runes:
            return
        seen_keys.add(k)
        ev = evaluate(gen, session)
        arch.evaluations += 1
        if len(arch.seen) < keep_seen:
            arch.seen.append(ev)
        score = objective(ev.program)
        cell = ev.niche
        if cell not in arch.scores or score > arch.scores[cell]:
            arch.elites[cell] = ev
            arch.scores[cell] = score

    for g in seeds:
        consider(g)
    for _ in range(init):
        consider(random_genome(rng))
    for i in range(iterations):
        if arch.elites and rng.random() < 0.9:
            parent = rng.choice(list(arch.elites.values())).genome
            child = mutate(parent, rng)
            if rng.random() < 0.3:
                child = mutate(child, rng)
        else:
            child = random_genome(rng)
        consider(child)
        if i % 250 == 0:
            arch.history.append((arch.evaluations, len(arch.elites), arch.qd_score))
    arch.history.append((arch.evaluations, len(arch.elites), arch.qd_score))
    return arch


# ═══════════════════════════════════════════
# Tree / beam search — inverse design
# ═══════════════════════════════════════════

def beam_search(
    objective: Callable[[SpellProgram], float],
    width: int = 24,
    depth: int = 6,
    expansions: int = 40,
    seed: int = 0,
    session=None,
    start: Iterable[SpellGenome] = (),
) -> list[tuple[float, Evaluated]]:
    """Grow spells rune-by-rune, keeping the `width` best at every depth.

    `objective` encodes the request, e.g. "a fire projectile that pierces":
        lambda p: p.power * (p.delivery.piercing) * (p.primary_verb == "damage")
    Returns the best spells found (score, evaluated), best first.
    """
    rng = random.Random(seed)
    frontier = [evaluate(g, session) for g in start] or [
        evaluate(SpellGenome((r,)), session) for r in RUNE_NAMES]
    best: dict[tuple, tuple[float, Evaluated]] = {}
    for ev in frontier:
        best[ev.genome.key()] = (objective(ev.program), ev)

    for _ in range(depth):
        children: dict[tuple, Evaluated] = {}
        for ev in frontier:
            for _ in range(expansions):
                child = mutate(ev.genome, rng)
                k = child.key()
                if k in best or k in children or not child.runes:
                    continue
                children[k] = evaluate(child, session)
        scored = sorted(((objective(c.program), c) for c in children.values()),
                        key=lambda t: -t[0])
        for s, c in scored:
            best[c.genome.key()] = (s, c)
        frontier = [c for _, c in scored[:width]] or frontier
    return sorted(best.values(), key=lambda t: -t[0])[:width]


# ═══════════════════════════════════════════
# Exploit / balance report
# ═══════════════════════════════════════════

def scaling_probe(base: SpellGenome, rune: str, steps: int = 6, session=None) -> list[float]:
    """Power as `rune` is stacked onto the strongest node again and again.

    A curve that keeps climbing is an exploit (unbounded stacking).
    """
    curve = []
    g = base
    target = 0
    for _ in range(steps):
        ev = evaluate(g, session)
        curve.append(raw_power(ev.program))
        runes = list(g.runes) + [rune]
        edges = list(g.edges) + [(len(runes) - 1, target, "DIRECT")]
        g = SpellGenome(tuple(runes), tuple(edges), g.circle_closed)
    return curve


def exploit_report(arch: Archive, session=None) -> dict:
    elites = list(arch.elites.values())
    seen = arch.seen or elites
    rune_use = Counter(r for e in elites for r in set(e.genome.runes))
    primary = Counter(e.program.primary_verb for e in seen)
    dead_verbs = [v for v in VERB_NAMES if primary.get(v, 0) == 0]
    never_elite = [r for r in RUNE_NAMES if rune_use.get(r, 0) == 0]

    ppr = sorted(seen, key=lambda e: -power_per_rune(e.program))[:8]
    ppr_vals = np.array([power_per_rune(e.program) for e in seen]) if seen else np.zeros(1)

    # per-verb dominance: how far the best is above the typical spell
    dominance = []
    by_verb: dict[str, list[float]] = {}
    for e in seen:
        by_verb.setdefault(e.program.primary_verb, []).append(efficiency(e.program))
    for verb, vals in by_verb.items():
        if len(vals) >= 5:
            dominance.append((verb, float(max(vals) / (np.median(vals) + 1e-6)), len(vals)))
    dominance.sort(key=lambda t: -t[1])

    stacking = {}
    for rune in ("amplify", "hold", "flow", "all"):
        curve = scaling_probe(SpellGenome(("fire",)), rune, session=session)
        stacking[rune] = [round(c, 4) for c in curve]

    from .combo_data import DISCOVERY_CATALOG
    reached = set()
    for e in seen:
        rs = set(e.genome.runes)
        for d in DISCOVERY_CATALOG.values():
            if set(d.runes) <= rs:
                reached.add(d.key)

    return {
        "evaluations": arch.evaluations,
        "niches_filled": len(arch.elites),
        "qd_score": round(arch.qd_score, 4),
        "dominant_runes": rune_use.most_common(8),
        "never_in_elites": never_elite,
        "primary_verb_share": {v: round(n / max(len(seen), 1), 4) for v, n in primary.most_common()},
        "dead_verbs": dead_verbs,
        "power_per_rune_outliers": [
            {"spell": e.genome.describe(), "ppr": round(power_per_rune(e.program), 4),
             "z": round(float((power_per_rune(e.program) - ppr_vals.mean()) / (ppr_vals.std() + 1e-9)), 2)}
            for e in ppr],
        "verb_dominance": [{"verb": v, "best_over_median": round(r, 2), "n": n}
                           for v, r, n in dominance[:8]],
        "stacking_curves": stacking,
        "catalog_coverage": round(len(reached) / max(len(DISCOVERY_CATALOG), 1), 4),
    }


# ═══════════════════════════════════════════
# Atlas (multi-dimensional → 2-D)
# ═══════════════════════════════════════════

def atlas_points(evs: list[Evaluated]) -> tuple[np.ndarray, np.ndarray]:
    """PCA of feature vectors → (N, 2) points and explained variance ratio."""
    X = np.stack([e.features for e in evs]).astype(np.float64)
    X = X - X.mean(0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    pts = X @ Vt[:2].T
    var = (S ** 2) / max((S ** 2).sum(), 1e-12)
    return pts, var[:2]


# ═══════════════════════════════════════════
# Gym-style environment for RL agents
# ═══════════════════════════════════════════

class SpellEnv:
    """Build a spell step by step. No gym dependency; same reset/step contract.

    Action = (kind, a, b, c) as a flat int via `encode_action` / `decode_action`:
        kind 0: add rune a                 (a ∈ runes)
        kind 1: connect node a → node b with edge type c
        kind 2: toggle circle
        kind 3: stop (cast)
    Observation: dict(tokens=list[int], roles, degree) — the SpellFormer input,
    so an agent can share the world model's encoder.
    Reward: `reward_fn(program)` on cast; small step cost otherwise.
    """

    MAX_NODES = 8

    def __init__(self, reward_fn: Callable[[SpellProgram], float] = efficiency,
                 max_steps: int = 16, step_cost: float = 0.002, session=None):
        from .tokenizer import SpellTokenizer
        self.reward_fn = reward_fn
        self.max_steps = max_steps
        self.step_cost = step_cost
        self.session = session
        self.tok = SpellTokenizer()
        self.n_runes = len(RUNE_NAMES)
        self.n_edges = len(EDGE_NAMES)
        self.action_space_n = (self.n_runes
                               + self.MAX_NODES * self.MAX_NODES * self.n_edges + 2)
        self.reset()

    def encode_action(self, kind: int, a: int = 0, b: int = 0, c: int = 0) -> int:
        if kind == 0:
            return a
        if kind == 1:
            return self.n_runes + (a * self.MAX_NODES + b) * self.n_edges + c
        if kind == 2:
            return self.action_space_n - 2
        return self.action_space_n - 1

    def decode_action(self, action: int) -> tuple[int, int, int, int]:
        if action < self.n_runes:
            return 0, action, 0, 0
        if action == self.action_space_n - 2:
            return 2, 0, 0, 0
        if action == self.action_space_n - 1:
            return 3, 0, 0, 0
        x = action - self.n_runes
        c = x % self.n_edges
        ab = x // self.n_edges
        return 1, ab // self.MAX_NODES, ab % self.MAX_NODES, c

    def valid_actions(self) -> list[int]:
        n = len(self.genome.runes)
        acts = []
        if n < self.MAX_NODES:
            acts += list(range(self.n_runes))
        for a in range(n):
            for b in range(n):
                if a != b:
                    acts += [self.encode_action(1, a, b, c) for c in range(self.n_edges)]
        acts.append(self.encode_action(2))
        if n:
            acts.append(self.encode_action(3))
        return acts

    def reset(self, seed: int | None = None):
        self.genome = SpellGenome(())
        self.steps = 0
        return self._obs()

    def _obs(self) -> dict:
        t = self.tok.tokenize(self.genome.graph())
        return {"tokens": t.token_ids, "roles": t.role_ids, "degree": t.degree,
                "n_nodes": len(self.genome.runes)}

    def step(self, action: int):
        kind, a, b, c = self.decode_action(action)
        g = self.genome
        self.steps += 1
        info: dict = {}
        if kind == 0 and len(g.runes) < self.MAX_NODES:
            self.genome = SpellGenome(g.runes + (RUNE_NAMES[a],), g.edges, g.circle_closed)
        elif kind == 1 and a < len(g.runes) and b < len(g.runes) and a != b:
            self.genome = SpellGenome(g.runes, g.edges + ((a, b, EDGE_NAMES[c]),), g.circle_closed)
        elif kind == 2:
            self.genome = SpellGenome(g.runes, g.edges, not g.circle_closed)
        done = kind == 3 or self.steps >= self.max_steps
        reward = -self.step_cost
        if done and self.genome.runes:
            ev = evaluate(self.genome, self.session)
            reward += self.reward_fn(ev.program)
            info["program"] = ev.program
            info["spell"] = self.genome.describe()
        return self._obs(), reward, done, info
