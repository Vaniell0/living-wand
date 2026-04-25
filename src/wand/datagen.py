"""Training data generator for SpellFormer.

Generates random spell graphs, evaluates them through algebra.py,
and converts (SpellGraph, SpellEffect) → (TokenizedSpell, target_vector).

Phases:
  1. Random graphs (uniform) — bulk of training data
  2. Biased toward synergies/cancellations — interesting spells
  3. Near-duplicates with perturbations — smooth interpolation
  4. Edge cases — open circles, disconnected, single-node
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

from .algebra import (
    EdgeType,
    Element,
    SignalKind,
    SpellGraph,
    SpellEffect,
    evaluate_spell,
)
from .tokenizer import SpellTokenizer, TokenizedSpell
from .tags import TAG_MATRIX, ALL_TAGS, SEMANTIC_DIM
from .transformer import EFFECT_DIM


# ═══════════════════════════════════════════
# Rune catalog (name → (SignalKind, Element|None))
# ═══════════════════════════════════════════

RUNE_CATALOG: list[tuple[str, SignalKind, Element | None]] = [
    # Elements
    ("fire", SignalKind.FORCE, Element.FIRE),
    ("water", SignalKind.FORCE, Element.WATER),
    ("air", SignalKind.FORCE, Element.AIR),
    ("earth", SignalKind.FORCE, Element.EARTH),
    ("light", SignalKind.FORCE, Element.LIGHT),
    ("dark", SignalKind.FORCE, Element.DARK),
    # Domains
    ("body", SignalKind.DOMAIN, None),
    ("mind", SignalKind.DOMAIN, None),
    ("matter", SignalKind.DOMAIN, None),
    ("space", SignalKind.DOMAIN, None),
    ("time", SignalKind.DOMAIN, None),
    ("life", SignalKind.DOMAIN, None),
    # Directions
    ("inward", SignalKind.VECTOR, None),
    ("outward", SignalKind.VECTOR, None),
    ("up", SignalKind.VECTOR, None),
    ("down", SignalKind.VECTOR, None),
    ("through", SignalKind.VECTOR, None),
    ("reverse", SignalKind.VECTOR, None),
    # Quantity
    ("point", SignalKind.SCALAR, None),
    ("flow", SignalKind.SCALAR, None),
    ("wave", SignalKind.SCALAR, None),
    ("all", SignalKind.SCALAR, None),
    # Modifiers
    ("amplify", SignalKind.MODIFIER, None),
    ("weaken", SignalKind.MODIFIER, None),
    ("hold", SignalKind.MODIFIER, None),
    ("dispel", SignalKind.MODIFIER, None),
    # Meta
    ("here", SignalKind.REFERENCE, None),
    ("there", SignalKind.REFERENCE, None),
    ("self", SignalKind.REFERENCE, None),
    ("target", SignalKind.REFERENCE, None),
]

# Category weights for sampling (not uniform — more elements/domains, fewer meta)
_CATEGORY_WEIGHTS = {
    SignalKind.FORCE: 3.0,
    SignalKind.DOMAIN: 3.0,
    SignalKind.VECTOR: 2.0,
    SignalKind.SCALAR: 1.0,
    SignalKind.MODIFIER: 2.0,
    SignalKind.REFERENCE: 1.0,
}

# Edge types and their relative frequency
EDGE_TYPES = list(EdgeType)
EDGE_WEIGHTS = [3.0, 3.0, 1.0, 0.5, 0.5, 1.0, 1.0]  # DIRECT, MUTUAL most common

# Domain name → pole index mapping
DOMAIN_POLE_MAP = {
    "body": 2,     # BIOLOGY
    "mind": 3,     # MENTAL
    "matter": 1,   # TRANSFORM
    "space": 4,    # SPACE
    "time": 5,     # TIME
    "life": 2,     # BIOLOGY
}


# ═══════════════════════════════════════════
# Semantic target computation
# ═══════════════════════════════════════════

def _tags_to_semantic(tags: set[str], force_power: dict, domain_power: dict) -> np.ndarray:
    """Convert spell tags + powers to a 16D semantic embedding.

    Strategy: compute a weighted average of the closest tag vectors
    based on keyword matching, element presence, and domain presence.
    Now covers combo/cross/modifier_fx tag domains too.
    """
    result = np.zeros(SEMANTIC_DIM, dtype=np.float32)
    weight_sum = 0.0

    # Pre-normalize spell tags for matching
    spell_tags_lower = {t.lower().replace("-", "_") for t in tags}

    for tag_obj in ALL_TAGS:
        score = 0.0
        tag_name_lower = tag_obj.name.lower()

        # 1. Direct/substring match between spell tags and tag name
        for t_lower in spell_tags_lower:
            if t_lower == tag_name_lower:
                score += 2.0
            elif t_lower in tag_name_lower or tag_name_lower in t_lower:
                score += 0.8

        # 2. Element match
        if tag_obj.domain in ("elemental", "combo"):
            for elem, power in force_power.items():
                if power > 0.3:
                    elem_name = elem.name.lower()
                    if elem_name in tag_name_lower:
                        score += 0.5

        # 3. Domain match
        if tag_obj.domain in ("biology", "mental", "space", "time", "transform",
                              "cross", "comfort"):
            for dom_name, power in domain_power.items():
                if power > 0.3:
                    if dom_name in tag_obj.domain or dom_name in tag_name_lower:
                        score += 0.5

        # 4. Contextual boosts
        for t in spell_tags_lower:
            if t == "reinforced" and tag_name_lower == "strengthen":
                score += 0.5
            if t == "self_cast" and tag_name_lower in ("healing", "regeneration", "rest"):
                score += 0.3
            if t == "ranged" and tag_name_lower in ("attack", "signaling", "projecting"):
                score += 0.3
            if t == "local" and tag_name_lower in ("cleaning", "repair", "shelter"):
                score += 0.3
            if t == "focused" and tag_name_lower in ("cutting", "polishing", "diagnosis"):
                score += 0.3
            if t == "sustained" and tag_name_lower in ("warming", "lighting", "growing"):
                score += 0.3
            if t == "infused" and "infus" in tag_name_lower:
                score += 0.3
            if t == "overcharged" and tag_name_lower in ("overcharged", "disruption"):
                score += 0.5
            if t == "amplified" and tag_name_lower in ("enhancement", "strengthen"):
                score += 0.3

        if score > 0:
            vec = np.array(tag_obj.vector, dtype=np.float32)
            result += score * vec
            weight_sum += score

    if weight_sum > 0:
        result /= weight_sum

    # L2-normalize
    norm = np.linalg.norm(result)
    if norm > 1e-8:
        result /= norm

    return result


# ═══════════════════════════════════════════
# SpellEffect → 38D target vector
# ═══════════════════════════════════════════

def effect_to_vector(effect: SpellEffect, graph: SpellGraph) -> np.ndarray:
    """Convert SpellEffect to a 38-dim target vector for training.

    Layout:
      [0:6]   pole_distribution (normalized)
      [6]     power (normalized to [0,1])
      [7]     stability [0,1]
      [8]     area (raw, clamped)
      [9]     duration (raw, clamped)
      [10:16] element_intensity [0,1] x 6
      [16:22] domain_intensity [0,1] x 6
      [22:38] semantic_embedding (16D)
    """
    vec = np.zeros(EFFECT_DIM, dtype=np.float32)

    # --- Pole distribution (6D) ---
    pole_scores = np.zeros(6, dtype=np.float32)

    # Element forces → ELEMENT pole (0)
    for elem, power in effect.force_power.items():
        if power > 0:
            pole_scores[0] += power

    # Domain powers → respective poles
    for dom_name, power in effect.domain_power.items():
        if power > 0:
            pole_idx = DOMAIN_POLE_MAP.get(dom_name, 0)
            pole_scores[pole_idx] += power

    # Normalize to distribution
    total = pole_scores.sum()
    if total > 0:
        vec[:6] = pole_scores / total
    else:
        vec[:6] = 1.0 / 6.0  # uniform

    # --- Power [0,1] ---
    # Normalize: typical total_power range is 0-10, sigmoid-like mapping
    vec[6] = min(effect.total_power / (effect.total_power + 3.0), 1.0)

    # --- Stability [0,1] ---
    vec[7] = max(0.0, min(1.0, effect.stability))

    # --- Area (clamped to reasonable range) ---
    vec[8] = max(0.0, min(10.0, effect.area))

    # --- Duration (clamped) ---
    vec[9] = max(0.0, min(10.0, effect.duration))

    # --- Element intensity [0,1] x 6 ---
    max_elem = max(effect.force_power.values(), default=0.0)
    if max_elem > 0:
        for elem, power in effect.force_power.items():
            vec[10 + elem.value] = max(0.0, min(1.0, power / max(max_elem, 1.0)))

    # --- Domain intensity [0,1] x 6 ---
    domain_names = ["body", "mind", "matter", "space", "time", "life"]
    max_dom = max(effect.domain_power.values(), default=0.0)
    for i, dname in enumerate(domain_names):
        power = effect.domain_power.get(dname, 0.0)
        if max_dom > 0:
            vec[16 + i] = max(0.0, min(1.0, power / max(max_dom, 1.0)))

    # --- Semantic embedding (16D) ---
    vec[22:38] = _tags_to_semantic(effect.tags, effect.force_power, effect.domain_power)

    return vec


# ═══════════════════════════════════════════
# Random spell generation
# ═══════════════════════════════════════════

def _sample_runes(n: int) -> list[tuple[str, SignalKind, Element | None]]:
    """Sample n runes from catalog with category weighting."""
    weights = [_CATEGORY_WEIGHTS[r[1]] for r in RUNE_CATALOG]
    return random.choices(RUNE_CATALOG, weights=weights, k=n)


def generate_random_graph(
    min_nodes: int = 2,
    max_nodes: int = 8,
    open_circle_prob: float = 0.15,
) -> SpellGraph:
    """Generate a random valid spell graph."""
    n_nodes = random.randint(min_nodes, max_nodes)
    runes = _sample_runes(n_nodes)

    graph = SpellGraph()
    node_ids = []
    for name, kind, elem in runes:
        nid = graph.add_node(name, kind, elem)
        node_ids.append(nid)

    # Ensure connectivity: build a spanning tree first
    shuffled = node_ids.copy()
    random.shuffle(shuffled)
    for i in range(1, len(shuffled)):
        edge_type = random.choices(EDGE_TYPES, weights=EDGE_WEIGHTS, k=1)[0]
        # Randomly pick direction
        if random.random() < 0.5:
            graph.connect(shuffled[i - 1], shuffled[i], edge_type)
        else:
            graph.connect(shuffled[i], shuffled[i - 1], edge_type)

    # Add extra edges (0 to n_nodes//2)
    n_extra = random.randint(0, max(1, n_nodes // 2))
    for _ in range(n_extra):
        a, b = random.sample(node_ids, 2)
        edge_type = random.choices(EDGE_TYPES, weights=EDGE_WEIGHTS, k=1)[0]
        graph.connect(a, b, edge_type)

    graph.circle_closed = random.random() > open_circle_prob
    graph.circle_radius = random.uniform(0.5, 2.0)

    return graph


def generate_synergy_graph() -> SpellGraph:
    """Generate a graph biased toward synergies/cancellations (all 15 element pairs)."""
    from .combo_data import ELEMENT_PAIRS

    pairs = list(ELEMENT_PAIRS.keys())
    pair = random.choice(pairs)
    elem_a, elem_b = pair

    graph = SpellGraph()
    a = graph.add_node(elem_a.name.lower(), SignalKind.FORCE, elem_a)
    b = graph.add_node(elem_b.name.lower(), SignalKind.FORCE, elem_b)
    graph.connect(a, b, EdgeType.MUTUAL)

    # Add 1-4 more runes for context
    n_extra = random.randint(1, 4)
    extra_runes = _sample_runes(n_extra)
    extra_ids = []
    for name, kind, elem in extra_runes:
        nid = graph.add_node(name, kind, elem)
        extra_ids.append(nid)

    for eid in extra_ids:
        target = random.choice([a, b])
        edge_type = random.choices(EDGE_TYPES, weights=EDGE_WEIGHTS, k=1)[0]
        graph.connect(target, eid, edge_type)

    graph.circle_closed = random.random() > 0.1
    return graph


def generate_element_domain_graph() -> SpellGraph:
    """Generate a graph with an element→domain or element═domain interaction."""
    from .combo_data import ELEMENT_DOMAIN

    pairs = list(ELEMENT_DOMAIN.keys())
    elem, domain_name = random.choice(pairs)

    graph = SpellGraph()
    e_id = graph.add_node(elem.name.lower(), SignalKind.FORCE, elem)
    d_id = graph.add_node(domain_name, SignalKind.DOMAIN)

    # Randomly MUTUAL or DIRECT
    if random.random() < 0.5:
        graph.connect(e_id, d_id, EdgeType.MUTUAL)
    else:
        graph.connect(e_id, d_id, EdgeType.DIRECT)

    # Add 1-3 context runes
    n_extra = random.randint(1, 3)
    extra_runes = _sample_runes(n_extra)
    for name, kind, el in extra_runes:
        nid = graph.add_node(name, kind, el)
        target = random.choice([e_id, d_id])
        edge_type = random.choices(EDGE_TYPES, weights=EDGE_WEIGHTS, k=1)[0]
        graph.connect(target, nid, edge_type)

    graph.circle_closed = random.random() > 0.1
    return graph


def generate_triple_graph() -> SpellGraph:
    """Generate a graph with a triple synergy (3 nodes all MUTUAL-connected)."""
    from .combo_data import TRIPLE_COMBOS

    triple_key = random.choice(list(TRIPLE_COMBOS.keys()))
    names = sorted(triple_key)

    graph = SpellGraph()
    node_ids = []
    for name in names:
        # Find in RUNE_CATALOG
        for rune_name, kind, elem in RUNE_CATALOG:
            if rune_name == name:
                nid = graph.add_node(rune_name, kind, elem)
                node_ids.append(nid)
                break
        else:
            # Domain fallback
            nid = graph.add_node(name, SignalKind.DOMAIN)
            node_ids.append(nid)

    # All MUTUAL (clique)
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            graph.connect(node_ids[i], node_ids[j], EdgeType.MUTUAL)

    # Add 0-3 context runes
    n_extra = random.randint(0, 3)
    extra_runes = _sample_runes(n_extra)
    for name, kind, elem in extra_runes:
        nid = graph.add_node(name, kind, elem)
        target = random.choice(node_ids)
        edge_type = random.choices(EDGE_TYPES, weights=EDGE_WEIGHTS, k=1)[0]
        graph.connect(target, nid, edge_type)

    graph.circle_closed = random.random() > 0.1
    return graph


def generate_modifier_combo_graph() -> SpellGraph:
    """Generate a graph where two modifiers target the same node."""
    from .combo_data import MODIFIER_COMBOS, SAME_MODIFIER_COMBOS

    # Pick: same-modifier combo or different-modifier combo
    if random.random() < 0.3 and SAME_MODIFIER_COMBOS:
        mod_name = random.choice(list(SAME_MODIFIER_COMBOS.keys()))
        mod_names = [mod_name, mod_name]
    else:
        pair_key = random.choice(list(MODIFIER_COMBOS.keys()))
        mod_names = list(pair_key)

    # Pick a target rune
    target_rune = random.choice(RUNE_CATALOG[:12])  # elements + domains

    graph = SpellGraph()
    t_id = graph.add_node(target_rune[0], target_rune[1], target_rune[2])
    for mn in mod_names:
        m_id = graph.add_node(mn, SignalKind.MODIFIER)
        graph.connect(m_id, t_id, EdgeType.DIRECT)

    # Add 0-2 context
    n_extra = random.randint(0, 2)
    extra_runes = _sample_runes(n_extra)
    for name, kind, elem in extra_runes:
        nid = graph.add_node(name, kind, elem)
        graph.connect(t_id, nid, random.choices(EDGE_TYPES, weights=EDGE_WEIGHTS, k=1)[0])

    graph.circle_closed = random.random() > 0.1
    return graph


def generate_recipe_graph() -> SpellGraph:
    """Generate a graph from a named recipe (with slight perturbation)."""
    from .combo_data import RECIPES

    recipe = random.choice(list(RECIPES.values()))

    graph = SpellGraph()
    name_to_id: dict[str, int] = {}

    for node_spec in recipe["nodes"]:
        name = node_spec["name"]
        kind = SignalKind[node_spec["kind"]]
        elem = Element[node_spec["element"]] if "element" in node_spec and node_spec["element"] else None
        nid = graph.add_node(name, kind, elem)
        # Handle duplicate names by using first occurrence
        if name not in name_to_id:
            name_to_id[name] = nid

    for edge_spec in recipe["edges"]:
        from_name, to_name, edge_type_name = edge_spec
        from_id = name_to_id.get(from_name)
        to_id = name_to_id.get(to_name)
        if from_id is not None and to_id is not None:
            edge_type = EdgeType[edge_type_name]
            # 5% chance to drop edge (perturbation)
            if random.random() > 0.05:
                graph.connect(from_id, to_id, edge_type)

    graph.circle_closed = recipe.get("circle_closed", True)
    # Slight radius perturbation
    graph.circle_radius = random.uniform(0.8, 1.2)

    return graph


def generate_edge_case_graph() -> SpellGraph:
    """Generate edge cases: single node, open circle, disconnected, etc."""
    case = random.choice(["single", "open", "many_edges", "all_same"])

    if case == "single":
        graph = SpellGraph()
        r = random.choice(RUNE_CATALOG)
        graph.add_node(r[0], r[1], r[2])
        graph.circle_closed = random.random() > 0.5
        return graph

    elif case == "open":
        graph = generate_random_graph(min_nodes=2, max_nodes=5, open_circle_prob=1.0)
        graph.circle_closed = False
        return graph

    elif case == "many_edges":
        graph = SpellGraph()
        runes = _sample_runes(4)
        ids = [graph.add_node(r[0], r[1], r[2]) for r in runes]
        # Connect everything to everything
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                edge_type = random.choices(EDGE_TYPES, weights=EDGE_WEIGHTS, k=1)[0]
                graph.connect(ids[i], ids[j], edge_type)
        return graph

    else:  # all_same
        graph = SpellGraph()
        elem = random.choice(RUNE_CATALOG[:6])  # elements only
        for _ in range(random.randint(2, 4)):
            graph.add_node(elem[0], elem[1], elem[2])
        ids = list(graph.nodes.keys())
        for i in range(1, len(ids)):
            graph.connect(ids[i - 1], ids[i], EdgeType.MUTUAL)
        return graph


# ═══════════════════════════════════════════
# Dataset generation
# ═══════════════════════════════════════════

@dataclass
class TrainingSample:
    tokens: TokenizedSpell
    target: np.ndarray  # (38,)


class SpellDataGenerator:
    """Generates training data for SpellFormer."""

    def __init__(self, seed: int | None = None):
        self.tokenizer = SpellTokenizer()
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate_sample(
        self,
        graph: SpellGraph | None = None,
        noise_std: float = 0.02,
    ) -> TrainingSample | None:
        """Generate a single (tokens, target) pair.

        Returns None if the graph produces trivially empty results.
        """
        if graph is None:
            graph = generate_random_graph()

        # Tokenize
        tokens = self.tokenizer.tokenize(graph)
        if tokens.node_count == 0:
            return None

        # Evaluate through algebra
        effect = evaluate_spell(graph)

        # Convert to target vector
        target = effect_to_vector(effect, graph)

        # Add noise for generalization
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, size=target.shape).astype(np.float32)
            # Don't add noise to pole_distribution (will be renormalized during training)
            # Don't add noise to semantic embedding (already noisy from tag matching)
            noise[:6] *= 0.5  # less noise on distribution
            noise[22:] = 0    # no noise on semantic
            target += noise
            # Clamp bounded values
            target[6:8] = np.clip(target[6:8], 0, 1)  # power, stability
            target[8:10] = np.clip(target[8:10], 0, 10)  # area, duration
            target[10:22] = np.clip(target[10:22], 0, 1)  # intensities

        return TrainingSample(tokens=tokens, target=target)

    def generate_dataset(
        self,
        n_random: int = 80_000,
        n_synergy: int = 30_000,
        n_element_domain: int = 15_000,
        n_triple: int = 15_000,
        n_recipe: int = 10_000,
        n_modifier: int = 10_000,
        n_perturbed: int = 10_000,
        n_edge: int = 10_000,
        noise_std: float = 0.02,
    ) -> list[TrainingSample]:
        """Generate a full training dataset in 8 phases.

        Default: ~180K samples, distributed across combo types.
        """
        samples: list[TrainingSample] = []

        # Phase 1: Random graphs
        for _ in range(n_random):
            s = self.generate_sample(noise_std=noise_std)
            if s is not None:
                samples.append(s)

        # Phase 2: Element synergy/cancel/transform (all 15 pairs)
        for _ in range(n_synergy):
            graph = generate_synergy_graph()
            s = self.generate_sample(graph, noise_std=noise_std)
            if s is not None:
                samples.append(s)

        # Phase 3: Element × Domain (36 combos)
        for _ in range(n_element_domain):
            graph = generate_element_domain_graph()
            s = self.generate_sample(graph, noise_std=noise_std)
            if s is not None:
                samples.append(s)

        # Phase 4: Triple synergies (28 combos)
        for _ in range(n_triple):
            graph = generate_triple_graph()
            s = self.generate_sample(graph, noise_std=noise_std)
            if s is not None:
                samples.append(s)

        # Phase 5: Named recipes (23 spells with perturbation)
        for _ in range(n_recipe):
            graph = generate_recipe_graph()
            s = self.generate_sample(graph, noise_std=noise_std)
            if s is not None:
                samples.append(s)

        # Phase 6: Modifier combos
        for _ in range(n_modifier):
            graph = generate_modifier_combo_graph()
            s = self.generate_sample(graph, noise_std=noise_std)
            if s is not None:
                samples.append(s)

        # Phase 7: Near-duplicates (take existing, slightly perturb)
        base_graphs = [generate_random_graph() for _ in range(n_perturbed // 5)]
        for _ in range(n_perturbed):
            graph = random.choice(base_graphs)
            perturbed = SpellGraph()
            for nid, node in graph.nodes.items():
                perturbed.add_node(node.name, node.signal_kind, node.element)
            for edge in graph.edges:
                if random.random() > 0.05:
                    perturbed.connect(edge.from_id, edge.to_id, edge.edge_type)
            perturbed.circle_closed = graph.circle_closed if random.random() > 0.1 else not graph.circle_closed
            perturbed.circle_radius = graph.circle_radius * random.uniform(0.8, 1.2)
            s = self.generate_sample(perturbed, noise_std=noise_std * 1.5)
            if s is not None:
                samples.append(s)

        # Phase 8: Edge cases
        for _ in range(n_edge):
            graph = generate_edge_case_graph()
            s = self.generate_sample(graph, noise_std=noise_std * 0.5)
            if s is not None:
                samples.append(s)

        return samples


def samples_to_tensors(
    samples: list[TrainingSample],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert list of TrainingSamples to batched tensors.

    Returns:
        (token_ids, role_ids, degree, targets) — all (N, ...) tensors
    """
    token_ids = torch.tensor([s.tokens.token_ids for s in samples], dtype=torch.long)
    role_ids = torch.tensor([s.tokens.role_ids for s in samples], dtype=torch.long)
    degree = torch.tensor([s.tokens.degree for s in samples], dtype=torch.long)
    targets = torch.tensor(np.stack([s.target for s in samples]), dtype=torch.float32)
    return token_ids, role_ids, degree, targets
