"""Effect interpretation — converts SpellFormer output to game semantics.

Takes a 38-dim effect vector and produces:
  - Human-readable description (Russian)
  - Particle/visualization parameters
  - Game-mechanical values (damage, healing, shield, utility)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .events import Pole
from .tags import (
    ALL_TAGS,
    SEMANTIC_DIM,
    SemanticTag,
    nearest_tags,
    tags_for_domain,
)

# Element names (indexed by element_intensity position)
ELEMENT_NAMES = ["fire", "water", "air", "earth", "light", "dark"]
ELEMENT_NAMES_RU = ["огонь", "вода", "воздух", "земля", "свет", "тьма"]

# Domain names (indexed by domain_intensity position)
DOMAIN_NAMES = ["body", "mind", "matter", "space", "time", "life"]
DOMAIN_NAMES_RU = ["тело", "разум", "материя", "пространство", "время", "жизнь"]

# Pole names
POLE_NAMES_RU = ["стихия", "трансформация", "биология", "ментал", "пространство", "время"]

# Particle colors per element (RGB)
ELEMENT_COLORS = {
    "fire":  (255, 120, 30),
    "water": (50, 150, 255),
    "air":   (200, 220, 255),
    "earth": (140, 100, 60),
    "light": (255, 255, 200),
    "dark":  (40, 20, 60),
}

# Particle behaviors per element
ELEMENT_BEHAVIORS = {
    "fire":  "rise",
    "water": "fall",
    "air":   "orbit",
    "earth": "pulse",
    "light": "scatter",
    "dark":  "implode",
}

# Particle colors per domain (RGB) — life близок к воде через бирюзовый
DOMAIN_COLORS = {
    "body":   (220, 120, 140),  # розовый — плоть
    "mind":   (160, 110, 220),  # фиолетовый — разум
    "matter": (170, 140, 90),   # охра — материя
    "space":  (90, 130, 220),   # индиго — пространство
    "time":   (230, 200, 90),   # янтарь — время
    "life":   (90, 210, 160),   # бирюзово-зелёный — жизнь
}

# Particle behaviors per domain
DOMAIN_BEHAVIORS = {
    "body":   "pulse",
    "mind":   "scatter",
    "matter": "default",
    "space":  "orbit",
    "time":   "orbit",
    "life":   "rise",
}

# Color per semantic tag.domain (None = use element/combo blend instead)
TAG_DOMAIN_COLORS: dict[str, tuple[int, int, int] | None] = {
    "household":     (200, 200, 220),
    "cooking":       (240, 180, 100),
    "crafting":      (200, 160, 100),
    "nature":        (120, 200, 110),
    "communication": (180, 220, 240),
    "navigation":    (160, 220, 200),
    "comfort":       (240, 210, 180),
    "combat":        (220, 80, 60),
    "elemental":     None,
    "biology":       (140, 230, 160),  # жизнь+исцеление
    "mental":        (180, 130, 230),
    "space":         (120, 150, 230),
    "time":          (240, 210, 100),
    "transform":     (210, 170, 230),
    "combo":         None,
    "cross":         (230, 200, 130),
    "modifier_fx":   (180, 180, 200),
}

# ═══════════════════════════════════════════
# Tag → game mechanic mappings
# ═══════════════════════════════════════════

DAMAGE_TAGS = {
    "attack", "burning", "disruption", "storm", "hellfire", "lava", "poison",
    "earthquake", "flash", "cutting", "volcano", "plague", "rage",
    "freezing_elem", "darkness", "void_wind", "dust_storm", "flood",
}
HEALING_TAGS = {
    "healing", "regeneration", "purification", "blessing", "phoenix",
    "nurture", "growing", "rest", "garden", "oasis", "vitality",
    "rejuvenation", "calm",
}
SHIELD_TAGS = {
    "defense", "barrier", "shelter", "wall", "hardening", "stasis",
    "suppress", "null_zone", "reservoir",
}
UTILITY_TAGS = {
    "cleaning", "lighting", "warming", "cooling", "compass", "pathfinding",
    "mapping", "signaling", "recording", "translating", "drying", "repair",
    "preserving", "cooking", "heating", "shaping", "joining", "polishing",
    "watering", "fragrance", "silence", "enchanting", "pottery_art",
}


@dataclass
class EffectDescription:
    """Interpreted spell effect for display and gameplay."""
    # Core values
    power: float          # [0, 1]
    stability: float      # [0, 1]
    area: float           # spatial extent
    duration: float       # temporal extent

    # Poles
    pole_distribution: list[float]  # 6 floats
    dominant_pole: int              # index
    dominant_pole_name: str         # Russian name

    # Elements & Domains
    active_elements: list[tuple[str, float]]  # (name, intensity) pairs, sorted desc
    active_domains: list[tuple[str, float]]   # (name, intensity) pairs, sorted desc

    # Semantic
    tags: list[tuple[str, float]]  # (tag_name, similarity) top matches
    primary_domain: str             # "household", "combat", "nature", etc.

    # Text
    title: str            # short effect name
    description: str      # full Russian description

    # Visualization
    particle_colors: list[tuple[int, int, int]]  # RGB tuples (per active element)
    particle_behaviors: list[str]                  # behavior names (mixed elem+domain)
    particle_count: int
    particle_lifetime: float
    dominant_color: tuple[int, int, int]           # weighted blend of all sources
    aura_color: tuple[int, int, int]               # color from top semantic tag
    palette: list[tuple[tuple[int, int, int], float]]  # (color, weight) pairs

    # Power descriptor
    power_word: str  # "слабый", "средний", "сильный", "могущественный"

    # Game mechanics
    damage: float = 0.0
    healing: float = 0.0
    shield: float = 0.0
    utility: float = 0.0


def _power_word(power: float) -> str:
    if power < 0.2:
        return "слабый"
    elif power < 0.45:
        return "умеренный"
    elif power < 0.7:
        return "сильный"
    else:
        return "могущественный"


def _stability_word(stability: float) -> str:
    if stability < 0.3:
        return "нестабильный"
    elif stability < 0.6:
        return "неуверенный"
    elif stability < 0.85:
        return "стабильный"
    else:
        return "точный"


def _match_combo_title(
    active_elements: list[tuple[str, float]],
    active_domains: list[tuple[str, float]],
) -> tuple[str, str] | None:
    """Try to match active elements/domains to a known combo → (name_ru, description_ru).

    Priority: triples → element pairs → element×domain → domain pairs.
    """
    from .combo_data import ELEMENT_PAIRS, ELEMENT_DOMAIN, DOMAIN_PAIRS, TRIPLE_COMBOS
    from .algebra import Element

    _ELEM_MAP = {e.name.lower(): e for e in Element}

    strong_elems = [name for name, intensity in active_elements if intensity > 0.3]
    strong_doms = [name for name, intensity in active_domains if intensity > 0.3]

    # Triples
    if len(strong_elems) >= 3:
        triple_key = frozenset(strong_elems[:3])
        combo = TRIPLE_COMBOS.get(triple_key)
        if combo and combo.description:
            return combo.description.split(" — ")[0], combo.description

    # Element pairs
    if len(strong_elems) >= 2:
        for i in range(len(strong_elems)):
            for j in range(i + 1, len(strong_elems)):
                a = _ELEM_MAP.get(strong_elems[i])
                b = _ELEM_MAP.get(strong_elems[j])
                if a is not None and b is not None:
                    combo = ELEMENT_PAIRS.get(frozenset({a, b}))
                    if combo and combo.description:
                        short = combo.products[0] if combo.products else combo.description.split(" — ")[0]
                        return short, combo.description

    # Element × Domain
    if strong_elems and strong_doms:
        elem = _ELEM_MAP.get(strong_elems[0])
        if elem is not None:
            combo = ELEMENT_DOMAIN.get((elem, strong_doms[0]))
            if combo and combo.description:
                return combo.description.split(" — ")[0], combo.description

    # Domain pairs
    if len(strong_doms) >= 2:
        pair_key = frozenset(strong_doms[:2])
        combo = DOMAIN_PAIRS.get(pair_key)
        if combo and combo.description:
            return combo.description.split(" — ")[0], combo.description

    return None


def _filter_tags(
    raw_tags: list[tuple[SemanticTag, float]],
    active_elements: list[tuple[str, float]],
    active_domains: list[tuple[str, float]],
) -> list[tuple[SemanticTag, float]]:
    """Reduce score for tags from incompatible domains."""
    has_elements = any(intensity > 0.3 for _, intensity in active_elements)
    has_domains = any(intensity > 0.3 for _, intensity in active_domains)

    # Build set of active contexts for boosting
    active_elem_names = {n for n, i in active_elements if i > 0.2}
    active_dom_names = {n for n, i in active_domains if i > 0.2}

    result = []
    for tag, sim in raw_tags:
        adjusted = sim
        # Penalize biology/mental tags when only elemental
        if has_elements and not has_domains:
            if tag.domain in ("biology", "mental", "comfort"):
                adjusted *= 0.5
        # Penalize elemental/combat tags when only domains
        if has_domains and not has_elements:
            if tag.domain in ("elemental", "combat"):
                adjusted *= 0.5
        # Boost tags from matching domains
        if tag.domain == "combo" or tag.domain == "cross":
            adjusted *= 1.2  # combo/cross tags are always relevant when matched
        result.append((tag, adjusted))

    result.sort(key=lambda x: x[1], reverse=True)
    return result


def _blend_palette(
    elem_pairs: list[tuple[str, float]],
    dom_pairs: list[tuple[str, float]],
    top_tag: SemanticTag | None,
    top_tag_sim: float,
) -> tuple[
    list[tuple[tuple[int, int, int], float]],  # palette
    tuple[int, int, int],                      # dominant
    tuple[int, int, int],                      # aura
]:
    """Weighted blend of element + domain + tag-domain colors.

    Elements carry full weight, domains 0.7×, top semantic tag 0.5× (if mapped).
    Dominant = weighted RGB mean. Aura = tag-domain color (or dominant fallback).
    """
    palette: list[tuple[tuple[int, int, int], float]] = []

    for name, intensity in elem_pairs:
        c = ELEMENT_COLORS.get(name)
        if c is not None:
            palette.append((c, float(intensity)))

    for name, intensity in dom_pairs:
        c = DOMAIN_COLORS.get(name)
        if c is not None:
            palette.append((c, float(intensity) * 0.7))

    aura: tuple[int, int, int] | None = None
    if top_tag is not None:
        tag_color = TAG_DOMAIN_COLORS.get(top_tag.domain)
        if tag_color is not None:
            w = max(0.0, float(top_tag_sim)) * 0.5
            if w > 0.05:
                palette.append((tag_color, w))
            aura = tag_color

    if palette:
        total_w = sum(w for _, w in palette) or 1.0
        r = sum(c[0] * w for c, w in palette) / total_w
        g = sum(c[1] * w for c, w in palette) / total_w
        b = sum(c[2] * w for c, w in palette) / total_w
        dominant = (
            max(0, min(255, int(r))),
            max(0, min(255, int(g))),
            max(0, min(255, int(b))),
        )
    else:
        dominant = (180, 180, 200)

    if aura is None:
        aura = dominant

    return palette, dominant, aura


def _compute_mechanics(
    power: float,
    tag_pairs: list[tuple[str, float]],
) -> tuple[float, float, float, float]:
    """Compute damage, healing, shield, utility from power and tag similarities."""
    damage = healing = shield = utility = 0.0
    d_count = h_count = s_count = u_count = 0

    for name, sim in tag_pairs:
        if name in DAMAGE_TAGS:
            damage += sim
            d_count += 1
        if name in HEALING_TAGS:
            healing += sim
            h_count += 1
        if name in SHIELD_TAGS:
            shield += sim
            s_count += 1
        if name in UTILITY_TAGS:
            utility += sim
            u_count += 1

    # Normalize by count, scale by power
    if d_count > 0:
        damage = min(power, power * damage / d_count)
    if h_count > 0:
        healing = min(power, power * healing / h_count)
    if s_count > 0:
        shield = min(power, power * shield / s_count)
    if u_count > 0:
        utility = min(power, power * utility / u_count)

    return damage, healing, shield, utility


def interpret_effect(
    pole_distribution: np.ndarray,
    power: float,
    stability: float,
    area: float,
    duration: float,
    element_intensity: np.ndarray,
    domain_intensity: np.ndarray,
    semantic_embedding: np.ndarray,
) -> EffectDescription:
    """Interpret a SpellFormer output into a full effect description."""

    # --- Poles ---
    poles = pole_distribution.tolist() if hasattr(pole_distribution, 'tolist') else list(pole_distribution)
    dominant_idx = int(np.argmax(poles))
    dominant_name = POLE_NAMES_RU[dominant_idx]

    # --- Active elements (threshold > 0.1) ---
    elem_pairs = [(ELEMENT_NAMES[i], float(element_intensity[i]))
                  for i in range(6) if element_intensity[i] > 0.1]
    elem_pairs.sort(key=lambda x: x[1], reverse=True)

    # --- Active domains (threshold > 0.1) ---
    dom_pairs = [(DOMAIN_NAMES[i], float(domain_intensity[i]))
                 for i in range(6) if domain_intensity[i] > 0.1]
    dom_pairs.sort(key=lambda x: x[1], reverse=True)

    # --- Semantic tags (filtered) ---
    raw_tags = nearest_tags(semantic_embedding, top_k=8)
    filtered_tags = _filter_tags(raw_tags, elem_pairs, dom_pairs)[:5]
    tag_pairs = [(t.name, sim) for t, sim in filtered_tags]

    # Primary domain from top tag
    primary_domain = filtered_tags[0][0].domain if filtered_tags else "unknown"

    # --- Combo-aware title ---
    combo_match = _match_combo_title(elem_pairs, dom_pairs)
    if combo_match:
        title, combo_desc = combo_match
    else:
        # Fallback: element — top tag
        title_parts = []
        if elem_pairs:
            elem_ru = ELEMENT_NAMES_RU[ELEMENT_NAMES.index(elem_pairs[0][0])]
            title_parts.append(elem_ru)
        if dom_pairs:
            dom_ru = DOMAIN_NAMES_RU[DOMAIN_NAMES.index(dom_pairs[0][0])]
            title_parts.append(dom_ru)
        if filtered_tags:
            title_parts.append(filtered_tags[0][0].description)
        title = " — ".join(title_parts[:2]) if title_parts else "неизвестный эффект"
        combo_desc = None

    # --- Description ---
    pw = _power_word(power)
    sw = _stability_word(stability)

    if combo_desc:
        # Use combo description as base
        desc_parts = [f"{pw.capitalize()} {combo_desc}"]
    else:
        desc_parts = [f"{pw.capitalize()}, {sw} эффект"]
        if elem_pairs:
            elem_str = ", ".join(
                ELEMENT_NAMES_RU[ELEMENT_NAMES.index(name)]
                for name, _ in elem_pairs[:3]
            )
            desc_parts.append(f"стихии: {elem_str}")
        if dom_pairs:
            dom_str = ", ".join(
                DOMAIN_NAMES_RU[DOMAIN_NAMES.index(name)]
                for name, _ in dom_pairs[:2]
            )
            desc_parts.append(f"домены: {dom_str}")

    if filtered_tags and not combo_desc:
        tag_str = ", ".join(t.description for t, _ in filtered_tags[:3])
        desc_parts.append(tag_str)

    if area > 2.0:
        desc_parts.append("широкая область")
    elif area < 0.5:
        desc_parts.append("точечный")

    if duration > 2.0:
        desc_parts.append("длительный")

    description = ". ".join(desc_parts) + "."

    # --- Game mechanics ---
    all_tag_pairs = [(t.name, sim) for t, sim in nearest_tags(semantic_embedding, top_k=15)]
    damage, healing, shield, utility = _compute_mechanics(power, all_tag_pairs)

    # --- Visualization ---
    colors = [ELEMENT_COLORS[name] for name, _ in elem_pairs[:3]]
    behaviors = [ELEMENT_BEHAVIORS[name] for name, _ in elem_pairs[:3]]

    # Mix in domain behaviors when domain is strong (or sole driver)
    for name, intensity in dom_pairs[:2]:
        if intensity > 0.25:
            beh = DOMAIN_BEHAVIORS.get(name)
            if beh and beh not in behaviors:
                behaviors.append(beh)

    if not colors and not behaviors:
        behaviors = ["pulse"]

    top_tag_obj = filtered_tags[0][0] if filtered_tags else None
    top_tag_sim = filtered_tags[0][1] if filtered_tags else 0.0
    palette, dominant_color, aura_color = _blend_palette(
        elem_pairs, dom_pairs, top_tag_obj, top_tag_sim,
    )

    if not colors:
        colors = [dominant_color]

    particle_count = int(10 + power * 40)
    particle_lifetime = 0.5 + duration * 0.3

    return EffectDescription(
        power=power,
        stability=stability,
        area=area,
        duration=duration,
        pole_distribution=poles,
        dominant_pole=dominant_idx,
        dominant_pole_name=dominant_name,
        active_elements=elem_pairs,
        active_domains=dom_pairs,
        tags=tag_pairs,
        primary_domain=primary_domain,
        title=title,
        description=description,
        particle_colors=colors,
        particle_behaviors=behaviors,
        particle_count=particle_count,
        particle_lifetime=particle_lifetime,
        dominant_color=dominant_color,
        aura_color=aura_color,
        palette=palette,
        power_word=pw,
        damage=damage,
        healing=healing,
        shield=shield,
        utility=utility,
    )


def interpret_model_output(output: dict) -> EffectDescription:
    """Convenience wrapper: takes SpellFormer forward() output dict directly."""
    return interpret_effect(
        pole_distribution=output["pole_distribution"][0].detach().numpy(),
        power=output["power"][0, 0].item(),
        stability=output["stability"][0, 0].item(),
        area=output["area"][0, 0].item(),
        duration=output["duration"][0, 0].item(),
        element_intensity=output["element_intensity"][0].detach().numpy(),
        domain_intensity=output["domain_intensity"][0].detach().numpy(),
        semantic_embedding=output["semantic_embedding"][0].detach().numpy(),
    )
