"""Data-driven combo loading for the magic algebra.

Loads JSON combo files from data/ and provides typed lookup dictionaries
for algebra.py and datagen.py. Also re-exports backward-compatible
SYNERGY_PAIRS and CANCEL_PAIRS.

Discovery catalog: builds a flat list of all discoverable combinations
for the grimoire discovery system.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

from .algebra import Element

# ═══════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════

@dataclass(frozen=True)
class ElementPairCombo:
    type: str              # "synergy" | "cancel" | "transform"
    multiplier: float
    products: tuple[str, ...]
    tags: tuple[str, ...]
    cancel_factor: float | None = None
    description: str = ""

@dataclass(frozen=True)
class ElementDomainCombo:
    multiplier: float
    tags: tuple[str, ...]
    description: str = ""

@dataclass(frozen=True)
class DomainPairCombo:
    multiplier: float
    stability_penalty: float
    tags: tuple[str, ...]
    description: str = ""

@dataclass(frozen=True)
class TripleCombo:
    multiplier: float
    tags: tuple[str, ...]
    description: str = ""

@dataclass(frozen=True)
class ModifierCombo:
    multiplier: float
    stability_penalty: float | None
    tags: tuple[str, ...]


# ═══════════════════════════════════════════
# Discovery catalog
# ═══════════════════════════════════════════

@dataclass(frozen=True)
class Discoverable:
    key: str                # canonical key: "air+fire" or "combat/fireball"
    name: str               # "storm", "candle", etc.
    name_ru: str            # short Russian name for UI
    description_ru: str     # full description from JSON
    category: str           # "element_pair", "element_domain", "domain_pair", "triple", "recipe_everyday", etc.
    runes: tuple[str, ...]  # what runes the player needs


# ═══════════════════════════════════════════
# Parsing helpers
# ═══════════════════════════════════════════

_ELEMENT_MAP = {e.name.lower(): e for e in Element}
_DOMAIN_NAMES = {"body", "mind", "matter", "space", "time", "life"}

DATA_DIR = Path(__file__).parent.parent.parent / "data"


def _load_json(name: str) -> dict:
    with open(DATA_DIR / name) as f:
        return json.load(f)


def _parse_element(name: str) -> Element | None:
    return _ELEMENT_MAP.get(name.lower())


def _parse_pair_key(key: str) -> tuple[str, str]:
    """Split 'fire+water' into ('fire', 'water')."""
    parts = key.split("+")
    assert len(parts) == 2, f"Bad pair key: {key}"
    return parts[0].strip(), parts[1].strip()


def _parse_triple_key(key: str) -> tuple[str, ...]:
    """Split 'fire+water+air' into ('fire', 'water', 'air')."""
    parts = key.split("+")
    assert len(parts) == 3, f"Bad triple key: {key}"
    return tuple(p.strip() for p in parts)


def _short_ru(desc: str) -> str:
    """Extract first short phrase from Russian description."""
    # Take text before first " — " or first "."
    for sep in (" — ", ". ", ", "):
        if sep in desc:
            return desc.split(sep)[0]
    return desc[:40]


# ═══════════════════════════════════════════
# Load element × element
# ═══════════════════════════════════════════

def _load_element_pairs() -> dict[frozenset[Element], ElementPairCombo]:
    raw = _load_json("combos_element_element.json")
    result = {}
    for key, val in raw.items():
        if key.startswith("_"):
            continue
        a_str, b_str = _parse_pair_key(key)
        a, b = _parse_element(a_str), _parse_element(b_str)
        if a is None or b is None:
            continue
        result[frozenset({a, b})] = ElementPairCombo(
            type=val["type"],
            multiplier=val["multiplier"],
            products=tuple(val.get("products", [])),
            tags=tuple(val.get("tags", [])),
            cancel_factor=val.get("cancel_factor"),
            description=val.get("description", ""),
        )
    return result


# ═══════════════════════════════════════════
# Load element × domain
# ═══════════════════════════════════════════

def _load_element_domain() -> dict[tuple[Element, str], ElementDomainCombo]:
    raw = _load_json("combos_element_domain.json")
    result = {}
    for key, val in raw.items():
        if key.startswith("_"):
            continue
        a_str, b_str = _parse_pair_key(key)
        elem = _parse_element(a_str)
        domain = b_str if b_str in _DOMAIN_NAMES else None
        if elem is None or domain is None:
            elem = _parse_element(b_str)
            domain = a_str if a_str in _DOMAIN_NAMES else None
        if elem is None or domain is None:
            continue
        result[(elem, domain)] = ElementDomainCombo(
            multiplier=val["multiplier"],
            tags=tuple(val.get("tags", [])),
            description=val.get("description", ""),
        )
    return result


# ═══════════════════════════════════════════
# Load domain × domain
# ═══════════════════════════════════════════

def _load_domain_pairs() -> dict[frozenset[str], DomainPairCombo]:
    raw = _load_json("combos_domain_domain.json")
    result = {}
    for key, val in raw.items():
        if key.startswith("_"):
            continue
        a_str, b_str = _parse_pair_key(key)
        if a_str not in _DOMAIN_NAMES or b_str not in _DOMAIN_NAMES:
            continue
        result[frozenset({a_str, b_str})] = DomainPairCombo(
            multiplier=val["multiplier"],
            stability_penalty=val.get("stability_penalty", 0.9),
            tags=tuple(val.get("tags", [])),
            description=val.get("description", ""),
        )
    return result


# ═══════════════════════════════════════════
# Load triple combos
# ═══════════════════════════════════════════

def _load_triples() -> dict[frozenset[str], TripleCombo]:
    raw = _load_json("combos_triple.json")
    result = {}
    for section_key in ("_elements_triple", "_mixed_triple"):
        section = raw.get(section_key, {})
        for key, val in section.items():
            parts = _parse_triple_key(key)
            result[frozenset(parts)] = TripleCombo(
                multiplier=val["multiplier"],
                tags=tuple(val.get("tags", [])),
                description=val.get("description", ""),
            )
    return result


# ═══════════════════════════════════════════
# Load modifier combos
# ═══════════════════════════════════════════

def _load_modifier_combos() -> dict[frozenset[str], ModifierCombo]:
    raw = _load_json("combos_edge_modifiers.json")
    section = raw.get("modifier_combos", {})
    result = {}
    for key, val in section.items():
        a_str, b_str = _parse_pair_key(key)
        result[frozenset({a_str, b_str})] = ModifierCombo(
            multiplier=val["multiplier"],
            stability_penalty=val.get("stability_penalty"),
            tags=tuple(val.get("tags", [])),
        )
    return result


def _load_same_modifier_combos() -> dict[str, ModifierCombo]:
    """Load combos where both modifiers are the same (amplify+amplify, hold+hold, etc.)."""
    raw = _load_json("combos_edge_modifiers.json")
    section = raw.get("modifier_combos", {})
    result = {}
    for key, val in section.items():
        a_str, b_str = _parse_pair_key(key)
        if a_str == b_str:
            result[a_str] = ModifierCombo(
                multiplier=val["multiplier"],
                stability_penalty=val.get("stability_penalty"),
                tags=tuple(val.get("tags", [])),
            )
    return result


# ═══════════════════════════════════════════
# Load recipes (raw, for datagen)
# ═══════════════════════════════════════════

def _load_recipes() -> dict[str, dict]:
    raw = _load_json("combos_recipes.json")
    result = {}
    for category, recipes in raw.items():
        if category.startswith("_"):
            continue
        if isinstance(recipes, dict):
            for name, recipe in recipes.items():
                if isinstance(recipe, dict) and "nodes" in recipe:
                    result[f"{category}/{name}"] = recipe
    return result


# ═══════════════════════════════════════════
# Module-level singletons (loaded once at import)
# ═══════════════════════════════════════════

ELEMENT_PAIRS: dict[frozenset[Element], ElementPairCombo] = _load_element_pairs()
ELEMENT_DOMAIN: dict[tuple[Element, str], ElementDomainCombo] = _load_element_domain()
DOMAIN_PAIRS: dict[frozenset[str], DomainPairCombo] = _load_domain_pairs()
TRIPLE_COMBOS: dict[frozenset[str], TripleCombo] = _load_triples()
MODIFIER_COMBOS: dict[frozenset[str], ModifierCombo] = _load_modifier_combos()
SAME_MODIFIER_COMBOS: dict[str, ModifierCombo] = _load_same_modifier_combos()
RECIPES: dict[str, dict] = _load_recipes()


# ═══════════════════════════════════════════
# Backward-compatible re-exports
# ═══════════════════════════════════════════

SYNERGY_PAIRS: dict[frozenset, tuple[str, float]] = {}
CANCEL_PAIRS: dict[frozenset, float] = {}

for pair, combo in ELEMENT_PAIRS.items():
    if combo.type == "synergy":
        name = combo.products[0] if combo.products else "synergy"
        SYNERGY_PAIRS[pair] = (name, combo.multiplier)
    elif combo.type in ("cancel", "transform"):
        factor = combo.cancel_factor if combo.cancel_factor is not None else 0.5
        CANCEL_PAIRS[pair] = factor


# ═══════════════════════════════════════════
# Discovery catalog
# ═══════════════════════════════════════════

def _build_discovery_catalog() -> dict[str, Discoverable]:
    catalog: dict[str, Discoverable] = {}

    # Element pairs (15)
    for pair, combo in ELEMENT_PAIRS.items():
        names = sorted(e.name.lower() for e in pair)
        key = "+".join(names)
        name = combo.products[0] if combo.products else combo.type
        catalog[key] = Discoverable(
            key=key,
            name=name,
            name_ru=_short_ru(combo.description) if combo.description else name,
            description_ru=combo.description,
            category="element_pair",
            runes=tuple(names),
        )

    # Element × Domain (36)
    for (elem, domain), combo in ELEMENT_DOMAIN.items():
        names = sorted([elem.name.lower(), domain])
        key = "+".join(names)
        name = combo.tags[0] if combo.tags else f"{elem.name.lower()}+{domain}"
        catalog[key] = Discoverable(
            key=key,
            name=name,
            name_ru=_short_ru(combo.description) if combo.description else name,
            description_ru=combo.description,
            category="element_domain",
            runes=tuple(names),
        )

    # Domain pairs (15)
    for pair, combo in DOMAIN_PAIRS.items():
        names = sorted(pair)
        key = "+".join(names)
        name = combo.tags[0] if combo.tags else "+".join(names)
        catalog[key] = Discoverable(
            key=key,
            name=name,
            name_ru=_short_ru(combo.description) if combo.description else name,
            description_ru=combo.description,
            category="domain_pair",
            runes=tuple(names),
        )

    # Triples (28)
    for triple_set, combo in TRIPLE_COMBOS.items():
        names = sorted(triple_set)
        key = "+".join(names)
        name = combo.tags[0] if combo.tags else "+".join(names)
        catalog[key] = Discoverable(
            key=key,
            name=name,
            name_ru=_short_ru(combo.description) if combo.description else name,
            description_ru=combo.description,
            category="triple",
            runes=tuple(names),
        )

    # Recipes (23)
    for recipe_key, recipe in RECIPES.items():
        rune_names = sorted(set(n["name"] for n in recipe["nodes"]))
        name = recipe_key.split("/")[-1]
        category = "recipe_" + recipe_key.split("/")[0]
        desc = recipe.get("description", "")
        catalog[recipe_key] = Discoverable(
            key=recipe_key,
            name=name,
            name_ru=_short_ru(desc) if desc else name,
            description_ru=desc,
            category=category,
            runes=tuple(rune_names),
        )

    return catalog


DISCOVERY_CATALOG: dict[str, Discoverable] = _build_discovery_catalog()


# ═══════════════════════════════════════════
# Discovery matching
# ═══════════════════════════════════════════

def match_cast(rune_names: list[str]) -> list[Discoverable]:
    """Find all discoverable combos fully covered by the given rune set."""
    rune_set = set(rune_names)
    matches = []
    for disc in DISCOVERY_CATALOG.values():
        if set(disc.runes).issubset(rune_set):
            matches.append(disc)
    return matches


def near_misses(rune_names: list[str], discovered: set[str]) -> list[tuple[Discoverable, list[str]]]:
    """Find discoverable combos that are 1 rune away from the cast and not yet discovered.

    Returns: list of (discoverable, missing_rune_names).
    """
    rune_set = set(rune_names)
    results = []
    for disc in DISCOVERY_CATALOG.values():
        if disc.key in discovered:
            continue
        missing = set(disc.runes) - rune_set
        if len(missing) == 1:
            results.append((disc, list(missing)))
    return results
