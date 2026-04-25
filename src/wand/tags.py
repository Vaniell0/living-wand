"""Semantic tag embeddings for magic effect interpretation.

Each tag is a 16D vector. Tags are grouped by domain. The model's semantic_embedding
output (16D) is compared via cosine similarity to find the closest tags, producing
human-readable effect descriptions.

Vectors are manually crafted to ensure:
- Tags within the same domain are close (high cosine sim)
- Tags from different domains are orthogonal or distant
- Combat tags occupy a small region — everyday magic dominates the space
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SEMANTIC_DIM = 16


@dataclass(frozen=True)
class SemanticTag:
    name: str
    domain: str
    vector: tuple[float, ...]
    description: str  # short Russian description for display


def _norm(v: list[float]) -> tuple[float, ...]:
    """L2-normalize a vector."""
    a = np.array(v, dtype=np.float32)
    n = np.linalg.norm(a)
    if n > 0:
        a = a / n
    return tuple(a.tolist())


# ═══════════════════════════════════════════
# Tag definitions
# ═══════════════════════════════════════════
# 16D layout (conceptual axes):
#   [0-1]   elemental:  0=fire/heat, 1=water/cold
#   [2-3]   physical:   2=force/destruction, 3=structure/creation
#   [4-5]   life:       4=growth/healing, 5=decay/harm
#   [6-7]   mental:     6=clarity/truth, 7=illusion/confusion
#   [8-9]   spatial:    8=movement/travel, 9=containment/boundary
#   [10-11] temporal:   10=speed/haste, 11=slow/stasis
#   [12-13] utility:    12=domestic/comfort, 13=craft/work
#   [14-15] scale:      14=precise/small, 15=vast/area

def _tags() -> list[SemanticTag]:
    tags = []

    def t(name: str, domain: str, v: list[float], desc: str):
        tags.append(SemanticTag(name, domain, _norm(v), desc))

    # --- HOUSEHOLD ---
    t("cleaning",     "household", [0,0.2, 0,0.3, 0,0, 0,0, 0.3,0, 0,0, 0.8,0, 0.5,0], "очистка, уборка")
    t("warming",      "household", [0.8,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0.7,0, 0.3,0], "обогрев, согревание")
    t("cooling",      "household", [0,0.8, 0,0, 0,0, 0,0, 0,0, 0,0, 0.7,0, 0.3,0], "охлаждение")
    t("lighting",     "household", [0.3,0, 0,0, 0,0, 0.6,0, 0,0, 0,0, 0.7,0, 0.3,0], "освещение")
    t("drying",       "household", [0.5,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0.8,0, 0.3,0], "сушка")
    t("repair",       "household", [0,0, 0,0.8, 0,0, 0,0, 0,0, 0,0, 0.6,0, 0.4,0], "починка, ремонт")
    t("shelter",      "household", [0,0, 0,0.5, 0,0, 0,0, 0,0.7, 0,0, 0.6,0, 0,0.4], "укрытие")

    # --- COOKING ---
    t("heating",      "cooking", [0.9,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0.5,0.4, 0.4,0], "нагрев для готовки")
    t("boiling",      "cooking", [0.7,0.3, 0,0, 0,0, 0,0, 0,0, 0,0, 0.4,0.5, 0.3,0], "кипячение")
    t("freezing",     "cooking", [0,0.9, 0,0, 0,0, 0,0, 0,0, 0,0.3, 0.4,0.4, 0.3,0], "заморозка")
    t("mixing",       "cooking", [0,0, 0,0.3, 0,0, 0,0, 0.3,0, 0,0, 0.4,0.6, 0.4,0], "смешивание")
    t("preserving",   "cooking", [0,0, 0,0.4, 0,0, 0,0, 0,0.3, 0,0.5, 0.3,0.5, 0.3,0], "консервация")
    t("fermenting",   "cooking", [0,0, 0,0, 0.4,0, 0,0, 0,0, 0,0.4, 0.3,0.5, 0.3,0], "ферментация")

    # --- CRAFTING ---
    t("shaping",      "crafting", [0,0, 0,0.7, 0,0, 0,0, 0,0, 0,0, 0,0.8, 0.5,0], "формовка")
    t("hardening",    "crafting", [0.3,0, 0,0.7, 0,0, 0,0, 0,0, 0,0, 0,0.7, 0.4,0], "закалка")
    t("softening",    "crafting", [0.2,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0.7, 0.4,0], "размягчение")
    t("joining",      "crafting", [0,0, 0,0.6, 0,0, 0,0, 0,0.3, 0,0, 0,0.7, 0.5,0], "соединение")
    t("cutting",      "crafting", [0,0, 0.5,0, 0,0, 0,0, 0,0, 0,0, 0,0.6, 0.6,0], "разрезание")
    t("polishing",    "crafting", [0,0, 0,0.4, 0,0, 0.3,0, 0,0, 0,0, 0,0.6, 0.5,0], "полировка")
    t("forging",      "crafting", [0.7,0, 0,0.6, 0,0, 0,0, 0,0, 0,0, 0,0.8, 0.3,0], "ковка")

    # --- NATURE / AGRICULTURE ---
    t("growing",      "nature", [0,0, 0,0.3, 0.8,0, 0,0, 0,0, 0,0, 0.3,0.3, 0,0.4], "выращивание")
    t("watering",     "nature", [0,0.5, 0,0, 0.5,0, 0,0, 0,0, 0,0, 0.3,0.3, 0,0.4], "полив")
    t("weather",      "nature", [0.3,0.3, 0,0, 0,0, 0,0, 0,0, 0,0, 0.2,0, 0,0.9], "погода")
    t("pest_control", "nature", [0,0, 0,0, 0,0.5, 0,0, 0.3,0, 0,0, 0.3,0.3, 0,0.4], "защита от вредителей")
    t("harvesting",   "nature", [0,0, 0.3,0, 0.3,0, 0,0, 0,0, 0.3,0, 0.2,0.5, 0,0.3], "сбор урожая")
    t("composting",   "nature", [0,0, 0,0, 0.4,0.3, 0,0, 0,0, 0,0.4, 0.2,0.4, 0.3,0], "компостирование")

    # --- COMMUNICATION ---
    t("signaling",    "communication", [0.2,0, 0,0, 0,0, 0.5,0, 0,0, 0,0, 0.3,0, 0,0.5], "подача сигнала")
    t("projecting",   "communication", [0.2,0, 0,0, 0,0, 0.4,0.3, 0.3,0, 0,0, 0.2,0, 0,0.5], "проецирование")
    t("recording",    "communication", [0,0, 0,0.3, 0,0, 0.5,0, 0,0.3, 0,0.3, 0.3,0, 0.3,0], "запись")
    t("translating",  "communication", [0,0, 0,0, 0,0, 0.8,0, 0,0, 0,0, 0.3,0, 0.4,0], "перевод")

    # --- NAVIGATION ---
    t("pathfinding",  "navigation", [0,0, 0,0, 0,0, 0.4,0, 0.7,0, 0,0, 0.2,0, 0,0.4], "поиск пути")
    t("mapping",      "navigation", [0,0, 0,0, 0,0, 0.5,0, 0.5,0, 0,0, 0.2,0, 0,0.5], "картография")
    t("compass",      "navigation", [0,0, 0,0, 0,0, 0.3,0, 0.8,0, 0,0, 0.3,0, 0.3,0], "компас")

    # --- COMFORT ---
    t("rest",         "comfort", [0,0, 0,0, 0.3,0, 0,0, 0,0.3, 0,0.3, 0.7,0, 0.3,0], "отдых")
    t("silence",      "comfort", [0,0, 0,0, 0,0, 0,0.4, 0,0.5, 0,0, 0.6,0, 0.3,0], "тишина")
    t("fragrance",    "comfort", [0,0, 0,0, 0.2,0, 0,0, 0.3,0, 0,0, 0.7,0, 0.3,0], "аромат")

    # --- COMBAT (one domain among many) ---
    t("attack",       "combat", [0.5,0, 0.7,0, 0,0.2, 0,0, 0.3,0, 0.3,0, 0,0, 0,0.3], "атака")
    t("defense",      "combat", [0,0, 0,0.6, 0,0, 0,0, 0,0.7, 0,0, 0,0, 0,0.3], "защита")
    t("disruption",   "combat", [0.3,0, 0.5,0, 0,0.3, 0,0.3, 0,0, 0.3,0, 0,0, 0,0.4], "разрушение")
    t("enhancement",  "combat", [0,0, 0,0, 0.3,0, 0,0, 0,0, 0.5,0, 0,0.3, 0.3,0], "усиление")

    # --- ELEMENTAL EFFECTS ---
    t("burning",      "elemental", [0.9,0, 0.4,0, 0,0.3, 0,0, 0,0, 0,0, 0,0, 0,0.3], "горение")
    t("freezing_elem","elemental", [0,0.9, 0,0.3, 0,0, 0,0, 0,0, 0,0.4, 0,0, 0,0.3], "замораживание")
    t("storm",        "elemental", [0.5,0, 0.3,0, 0,0, 0,0, 0.4,0, 0.3,0, 0,0, 0,0.9], "шторм")
    t("earthquake",   "elemental", [0,0, 0.6,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0.9], "землетрясение")
    t("flash",        "elemental", [0.4,0, 0.2,0, 0,0, 0.5,0, 0,0, 0.4,0, 0,0, 0,0.3], "вспышка")
    t("darkness",     "elemental", [0,0, 0,0, 0,0, 0,0.7, 0,0.3, 0,0, 0,0, 0,0.4], "тьма")
    t("mist",         "elemental", [0,0.4, 0,0, 0,0, 0,0.4, 0,0.3, 0,0, 0,0, 0,0.5], "туман")

    # --- HEALING / BIOLOGY ---
    t("healing",      "biology", [0,0, 0,0.3, 0.9,0, 0,0, 0,0, 0,0, 0.2,0, 0.4,0], "исцеление")
    t("regeneration", "biology", [0,0, 0,0.2, 0.8,0, 0,0, 0,0, 0.3,0, 0.2,0, 0.3,0], "регенерация")
    t("purification", "biology", [0.2,0, 0,0.3, 0.6,0, 0.3,0, 0,0, 0,0, 0.4,0, 0.3,0], "очищение светом")
    t("poison",       "biology", [0,0, 0,0, 0,0.8, 0,0, 0,0, 0,0, 0,0, 0.3,0], "яд")
    t("diagnosis",    "biology", [0,0, 0,0, 0.4,0, 0.5,0, 0,0, 0,0, 0.2,0, 0.4,0], "диагностика")
    t("empathy",      "biology", [0,0, 0,0, 0.3,0, 0.6,0, 0,0, 0,0, 0.2,0, 0.4,0], "эмпатия")

    # --- MENTAL ---
    t("illusion",     "mental", [0,0, 0,0, 0,0, 0,0.8, 0,0, 0,0, 0,0, 0.3,0.3], "иллюзия")
    t("clarity",      "mental", [0,0, 0,0, 0,0, 0.9,0, 0,0, 0,0, 0.2,0, 0.3,0], "ясность")
    t("fear",         "mental", [0,0, 0,0, 0,0.3, 0,0.6, 0,0, 0,0, 0,0, 0,0.4], "страх")
    t("calm",         "mental", [0,0, 0,0, 0.2,0, 0.3,0, 0,0.3, 0,0.3, 0.5,0, 0.3,0], "спокойствие")
    t("memory",       "mental", [0,0, 0,0, 0,0, 0.6,0, 0,0.2, 0,0.3, 0.2,0, 0.3,0], "память")
    t("telepathy",    "mental", [0,0, 0,0, 0,0, 0.5,0, 0.5,0, 0,0, 0,0, 0,0.3], "телепатия")

    # --- SPACE ---
    t("teleportation","space", [0,0, 0,0, 0,0, 0,0, 0.9,0, 0,0, 0,0, 0.3,0], "телепортация")
    t("barrier",      "space", [0,0, 0,0.4, 0,0, 0,0, 0,0.9, 0,0, 0,0, 0,0.3], "барьер")
    t("compression",  "space", [0,0, 0,0, 0,0, 0,0, 0.5,0.5, 0,0, 0,0.3, 0.5,0], "сжатие пространства")
    t("expansion",    "space", [0,0, 0,0, 0,0, 0,0, 0.5,0, 0,0, 0,0, 0,0.8], "расширение")
    t("portal",       "space", [0,0, 0,0, 0,0, 0,0, 0.8,0, 0,0, 0,0.2, 0,0.4], "портал")

    # --- TIME ---
    t("haste",        "time", [0,0, 0,0, 0,0, 0,0, 0.3,0, 0.9,0, 0,0, 0.3,0], "ускорение")
    t("slow",         "time", [0,0, 0,0, 0,0, 0,0, 0,0, 0,0.9, 0,0, 0,0.3], "замедление")
    t("stasis",       "time", [0,0, 0,0.3, 0,0, 0,0, 0,0.4, 0,0.8, 0,0, 0.3,0], "стазис")
    t("rewind",       "time", [0,0, 0,0, 0,0, 0,0, 0,0, 0.4,0.4, 0,0, 0.3,0], "откат")
    t("prophecy",     "time", [0,0, 0,0, 0,0, 0.5,0, 0,0, 0.3,0.3, 0,0, 0.3,0], "предвидение")

    # --- TRANSFORMATION ---
    t("transmutation","transform", [0,0, 0,0.5, 0,0, 0,0, 0,0, 0,0, 0,0.6, 0.4,0], "трансмутация")
    t("enlargement",  "transform", [0,0, 0,0.3, 0,0, 0,0, 0,0, 0,0, 0,0.4, 0,0.7], "увеличение")
    t("miniaturize",  "transform", [0,0, 0,0.3, 0,0, 0,0, 0,0, 0,0, 0,0.4, 0.7,0], "уменьшение")
    t("strengthen",   "transform", [0,0, 0,0.7, 0,0, 0,0, 0,0, 0,0, 0,0.5, 0.3,0], "упрочнение")

    # --- COMBO ELEMENTAL (emergent element×element effects) ---
    t("steam",        "combo", [0.5,0.5, 0,0, 0,0, 0,0.3, 0,0, 0,0, 0.4,0, 0,0.4], "пар, обжигающий туман")
    t("lava",         "combo", [0.8,0, 0.3,0.3, 0,0, 0,0, 0,0, 0,0, 0,0.5, 0,0.4], "лава, расплавленный камень")
    t("radiance",     "combo", [0.5,0, 0,0.3, 0.3,0, 0.5,0, 0,0, 0,0, 0.3,0, 0,0.5], "сияние, солнечный свет")
    t("hellfire",     "combo", [0.7,0, 0.5,0, 0,0.5, 0,0.3, 0,0, 0,0, 0,0, 0,0.4], "адский огонь, проклятое пламя")
    t("twilight",     "combo", [0.2,0.2, 0,0, 0,0, 0.3,0.3, 0,0, 0,0.3, 0.4,0, 0.3,0], "сумерки, граница света и тьмы")
    t("eclipse_effect","combo",[0.3,0, 0.2,0, 0,0, 0.2,0.4, 0,0, 0,0.2, 0,0, 0,0.5], "затмение, двойственность")
    t("dust_storm",   "combo", [0.2,0, 0.3,0, 0,0, 0,0, 0.4,0, 0,0, 0,0, 0,0.8], "пыльная буря, эрозия")
    t("void_wind",    "combo", [0,0, 0.2,0, 0,0.3, 0,0.4, 0.3,0, 0,0, 0,0, 0,0.3], "пустотный ветер, вакуум")
    t("crystal_gem",  "combo", [0,0, 0,0.6, 0,0, 0.4,0, 0,0, 0,0, 0,0.5, 0.4,0], "кристалл, драгоценный камень")
    t("rain",         "combo", [0,0.5, 0,0, 0.2,0, 0,0, 0.3,0, 0,0, 0.3,0, 0,0.6], "дождь, влажность")
    t("volcano",      "combo", [0.7,0, 0.5,0, 0,0.2, 0,0, 0,0, 0,0, 0,0.3, 0,0.9], "вулкан, извержение")
    t("oasis",        "combo", [0,0.4, 0,0.3, 0.5,0, 0.3,0, 0,0, 0,0, 0.4,0, 0,0.3], "оазис, источник жизни")
    t("rainbow_effect","combo",[0.2,0.2, 0,0.2, 0.2,0, 0.5,0, 0.2,0, 0,0, 0.3,0, 0,0.4], "радуга, преломление")

    # --- COMBO CROSS-DOMAIN (element×domain effects) ---
    t("fever",        "cross", [0.6,0, 0,0, 0,0.3, 0,0, 0,0, 0,0, 0.2,0, 0.4,0], "лихорадка, жар тела")
    t("rage",         "cross", [0.5,0, 0.4,0, 0,0, 0,0.2, 0,0, 0.3,0, 0,0, 0,0.3], "ярость, боевое безумие")
    t("phoenix",      "cross", [0.6,0, 0,0.3, 0.6,0, 0,0, 0,0, 0.2,0, 0,0.3, 0,0.3], "феникс, возрождение через огонь")
    t("necromancy_art","cross",[0,0, 0,0, 0,0.6, 0,0.5, 0,0, 0,0, 0,0.3, 0,0.3], "некромантия, нежить")
    t("plague",       "cross", [0,0, 0,0, 0,0.7, 0,0.3, 0,0, 0,0.3, 0,0, 0,0.5], "чума, болезнь")
    t("pottery_art",  "cross", [0.3,0.2, 0,0.6, 0,0, 0,0, 0,0, 0,0, 0,0.7, 0.4,0], "гончарное дело, керамика")
    t("enchanting",   "cross", [0,0, 0,0.4, 0,0, 0.5,0, 0,0, 0,0, 0,0.6, 0.4,0], "зачарование предметов")
    t("golem",        "cross", [0,0, 0,0.5, 0,0, 0.3,0, 0,0, 0,0, 0,0.5, 0,0.5], "голем, оживлённая материя")
    t("blessing",     "cross", [0.2,0, 0,0.3, 0.5,0, 0.4,0, 0,0, 0,0, 0.3,0, 0,0.3], "благословение, аура света")
    t("nightmare",    "cross", [0,0, 0,0, 0,0.3, 0,0.7, 0,0, 0,0.3, 0,0, 0.3,0], "кошмар, тёмные сны")
    t("flood",        "cross", [0,0.6, 0.3,0, 0,0, 0,0, 0.3,0, 0,0, 0,0, 0,0.8], "потоп, наводнение")
    t("garden",       "cross", [0,0.3, 0,0.3, 0.6,0, 0,0, 0,0, 0,0, 0.5,0.3, 0,0.4], "сад, выращивание")

    # --- MODIFIER COMBO EFFECTS ---
    t("overcharged",  "modifier_fx", [0.3,0, 0.5,0, 0,0.2, 0,0, 0,0, 0.3,0, 0,0.3, 0,0.4], "перегрузка, нестабильная мощь")
    t("suppress",     "modifier_fx", [0,0, 0,0.3, 0,0, 0,0, 0,0.6, 0,0.5, 0,0, 0.4,0], "подавление, печать")
    t("drain",        "modifier_fx", [0,0, 0,0, 0,0.5, 0,0.2, 0,0, 0,0.3, 0,0.2, 0.3,0], "вытягивание, сифон")
    t("null_zone",    "modifier_fx", [0,0, 0,0, 0,0, 0,0.3, 0,0.4, 0,0.4, 0,0, 0.3,0.3], "мёртвая зона магии")
    t("reservoir",    "modifier_fx", [0,0, 0,0.4, 0,0, 0,0, 0,0.5, 0,0.3, 0.3,0.3, 0.3,0], "резервуар силы, батарея")

    return tags


# Singleton — all tags
ALL_TAGS: list[SemanticTag] = _tags()
TAG_BY_NAME: dict[str, SemanticTag] = {t.name: t for t in ALL_TAGS}
TAG_NAMES: list[str] = [t.name for t in ALL_TAGS]
NUM_TAGS = len(ALL_TAGS)

# Pre-computed tag matrix for fast nearest-neighbor lookup
TAG_MATRIX = np.array([t.vector for t in ALL_TAGS], dtype=np.float32)  # (NUM_TAGS, 16)


def nearest_tags(embedding: np.ndarray, top_k: int = 3) -> list[tuple[SemanticTag, float]]:
    """Find the top_k closest semantic tags to a 16D embedding vector.

    Uses cosine similarity.
    """
    emb = np.asarray(embedding, dtype=np.float32).flatten()
    norm = np.linalg.norm(emb)
    if norm < 1e-8:
        return []
    emb = emb / norm

    # TAG_MATRIX is already L2-normalized
    sims = TAG_MATRIX @ emb  # (NUM_TAGS,)
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [(ALL_TAGS[i], float(sims[i])) for i in top_idx]


def tags_for_domain(domain: str) -> list[SemanticTag]:
    """Get all tags belonging to a domain."""
    return [t for t in ALL_TAGS if t.domain == domain]
