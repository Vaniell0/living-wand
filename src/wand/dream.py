"""Dreams — the gameplay space where the player learns the language of magic.

A new player does not know the runes. Handing them the 30-rune alphabet is a
spreadsheet; making them guess blindly is noise. The wand teaches instead:

  Lexicon   — the player's own dictionary: which runes they know, and what
              they have SEEN each rune do (notes sharpen with every cast —
              learning by correlation, like a real experimenter).
  Vision    — while the wand sleeps it dreams a spell the player could cast
              if they knew ONE more word (or one more combination). The
              dream shows the effect, not the recipe.
  attempt() — in the dream casting is free and safe; every attempt gets
              hot/cold feedback per aspect ("жар — да, полёт — нет"), so the
              player learns what each part of their spell is responsible for.
              Reproduce the vision → the new rune is learned.

The same search machinery as explore.py (sampling + scoring of programs)
runs here as a game mechanic: the wand is a research partner, not a wiki.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

from .actions import SpellProgram
from .algebra import RUNES, EdgeType, SignalKind
from .explore import Evaluated, SpellGenome, evaluate

STARTER_RUNES = ("fire", "water", "here", "self", "target", "outward")

RUNE_RU = {
    "fire": "огонь", "water": "вода", "air": "воздух", "earth": "земля", "light": "свет",
    "dark": "тьма", "body": "тело", "mind": "разум", "matter": "материя",
    "space": "пространство", "time": "время", "life": "жизнь", "inward": "внутрь",
    "outward": "наружу", "up": "вверх", "down": "вниз", "through": "сквозь",
    "reverse": "вспять", "point": "точка", "flow": "поток", "wave": "волна", "all": "всё",
    "amplify": "усиление", "weaken": "ослабление", "hold": "удержание", "dispel": "развеяние",
    "here": "здесь", "there": "там", "self": "сам", "target": "цель",
}
VERB_RU = {
    "damage": "ранит", "heal": "исцеляет", "shield": "защищает", "status": "насылает морок",
    "push": "толкает", "pull": "притягивает", "move": "переносит", "timescale": "меняет ход времени",
    "light": "светит", "obscure": "скрывает", "transform": "преображает материю",
    "grow": "растит", "temperature": "меняет тепло", "cleanse": "очищает",
    "dispel": "развеивает магию", "reveal": "открывает скрытое", "communicate": "передаёт мысль",
}
DELIVERY_RU = {
    "projectile": "летит к цели", "beam": "тянется лучом", "zone": "ложится на место",
    "aura": "окутывает", "touch": "касается", "instant": "вспыхивает разом",
}
TARGET_RU = {
    "self": "тебя самого", "target": "кого-то вдали", "point": "точку вдали",
    "caster_area": "всё вокруг тебя",
}
FIELD_RU = {
    "heat": ("жар", "холод"), "moisture": ("влага", "сухость"), "motion": ("напор", "затишье"),
    "solidity": ("твёрдость", "хрупкость"), "light": ("свет", "тьма"),
    "vitality": ("жизнь", "увядание"), "will": ("ясность", "смятение"),
    "position": ("сдвиг", "сжатие"), "timeflow": ("спешка", "замедление"),
}
CHANNEL_IMAGERY = {
    "fire": "алые языки пламени", "water": "синие капли", "air": "белёсые вихри",
    "earth": "бурая пыль", "light": "золотые лучи", "dark": "лиловая мгла",
    "body": "розоватое биение", "mind": "сиреневые искры", "matter": "охристые осколки",
    "space": "индиговые кольца", "time": "янтарные песчинки", "life": "бирюзовые ростки",
    "arcane": "бесцветное мерцание",
}


def field_words(program: SpellProgram, k: int = 3) -> list[str]:
    out = []
    for f in program.fields[:k]:
        pos, neg = FIELD_RU.get(f.property, (f.property, f.property))
        out.append(pos if f.delta > 0 else neg)
    return out


def describe_effect(program: SpellProgram) -> str:
    """What the dreamer SEES — effect only, no runes."""
    channels = []
    for op in program.ops:
        if op.channel not in channels:
            channels.append(op.channel)
    imagery = ", ".join(CHANNEL_IMAGERY.get(c, c) for c in channels[:2]) or "мерцание"
    verbs = " и ".join(VERB_RU.get(op.verb, op.verb) for op in program.ops[:2])
    feel = ", ".join(field_words(program))
    return (f"Ты видишь {imagery}. Это {DELIVERY_RU.get(program.delivery.kind, '')} "
            f"и {verbs} — {TARGET_RU.get(program.target.selector, '')}. "
            f"Ты чувствуешь: {feel}.")


# ═══════════════════════════════════════════
# Lexicon
# ═══════════════════════════════════════════

@dataclass
class RuneNote:
    """What the player has observed a rune do (correlational, gets sharper)."""
    casts: int = 0
    fields: Counter = field(default_factory=Counter)   # property → signed sum
    verbs: Counter = field(default_factory=Counter)

    def summary(self, k: int = 3) -> str:
        if not self.casts:
            return "ещё не испытана"
        props = sorted(self.fields.items(), key=lambda kv: -abs(kv[1]))[:k]
        words = []
        for prop, v in props:
            if abs(v) / self.casts < 0.02:
                continue
            pos, neg = FIELD_RU.get(prop, (prop, prop))
            words.append(pos if v > 0 else neg)
        verbs = [VERB_RU.get(v, v) for v, _ in self.verbs.most_common(2)]
        sure = "уверенно" if self.casts >= 8 else ("похоже" if self.casts >= 3 else "смутно")
        return f"{sure}: {', '.join(words) or '—'}; {', '.join(verbs) or '—'}"


@dataclass
class Lexicon:
    known: set[str] = field(default_factory=lambda: set(STARTER_RUNES))
    notes: dict[str, RuneNote] = field(default_factory=dict)
    encounters: Counter = field(default_factory=Counter)  # unknown runes met while awake
    discovered: set[str] = field(default_factory=set)
    dreams_solved: int = 0

    def knows(self, rune: str) -> bool:
        return rune in self.known

    def unknown(self) -> list[str]:
        return [r for r in RUNES if r not in self.known]

    def learn(self, rune: str) -> bool:
        if rune in self.known or rune not in RUNES:
            return False
        self.known.add(rune)
        return True

    def familiarity(self, runes) -> float:
        runes = list(runes)
        return sum(r in self.known for r in runes) / len(runes) if runes else 1.0

    def observe(self, program: SpellProgram) -> list[str]:
        """Record what a cast did against every rune in it.

        Unknown runes met three times in waking casts are learned the hard
        way. Returns runes learned by this observation.
        """
        learned = []
        for rune in set(program.runes):
            note = self.notes.setdefault(rune, RuneNote())
            note.casts += 1
            for f in program.fields:
                note.fields[f.property] += f.delta
            for op in program.ops[:2]:
                note.verbs[op.verb] += 1
            if rune not in self.known:
                self.encounters[rune] += 1
                if self.encounters[rune] >= 3 and self.learn(rune):
                    learned.append(rune)
        return learned

    def to_dict(self) -> dict:
        return {
            "known": sorted(self.known),
            "notes": {r: {"casts": n.casts, "fields": dict(n.fields), "verbs": dict(n.verbs)}
                      for r, n in self.notes.items()},
            "encounters": dict(self.encounters),
            "discovered": sorted(self.discovered),
            "dreams_solved": self.dreams_solved,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Lexicon":
        lex = cls(set(d.get("known", STARTER_RUNES)))
        for r, n in d.get("notes", {}).items():
            lex.notes[r] = RuneNote(n["casts"], Counter(n["fields"]), Counter(n["verbs"]))
        lex.encounters = Counter(d.get("encounters", {}))
        lex.discovered = set(d.get("discovered", []))
        lex.dreams_solved = int(d.get("dreams_solved", 0))
        return lex


# ═══════════════════════════════════════════
# Visions
# ═══════════════════════════════════════════

ASPECT_WEIGHTS = {"verb": 0.3, "verbs": 0.15, "delivery": 0.15, "target": 0.1, "feel": 0.2,
                  "new_rune": 0.1}
SOLVED_AT = 0.8


@dataclass
class Vision:
    solution: SpellGenome
    target: SpellProgram
    new_runes: list[str]
    hints: int = 0
    attempts: int = 0
    best: float = 0.0

    @property
    def text(self) -> str:
        return describe_effect(self.target)

    def hint(self) -> str:
        """Each call reveals a bit more of the recipe."""
        self.hints += 1
        sol = self.solution
        if self.hints == 1 and self.new_runes:
            r = self.new_runes[0]
            return f"В видении мерцал незнакомый знак — «{RUNE_RU.get(r, r)}» ({r})."
        if self.hints <= 2:
            known = [RUNE_RU.get(r, r) for r in sol.runes if r not in self.new_runes]
            return (f"Знаков было {len(sol.runes)}; среди знакомых: "
                    f"{', '.join(known) or 'ни одного'}.")
        links = []
        for a, b, t in sol.edges:
            word = {"DIRECT": "→", "MUTUAL": "смешан с", "SEQUENCE": "затем",
                    "CONDITION": "если удастся →", "NEGATE": "отрицает", "ITERATE": "повторяет",
                    "LIMIT": "сдерживает"}[t]
            links.append(f"{RUNE_RU.get(sol.runes[a])} {word} {RUNE_RU.get(sol.runes[b])}")
        return "Связи в видении: " + "; ".join(links)


def aspects(p: SpellProgram) -> dict:
    return {
        "verb": p.primary_verb,
        "verbs": {op.verb for op in p.ops},
        "delivery": p.delivery.kind,
        "target": p.target.selector,
        "feel": {(f.property, f.delta > 0) for f in p.fields[:3]},
    }


@dataclass
class AttemptResult:
    score: float
    feedback: list[tuple[str, bool]]   # (aspect description in Russian, matched)
    solved: bool
    learned: list[str]
    program: SpellProgram

    def text(self) -> str:
        marks = "; ".join(f"{d} — {'да' if ok else 'нет'}" for d, ok in self.feedback)
        heat = ("Видение ясно!" if self.solved else
                "Горячо." if self.score >= 0.6 else "Тепло." if self.score >= 0.35 else "Холодно.")
        return f"{heat} {marks}"


def compare(target: SpellProgram, attempt: SpellProgram, new_runes: list[str]) -> tuple[float, list]:
    a, b = aspects(target), aspects(attempt)
    fb: list[tuple[str, bool]] = []
    score = 0.0

    ok = a["verb"] == b["verb"]
    score += ASPECT_WEIGHTS["verb"] * ok
    fb.append((f"суть («{VERB_RU.get(a['verb'], a['verb'])}»)", ok))

    j = len(a["verbs"] & b["verbs"]) / max(len(a["verbs"] | b["verbs"]), 1)
    score += ASPECT_WEIGHTS["verbs"] * j

    ok = a["delivery"] == b["delivery"]
    score += ASPECT_WEIGHTS["delivery"] * ok
    fb.append((f"движение («{DELIVERY_RU.get(a['delivery'], a['delivery'])}»)", ok))

    ok = a["target"] == b["target"]
    score += ASPECT_WEIGHTS["target"] * ok
    fb.append(("на кого", ok))

    feel = len(a["feel"] & b["feel"]) / max(len(a["feel"]), 1)
    score += ASPECT_WEIGHTS["feel"] * feel
    for prop, positive in sorted(a["feel"]):
        pos, neg = FIELD_RU.get(prop, (prop, prop))
        fb.append((pos if positive else neg, (prop, positive) in b["feel"]))

    used = all(r in attempt.runes for r in new_runes) if new_runes else True
    score += ASPECT_WEIGHTS["new_rune"] * used
    return round(score, 4), fb


class DreamWeaver:
    """Conjures visions just beyond the player's vocabulary and judges attempts."""

    def __init__(self, lexicon: Lexicon,
                 evaluator: Callable[[SpellGenome], Evaluated] = evaluate):
        self.lexicon = lexicon
        self.evaluate = evaluator

    def conjure(self, rng: random.Random | None = None, samples: int = 400) -> Vision | None:
        rng = rng or random.Random()
        lex = self.lexicon
        unknown = lex.unknown()
        # Prefer words that combine with what the player already knows.
        new = rng.choice(unknown) if unknown else None
        known = sorted(lex.known)
        best: tuple[float, Evaluated] | None = None
        seen = set()
        for _ in range(samples):
            gen = self._sample(rng, known, new)
            k = gen.key()
            if k in seen:
                continue
            seen.add(k)
            ev = self.evaluate(gen)
            s = self._interest(ev, new)
            if best is None or s > best[0]:
                best = (s, ev)
        if best is None or best[0] <= 0:
            return None
        ev = best[1]
        return Vision(ev.genome, ev.program, [new] if new else [])

    def attempt(self, vision: Vision, program: SpellProgram) -> AttemptResult:
        """Judge a dream cast. Dreams are safe: no cost, no backfire, no XP."""
        vision.attempts += 1
        score, fb = compare(vision.target, program, vision.new_runes)
        vision.best = max(vision.best, score)
        solved = score >= SOLVED_AT
        learned = []
        if solved:
            for r in vision.new_runes:
                if self.lexicon.learn(r):
                    learned.append(r)
            self.lexicon.dreams_solved += 1
        return AttemptResult(score, fb, solved, learned, program)

    # ── internals ──

    def _sample(self, rng: random.Random, known: list[str], new: str | None) -> SpellGenome:
        """Small spell: the new rune + 1–3 known runes, sensibly wired."""
        n_known = rng.randint(1, 3)
        runes = ([new] if new else []) + [rng.choice(known) for _ in range(n_known)]
        if not new:
            runes.append(rng.choice(known))
        rng.shuffle(runes)
        edges = []
        for i in range(1, len(runes)):
            j = rng.randrange(i)
            kind_i = RUNES[runes[i]][0]
            kind_j = RUNES[runes[j]][0]
            core = {SignalKind.FORCE, SignalKind.DOMAIN}
            if kind_i in core and kind_j in core:
                t = rng.choice(["MUTUAL", "DIRECT"])
            else:
                t = "DIRECT"
            a, b = (i, j) if rng.random() < 0.5 else (j, i)
            # references are sinks
            if RUNES[runes[a]][0] == SignalKind.REFERENCE:
                a, b = b, a
            edges.append((a, b, t))
        return SpellGenome(tuple(runes), tuple(edges), True)

    def _interest(self, ev: Evaluated, new: str | None) -> float:
        p = ev.program
        if new and new not in p.runes:
            return 0.0
        if p.stability < 0.7 or p.risk.backfire_chance > 0.05 or not p.ops:
            return 0.0
        from .combo_data import match_cast
        fresh = [d for d in match_cast(list(p.runes)) if d.key not in self.lexicon.discovered]
        legible = 1.0 if p.fields and abs(p.fields[0].delta) > 0.08 else 0.5
        small = 1.0 / (1.0 + 0.25 * max(len(p.runes) - 2, 0))
        return (p.power + 0.3 * bool(fresh)) * legible * small
