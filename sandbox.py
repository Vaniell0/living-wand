"""
Living Wand — Spell Sandbox

Minimal text REPL for exploring the magic system.
No tutorials, no hints. Just you and the wand.

Commands:
  add <symbol>          — place a symbol in the circle
  link <a> <b> [type]   — connect two symbols (type: direct/mutual/seq/cond/neg/iter/limit)
  circle open|closed    — toggle circle state
  cast                  — cast the spell
  clear                 — clear the circle
  wand                  — show wand state
  symbols               — list known symbols
  history               — show casting history
  sleep                 — train wand on recent casts (the wand "dreams")
  quit
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import random
import numpy as np
import torch

from wand.algebra import (
    SpellGraph, evaluate_spell,
    SignalKind, Element, EdgeType,
)
from wand.model import create_wand, WandTrainer, affinity_multiplier
from wand.events import Event, EventType, Pole, NUM_POLES, Emotion

# Symbol catalog — what the player can discover
SYMBOLS = {
    # Elements (FORCE)
    "fire":   (SignalKind.FORCE, Element.FIRE),
    "water":  (SignalKind.FORCE, Element.WATER),
    "air":    (SignalKind.FORCE, Element.AIR),
    "earth":  (SignalKind.FORCE, Element.EARTH),
    "light":  (SignalKind.FORCE, Element.LIGHT),
    "dark":   (SignalKind.FORCE, Element.DARK),
    # Domains
    "body":   (SignalKind.DOMAIN, None),
    "mind":   (SignalKind.DOMAIN, None),
    "matter": (SignalKind.DOMAIN, None),
    "space":  (SignalKind.DOMAIN, None),
    "time":   (SignalKind.DOMAIN, None),
    "life":   (SignalKind.DOMAIN, None),
    # Directions
    "inward":  (SignalKind.VECTOR, None),
    "outward": (SignalKind.VECTOR, None),
    "up":      (SignalKind.VECTOR, None),
    "down":    (SignalKind.VECTOR, None),
    "through": (SignalKind.VECTOR, None),
    "reverse": (SignalKind.VECTOR, None),
    # Quantity
    "point": (SignalKind.SCALAR, None),
    "flow":  (SignalKind.SCALAR, None),
    "wave":  (SignalKind.SCALAR, None),
    "all":   (SignalKind.SCALAR, None),
    # Modifiers
    "amplify": (SignalKind.MODIFIER, None),
    "weaken":  (SignalKind.MODIFIER, None),
    "hold":    (SignalKind.MODIFIER, None),
    "dispel":  (SignalKind.MODIFIER, None),
    # References
    "here":   (SignalKind.REFERENCE, None),
    "there":  (SignalKind.REFERENCE, None),
    "self":   (SignalKind.REFERENCE, None),
    "target": (SignalKind.REFERENCE, None),
}

EDGE_TYPES = {
    "direct": EdgeType.DIRECT,
    "mutual": EdgeType.MUTUAL,
    "seq":    EdgeType.SEQUENCE,
    "cond":   EdgeType.CONDITION,
    "neg":    EdgeType.NEGATE,
    "iter":   EdgeType.ITERATE,
    "limit":  EdgeType.LIMIT,
    "d": EdgeType.DIRECT,
    "m": EdgeType.MUTUAL,
    "s": EdgeType.SEQUENCE,
    "c": EdgeType.CONDITION,
    "n": EdgeType.NEGATE,
    "i": EdgeType.ITERATE,
    "l": EdgeType.LIMIT,
}

POLE_NAMES = [p.name.lower() for p in Pole]
POLE_BAR = "▰▱"


def pole_bar(value: float, width: int = 10) -> str:
    filled = int(round(value * width))
    return POLE_BAR[0] * filled + POLE_BAR[1] * (width - filled)


def format_effect(effect, wand_mod: dict | None = None) -> str:
    """Format a SpellEffect for display."""
    lines = []

    if effect.synergies:
        for s in effect.synergies:
            lines.append(f"  ✦ {s}")
    if effect.cancellations:
        for c in effect.cancellations:
            lines.append(f"  ✗ {c}")

    # Force
    if effect.force_power:
        for elem, power in sorted(effect.force_power.items(), key=lambda x: -x[1]):
            bar = pole_bar(min(power / 3.0, 1.0), 15)
            lines.append(f"  {elem.name:<6s} {bar} {power:.2f}")

    # Domain
    if effect.domain_power:
        for domain, power in sorted(effect.domain_power.items(), key=lambda x: -x[1]):
            bar = pole_bar(min(power / 3.0, 1.0), 15)
            lines.append(f"  {domain:<6s} {bar} {power:.2f}")

    lines.append(f"  power={effect.total_power:.2f}  stability={effect.stability:.2f}  "
                 f"area={effect.area:.2f}  duration={effect.duration:.2f}")

    if effect.tags:
        lines.append(f"  [{', '.join(sorted(effect.tags))}]")

    if effect.warnings:
        for w in effect.warnings:
            lines.append(f"  ⚠ {w}")

    if wand_mod:
        lines.append(f"  wand: ×{wand_mod['multiplier']:.2f} → final={wand_mod['final_power']:.2f}")

    return "\n".join(lines)


def effect_to_events(effect, success: float) -> list[Event]:
    """Convert a cast result to training events for the wand."""
    events = []

    # The act of casting itself
    pole_aff = [0.0] * NUM_POLES
    for elem, power in effect.force_power.items():
        pole_aff[Pole.ELEMENT] += abs(power)
    domain_map = {"body": Pole.BIOLOGY, "life": Pole.BIOLOGY,
                  "mind": Pole.MENTAL, "matter": Pole.TRANSFORM,
                  "space": Pole.SPACE, "time": Pole.TIME}
    for domain, power in effect.domain_power.items():
        p = domain_map.get(domain, Pole.ELEMENT)
        pole_aff[p] += abs(power)

    # Normalize
    total = sum(pole_aff)
    if total > 0:
        pole_aff = [v / total for v in pole_aff]

    weight = 0.1 + 0.3 * min(effect.total_power, 3.0) / 3.0
    valence = success * 0.5 + 0.2  # success feels good

    events.append(Event(
        event_type=EventType.ACTION,
        pole_affinity=pole_aff,
        emotion=Emotion(valence=valence, intensity=weight, arousal=0.4),
        weight=weight,
    ))

    return events


class Sandbox:
    def __init__(self):
        self.wand = create_wand(seed=42)
        self.trainer = WandTrainer(self.wand, lr=5e-4)
        self.graph = SpellGraph()
        self.node_names: list[str] = []  # ordered list of placed symbols
        self.history: list[dict] = []
        self.pending_events: list[Event] = []
        self.cast_count = 0
        self.sleep_count = 0

    def cmd_add(self, name: str):
        if name not in SYMBOLS:
            print(f"  unknown symbol: {name}")
            return
        kind, elem = SYMBOLS[name]
        nid = self.graph.add_node(name, kind, elem)
        self.node_names.append(name)
        print(f"  [{nid}] {name}")

    def cmd_link(self, parts: list[str]):
        if len(parts) < 2:
            print("  usage: link <a> <b> [type]")
            return

        # Parse a, b as indices or names
        a, b = parts[0], parts[1]
        edge_type = EdgeType.DIRECT
        if len(parts) >= 3 and parts[2] in EDGE_TYPES:
            edge_type = EDGE_TYPES[parts[2]]

        a_id = self._resolve_node(a)
        b_id = self._resolve_node(b)
        if a_id is None or b_id is None:
            return

        self.graph.connect(a_id, b_id, edge_type)
        a_name = self.graph.nodes[a_id].name
        b_name = self.graph.nodes[b_id].name
        print(f"  {a_name} —{edge_type.name}→ {b_name}")

    def _resolve_node(self, ref: str) -> int | None:
        # Try as index
        try:
            idx = int(ref)
            if idx in self.graph.nodes:
                return idx
        except ValueError:
            pass
        # Try as name (last matching)
        for nid in reversed(list(self.graph.nodes.keys())):
            if self.graph.nodes[nid].name == ref:
                return nid
        print(f"  node not found: {ref}")
        return None

    def cmd_circle(self, state: str):
        if state in ("open", "o"):
            self.graph.circle_closed = False
            print("  circle: open")
        elif state in ("closed", "c"):
            self.graph.circle_closed = True
            print("  circle: closed")

    def cmd_cast(self):
        if not self.graph.nodes:
            print("  nothing to cast")
            return

        # Show what's in the circle
        print(f"\n  ╭{'─' * 40}╮")
        print(f"  │ {'○' if self.graph.circle_closed else '◌'}  ", end="")
        print(" ".join(f"{self.graph.nodes[i].name}" for i in sorted(self.graph.nodes)))
        print(f"  │ edges: ", end="")
        for e in self.graph.edges:
            a = self.graph.nodes[e.from_id].name
            b = self.graph.nodes[e.to_id].name
            sym = {"DIRECT":"→","MUTUAL":"═","SEQUENCE":"⋯","CONDITION":"?→",
                   "NEGATE":"✗→","ITERATE":"↻","LIMIT":"│→"}
            print(f"{a}{sym.get(e.edge_type.name,'→')}{b} ", end="")
        print(f"\n  ╰{'─' * 40}╯")

        # Evaluate
        effect = evaluate_spell(self.graph)

        # Wand modifier
        profile = self.wand.affinity_profile().numpy()
        wand_dominant = int(np.argmax(profile))
        # Simple modifier: how well the wand's personality aligns with the spell
        spell_poles = [0.0] * NUM_POLES
        for elem in effect.force_power:
            spell_poles[Pole.ELEMENT] += abs(effect.force_power[elem])
        domain_map = {"body": Pole.BIOLOGY, "life": Pole.BIOLOGY,
                      "mind": Pole.MENTAL, "matter": Pole.TRANSFORM,
                      "space": Pole.SPACE, "time": Pole.TIME}
        for domain in effect.domain_power:
            p = domain_map.get(domain, Pole.ELEMENT)
            spell_poles[p] += abs(effect.domain_power[domain])
        total_sp = sum(spell_poles)
        if total_sp > 0:
            spell_poles = [v / total_sp for v in spell_poles]
        # Dot product with profile = alignment
        alignment = sum(p * s for p, s in zip(profile, spell_poles))
        multiplier = 0.5 + alignment  # range ~0.5 to ~1.0
        final_power = effect.total_power * multiplier

        wand_mod = {"multiplier": multiplier, "final_power": final_power}

        print(format_effect(effect, wand_mod))

        # Record
        self.cast_count += 1
        success = min(final_power / 3.0, 1.0)
        events = effect_to_events(effect, success)
        self.pending_events.extend(events)

        self.history.append({
            "nodes": [self.graph.nodes[i].name for i in sorted(self.graph.nodes)],
            "effect": effect,
            "final_power": final_power,
        })

    def cmd_clear(self):
        self.graph = SpellGraph()
        self.node_names = []
        print("  cleared")

    def cmd_wand(self):
        profile = self.wand.affinity_profile().numpy()
        print(f"  wand ({self.wand.param_count()} params, {self.cast_count} casts, {self.sleep_count} sleeps)")
        for i, pole in enumerate(Pole):
            bar = pole_bar(profile[i] / 0.5, 20)
            marker = " ◆" if profile[i] == max(profile) else ""
            print(f"    {pole.name:<12s} {bar} {profile[i]:.3f}{marker}")

    def cmd_symbols(self):
        by_kind = {}
        for name, (kind, _) in SYMBOLS.items():
            by_kind.setdefault(kind.name, []).append(name)
        for kind, names in by_kind.items():
            print(f"  {kind}: {', '.join(names)}")

    def cmd_history(self):
        if not self.history:
            print("  no casts yet")
            return
        for i, h in enumerate(self.history[-10:]):
            syms = " ".join(h["nodes"])
            pwr = h["final_power"]
            tags = ", ".join(sorted(h["effect"].tags)) if h["effect"].tags else ""
            print(f"  {i+1:3d}. [{syms}] → {pwr:.2f}  {tags}")

    def cmd_sleep(self):
        if not self.pending_events:
            print("  nothing to dream about")
            return
        n = len(self.pending_events)
        losses = self.trainer.train_biography(self.pending_events, epochs=3, batch_size=max(n, 8))
        self.sleep_count += 1
        self.pending_events = []
        print(f"  ...the wand dreams of {n} moments...")
        print(f"  ...loss: {losses[-1]:.5f}...")
        # Show profile shift
        self.cmd_wand()

    def cmd_show(self):
        """Show current circle contents."""
        if not self.graph.nodes:
            print("  circle is empty")
            return
        print(f"  circle: {'closed' if self.graph.circle_closed else 'open'}")
        for nid, node in sorted(self.graph.nodes.items()):
            print(f"    [{nid}] {node.name} ({node.signal_kind.name})")
        for e in self.graph.edges:
            a = self.graph.nodes[e.from_id].name
            b = self.graph.nodes[e.to_id].name
            print(f"    {a} —{e.edge_type.name}→ {b}")

    def run(self):
        print("Living Wand — Spell Sandbox")
        print("type 'symbols' to see what you can draw")
        print()

        while True:
            try:
                line = input("⟩ ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not line:
                continue

            parts = line.split()
            cmd = parts[0].lower()

            if cmd == "quit" or cmd == "q":
                break
            elif cmd == "add" or cmd == "a":
                if len(parts) >= 2:
                    for sym in parts[1:]:
                        self.cmd_add(sym)
                else:
                    print("  usage: add <symbol> [symbol2 ...]")
            elif cmd == "link" or cmd == "l":
                self.cmd_link(parts[1:])
            elif cmd == "circle":
                if len(parts) >= 2:
                    self.cmd_circle(parts[1])
            elif cmd == "cast":
                self.cmd_cast()
            elif cmd == "clear":
                self.cmd_clear()
            elif cmd == "wand" or cmd == "w":
                self.cmd_wand()
            elif cmd == "symbols" or cmd == "sym":
                self.cmd_symbols()
            elif cmd == "history" or cmd == "h":
                self.cmd_history()
            elif cmd == "sleep":
                self.cmd_sleep()
            elif cmd == "show" or cmd == "s":
                self.cmd_show()
            else:
                # Try as symbol shorthand: just type the symbol name to add it
                if cmd in SYMBOLS:
                    for sym in parts:
                        if sym in SYMBOLS:
                            self.cmd_add(sym)
                else:
                    print(f"  ? {cmd}")


if __name__ == "__main__":
    Sandbox().run()
