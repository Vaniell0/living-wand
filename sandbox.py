"""
Living Wand — Spell Sandbox (v2: SpellFormer)

Minimal text REPL for exploring the magic system.
No tutorials, no hints. Just you and the wand.

Commands:
  add <symbol>          — place a symbol in the circle
  link <a> <b> [type]   — connect two symbols (type: direct/mutual/seq/cond/neg/iter/limit)
  circle open|closed    — toggle circle state
  cast                  — cast the spell (SpellFormer + algebra comparison)
  clear                 — clear the circle
  wand                  — show wand state
  symbols               — list known symbols
  history               — show casting history
  sleep                 — train wand on recent casts (the wand "dreams")
  crystal               — list crystallized spells
  tags                  — show semantic tags of last cast
  show                  — show current circle
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
from wand.events import Event, EventType, Pole, NUM_POLES, Emotion
from wand.transformer import SpellFormer
from wand.tokenizer import SpellTokenizer
from wand.wand_adapter import WandAdapter, AffinityHexagon, WandAdapterTrainer
from wand.crystal import CrystalRegistry
from wand.effect import interpret_effect
from wand.grimoire import Grimoire

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
POLE_BAR = "\u25b0\u25b1"


def pole_bar(value: float, width: int = 10) -> str:
    filled = int(round(value * width))
    return POLE_BAR[0] * filled + POLE_BAR[1] * (width - filled)


class Sandbox:
    def __init__(self):
        # SpellFormer (shared magic knowledge)
        self.spell_former = SpellFormer()
        model_path = os.path.join(os.path.dirname(__file__), "data", "spellformer.pt")
        if os.path.exists(model_path):
            self.spell_former.load_state_dict(torch.load(model_path, weights_only=True))
            print("SpellFormer loaded")
        else:
            print("SpellFormer: no trained model found, using random init")
        self.spell_former.eval()

        # Wand (personal)
        self.adapter = WandAdapter()
        self.hexagon = AffinityHexagon()
        self.trainer = WandAdapterTrainer(self.adapter, self.hexagon)

        # Crystallization
        self.crystal = CrystalRegistry(threshold=500)
        from pathlib import Path
        crystal_dir = Path(os.path.dirname(__file__)) / "data" / "crystals"
        self.crystal.save_dir = crystal_dir
        self.crystal.load()

        # Tokenizer
        self.tokenizer = SpellTokenizer()

        # Grimoire
        self.grimoire = Grimoire(os.path.join(os.path.dirname(__file__), "data", "grimoire.db"))

        # Load saved state
        save_dir = os.path.join(os.path.dirname(__file__), "data", "save")
        adapter_path = os.path.join(save_dir, "adapter.pt")
        if os.path.exists(adapter_path):
            self.adapter.load_state_dict(torch.load(adapter_path, weights_only=True))
            print("Wand adapter loaded.")
        self.hexagon.load(os.path.join(save_dir, "hexagon.json"))
        self.crystal.save_dir = Path(save_dir) / "crystals"
        self.crystal.load()

        # State
        self.graph = SpellGraph()
        self.node_names: list[str] = []
        self.history: list[dict] = []
        self.pending_events: list[Event] = []
        self.cast_count = self.grimoire.total_casts()
        self.sleep_count = 0
        self.last_effect_desc = None
        self._save_dir = save_dir

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
        print(f"  {a_name} \u2014{edge_type.name}\u2192 {b_name}")

    def _resolve_node(self, ref: str) -> int | None:
        try:
            idx = int(ref)
            if idx in self.graph.nodes:
                return idx
        except ValueError:
            pass
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

        # Show circle
        print(f"\n  \u256d{'\u2500' * 40}\u256e")
        print(f"  \u2502 {'\u25cb' if self.graph.circle_closed else '\u25cc'}  ", end="")
        print(" ".join(f"{self.graph.nodes[i].name}" for i in sorted(self.graph.nodes)))
        print(f"  \u2502 edges: ", end="")
        sym = {"DIRECT":"\u2192","MUTUAL":"\u2550","SEQUENCE":"\u22ef","CONDITION":"?\u2192",
               "NEGATE":"\u2717\u2192","ITERATE":"\u21bb","LIMIT":"\u2502\u2192"}
        for e in self.graph.edges:
            a = self.graph.nodes[e.from_id].name
            b = self.graph.nodes[e.to_id].name
            print(f"{a}{sym.get(e.edge_type.name,'\u2192')}{b} ", end="")
        print(f"\n  \u2570{'\u2500' * 40}\u256f")

        # --- SpellFormer inference ---
        tokens = self.tokenizer.tokenize(self.graph)
        ids = torch.tensor([tokens.token_ids])
        roles = torch.tensor([tokens.role_ids])
        degs = torch.tensor([tokens.degree])

        with torch.no_grad():
            output = self.spell_former(ids, roles, degs)
            base_raw = output["raw"]
            spell_poles = output["pole_distribution"]

            # WandAdapter
            modified = self.adapter(base_raw, spell_poles)

            poles = torch.softmax(modified[:, :6], dim=-1)[0].numpy()
            power = torch.sigmoid(modified[:, 6:7])[0, 0].item()
            stability = torch.sigmoid(modified[:, 7:8])[0, 0].item()
            area = torch.nn.functional.softplus(modified[:, 8:9])[0, 0].item()
            duration = torch.nn.functional.softplus(modified[:, 9:10])[0, 0].item()
            elem_int = torch.sigmoid(modified[:, 10:16])[0].numpy()
            dom_int = torch.sigmoid(modified[:, 16:22])[0].numpy()
            sem_emb = torch.tanh(modified[:, 22:38])[0].numpy()

        # Interpret
        desc = interpret_effect(poles, power, stability, area, duration, elem_int, dom_int, sem_emb)
        self.last_effect_desc = desc

        # Algebra comparison
        alg_effect = evaluate_spell(self.graph)

        # Display SpellFormer result
        print(f"\n  SpellFormer: {desc.title}")
        print(f"  {desc.description}")
        print(f"  power={desc.power:.3f}  stability={desc.stability:.3f}  "
              f"area={desc.area:.2f}  duration={desc.duration:.2f}")

        # Element bars
        for name, intensity in desc.active_elements[:4]:
            bar = pole_bar(intensity, 15)
            print(f"    {name:<8s} {bar} {intensity:.2f}")

        # Domain bars
        for name, intensity in desc.active_domains[:3]:
            bar = pole_bar(intensity, 15)
            print(f"    {name:<8s} {bar} {intensity:.2f}")

        # Tags
        if desc.tags:
            print(f"  tags: {', '.join(f'{t}({s:.2f})' for t, s in desc.tags[:4])}")

        # Crystallization
        canon, crystallized = self.crystal.record_cast(tokens)
        progress = self.crystal.approaching_crystal(tokens)
        if crystallized:
            entry = self.crystal.get_crystal_by_hash(canon)
            print(f"  ** CRYSTALLIZED: {entry.name} -> token #{entry.token_id} **")
            # Set embedding
            with torch.no_grad():
                h = self.spell_former.encode(ids, roles, degs)
                pooled = self.spell_former.pool(h, ids)
                self.crystal.set_crystal_embedding(canon, pooled[0])
                self.spell_former.set_crystal_embedding(entry.crystal_id, pooled[0])
        elif progress > 0 and progress < 1.0:
            print(f"  crystal: {progress:.0%}")

        # Algebra comparison (dimmed)
        print(f"\n  (algebra: power={alg_effect.total_power:.2f} "
              f"stability={alg_effect.stability:.2f} "
              f"synergies={alg_effect.synergies[:2]} "
              f"tags={sorted(alg_effect.tags)[:3]})")

        # Record
        self.cast_count += 1
        pole_aff = poles.tolist()
        weight = 0.1 + 0.3 * min(power * 3, 1.0)
        self.pending_events.append(Event(
            EventType.ACTION, pole_aff,
            Emotion(min(power, 0.9), weight, 0.4),
            weight,
        ))

        self.history.append({
            "nodes": [self.graph.nodes[i].name for i in sorted(self.graph.nodes)],
            "title": desc.title,
            "power": power,
            "tags": [t for t, _ in desc.tags[:3]],
        })

        # Grimoire
        rune_names = [self.graph.nodes[i].name for i in sorted(self.graph.nodes)]
        edge_triples = [(self.graph.nodes[e.from_id].name, e.edge_type.name, self.graph.nodes[e.to_id].name)
                        for e in self.graph.edges]
        self.grimoire.record_cast(
            runes=rune_names, edges=edge_triples,
            circle_closed=self.graph.circle_closed,
            power=power, stability=stability,
            title=desc.title, tags=[t for t, _ in desc.tags[:3]],
        )

    def cmd_clear(self):
        self.graph = SpellGraph()
        self.node_names = []
        print("  cleared")

    def cmd_wand(self):
        profile = self.adapter.affinity_profile().numpy()
        hex_profile = self.hexagon.profile()
        levels = self.hexagon.levels()
        print(f"  wand ({self.adapter.param_count()} params, {self.cast_count} casts, {self.sleep_count} sleeps)")
        print(f"  crystals: {self.crystal.num_crystals}")
        for i, pole in enumerate(Pole):
            bar = pole_bar(profile[i] / 0.5, 20)
            xp_bar = pole_bar(hex_profile[i], 10)
            marker = " \u25c6" if profile[i] == max(profile) else ""
            print(f"    {pole.name:<12s} {bar} {profile[i]:.3f}  xp:{xp_bar} lv{levels[i]}{marker}")

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
            tags = ", ".join(h["tags"]) if h["tags"] else ""
            print(f"  {i+1:3d}. [{syms}] -> {h['title']}  pwr={h['power']:.2f}  {tags}")

    def cmd_sleep(self):
        if not self.pending_events:
            print("  nothing to dream about")
            return
        n = len(self.pending_events)

        # Train adapter on pending events
        # Use a dummy base effect (uniform) for training context
        dummy = torch.zeros(1, self.adapter.effect_dim)
        total_loss = self.trainer.train_events(self.pending_events, dummy)

        self.sleep_count += 1
        self.pending_events = []
        print(f"  ...the wand dreams of {n} moments...")
        print(f"  ...loss: {total_loss:.5f}...")
        self.cmd_wand()

    def cmd_show(self):
        if not self.graph.nodes:
            print("  circle is empty")
            return
        print(f"  circle: {'closed' if self.graph.circle_closed else 'open'}")
        for nid, node in sorted(self.graph.nodes.items()):
            print(f"    [{nid}] {node.name} ({node.signal_kind.name})")
        for e in self.graph.edges:
            a = self.graph.nodes[e.from_id].name
            b = self.graph.nodes[e.to_id].name
            print(f"    {a} \u2014{e.edge_type.name}\u2192 {b}")

    def cmd_crystal(self):
        crystals = self.crystal.list_crystals()
        if not crystals:
            print("  no crystallized spells yet")
            return
        for c in crystals:
            emb = "embedded" if c.embedding is not None else "pending"
            print(f"  #{c.token_id}: {c.name} (casts={c.cast_count}, {emb})")

    def cmd_grimoire(self):
        casts = self.grimoire.recent_casts(10)
        recipes = self.grimoire.top_recipes(5)
        total = self.grimoire.total_casts()
        print(f"  grimoire: {total} total casts, {len(recipes)} recipes")
        if recipes:
            print("  top recipes:")
            for r in recipes:
                print(f"    {r.name} ({'+'.join(r.runes)}) x{r.times_cast} best={r.best_power:.2f}")
        if casts:
            print("  recent casts:")
            for c in casts[:5]:
                print(f"    {c.timestamp[11:19]} {c.title} pwr={c.power:.2f} [{','.join(c.tags)}]")

    def cmd_save(self):
        os.makedirs(self._save_dir, exist_ok=True)
        torch.save(self.adapter.state_dict(), os.path.join(self._save_dir, "adapter.pt"))
        self.hexagon.save(os.path.join(self._save_dir, "hexagon.json"))
        self.crystal.save()
        print("  saved")

    def cmd_tags(self):
        if self.last_effect_desc is None:
            print("  cast a spell first")
            return
        desc = self.last_effect_desc
        print(f"  domain: {desc.primary_domain}")
        for tag_name, sim in desc.tags:
            bar = pole_bar(max(0, sim), 10)
            print(f"    {tag_name:<16s} {bar} {sim:.3f}")

    def run(self):
        print("Living Wand v2 \u2014 Spell Sandbox (SpellFormer)")
        print("type 'symbols' to see what you can draw")
        print()

        while True:
            try:
                line = input("\u27e9 ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not line:
                continue

            parts = line.split()
            cmd = parts[0].lower()

            if cmd in ("quit", "q"):
                self.cmd_save()
                self.grimoire.close()
                break
            elif cmd in ("add", "a"):
                if len(parts) >= 2:
                    for sym in parts[1:]:
                        self.cmd_add(sym)
                else:
                    print("  usage: add <symbol> [symbol2 ...]")
            elif cmd in ("link", "l"):
                self.cmd_link(parts[1:])
            elif cmd == "circle":
                if len(parts) >= 2:
                    self.cmd_circle(parts[1])
            elif cmd == "cast":
                self.cmd_cast()
            elif cmd == "clear":
                self.cmd_clear()
            elif cmd in ("wand", "w"):
                self.cmd_wand()
            elif cmd in ("symbols", "sym"):
                self.cmd_symbols()
            elif cmd in ("history", "h"):
                self.cmd_history()
            elif cmd == "sleep":
                self.cmd_sleep()
            elif cmd in ("show", "s"):
                self.cmd_show()
            elif cmd in ("crystal", "cr"):
                self.cmd_crystal()
            elif cmd in ("tags", "t"):
                self.cmd_tags()
            elif cmd in ("grimoire", "gr"):
                self.cmd_grimoire()
            elif cmd == "save":
                self.cmd_save()
            else:
                # Symbol shorthand
                if cmd in SYMBOLS:
                    for sym in parts:
                        if sym in SYMBOLS:
                            self.cmd_add(sym)
                else:
                    print(f"  ? {cmd}")


if __name__ == "__main__":
    Sandbox().run()
