"""
Living Wand — text sandbox (WandSession REPL)

Build a spell by hand, cast it, read the program the engine would receive.

  add <rune>                 place a rune            link <a> <b> [type]   connect (d m s c n i l)
  circle open|closed         toggle the circle       clear / show          reset / print the circle
  cast                       cast it                 json                  last program as JSON
  world <axis> <value>       weather|time|place      world clear           ordinary world
  dream                      sleep and see a vision  hint                  ask the dream
  try                        reproduce the vision with the current circle
  brew <ing ...> [/ <step ...>] [in <flask>] [throw]      e.g. brew heartroot sunpetal / steep in crystal
  enchant <material> [trigger]                          bind the current circle to an item
  lexicon                    your dictionary         wand                  wand profile & signatures
  crafts                     ingredients, steps, materials
  explore [n]                run the magic explorer and print the balance report
  quit
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from wand.algebra import RUNES, EdgeType, SpellGraph  # noqa: E402
from wand.dream import FIELD_RU, RUNE_RU, STATUS_RU, VERB_RU  # noqa: E402
from wand.events import Pole  # noqa: E402
from wand.session import WandSession  # noqa: E402

EDGE_ALIASES = {
    "d": "DIRECT", "m": "MUTUAL", "s": "SEQUENCE", "c": "CONDITION", "n": "NEGATE",
    "i": "ITERATE", "l": "LIMIT", "direct": "DIRECT", "mutual": "MUTUAL", "seq": "SEQUENCE",
    "cond": "CONDITION", "neg": "NEGATE", "iter": "ITERATE", "limit": "LIMIT",
}
EDGE_SYM = {"DIRECT": "→", "MUTUAL": "═", "SEQUENCE": "⋯", "CONDITION": "?→",
            "NEGATE": "¬", "ITERATE": "↻", "LIMIT": "⊣"}
BOX_TOP = "╭" + "─" * 46 + "╮"
BOX_BOTTOM = "╰" + "─" * 46 + "╯"


def bar(v: float, width: int = 14) -> str:
    n = int(round(max(0.0, min(v, 1.0)) * width))
    return "▰" * n + "▱" * (width - n)


def print_program(p) -> None:
    print(f"\n  ✧ {p.title}")
    print(f"    сила {bar(p.power)} {p.power:.2f}   стабильность {bar(p.stability)} {p.stability:.2f}")
    print(f"    цена {p.cost:.0f}   заряд {p.cast_time:.2f}s   риск отдачи {p.risk.backfire_chance:.0%}"
          + (f"  ({'; '.join(p.risk.reasons)})" if p.risk.reasons else ""))
    print(f"    {p.delivery.kind} → {p.target.selector}  ·  форма {p.shape.kind} r={p.shape.radius}")
    for op in p.ops:
        status = op.params.get("status")
        extra = f" ({STATUS_RU.get(status, status)})" if status else ""
        trig = "" if op.trigger == "cast" else f" [{op.trigger}]"
        print(f"      • {VERB_RU.get(op.verb, op.verb)}{extra}  {op.channel}  {op.amount:.2f}{trig}")
    if p.fields:
        feel = []
        for f in p.fields:
            pos, neg = FIELD_RU.get(f.property, (f.property, f.property))
            feel.append(f"{pos if f.delta > 0 else neg} {abs(f.delta):.2f}")
        print(f"    в мире: {', '.join(feel)}")
    sig = p.provenance.get("signatures")
    if sig:
        print(f"    подпись палочки: {', '.join(sig)}")
    ctx = p.provenance.get("context_effects")
    if ctx:
        print(f"    мир повлиял: {ctx}")


class Sandbox:
    def __init__(self):
        self.s = WandSession(load_ocr=False)
        self.graph = SpellGraph()
        self.last = None
        print(f"SpellFormer: {'обучен' if self.s.model_trained else 'нет весов — работает алгебра'}"
              f"  ·  слов знаешь: {len(self.s.lexicon.known)}/30")

    # ── circle ──

    def cmd_add(self, args):
        if not args or args[0] not in RUNES:
            print("  неизвестная руна; руны:", " ".join(RUNES))
            return
        nid = self.graph.add_rune(args[0])
        known = "" if self.s.lexicon.knows(args[0]) else "  (палочка ещё не знает этого слова)"
        print(f"  [{nid}] {args[0]}{known}")

    def _resolve(self, ref: str):
        if ref.isdigit() and int(ref) in self.graph.nodes:
            return int(ref)
        for nid in reversed(list(self.graph.nodes)):
            if self.graph.nodes[nid].name == ref:
                return nid
        print(f"  нет такого узла: {ref}")
        return None

    def cmd_link(self, args):
        if len(args) < 2:
            print("  link <a> <b> [type]")
            return
        a, b = self._resolve(args[0]), self._resolve(args[1])
        if a is None or b is None:
            return
        etype = EdgeType[EDGE_ALIASES.get(args[2], "DIRECT")] if len(args) > 2 else EdgeType.DIRECT
        self.graph.connect(a, b, etype)
        print(f"  {self.graph.nodes[a].name} {EDGE_SYM[etype.name]} {self.graph.nodes[b].name}")

    def cmd_show(self, _=None):
        g = self.graph
        print("  " + BOX_TOP)
        mark = "○" if g.circle_closed else "◌"
        print(f"  │ {mark} " + " ".join(g.nodes[i].name for i in sorted(g.nodes)))
        edges = " ".join(f"{g.nodes[e.from_id].name}{EDGE_SYM[e.edge_type.name]}{g.nodes[e.to_id].name}"
                         for e in g.edges)
        print(f"  │ {edges}")
        print("  " + BOX_BOTTOM)

    def cmd_circle(self, args):
        self.graph.circle_closed = not (args and args[0].startswith("o"))
        print(f"  круг {'замкнут' if self.graph.circle_closed else 'разомкнут'}")

    def cmd_cast(self, _=None):
        if not self.graph.nodes:
            print("  круг пуст")
            return
        self.cmd_show()
        report = self.s.cast(self.graph)
        self.last = report.program
        print_program(report.program)
        # no world here: report the program's own intent as its outcome
        from wand.actions import Outcome
        p = report.program
        self.s.report_outcome(Outcome(p.id, totals={op.verb: op.amount * 100 for op in p.ops}))
        for r in report.program.provenance.get("learned", []):
            print(f"  ✦ палочка поняла слово «{RUNE_RU[r]}»")
        for d in report.discoveries:
            print(f"  открытие: {d.name_ru}")

    def cmd_json(self, _=None):
        print(self.last.to_json(indent=1) if self.last else "  ещё ничего не сотворено")

    # ── world, dreams, crafts ──

    def cmd_world(self, args):
        if not args or args[0] == "clear":
            self.s.context = None
            print("  мир обычный")
            return
        ctx = dict(self.s.context or {})
        ctx[args[0]] = args[1] if len(args) > 1 else ""
        self.s.context = ctx
        print(f"  мир: {ctx}")

    def cmd_dream(self, _=None):
        v = self.s.dream()
        if v is None:
            print("  сон без видений")
            return
        print(f"\n  ☾ {v.text}\n  повтори увиденное: собери круг и скажи `try` (подсказка: `hint`)")

    def cmd_hint(self, _=None):
        print("  " + (self.s.vision.hint() if self.s.vision else "ты не спишь"))

    def cmd_try(self, _=None):
        res = self.s.dream_attempt(self.graph)
        if res is None:
            print("  ты не спишь — `dream`")
            return
        print(f"  {res.text()}  ({res.score:.2f})")
        for r in res.learned:
            print(f"  ✦ новое слово: «{RUNE_RU[r]}» ({r})")

    def cmd_brew(self, args):
        from wand.craft import brew
        mode = "throw" if "throw" in args else "drink"
        args = [a for a in args if a != "throw"]
        flask = "glass"
        if "in" in args:
            k = args.index("in")
            flask, args = args[k + 1], args[:k]
        ings, steps = (args[:args.index("/")], args[args.index("/") + 1:]) if "/" in args else (args, [])
        try:
            potion = brew(ings, steps, flask, mode, context=self.s.context)
        except KeyError as e:
            print(f"  нет такого: {e} — смотри `crafts`")
            return
        self.last = potion.program
        print(f"\n  ⚗ {potion.name}  · качество {potion.quality:.2f}")
        print_program(potion.program)

    def cmd_enchant(self, args):
        from wand.craft import enchant
        if not args or not self.graph.nodes:
            print("  enchant <material> [trigger] — сначала собери круг")
            return
        try:
            ench = enchant(self.graph, args[0], args[1] if len(args) > 1 else "on_hit")
        except KeyError as e:
            print(f"  нет такого: {e} — смотри `crafts`")
            return
        self.last = ench.program
        charges = "навсегда" if ench.charges is None else f"{ench.charges} зарядов"
        print(f"\n  ✠ {ench.item}: {ench.trigger}, {charges}")
        print_program(ench.program)

    def cmd_crafts(self, _=None):
        from wand.actions import _data
        ing, mat = _data("ingredients.json"), _data("materials.json")
        print("  ингредиенты:", ", ".join(f"{k} ({v['name_ru']})" for k, v in ing["ingredients"].items()))
        print("  шаги:", ", ".join(f"{k} ({v['name_ru']})" for k, v in ing["steps"].items() if not k.startswith("_")))
        print("  колбы:", ", ".join(ing["flasks"]))
        print("  материалы:", ", ".join(f"{k} ({v['name_ru']})" for k, v in mat["materials"].items()))
        print("  триггеры:", ", ".join(mat["triggers"]))

    # ── wand ──

    def cmd_lexicon(self, _=None):
        lex = self.s.lexicon
        print(f"  слов: {len(lex.known)}/30   видений разгадано: {lex.dreams_solved}")
        for r in RUNES:
            if lex.knows(r):
                note = lex.notes.get(r)
                print(f"    {RUNE_RU[r]:<12} {note.summary() if note else 'ещё не испытана'}")

    def cmd_wand(self, _=None):
        levels = self.s.pole_levels()
        prof = self.s.hexagon.profile()
        for i, pole in enumerate(Pole):
            print(f"    {pole.name.lower():<10} {bar(prof[i] / 0.5, 12)} ур. {levels[pole.name]}")
        from wand.actions import signature_catalog
        for sig in signature_catalog():
            got = levels.get(sig["pole"], 0) >= sig["level"]
            print(f"    {'✓' if got else '·'} {sig['name']} ({sig['pole'].lower()} {sig['level']}): {sig['desc']}")

    def cmd_explore(self, args):
        from wand.explore import exploit_report, map_elites
        n = int(args[0]) if args else 3000
        arch = map_elites(iterations=n)
        print(json.dumps(exploit_report(arch), ensure_ascii=False, indent=1))

    def cmd_clear(self, _=None):
        self.graph = SpellGraph()
        print("  круг очищен")

    def run(self):
        print(__doc__)
        while True:
            try:
                line = input("wand> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            cmd, *args = line.split()
            if cmd in ("quit", "exit", "q"):
                break
            fn = getattr(self, f"cmd_{cmd}", None)
            if fn is None:
                print("  неизвестная команда")
                continue
            fn(args)
        self.s.save()
        self.s.close()


if __name__ == "__main__":
    Sandbox().run()
