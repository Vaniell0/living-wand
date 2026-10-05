"""Alchemy and enchanting — the same magic substrate, different hands.

Nothing here has its own rules engine. Both crafts build an ordinary
SpellGraph and compile it through the same algebra + compiler as a drawn
spell; only three things change:

                 spell (drawn)        alchemy (brewed)        enchanting (bound)
    words        runes                ingredient natures       runes
    vessel       the drawn circle     the flask               the item's material
    delivery     from vectors/refs    consume / splash         imbue + trigger
    world        weather/time/place   (the cauldron)          material affinity

So every discovery transfers: what the player learns about fire+earth while
casting is true in the cauldron and in the forge — and the explorer, dreams
and the engine API work on potions and enchantments unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .actions import Delivery, SpellProgram, Target, _data, compile_program
from .algebra import RUNES, EdgeType, SignalKind, SpellGraph, evaluate_spell
from .checker import check_graph


def ingredients() -> dict:
    return _data("ingredients.json")["ingredients"]


def materials() -> dict:
    return _data("materials.json")["materials"]


# ═══════════════════════════════════════════
# Alchemy
# ═══════════════════════════════════════════

@dataclass
class Potion:
    name: str
    ingredients: list[str]
    steps: list[str]
    flask: str
    program: SpellProgram
    graph: SpellGraph = field(repr=False)

    @property
    def quality(self) -> float:
        """Stable brews are good brews; overloading the flask clouds them."""
        return round(self.program.stability * (1.0 - self.program.risk.backfire_chance), 3)


def brew_graph(ingredient_ids: list[str], steps: list[str]) -> SpellGraph:
    """Ingredients' natures, mixed (MUTUAL between their cores), then processed."""
    table = ingredients()
    step_table = _data("ingredients.json")["steps"]
    g = SpellGraph(circle_closed=True)
    cores: list[int] = []
    for ing in ingredient_ids:
        spec = table[ing]
        ids = [g.add_rune(r) for r in spec["runes"]]
        for a, b, t in spec["edges"]:
            g.connect(ids[a], ids[b], EdgeType[t])
        core = next((i for i in ids if g.nodes[i].signal_kind in
                     (SignalKind.FORCE, SignalKind.DOMAIN)), ids[0])
        if cores:
            g.connect(cores[-1], core, EdgeType.MUTUAL)  # stirring mixes natures
        cores.append(core)
    for step in steps:
        st = step_table[step]
        anchor = cores[0] if cores else None
        for rune in st["add"]:
            nid = g.add_rune(rune)
            if anchor is not None:
                g.connect(nid, anchor, EdgeType[st["edge"]])
    return g


def brew(ingredient_ids: list[str], steps: list[str] = (), flask: str = "glass",
         mode: str = "drink", **compile_kw) -> Potion:
    """Brew a potion. mode: "drink" (consume, self) or "throw" (splash, point)."""
    graph = brew_graph(list(ingredient_ids), list(steps))
    flasks = _data("ingredients.json")["flasks"]
    program = compile_program(graph, evaluate_spell(graph), check=check_graph(graph),
                              vessel_capacity=flasks[flask]["capacity"], **compile_kw)
    if mode == "throw":
        program.delivery = Delivery("splash", speed=8.0)
        program.target = Target("point", 6.0)
    else:
        program.delivery = Delivery("consume")
        program.target = Target("self", 0.0)
    names = [ingredients()[i]["name_ru"] for i in ingredient_ids]
    name = f"Зелье: {program.title}" if "+" not in program.title else "Зелье из " + ", ".join(names)
    program.provenance["craft"] = {"kind": "alchemy", "ingredients": list(ingredient_ids),
                                   "steps": list(steps), "flask": flask, "mode": mode}
    return Potion(name, list(ingredient_ids), list(steps), flask, program, graph)


# ═══════════════════════════════════════════
# Enchanting
# ═══════════════════════════════════════════

@dataclass
class Enchantment:
    item: str
    material: str
    trigger: str
    charges: int | None          # None = permanent (the item holds it easily)
    program: SpellProgram
    graph: SpellGraph = field(repr=False)


def enchant(graph: SpellGraph, material: str, trigger: str = "on_hit", item: str = "",
            **compile_kw) -> Enchantment:
    """Bind a spell to an item. The material is the vessel and colours the magic."""
    mat = materials()[material]
    trig = _data("materials.json")["triggers"][trigger]
    affinity = dict(mat.get("affinity", {}))
    if "stability" in mat:
        affinity["_stability"] = mat["stability"]
    program = compile_program(graph, evaluate_spell(graph), check=check_graph(graph),
                              affinity=affinity, vessel_capacity=mat["capacity"], **compile_kw)
    if trig["target"] is not None:
        rng = 1.5 if trig["target"] == "target" else 0.0
        program.target = Target(trig["target"], rng)
    program.delivery = Delivery("imbue", direction=trigger)
    for op in program.ops:
        op.trigger = trigger
    for f in program.fields:
        f.trigger = trigger
    load = program.provenance["vessel"]["load"]
    room = mat["capacity"] - load
    # an item that holds the spell with room to spare keeps it forever;
    # a tight fit burns out after a few discharges
    charges = None if room >= mat["capacity"] * 0.4 and program.stability > 0.8 else max(1, int(3 + 2 * room))
    program.provenance["craft"] = {"kind": "enchanting", "material": material,
                                   "trigger": trigger, "charges": charges}
    return Enchantment(item or mat["name_ru"], material, trigger, charges, program, graph)
