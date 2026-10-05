from wand.actions import (Outcome, apply_signatures, compile_program, json_schema,
                          outcome_to_event)
from wand.algebra import build_fireball, evaluate_spell, graph_from_spec
from wand.checker import check_graph
from wand.combo_data import RECIPES


def prog(g, **kw):
    return compile_program(g, evaluate_spell(g), check=check_graph(g), **kw)


def test_every_recipe_compiles_to_valid_program():
    schema = json_schema()
    for name, r in RECIPES.items():
        p = prog(graph_from_spec(r))
        d = p.to_dict()
        for key in schema["required"]:
            assert key in d, (name, key)
        assert p.ops, name
        assert 0 <= p.power <= 1 and 0 <= p.stability <= 1


def test_fireball_is_a_burning_projectile():
    p = prog(build_fireball())
    assert p.primary_verb == "damage" and p.ops[0].params.get("status") == "burning"
    assert p.delivery.kind == "projectile" and p.target.selector == "target"
    assert p.fields[0].property == "heat" and p.fields[0].delta > 0


def test_intent_candle_warms_instead_of_burning():
    p = prog(graph_from_spec(RECIPES["everyday/candle"]))
    assert p.primary_verb in ("temperature", "light")


def test_world_context_changes_magic():
    g = build_fireball()
    assert prog(g, context={"weather": "rain"}).power < prog(g).power
    assert prog(g, context={"weather": "heat"}).power > prog(g).power


def test_vessel_overload_costs_stability():
    spec = {"nodes": [{"name": n} for n in ["fire", "air", "earth", "water", "light", "dark", "life", "mind"]],
            "edges": [[a, b, "MUTUAL"] for a, b in [("fire", "air"), ("air", "earth"), ("earth", "water"),
                                                     ("water", "light"), ("light", "dark"), ("dark", "life"),
                                                     ("life", "mind"), ("mind", "fire")]]}
    p = prog(graph_from_spec(spec))
    assert p.provenance["vessel"]["overload"] > 1.0
    assert any("overloaded" in r for r in p.risk.reasons)


def test_signatures_visibly_change_spells():
    p = apply_signatures(prog(build_fireball()), {"BIOLOGY": 3, "ELEMENT": 5})
    assert "cauterize" in p.provenance["signatures"] and "blast" in p.provenance["signatures"]
    assert p.ops[0].params.get("lifesteal") == 0.2
    assert any(op.trigger == "on_hit" and op.verb == "push" for op in p.ops)


def test_outcomes_teach_less_when_repeated():
    p = prog(build_fireball())
    o = Outcome(p.id, totals={"damage": 20})
    first, tenth = outcome_to_event(p, o, 0), outcome_to_event(p, o, 9)
    assert tenth.weight < first.weight
    assert first.pole_affinity[0] == 1.0  # fire → ELEMENT pole
