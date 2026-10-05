from wand.algebra import (ALGEBRA_PRESETS, EdgeType, SpellGraph, build_fireball, evaluate_spell,
                          graph_from_spec)
from wand.checker import check_graph
from wand.combo_data import RECIPES
from wand.tokenizer import SpellTokenizer


def _mod(mod):
    g = SpellGraph()
    f = g.add_rune("fire")
    m = g.add_rune(mod)
    g.connect(m, f)
    return evaluate_spell(g)


def test_modifiers_have_distinct_meaning():
    base = SpellGraph()
    base.add_rune("fire")
    p0 = evaluate_spell(base).total_power
    assert _mod("amplify").total_power > p0
    assert _mod("weaken").total_power < p0
    assert "sustained" in _mod("hold").tags and _mod("hold").duration > 1.0
    assert _mod("dispel").total_power < _mod("weaken").total_power
    assert "amplified" not in _mod("weaken").tags


def test_open_circle_leaks_power():
    g = build_fireball()
    closed = evaluate_spell(g).total_power
    g.circle_closed = False
    assert evaluate_spell(g).total_power < closed * 0.5


def _two_orders():
    a = SpellGraph()
    w, f, h = a.add_rune("weaken"), a.add_rune("fire"), a.add_rune("here")
    a.connect(f, h)
    a.connect(w, f)
    b = SpellGraph()
    h2, f2, w2 = b.add_rune("here"), b.add_rune("fire"), b.add_rune("weaken")
    b.connect(w2, f2)
    b.connect(f2, h2)
    return a, b


def test_evaluation_independent_of_drawing_order():
    a, b = _two_orders()
    ea, eb = evaluate_spell(a), evaluate_spell(b)
    assert abs(ea.total_power - eb.total_power) < 1e-9
    assert ea.tags == eb.tags


def test_tokens_independent_of_drawing_order():
    a, b = _two_orders()
    t = SpellTokenizer()
    assert t.tokenize(a).token_ids == t.tokenize(b).token_ids


def test_spec_roundtrip_and_recipes():
    g = build_fireball()
    g2 = graph_from_spec(g.to_spec())
    assert evaluate_spell(g).total_power == evaluate_spell(g2).total_power
    for name, recipe in RECIPES.items():
        g = graph_from_spec(recipe)
        assert g.nodes, name
        evaluate_spell(g)
    for build in ALGEBRA_PRESETS.values():
        evaluate_spell(build())


def test_time_paradox_is_critical_not_a_crash():
    g = SpellGraph()
    a, b = g.add_rune("time"), g.add_rune("time")
    g.connect(a, b, EdgeType.MUTUAL)
    assert check_graph(g).verdict.name == "CRITICAL"
