import random

from wand.algebra import build_fireball, graph_from_spec
from wand.craft import brew, enchant
from wand.dream import DreamWeaver, Lexicon


def test_healing_potion():
    potion = brew(["heartroot", "sunpetal"], ["steep"], "crystal")
    assert potion.program.delivery.kind == "consume" and potion.program.target.selector == "self"
    assert potion.program.primary_verb == "heal"
    assert potion.quality > 0.8


def test_overfilled_flask_ruins_the_brew():
    big = brew(["sulfur", "night_mold", "fire_salt", "quicksilver"], ["heat", "charge", "steep"], "clay")
    assert big.quality < 0.3


def test_enchantment_binds_to_trigger_and_material():
    silver = enchant(build_fireball(), "silver", "on_hit")
    iron = enchant(build_fireball(), "iron", "on_hit")
    assert silver.program.delivery.kind == "imbue"
    assert all(op.trigger == "on_hit" for op in silver.program.ops)
    assert silver.charges is None and iron.charges is not None
    assert iron.program.power < silver.program.power


def test_dream_teaches_a_word():
    lex = Lexicon()
    weaver = DreamWeaver(lex)
    vision = weaver.conjure(random.Random(3))
    assert vision is not None and len(vision.new_runes) == 1
    new = vision.new_runes[0]
    assert new in vision.solution.runes and not lex.knows(new)
    wrong = weaver.attempt(vision, compile_any("water", "self"))
    assert not wrong.solved and wrong.feedback
    from wand.explore import evaluate
    right = weaver.attempt(vision, evaluate(vision.solution).program)
    assert right.solved and lex.knows(new) and right.learned == [new]


def compile_any(a, b):
    from wand.actions import compile_program
    from wand.algebra import evaluate_spell
    g = graph_from_spec({"nodes": [{"name": a}, {"name": b}], "edges": [[a, b, "DIRECT"]]})
    return compile_program(g, evaluate_spell(g))


def test_lexicon_roundtrip_and_learning_by_use():
    lex = Lexicon()
    p = compile_any("air", "target")
    for _ in range(3):
        lex.observe(p)
    assert lex.knows("air")                    # learned the hard way
    again = Lexicon.from_dict(lex.to_dict())
    assert again.known == lex.known and again.notes["air"].casts == 3
