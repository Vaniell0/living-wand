import numpy as np

from wand.algebra import EdgeType
from wand.sketch import SketchParser, auto_wire_edges, render_strokes


def box(x, y, s=60):
    return [(x, y), (x + s, y), (x + s, y + s), (x, y + s), (x, y)]


def tri(x, y, s=60):
    return [(x, y + s), (x + s / 2, y), (x + s, y + s), (x, y + s)]


def fake_recognizer(table):
    """Recognise by stroke count — enough to test segmentation & wiring."""
    def rec(img):
        return table
    return rec


def test_render_is_normalised_and_keeps_dots():
    a = render_strokes([box(0, 0, 50)])
    b = render_strokes([box(300, 200, 150)])
    assert np.abs(a - b).max() < 0.35          # position/size do not matter
    dot = render_strokes([[(10, 10)]], min_extent=30)
    assert dot.sum() > 5 and dot[32, 32] > 0.5  # a click is a visible dot


def test_connector_between_two_glyphs_becomes_edge():
    strokes = [tri(100, 100), box(300, 100), [(170, 130), (290, 130)]]
    glyphs, connectors = SketchParser().segment(strokes)
    assert connectors == [2]
    assert sorted(map(sorted, glyphs)) == [[0], [1]]


def test_crossing_stroke_is_part_of_the_rune():
    # a crosshair: the long horizontal crosses the vertical → glyph, not a link
    strokes = [[(200, 100), (200, 200)], [(140, 150), (260, 150)]]
    glyphs, connectors = SketchParser().segment(strokes)
    assert connectors == [] and glyphs == [[0, 1]]


def test_small_mark_inside_a_rune_merges():
    circle = [(200 + 40 * np.cos(a), 200 + 40 * np.sin(a)) for a in np.linspace(0, 6.3, 40)]
    glyphs, _ = SketchParser().segment([circle, [(200, 200)]])
    assert glyphs == [[0, 1]]


def test_double_connector_means_mutual_and_parse_builds_graph():
    labels = iter(["fire", "air"])

    def rec(img):
        return [(next(labels), 0.95), ("noise", 0.01)]

    strokes = [tri(100, 100), box(300, 100), [(170, 120), (290, 120)], [(290, 140), (170, 140)]]
    res = SketchParser(rec).parse(strokes)
    assert [g.label for g in res.glyphs] == ["fire", "air"]
    assert len(res.edges) == 1 and res.edges[0].edge_type == EdgeType.MUTUAL
    assert len(res.graph.edges) == 1


def test_overrides_follow_the_strokes_not_the_index():
    rec = lambda img: [("fire", 0.4), ("amplify", 0.38)]
    strokes = [tri(100, 100)]
    res = SketchParser(rec).parse(strokes, overrides={frozenset([0]): "amplify"})
    assert res.glyphs[0].label == "amplify" and res.glyphs[0].confirmed


def test_grammar_auto_wiring():
    from wand.sketch import Glyph
    gl = [Glyph([0], (0, 0, 50, 50), label="hold"), Glyph([1], (100, 0, 150, 50), label="fire"),
          Glyph([2], (220, 0, 270, 50), label="here")]
    edges = {(gl[e.src].label, gl[e.dst].label, e.edge_type) for e in auto_wire_edges(gl, [0, 1, 2])}
    assert ("hold", "fire", EdgeType.DIRECT) in edges
    assert ("fire", "here", EdgeType.DIRECT) in edges
