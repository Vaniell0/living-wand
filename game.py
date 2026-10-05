"""
Living Wand — graphical prototype (pygame-ce)

Draw runes inside the circle, connect them with straight lines (or let the
wand wire them by grammar), cast, and watch the program play out.

Controls
  LMB drag ............ draw a stroke (a long straight line between two runes = a link;
                        draw it twice = the runes mix)
  RMB / Enter ......... cast (in a dream: try to reproduce the vision)
  Wheel over a rune ... pick another reading of an uncertain rune (teaches the OCR)
  T over a link ....... cycle the link type (→ ═ ⋯ ? ¬ ↻ ⊣)
  Z ................... undo last stroke         Space / MMB ... clear
  C ................... open / close the circle E ............. change the world (weather, place)
  D ................... sleep & dream / wake    H ............. ask the dream for a hint
  Tab ................. your lexicon of runes    Esc ........... quit (state is saved)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Find system SDL2 with Wayland support (pip pygame-ce ships without it)
import glob
_sdl = glob.glob("/nix/store/*sdl2-compat*/lib/libSDL2-2.0.so")
if _sdl:
    os.environ["SDL_DYNAMIC_API"] = sorted(_sdl)[-1]  # latest version

import math
import random

import numpy as np
import pygame
import torch

from wand.actions import Outcome
from wand.algebra import RUNES, EdgeType
from wand.dream import DELIVERY_RU, FIELD_RU, RUNE_RU, STATUS_RU, TARGET_RU, VERB_RU
from wand.effect import DOMAIN_COLORS, ELEMENT_COLORS
from wand.events import Pole
from wand.session import WandSession

from vfx import VFX, SpellPlayer

# ═══ Layout & palette ═══
W, H = 1180, 720
CANVAS = pygame.Rect(30, 70, 560, 560)
CENTER = CANVAS.center
CIRCLE_R = 255
PANEL_X = CANVAS.right + 30
PANEL_W = W - PANEL_X - 20

INK = (150, 170, 255)
INK_CORE = (225, 232, 255)
TEXT = (205, 200, 235)
DIM = (110, 105, 150)
ACCENT = (170, 160, 245)
AMBER = (255, 190, 90)
GOOD = (130, 220, 150)
BAD = (235, 110, 100)

POLE_COLORS = {
    Pole.ELEMENT: (232, 93, 58), Pole.TRANSFORM: (200, 200, 210), Pole.BIOLOGY: (76, 190, 110),
    Pole.MENTAL: (170, 90, 220), Pole.SPACE: (60, 150, 243), Pole.TIME: (255, 193, 7),
}
POLE_RU = ["стихия", "трансф.", "биология", "ментал", "простр.", "время"]

CONTEXTS = [
    (None, "ясный день"),
    ({"weather": "rain"}, "дождь"),
    ({"time": "night"}, "ночь"),
    ({"place": "forest"}, "лес"),
    ({"weather": "storm"}, "гроза"),
    ({"place": "water"}, "у воды"),
    ({"place": "ley_line"}, "лей-линия"),
    ({"place": "graveyard", "time": "night"}, "кладбище ночью"),
]
EDGE_CYCLE = [EdgeType.DIRECT, EdgeType.MUTUAL, EdgeType.SEQUENCE, EdgeType.CONDITION,
              EdgeType.NEGATE, EdgeType.ITERATE, EdgeType.LIMIT]
EDGE_SYM = {"DIRECT": "→", "MUTUAL": "═", "SEQUENCE": "⋯", "CONDITION": "?→",
            "NEGATE": "¬", "ITERATE": "↻", "LIMIT": "⊣"}
GAP_ANGLE = -math.pi / 4  # where an open circle leaks


def rune_color(name: str) -> tuple[int, int, int]:
    return ELEMENT_COLORS.get(name) or DOMAIN_COLORS.get(name) or INK


def font(size: int, bold: bool = False) -> pygame.font.Font:
    path = pygame.font.match_font("dejavusans,freesans,liberationsans,notosans", bold=bold)
    return pygame.font.Font(path, size)


def ensure_ocr(session: WandSession):
    """First run: train the stroke-based rune reader (≈ a few minutes on CPU)."""
    if session.ocr is not None:
        return
    from wand.ocr import train_symbol_net
    from wand.synth_data import generate_stroke_dataset
    print("Training rune reader (first run; `python tools/train_ocr.py` does it with more data)...")
    images, labels = generate_stroke_dataset(300, seed=0)
    net, hist = train_symbol_net(images, labels, epochs=15)
    out = session.data_dir / "ocr_symbolnet_v2.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out)
    print(f"done ({hist['val_acc'][-1]:.0%}), cached to {out}")
    session.attach_ocr(net)


class Game:
    def __init__(self, headless: bool = False):
        pygame.init()
        flags = 0
        self.screen = pygame.display.set_mode((W, H), flags)
        pygame.display.set_caption("Living Wand")
        self.clock = pygame.time.Clock()
        self.f_title = font(22, True)
        self.f_big = font(17, True)
        self.f = font(14)
        self.f_small = font(12)

        self.session = WandSession()
        self.session.vocabulary_mode = True
        ensure_ocr(self.session)

        self.vfx = VFX((W, H))
        self.bg = self._make_background(dream=False)
        self.bg_dream = self._make_background(dream=True)

        # drawing state
        self.strokes: list[list[tuple[int, int]]] = []
        self.current: list[tuple[int, int]] = []
        self.drawing = False
        self.overrides: dict[frozenset, str] = {}
        self.edge_overrides: dict[tuple[frozenset, frozenset], EdgeType] = {}
        self.circle_closed = True
        self.sketch = None

        # spell state
        self.player: SpellPlayer | None = None
        self.burning: list[list[tuple[int, int]]] = []
        self.burning_colors: list[tuple[int, int, int]] = []
        self.program: dict | None = None
        self.context_idx = 0
        self.dreaming = False
        self.dream_lines: list[tuple[str, tuple]] = []
        self.show_lexicon = False
        self.log: list[tuple[str, tuple]] = []
        self.time = 0.0
        self.add_log("Нарисуй руны в круге. ПКМ — сотворить. D — уснуть.", DIM)

    # ═══ helpers ═══

    def add_log(self, msg: str, color=TEXT):
        self.log.append((msg, color))
        self.log = self.log[-9:]

    def _make_background(self, dream: bool) -> pygame.Surface:
        y, x = np.mgrid[0:H, 0:W].astype(np.float32)
        cx, cy = CENTER
        d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / 700.0
        base = np.array([24, 18, 46] if dream else [16, 13, 26], dtype=np.float32)
        glow = np.array([40, 50, 110] if dream else [26, 22, 48], dtype=np.float32)
        k = np.clip(1.0 - d, 0, 1)[..., None] ** 2
        img = base + glow * k
        rng = np.random.default_rng(1 if dream else 0)
        img += rng.normal(0, 2.0, img.shape[:2])[..., None]  # paper grain
        if dream:
            stars = rng.random((H, W)) > 0.9985
            img[stars] = [200, 210, 255]
        arr = np.clip(img, 0, 255).astype(np.uint8).transpose(1, 0, 2)
        return pygame.surfarray.make_surface(arr).convert()

    def reparse(self):
        if not self.strokes:
            self.sketch = None
            return
        self.sketch = self.session.parse_strokes(
            self.strokes, self.circle_closed, self.overrides, self.edge_overrides)

    def label_text(self, label: str) -> str:
        if label not in RUNES:
            return "·"
        if self.session.lexicon.knows(label):
            return RUNE_RU.get(label, label)
        return "?"

    def glyph_at(self, pos):
        if not self.sketch:
            return None
        for i, g in enumerate(self.sketch.glyphs):
            x0, y0, x1, y1 = g.bbox
            if x0 - 8 <= pos[0] <= x1 + 8 and y0 - 8 <= pos[1] <= y1 + 8:
                return i
        return None

    def edge_at(self, pos):
        if not self.sketch:
            return None
        best, best_d = None, 28.0
        for e in self.sketch.edges:
            (ax, ay), (bx, by) = self.sketch.glyphs[e.src].center, self.sketch.glyphs[e.dst].center
            mx, my = (ax + bx) / 2, (ay + by) / 2
            d = math.hypot(pos[0] - mx, pos[1] - my)
            if d < best_d:
                best, best_d = e, d
        return best

    def cycle_reading(self, idx: int, direction: int):
        g = self.sketch.glyphs[idx]
        options = [name for name, _ in g.alternatives if name in RUNES]
        if not options:
            return
        cur = options.index(g.label) if g.label in options else -1
        new = options[(cur + direction) % len(options)]
        self.overrides[frozenset(g.stroke_ids)] = new
        self.reparse()
        self.add_log(f"руна прочитана как «{self.label_text(new)}»", ACCENT)

    def cycle_edge(self, e):
        key = (frozenset(self.sketch.glyphs[e.src].stroke_ids),
               frozenset(self.sketch.glyphs[e.dst].stroke_ids))
        cur = EDGE_CYCLE.index(e.edge_type)
        self.edge_overrides[key] = EDGE_CYCLE[(cur + 1) % len(EDGE_CYCLE)]
        self.reparse()

    def clear(self):
        self.strokes, self.current = [], []
        self.overrides, self.edge_overrides = {}, {}
        self.sketch = None

    # ═══ casting ═══

    def cast(self):
        self.reparse()
        if not self.sketch or not self.sketch.graph.nodes:
            self.add_log("в круге нет ни одной руны", DIM)
            return
        sk = self.sketch
        glyph_strokes = [[p for sid in g.stroke_ids for p in self.strokes[sid]]
                         for g in sk.glyphs if g.is_rune]
        colors = [g.label for g in sk.glyphs if g.is_rune]
        target_pos = (CENTER[0] + 190, CENTER[1] - 60)

        if self.dreaming and self.session.vision is not None:
            res = self.session.dream_attempt(sk.graph)
            program = res.program
            self.program = program.to_dict()
            self.player = SpellPlayer(self.program, CENTER, target_pos, glyph_strokes, self.vfx,
                                      backfired=False, dream=True)
            self.dream_lines = self.dream_lines[:1] + [(res.text(), GOOD if res.solved else ACCENT)]
            if res.solved:
                for r in res.learned:
                    self.add_log(f"✦ новое слово: «{RUNE_RU[r]}» ({r})", AMBER)
                self.add_log("видение рассеялось — проснись (D)", DIM)
        else:
            self.session.context = CONTEXTS[self.context_idx][0]
            _, report = self.session.cast_strokes(self.strokes, self.circle_closed,
                                                  self.overrides, self.edge_overrides)
            if report is None:
                self.add_log("палочка не поняла ни одной руны", DIM)
                return
            program = report.program
            self.program = program.to_dict()
            backfired = program.roll_backfire()
            self.player = SpellPlayer(self.program, CENTER, target_pos, glyph_strokes, self.vfx,
                                      backfired=backfired)
            # This prototype has no world simulation; a real engine reports what
            # actually happened. We report the program's own intent.
            totals = {op.verb: round(op.amount * 100, 1) for op in program.ops}
            self.session.report_outcome(Outcome(program.id, hit=not backfired,
                                                backfired=backfired, totals=totals))
            shown = program.title if "+" not in program.title else \
                " + ".join(self.label_text(r) for r in program.runes)
            self.add_log(f"✧ {shown}", ACCENT)
            if backfired:
                self.add_log("  отдача! заклинание ударило по тебе", BAD)
            for r in program.provenance.get("learned", []):
                self.add_log(f"  ✦ палочка поняла слово «{RUNE_RU[r]}»", AMBER)
            for d in report.discoveries[:2]:
                self.add_log(f"  открытие: {d.name_ru}", GOOD)
            if report.newly_crystallized:
                self.add_log(f"  ** кристалл: {report.crystal_name} **", AMBER)
        core = tuple(self.program["visual"]["core_color"])
        self.burning = glyph_strokes
        self.burning_colors = [rune_color(n) if (n in ELEMENT_COLORS or n in DOMAIN_COLORS) else core
                               for n in colors]
        self.vfx.play(self.player)
        self.clear()

    def toggle_dream(self):
        if self.dreaming:
            self.dreaming = False
            self.session.vision = None
            self.dream_lines = []
            self.add_log("ты просыпаешься", DIM)
            return
        n = len(self.session.pending)
        vision = self.session.dream(seed=random.randrange(1 << 30))
        self.dreaming = True
        if n:
            self.add_log(f"палочка видит во сне {n} дневных заклинаний", DIM)
        if vision is None:
            self.dream_lines = [("Сон без видений. Ты знаешь все слова.", DIM)]
        else:
            self.dream_lines = [(vision.text, (200, 210, 255)),
                                ("Повтори увиденное. H — подсказка.", DIM)]

    def hint(self):
        if self.dreaming and self.session.vision is not None:
            self.dream_lines = self.dream_lines[:1] + [(self.session.vision.hint(), AMBER)]

    # ═══ frame ═══

    def update(self, dt: float):
        self.time += dt
        self.vfx.ambient(CENTER, CIRCLE_R - 10, (120, 140, 255) if self.dreaming else INK,
                         self.circle_closed, GAP_ANGLE)
        self.vfx.update(dt)
        if self.player is not None and self.player not in self.vfx.players:
            self.player = None
            self.burning = []

    def draw(self):
        scr = self.screen
        scr.blit(self.bg_dream if self.dreaming else self.bg, (0, 0))
        ox, oy = self.vfx.offset()
        self._draw_circle(ox, oy)
        self._draw_strokes(ox, oy)
        self._draw_glyphs()
        self.vfx.draw(scr)
        self._draw_sentence()
        if self.show_lexicon:
            self._draw_lexicon()
        else:
            self._draw_panel()
        hint = "ЛКМ рисовать · ПКМ сотворить · колесо: другое прочтение · T: тип связи · Z отмена · C круг · E мир · D сон · H подсказка · Tab словарь"
        scr.blit(self.f_small.render(hint, True, (70, 66, 100)), (20, H - 22))
        pygame.display.flip()

    def _draw_circle(self, ox, oy):
        cx, cy = CENTER[0] + ox, CENTER[1] + oy
        base = (120, 140, 255) if self.dreaming else (95, 88, 170)
        pulse = 0.75 + 0.25 * math.sin(self.time * 1.7)
        charge = self.player.charge_level if (self.player and not self.player.released) else 0.0
        col = tuple(min(255, int(c * (pulse + charge * 0.8))) for c in base)
        if self.circle_closed:
            pygame.draw.circle(self.screen, tuple(c // 3 for c in col), (cx, cy), CIRCLE_R + 3, 6)
            pygame.draw.circle(self.screen, col, (cx, cy), CIRCLE_R, 2)
        else:
            gap = 0.35
            pts = [(cx + CIRCLE_R * math.cos(a), cy + CIRCLE_R * math.sin(a))
                   for a in np.linspace(GAP_ANGLE + gap, GAP_ANGLE + math.tau - gap, 120)]
            pygame.draw.lines(self.screen, tuple(c // 3 for c in col), False, pts, 6)
            pygame.draw.lines(self.screen, col, False, pts, 2)
        # inner ring of tick marks — a sense of a ritual instrument
        for k in range(36):
            a = k / 36 * math.tau + self.time * 0.05
            r0, r1 = CIRCLE_R - 12, CIRCLE_R - (20 if k % 3 == 0 else 15)
            pygame.draw.line(self.screen, tuple(c // 2 for c in col),
                             (cx + r0 * math.cos(a), cy + r0 * math.sin(a)),
                             (cx + r1 * math.cos(a), cy + r1 * math.sin(a)), 1)

    def _ink(self, pts, color, core, width=6):
        if len(pts) >= 2:
            pygame.draw.lines(self.screen, tuple(c // 3 for c in color), False, pts, width + 4)
            pygame.draw.lines(self.screen, color, False, pts, width // 2 + 1)
            pygame.draw.aalines(self.screen, core, False, pts)
        elif len(pts) == 1:
            pygame.draw.circle(self.screen, color, pts[0], 4)

    def _draw_strokes(self, ox, oy):
        shift = lambda s: [(x + ox, y + oy) for x, y in s]
        for s in self.strokes:
            self._ink(shift(s), INK, INK_CORE)
        if self.current:
            self._ink(shift(self.current), INK, INK_CORE)
        if self.player and not self.player.released:
            k = self.player.charge_level
            for s, c in zip(self.burning, self.burning_colors):
                col = tuple(int(INK[i] + (c[i] - INK[i]) * k) for i in range(3))
                core = tuple(min(255, int(v + 120 * k)) for v in col)
                self._ink(shift(s), col, core, width=6 + int(4 * k))

    def _draw_glyphs(self):
        if not self.sketch:
            return
        sk = self.sketch
        for e in sk.edges:
            a, b = sk.glyphs[e.src].center, sk.glyphs[e.dst].center
            col = DIM if e.inferred else AMBER
            if e.inferred:
                n = max(int(math.hypot(b[0] - a[0], b[1] - a[1]) // 10), 1)
                for k in range(0, n, 2):
                    p0 = (a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n)
                    p1 = (a[0] + (b[0] - a[0]) * (k + 1) / n, a[1] + (b[1] - a[1]) * (k + 1) / n)
                    pygame.draw.line(self.screen, col, p0, p1, 1)
            mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            sym = self.f.render(EDGE_SYM[e.edge_type.name], True, col)
            self.screen.blit(sym, (mid[0] - sym.get_width() / 2, mid[1] - sym.get_height() / 2))
        for g in sk.glyphs:
            x0, y0, x1, y1 = g.bbox
            rect = pygame.Rect(x0 - 6, y0 - 6, x1 - x0 + 12, y1 - y0 + 12)
            if not g.is_rune:
                pygame.draw.rect(self.screen, (60, 56, 80), rect, 1, border_radius=6)
                continue
            known = self.session.lexicon.knows(g.label)
            if g.uncertain:
                k = 0.5 + 0.5 * math.sin(self.time * 6)
                col = tuple(int(c * (0.5 + 0.5 * k)) for c in AMBER)
            else:
                col = rune_color(g.label) if known else (150, 140, 120)
            pygame.draw.rect(self.screen, col, rect, 1, border_radius=6)
            txt = f"{self.label_text(g.label)} {g.confidence:.0%}"
            self.screen.blit(self.f_small.render(txt, True, col), (rect.x, rect.y - 16))

    def _draw_sentence(self):
        """The spell read aloud — the grammar made visible."""
        if not self.sketch or not self.sketch.graph.nodes:
            return
        sk = self.sketch
        parts = []
        used = set()
        for e in sk.edges:
            a = self.label_text(sk.glyphs[e.src].label)
            b = self.label_text(sk.glyphs[e.dst].label)
            parts.append(f"{a} {EDGE_SYM[e.edge_type.name]} {b}")
            used.update((e.src, e.dst))
        for i, g in enumerate(sk.glyphs):
            if g.is_rune and i not in used:
                parts.append(self.label_text(g.label))
        text = " · ".join(parts)
        surf = self.f.render(text[:90], True, TEXT)
        self.screen.blit(surf, (CANVAS.centerx - surf.get_width() / 2, CANVAS.bottom + 6))

    # ═══ right panel ═══

    def _bar(self, x, y, w, value, color, label):
        pygame.draw.rect(self.screen, (34, 30, 52), (x, y, w, 8), border_radius=4)
        pygame.draw.rect(self.screen, color, (x, y, int(w * max(0, min(value, 1))), 8), border_radius=4)
        self.screen.blit(self.f_small.render(label, True, DIM), (x + w + 8, y - 3))

    def _draw_panel(self):
        x, y = PANEL_X, 22
        scr = self.screen
        scr.blit(self.f_title.render("Living Wand" + ("  ·  сон" if self.dreaming else ""),
                                     True, (200, 210, 255) if self.dreaming else ACCENT), (x, y))
        y += 34
        ctx = CONTEXTS[self.context_idx][1]
        lex = self.session.lexicon
        scr.blit(self.f_small.render(
            f"мир: {ctx}   ·   слов: {len(lex.known)}/30   ·   видений разгадано: {lex.dreams_solved}",
            True, DIM), (x, y))
        y += 22

        if self.dreaming:
            for line, col in self.dream_lines:
                for chunk in _wrap(line, self.f, PANEL_W):
                    scr.blit(self.f.render(chunk, True, col), (x, y))
                    y += 19
                y += 6
            y += 6

        p = self.program
        if p:
            title = p["title"]
            if "+" in title:  # no catalog name: never spell out words the player hasn't learned
                title = " + ".join(self.label_text(r) for r in p["runes"])
            scr.blit(self.f_big.render(title[:40], True, TEXT), (x, y))
            y += 26
            self._bar(x, y, 160, p["power"], (255, 150, 90), f"сила {p['power']:.2f}")
            y += 16
            self._bar(x, y, 160, p["stability"], (120, 200, 255), f"стабильность {p['stability']:.2f}")
            y += 16
            risk = p["risk"]["backfire_chance"]
            self._bar(x, y, 160, risk, BAD, f"риск отдачи {risk:.0%}  ·  цена {p['cost']:.0f}")
            y += 22
            d = p["delivery"]["kind"]
            sel = p["target"]["selector"]
            scr.blit(self.f.render(f"{DELIVERY_RU.get(d, d)} → {TARGET_RU.get(sel, sel)}",
                                   True, DIM), (x, y))
            y += 22
            for op in p["ops"]:
                verb = VERB_RU.get(op["verb"], op["verb"])
                status = op["params"].get("status")
                extra = f"({STATUS_RU.get(status, status)})" if status else ""
                trig = "" if op["trigger"] == "cast" else f" [{op['trigger']}]"
                col = rune_color(op["channel"])
                pygame.draw.circle(scr, col, (x + 5, y + 8), 4)
                scr.blit(self.f.render(f"{verb} {extra}{trig}", True, TEXT), (x + 16, y))
                self._bar(x + 230, y + 5, 80, op["amount"] * 2, col, f"{op['amount']:.2f}")
                y += 19
            feel = []
            for f in p["fields"][:4]:
                pos, neg = FIELD_RU.get(f["property"], (f["property"],) * 2)
                feel.append(f"{pos if f['delta'] > 0 else neg} {abs(f['delta']):.2f}")
            if feel:
                y += 4
                scr.blit(self.f_small.render("в мире: " + ", ".join(feel), True, DIM), (x, y))
                y += 16
            sigs = p["provenance"].get("signatures") or []
            if sigs:
                scr.blit(self.f_small.render("подпись палочки: " + ", ".join(sigs), True, AMBER), (x, y))
                y += 16
            reasons = p["risk"]["reasons"]
            if reasons:
                scr.blit(self.f_small.render("⚠ " + "; ".join(reasons)[:60], True, BAD), (x, y))
                y += 16
            y += 8

        y = max(y, 330)
        self._draw_hexagon(x + 70, y + 70)
        lv = self.session.hexagon.levels()
        for i, pole in enumerate(Pole):
            scr.blit(self.f_small.render(f"{POLE_RU[i]} {lv[i]}", True, POLE_COLORS[pole]),
                     (x + 170, y + 10 + i * 18))
        y += 160
        for msg, col in self.log[-8:]:
            scr.blit(self.f_small.render(msg[:64], True, col), (x, y))
            y += 16

    def _draw_hexagon(self, cx, cy, r=58):
        profile = self.session.hexagon.profile()
        angles = [math.pi / 2 + i * math.pi / 3 for i in range(6)]
        outline = [(cx + r * math.cos(a), cy - r * math.sin(a)) for a in angles]
        pygame.draw.polygon(self.screen, (50, 46, 72), outline, 1)
        pts = [(cx + r * min(profile[i] / 0.4, 1) * math.cos(a),
                cy - r * min(profile[i] / 0.4, 1) * math.sin(a)) for i, a in enumerate(angles)]
        s = pygame.Surface((W, H), pygame.SRCALPHA)
        pygame.draw.polygon(s, (*ACCENT, 50), pts)
        self.screen.blit(s, (0, 0))
        pygame.draw.polygon(self.screen, ACCENT, pts, 2)
        for i, a in enumerate(angles):
            pygame.draw.circle(self.screen, POLE_COLORS[Pole(i)],
                               (int(cx + (r + 8) * math.cos(a)), int(cy - (r + 8) * math.sin(a))), 3)

    def _draw_lexicon(self):
        x, y = PANEL_X, 22
        lex = self.session.lexicon
        self.screen.blit(self.f_title.render(f"Словарь рун  {len(lex.known)}/30", True, ACCENT), (x, y))
        y += 34
        for name in RUNES:
            if lex.knows(name):
                note = lex.notes.get(name)
                summary = note.summary() if note else "ещё не испытана"
                self.screen.blit(self.f_small.render(f"{RUNE_RU[name]:<12}", True, rune_color(name)), (x, y))
                self.screen.blit(self.f_small.render(summary[:46], True, DIM), (x + 105, y))
            else:
                met = lex.encounters.get(name, 0)
                tag = "незнакомый знак" + (f" · встречен {met}×" if met else "")
                self.screen.blit(self.f_small.render(f"{'· · ·':<12}{tag}", True, (70, 66, 100)), (x, y))
            y += 18

    # ═══ loop ═══

    def handle(self, event) -> bool:
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            k = event.key
            if k == pygame.K_ESCAPE:
                return False
            if k == pygame.K_SPACE:
                self.clear()
            elif k == pygame.K_RETURN:
                self.cast()
            elif k == pygame.K_z and self.strokes:
                self.strokes.pop()
                self.reparse()
            elif k == pygame.K_c:
                self.circle_closed = not self.circle_closed
                self.reparse()
            elif k == pygame.K_e:
                self.context_idx = (self.context_idx + 1) % len(CONTEXTS)
                self.add_log(f"мир: {CONTEXTS[self.context_idx][1]}", DIM)
            elif k == pygame.K_d:
                self.toggle_dream()
            elif k == pygame.K_h:
                self.hint()
            elif k == pygame.K_TAB:
                self.show_lexicon = not self.show_lexicon
            elif k == pygame.K_t:
                e = self.edge_at(pygame.mouse.get_pos())
                if e is not None:
                    self.cycle_edge(e)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and CANVAS.collidepoint(event.pos):
                self.drawing = True
                self.current = [event.pos]
            elif event.button == 3:
                self.cast()
            elif event.button == 2:
                self.clear()
        elif event.type == pygame.MOUSEWHEEL:
            i = self.glyph_at(pygame.mouse.get_pos())
            if i is not None:
                self.cycle_reading(i, -1 if event.y > 0 else 1)
        elif event.type == pygame.MOUSEMOTION and self.drawing:
            if CANVAS.collidepoint(event.pos):
                self.current.append(event.pos)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.drawing:
            self.drawing = False
            if self.current:
                self.strokes.append(self.current)
            self.current = []
            self.reparse()
        return True

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(60) / 1000.0
            for event in pygame.event.get():
                running = self.handle(event) and running
            self.update(dt)
            self.draw()
        self.session.save()
        self.session.close()
        pygame.quit()


def _wrap(text: str, f: pygame.font.Font, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if f.size(trial)[0] > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


if __name__ == "__main__":
    Game().run()
