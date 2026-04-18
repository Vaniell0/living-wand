"""
Living Wand — Graphical Prototype (pygame-ce)

Minimal graphical sandbox for exploring magic.
Draw symbols, connect them, cast spells, watch the wand evolve.

Controls:
  Left mouse:  draw strokes
  Right mouse: hold to enter cast mode, release to cast
  Middle mouse / Space: clear canvas
  Tab: toggle wand profile view
  1-6: life events (fire the wand's experience)
  S: sleep (wand trains on accumulated experience)
  Esc: quit
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Find system SDL2 with Wayland support (pip pygame-ce ships without it)
import glob
_sdl = glob.glob("/nix/store/*sdl2-compat*/lib/libSDL2-2.0.so")
if _sdl:
    os.environ["SDL_DYNAMIC_API"] = sorted(_sdl)[-1]  # latest version

import pygame
import numpy as np
import math
import random
import torch

from wand.algebra import (
    SpellGraph, evaluate_spell, SpellEffect,
    SignalKind, Element, EdgeType,
    SYNERGY_PAIRS, CANCEL_PAIRS,
)
from wand.model import create_wand, WandTrainer, affinity_multiplier
from wand.events import Event, EventType, Pole, NUM_POLES, Emotion
from wand.ocr import SymbolCNN, train_symbol_cnn
from wand.synth_data import generate_dataset, SYMBOL_NAMES, IMG_SIZE

# ═══ Constants ═══
W, H = 1100, 700
CANVAS_X, CANVAS_Y = 50, 80
CANVAS_W, CANVAS_H = 500, 500
PANEL_X = CANVAS_X + CANVAS_W + 40
PANEL_W = W - PANEL_X - 20

BG = (18, 16, 28)
CANVAS_BG = (24, 22, 38)
CIRCLE_COLOR = (60, 55, 100)
STROKE_COLOR = (180, 175, 220)
TEXT_COLOR = (200, 195, 240)
DIM_TEXT = (100, 95, 140)
ACCENT = (159, 154, 230)

POLE_COLORS = {
    Pole.ELEMENT: (232, 93, 58),
    Pole.TRANSFORM: (192, 192, 192),
    Pole.BIOLOGY: (76, 175, 80),
    Pole.MENTAL: (156, 39, 176),
    Pole.SPACE: (33, 150, 243),
    Pole.TIME: (255, 193, 7),
}

# Life event presets
LIFE_EVENTS = {
    pygame.K_1: ("battle fury", EventType.ACTION,
                 [0.5, 0.3, 0, 0, 0, 0.1],
                 Emotion(-0.1, 0.8, 0.9)),
    pygame.K_2: ("healed someone", EventType.ACTION,
                 [0, 0, 0.6, 0.2, 0, 0],
                 Emotion(0.7, 0.5, 0.3)),
    pygame.K_3: ("read ancient text", EventType.LEARNING,
                 [0, 0, 0, 0.5, 0.3, 0.1],
                 Emotion(0.3, 0.4, 0.2)),
    pygame.K_4: ("lost a friend", EventType.LOSS,
                 [0.2, 0, 0, 0.1, 0, 0.4],
                 Emotion(-0.8, 0.9, 0.7)),
    pygame.K_5: ("traveled far", EventType.OBSERVATION,
                 [0, 0, 0, 0, 0.6, 0.2],
                 Emotion(0.2, 0.3, 0.3)),
    pygame.K_6: ("revelation", EventType.REVELATION,
                 [0.1, 0.1, 0.1, 0.3, 0.1, 0.2],
                 Emotion(0.9, 1.0, 0.8)),
}


class Particle:
    """Particle with behavior pattern."""
    def __init__(self, x, y, color, vx=0, vy=0, life=1.0,
                 gravity=0.05, size=4, behavior="default"):
        self.x, self.y = x, y
        self.color = color
        self.vx, self.vy = vx, vy
        self.life = life
        self.decay = random.uniform(0.01, 0.03)
        self.gravity = gravity
        self.size = size
        self.behavior = behavior
        self.age = 0
        self.ox, self.oy = x, y  # origin for orbit

    def update(self):
        self.age += 1

        if self.behavior == "rise":
            self.vy -= 0.08  # float upward
            self.vx *= 0.98
        elif self.behavior == "orbit":
            angle = self.age * 0.05
            r = 30 + self.age * 0.5
            self.x = self.ox + r * math.cos(angle + hash(id(self)) % 100)
            self.y = self.oy + r * math.sin(angle + hash(id(self)) % 100)
            self.life -= self.decay
            return self.life > 0
        elif self.behavior == "fall":
            self.vy += 0.12
            self.vx *= 0.95
        elif self.behavior == "pulse":
            self.size = 3 + 3 * abs(math.sin(self.age * 0.1))
            self.vy -= 0.02
        elif self.behavior == "scatter":
            self.vx += random.gauss(0, 0.3)
            self.vy += random.gauss(0, 0.3)
        elif self.behavior == "implode":
            dx = self.ox - self.x
            dy = self.oy - self.y
            self.vx += dx * 0.01
            self.vy += dy * 0.01
        else:
            self.vy += self.gravity

        self.x += self.vx
        self.y += self.vy
        self.life -= self.decay
        return self.life > 0

    def draw(self, surface):
        r = max(1, int(self.life * self.size))
        c = tuple(max(0, min(255, int(ch * self.life))) for ch in self.color)
        pygame.draw.circle(surface, c, (int(self.x), int(self.y)), r)


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("Living Wand")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 14)
        self.font_big = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 11)

        # Wand
        self.wand = create_wand(seed=42)
        self.trainer = WandTrainer(self.wand, lr=5e-4)
        self.pending_events: list[Event] = []
        self.cast_count = 0
        self.sleep_count = 0

        # OCR — load cached or train
        ocr_cache = os.path.join(os.path.dirname(__file__), "data", "ocr_model.pt")
        if os.path.exists(ocr_cache):
            print("Loading OCR... ", end="", flush=True)
            self.symbol_cnn = SymbolCNN()
            self.symbol_cnn.load_state_dict(torch.load(ocr_cache, weights_only=True))
            print("done.")
        else:
            print("Training OCR (first run)... ", end="", flush=True)
            images, labels = generate_dataset(n_per_class=400)
            self.symbol_cnn, _ = train_symbol_cnn(images, labels, epochs=30)
            os.makedirs(os.path.dirname(ocr_cache), exist_ok=True)
            torch.save(self.symbol_cnn.state_dict(), ocr_cache)
            print("done (cached for next time).")

        # Drawing state
        self.strokes: list[list[tuple[int, int]]] = []  # list of stroke point lists
        self.current_stroke: list[tuple[int, int]] = []
        self.drawing = False
        self.circle_closed = True

        # Recognized symbols
        self.recognized: list[tuple[str, float, pygame.Rect]] = []

        # Spell state
        self.spell_graph: SpellGraph | None = None
        self.last_effect: SpellEffect | None = None
        self.last_cast_info: dict | None = None
        self.cast_flash = 0  # frames of flash animation
        self.particles: list[Particle] = []

        # UI state
        self.show_profile = False
        self.log: list[str] = []
        self.history: list[dict] = []

    def add_log(self, msg: str):
        self.log.append(msg)
        if len(self.log) > 15:
            self.log.pop(0)

    def recognize_strokes(self):
        """Segment and recognize symbols from drawn strokes."""
        self.recognized = []
        if not self.strokes:
            return

        clusters = self._cluster_strokes()

        for cluster_strokes, bbox in clusters:
            img = self._render_to_image(cluster_strokes, bbox)
            name, conf = self.symbol_cnn.predict(img)
            # Skip noise class and very low confidence
            if name != "noise" and conf > 0.35:
                self.recognized.append((name, conf, bbox))

    def _stroke_compactness(self, stroke) -> float:
        """How compact is a stroke? 1.0 = square/circle, 0.0 = thin line."""
        xs = [p[0] for p in stroke]
        ys = [p[1] for p in stroke]
        w = max(xs) - min(xs) + 1
        h = max(ys) - min(ys) + 1
        # Ratio of short to long side
        return min(w, h) / max(w, h, 1)

    def _cluster_strokes(self) -> list[tuple[list, pygame.Rect]]:
        """Each stroke = potential symbol. Noise class handles rejection."""
        clusters = []
        for stroke in self.strokes:
            if len(stroke) < 3:
                continue
            xs = [p[0] for p in stroke]
            ys = [p[1] for p in stroke]
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
            if w < 5 and h < 5:
                continue  # skip accidental dots only
            bbox = pygame.Rect(min(xs) - 5, min(ys) - 5, max(w, 10) + 10, max(h, 10) + 10)
            clusters.append(([stroke], bbox))
        return clusters

    def _render_to_image(self, strokes, bbox: pygame.Rect) -> np.ndarray:
        """Render strokes into a 64x64 numpy image for OCR."""
        img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        if bbox.w == 0 or bbox.h == 0:
            return img

        scale = min(IMG_SIZE / max(bbox.w, 1), IMG_SIZE / max(bbox.h, 1)) * 0.8
        ox = IMG_SIZE / 2 - (bbox.centerx - bbox.x) * scale
        oy = IMG_SIZE / 2 - (bbox.centery - bbox.y) * scale

        for stroke in strokes:
            for i in range(len(stroke) - 1):
                x0 = int((stroke[i][0] - bbox.x) * scale + ox)
                y0 = int((stroke[i][1] - bbox.y) * scale + oy)
                x1 = int((stroke[i+1][0] - bbox.x) * scale + ox)
                y1 = int((stroke[i+1][1] - bbox.y) * scale + oy)
                # Draw thick line
                n = max(abs(x1 - x0), abs(y1 - y0), 1)
                for t in range(n + 1):
                    px = int(x0 + (x1 - x0) * t / n)
                    py = int(y0 + (y1 - y0) * t / n)
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            nx, ny = px + dx, py + dy
                            if 0 <= nx < IMG_SIZE and 0 <= ny < IMG_SIZE:
                                img[ny, nx] = 1.0
        return img

    def build_spell(self):
        """Build spell graph from recognized symbols with auto-connections."""
        if not self.recognized:
            return None

        graph = SpellGraph(circle_closed=self.circle_closed)
        nodes = []
        for name, conf, bbox in self.recognized:
            sym_info = {
                "fire": (SignalKind.FORCE, Element.FIRE),
                "water": (SignalKind.FORCE, Element.WATER),
                "air": (SignalKind.FORCE, Element.AIR),
                "earth": (SignalKind.FORCE, Element.EARTH),
                "light": (SignalKind.FORCE, Element.LIGHT),
                "dark": (SignalKind.FORCE, Element.DARK),
                "life": (SignalKind.DOMAIN, None),
                "mind": (SignalKind.DOMAIN, None),
                "space": (SignalKind.DOMAIN, None),
                "time": (SignalKind.DOMAIN, None),
            }.get(name, (SignalKind.DOMAIN, None))

            nid = graph.add_node(name, sym_info[0], sym_info[1])
            nodes.append((nid, bbox))

        # Auto-connect by spatial proximity
        # Symbols drawn close together → mutual; far apart → direct
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                nid_a, bbox_a = nodes[i]
                nid_b, bbox_b = nodes[j]
                dist = math.hypot(bbox_a.centerx - bbox_b.centerx,
                                  bbox_a.centery - bbox_b.centery)
                if dist < 150:
                    # Close = mutual (mixing)
                    graph.connect(nid_a, nid_b, EdgeType.MUTUAL)
                elif dist < 300:
                    # Medium = direct (one affects other)
                    # Left-to-right or top-to-bottom ordering
                    if bbox_a.centerx < bbox_b.centerx or bbox_a.centery < bbox_b.centery:
                        graph.connect(nid_a, nid_b, EdgeType.DIRECT)
                    else:
                        graph.connect(nid_b, nid_a, EdgeType.DIRECT)

        return graph

    def cast(self):
        """Execute a spell cast."""
        self.recognize_strokes()
        graph = self.build_spell()
        if graph is None or not graph.nodes:
            self.add_log("nothing to cast")
            return

        self.spell_graph = graph
        effect = evaluate_spell(graph)
        self.last_effect = effect

        # Wand modifier
        profile = self.wand.affinity_profile().numpy()
        wand_dominant = int(np.argmax(profile))

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
        alignment = sum(p * s for p, s in zip(profile, spell_poles))
        multiplier = 0.5 + alignment
        final_power = effect.total_power * multiplier

        self.last_cast_info = {
            "symbols": [graph.nodes[i].name for i in sorted(graph.nodes)],
            "power": final_power,
            "multiplier": multiplier,
            "effect": effect,
        }

        # Log
        syms = " + ".join(self.last_cast_info["symbols"])
        self.add_log(f"cast: {syms}")
        for s in effect.synergies:
            self.add_log(f"  ✦ {s}")
        for c in effect.cancellations:
            self.add_log(f"  ✗ {c}")
        if effect.tags:
            self.add_log(f"  [{', '.join(sorted(effect.tags))}]")
        self.add_log(f"  power={final_power:.2f} (wand ×{multiplier:.2f})")

        # Spawn particles
        self._spawn_cast_particles(effect)

        # Create experience events
        pole_aff = spell_poles
        weight = 0.1 + 0.3 * min(final_power, 3.0) / 3.0
        self.pending_events.append(Event(
            EventType.ACTION, pole_aff,
            Emotion(min(final_power / 3, 0.9), weight, 0.5),
            weight,
        ))

        self.cast_count += 1
        self.cast_flash = 30
        self.history.append(self.last_cast_info)

    def _spawn_cast_particles(self, effect: SpellEffect):
        cx = CANVAS_X + CANVAS_W // 2
        cy = CANVAS_Y + CANVAS_H // 2
        power = effect.total_power

        ELEM_COLORS = {
            Element.FIRE: (255, 120, 50),
            Element.WATER: (50, 150, 255),
            Element.AIR: (180, 220, 255),
            Element.EARTH: (160, 120, 60),
            Element.LIGHT: (255, 255, 200),
            Element.DARK: (80, 40, 100),
        }

        # --- Element-specific effects ---
        for elem, ep in effect.force_power.items():
            if ep <= 0:
                continue
            n = int(10 + ep * 20)
            color = ELEM_COLORS.get(elem, ACCENT)

            if elem == Element.FIRE:
                for _ in range(n):
                    self.particles.append(Particle(
                        cx + random.gauss(0, 25), cy + random.gauss(0, 10),
                        color, random.gauss(0, 1.5), random.uniform(-3, -1),
                        random.uniform(0.6, 1.0), gravity=-0.03, size=5,
                        behavior="rise"))

            elif elem == Element.WATER:
                for _ in range(n):
                    self.particles.append(Particle(
                        cx + random.gauss(0, 40), cy - random.uniform(20, 80),
                        color, random.gauss(0, 0.5), random.uniform(0.5, 2),
                        random.uniform(0.4, 0.8), gravity=0.1, size=3,
                        behavior="fall"))

            elif elem == Element.AIR:
                for _ in range(n):
                    self.particles.append(Particle(
                        cx + random.gauss(0, 15), cy + random.gauss(0, 15),
                        color, random.gauss(0, 0.3), random.gauss(0, 0.3),
                        random.uniform(0.5, 1.0), size=2,
                        behavior="orbit"))

            elif elem == Element.EARTH:
                for _ in range(n):
                    angle = random.uniform(0, 2 * math.pi)
                    r = random.uniform(10, 50)
                    self.particles.append(Particle(
                        cx + r * math.cos(angle), cy + r * math.sin(angle),
                        color, random.gauss(0, 0.5), random.uniform(1, 3),
                        random.uniform(0.3, 0.7), gravity=0.15, size=6,
                        behavior="default"))

            elif elem == Element.LIGHT:
                for _ in range(n):
                    angle = random.uniform(0, 2 * math.pi)
                    speed = random.uniform(2, 5)
                    self.particles.append(Particle(
                        cx, cy, color,
                        math.cos(angle) * speed, math.sin(angle) * speed,
                        random.uniform(0.3, 0.6), gravity=0, size=3,
                        behavior="default"))

            elif elem == Element.DARK:
                for _ in range(n):
                    angle = random.uniform(0, 2 * math.pi)
                    r = random.uniform(40, 80)
                    self.particles.append(Particle(
                        cx + r * math.cos(angle), cy + r * math.sin(angle),
                        color, 0, 0,
                        random.uniform(0.5, 1.0), size=5,
                        behavior="implode"))

        # --- Domain effects ---
        for domain, dp in effect.domain_power.items():
            if dp <= 0:
                continue
            n = int(8 + dp * 12)
            if domain in ("life", "body"):
                for _ in range(n):
                    self.particles.append(Particle(
                        cx + random.gauss(0, 30), cy + random.gauss(0, 30),
                        (80, 220, 100), random.gauss(0, 0.3), random.gauss(0, 0.3),
                        random.uniform(0.6, 1.2), size=4, behavior="pulse"))
            elif domain == "mind":
                for _ in range(n):
                    self.particles.append(Particle(
                        cx + random.gauss(0, 20), cy + random.gauss(0, 20),
                        (180, 100, 220), random.gauss(0, 0.5), random.gauss(0, 0.5),
                        random.uniform(0.4, 0.9), size=3, behavior="scatter"))
            elif domain in ("space", "time"):
                color = (100, 180, 255) if domain == "space" else (255, 200, 60)
                for _ in range(n):
                    self.particles.append(Particle(
                        cx + random.gauss(0, 10), cy + random.gauss(0, 10),
                        color, 0, 0,
                        random.uniform(0.5, 1.0), size=3, behavior="orbit"))

        # --- Tag-based overlays ---
        if "storm" in effect.tags:
            # Swirling dual-color particles
            for _ in range(30):
                angle = random.uniform(0, 2 * math.pi)
                c = random.choice([(255, 140, 60), (200, 230, 255)])
                self.particles.append(Particle(
                    cx, cy, c, math.cos(angle) * 3, math.sin(angle) * 3,
                    random.uniform(0.5, 1.0), size=4, behavior="orbit"))

        if "reinforced" in effect.tags:
            # Bright central burst
            for _ in range(20):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(3, 6)
                self.particles.append(Particle(
                    cx, cy, (255, 255, 255),
                    math.cos(angle) * speed, math.sin(angle) * speed,
                    random.uniform(0.2, 0.4), gravity=0, size=6, behavior="default"))

        if "solid" in effect.tags:
            # Ring of earth particles
            for i in range(24):
                angle = i / 24 * 2 * math.pi
                r = 60
                self.particles.append(Particle(
                    int(cx + r * math.cos(angle)), int(cy + r * math.sin(angle)),
                    (140, 110, 50), 0, 0,
                    random.uniform(0.8, 1.5), gravity=0, size=5, behavior="default"))

        if "sustained" in effect.tags:
            # Trailing lingering particles
            for _ in range(15):
                self.particles.append(Particle(
                    cx + random.gauss(0, 10), cy + random.gauss(0, 10),
                    (255, 180, 80), random.gauss(0, 0.2), random.gauss(0, 0.2),
                    random.uniform(1.0, 2.0), gravity=0, size=2, behavior="pulse"))

        # --- Cancellation: sad fizzle ---
        if effect.cancellations:
            self.particles.clear()  # kill existing
            for _ in range(12):
                self.particles.append(Particle(
                    cx + random.gauss(0, 15), cy + random.gauss(0, 15),
                    (100, 80, 80), random.gauss(0, 0.5), random.uniform(0.5, 1.5),
                    random.uniform(0.2, 0.4), gravity=0.08, size=2, behavior="default"))

        # --- Stability visual ---
        if effect.stability < 0.7:
            # Unstable: jittery particles
            for _ in range(10):
                self.particles.append(Particle(
                    cx + random.gauss(0, 40), cy + random.gauss(0, 40),
                    (200, 50, 50), random.gauss(0, 2), random.gauss(0, 2),
                    random.uniform(0.2, 0.5), size=2, behavior="scatter"))

        # --- Fallback if nothing generated ---
        if not self.particles:
            for _ in range(10):
                angle = random.uniform(0, 2 * math.pi)
                self.particles.append(Particle(
                    cx, cy, DIM_TEXT,
                    math.cos(angle), math.sin(angle),
                    0.3, size=2, behavior="default"))

    def add_life_event(self, key):
        if key not in LIFE_EVENTS:
            return
        name, etype, poles, emotion = LIFE_EVENTS[key]
        weight = 0.2 + emotion.intensity * 0.5
        self.pending_events.append(Event(etype, poles, emotion, weight))
        self.add_log(f"lived: {name}")

    def sleep(self):
        if not self.pending_events:
            self.add_log("nothing to dream about")
            return
        n = len(self.pending_events)
        self.trainer.train_biography(self.pending_events, epochs=3, batch_size=max(n, 8))
        self.sleep_count += 1
        self.pending_events = []
        self.add_log(f"...dreamed of {n} moments...")

    def clear_canvas(self):
        self.strokes = []
        self.current_stroke = []
        self.recognized = []
        self.last_effect = None
        self.last_cast_info = None
        self.spell_graph = None

    # ═══ Drawing ═══

    def draw_ui(self):
        self.screen.fill(BG)

        # Canvas area
        canvas_rect = pygame.Rect(CANVAS_X, CANVAS_Y, CANVAS_W, CANVAS_H)

        # Circle
        if self.cast_flash > 0:
            flash_alpha = self.cast_flash / 30.0
            flash_color = tuple(int(c * flash_alpha + CANVAS_BG[i] * (1 - flash_alpha))
                               for i, c in enumerate(ACCENT))
            pygame.draw.rect(self.screen, flash_color, canvas_rect, border_radius=10)
            self.cast_flash -= 1
        else:
            pygame.draw.rect(self.screen, CANVAS_BG, canvas_rect, border_radius=10)

        # Circle border
        circle_center = (CANVAS_X + CANVAS_W // 2, CANVAS_Y + CANVAS_H // 2)
        circle_r = min(CANVAS_W, CANVAS_H) // 2 - 20
        color = CIRCLE_COLOR if self.circle_closed else (80, 40, 40)
        if self.circle_closed:
            pygame.draw.circle(self.screen, color, circle_center, circle_r, 2)
        else:
            # Draw broken circle
            for i in range(0, 350, 20):
                a0 = math.radians(i)
                a1 = math.radians(i + 15)
                start = (int(circle_center[0] + circle_r * math.cos(a0)),
                        int(circle_center[1] + circle_r * math.sin(a0)))
                end = (int(circle_center[0] + circle_r * math.cos(a1)),
                      int(circle_center[1] + circle_r * math.sin(a1)))
                pygame.draw.line(self.screen, color, start, end, 2)

        # Draw strokes
        for stroke in self.strokes:
            if len(stroke) >= 2:
                pygame.draw.lines(self.screen, STROKE_COLOR, False, stroke, 3)
        if len(self.current_stroke) >= 2:
            pygame.draw.lines(self.screen, STROKE_COLOR, False, self.current_stroke, 3)

        # Draw recognized symbol labels
        for name, conf, bbox in self.recognized:
            # Bbox outline
            pygame.draw.rect(self.screen, (60, 80, 60), bbox, 1)
            # Label
            label = self.font_small.render(f"{name} {conf:.0%}", True, (120, 200, 120))
            self.screen.blit(label, (bbox.x, bbox.y - 14))

        # Particles
        self.particles = [p for p in self.particles if p.update()]
        for p in self.particles:
            p.draw(self.screen)

        # ═══ Right panel ═══
        y = 20

        # Title
        title = self.font_big.render("Living Wand", True, ACCENT)
        self.screen.blit(title, (PANEL_X, y))
        y += 30

        # Wand profile (always visible, compact)
        profile = self.wand.affinity_profile().numpy()
        dominant = Pole(np.argmax(profile))

        # Hexagon visualization
        hex_cx = PANEL_X + PANEL_W // 2
        hex_cy = y + 80
        hex_r = 60
        angles = [math.pi / 2 + i * math.pi / 3 for i in range(NUM_POLES)]

        # Draw hexagon outline
        hex_points = [(int(hex_cx + hex_r * math.cos(a)), int(hex_cy - hex_r * math.sin(a)))
                      for a in angles]
        pygame.draw.polygon(self.screen, (40, 38, 55), hex_points, 1)

        # Draw profile fill
        profile_points = [
            (int(hex_cx + hex_r * profile[i] / 0.5 * math.cos(angles[i])),
             int(hex_cy - hex_r * profile[i] / 0.5 * math.sin(angles[i])))
            for i in range(NUM_POLES)
        ]
        if len(profile_points) >= 3:
            # Fill
            s = pygame.Surface((W, H), pygame.SRCALPHA)
            pygame.draw.polygon(s, (*ACCENT, 40), profile_points)
            self.screen.blit(s, (0, 0))
            pygame.draw.polygon(self.screen, ACCENT, profile_points, 2)

        # Labels
        label_offset = 18
        for i, pole in enumerate(Pole):
            lx = hex_cx + (hex_r + label_offset) * math.cos(angles[i])
            ly = hex_cy - (hex_r + label_offset) * math.sin(angles[i])
            color = POLE_COLORS[pole]
            text = self.font_small.render(pole.name[:4], True, color)
            self.screen.blit(text, (lx - text.get_width() // 2, ly - text.get_height() // 2))

        y = hex_cy + hex_r + 30

        # Stats
        stats = self.font_small.render(
            f"casts: {self.cast_count}  sleeps: {self.sleep_count}  "
            f"pending: {len(self.pending_events)}", True, DIM_TEXT)
        self.screen.blit(stats, (PANEL_X, y))
        y += 20

        # Last cast info
        if self.last_cast_info:
            y += 5
            line = self.font.render("last cast:", True, DIM_TEXT)
            self.screen.blit(line, (PANEL_X, y))
            y += 18

            effect = self.last_cast_info["effect"]
            # Force bars
            for elem, power in sorted(effect.force_power.items(), key=lambda x: -x[1]):
                bar_w = int(min(power / 3.0, 1.0) * 150)
                elem_colors = {
                    Element.FIRE: (232, 93, 58), Element.WATER: (33, 150, 243),
                    Element.AIR: (144, 202, 249), Element.EARTH: (139, 105, 20),
                    Element.LIGHT: (255, 236, 179), Element.DARK: (80, 50, 120),
                }
                c = elem_colors.get(elem, ACCENT)
                pygame.draw.rect(self.screen, c, (PANEL_X, y, bar_w, 12))
                label = self.font_small.render(f"{elem.name} {power:.2f}", True, TEXT_COLOR)
                self.screen.blit(label, (PANEL_X + bar_w + 5, y - 1))
                y += 16

            for domain, power in sorted(effect.domain_power.items(), key=lambda x: -x[1]):
                bar_w = int(min(power / 3.0, 1.0) * 150)
                pygame.draw.rect(self.screen, (76, 175, 80), (PANEL_X, y, bar_w, 12))
                label = self.font_small.render(f"{domain} {power:.2f}", True, TEXT_COLOR)
                self.screen.blit(label, (PANEL_X + bar_w + 5, y - 1))
                y += 16

            pwr_text = self.font.render(
                f"power: {self.last_cast_info['power']:.2f} "
                f"(×{self.last_cast_info['multiplier']:.2f})", True, ACCENT)
            self.screen.blit(pwr_text, (PANEL_X, y))
            y += 20

        # Log
        y = max(y + 10, H - 220)
        line = self.font_small.render("─── log ───", True, DIM_TEXT)
        self.screen.blit(line, (PANEL_X, y))
        y += 15
        for msg in self.log[-10:]:
            color = TEXT_COLOR
            if msg.startswith("  ✦"):
                color = (120, 200, 120)
            elif msg.startswith("  ✗"):
                color = (200, 100, 100)
            line = self.font_small.render(msg[:50], True, color)
            self.screen.blit(line, (PANEL_X, y))
            y += 14

        # Bottom hints
        hints = "LMB: draw | RMB: cast | Space: clear | 1-6: life events | S: sleep | C: circle | Tab: profile"
        hint_text = self.font_small.render(hints, True, (50, 48, 70))
        self.screen.blit(hint_text, (20, H - 20))

        pygame.display.flip()

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.clear_canvas()
                    elif event.key == pygame.K_TAB:
                        self.show_profile = not self.show_profile
                    elif event.key == pygame.K_s:
                        self.sleep()
                    elif event.key == pygame.K_c:
                        self.circle_closed = not self.circle_closed
                        self.add_log(f"circle: {'closed' if self.circle_closed else 'open'}")
                    elif event.key in LIFE_EVENTS:
                        self.add_life_event(event.key)

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click — draw
                        mx, my = event.pos
                        canvas_rect = pygame.Rect(CANVAS_X, CANVAS_Y, CANVAS_W, CANVAS_H)
                        if canvas_rect.collidepoint(mx, my):
                            self.drawing = True
                            self.current_stroke = [(mx, my)]
                    elif event.button == 3:  # Right click — cast
                        self.cast()

                elif event.type == pygame.MOUSEMOTION:
                    if self.drawing:
                        mx, my = event.pos
                        canvas_rect = pygame.Rect(CANVAS_X, CANVAS_Y, CANVAS_W, CANVAS_H)
                        if canvas_rect.collidepoint(mx, my):
                            self.current_stroke.append((mx, my))

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1 and self.drawing:
                        self.drawing = False
                        if len(self.current_stroke) >= 3:
                            self.strokes.append(self.current_stroke)
                        self.current_stroke = []

            self.draw_ui()
            self.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    Game().run()
