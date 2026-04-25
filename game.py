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
from pathlib import Path
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
from wand.events import Event, EventType, Pole, NUM_POLES, Emotion
from wand.ocr import SymbolNet, AdaptiveOCR, train_symbol_net
from wand.synth_data import generate_dataset, SYMBOL_NAMES, IMG_SIZE
from wand.transformer import SpellFormer
from wand.tokenizer import SpellTokenizer
from wand.wand_adapter import WandAdapter, AffinityHexagon, WandAdapterTrainer
from wand.crystal import CrystalRegistry
from wand.effect import interpret_effect, EffectDescription
from wand.tags import SemanticTag, TAG_BY_NAME
from wand.grimoire import Grimoire

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

        # SpellFormer (shared magic knowledge)
        self.spell_former = SpellFormer()
        sf_path = os.path.join(os.path.dirname(__file__), "data", "spellformer.pt")
        if os.path.exists(sf_path):
            self.spell_former.load_state_dict(torch.load(sf_path, weights_only=True))
            print("SpellFormer loaded.")
        else:
            print("SpellFormer: no model, using random init")
        self.spell_former.eval()

        # Wand (personal)
        self.adapter = WandAdapter()
        self.hexagon = AffinityHexagon()
        self.trainer = WandAdapterTrainer(self.adapter, self.hexagon)
        self.tokenizer = SpellTokenizer()
        self.crystal = CrystalRegistry(threshold=500)

        self.pending_events: list[Event] = []
        self.cast_count = 0
        self.sleep_count = 0

        # Grimoire
        self.grimoire = Grimoire(os.path.join(os.path.dirname(__file__), "data", "grimoire.db"))

        # Load saved state
        self._load_state()

        # OCR — SymbolNet (CoordConv + SE + multi-scale)
        ocr_cache = os.path.join(os.path.dirname(__file__), "data", "ocr_symbolnet.pt")
        if os.path.exists(ocr_cache):
            print("Loading SymbolNet... ", end="", flush=True)
            base_cnn = SymbolNet()
            base_cnn.load_state_dict(torch.load(ocr_cache, weights_only=True))
            print("done.")
        else:
            print("Training SymbolNet (first run)... ", end="", flush=True)
            images, labels = generate_dataset(n_per_class=400)
            base_cnn, _ = train_symbol_net(images, labels, epochs=30)
            os.makedirs(os.path.dirname(ocr_cache), exist_ok=True)
            torch.save(base_cnn.state_dict(), ocr_cache)
            print("done (cached).")

        # Wrap in adaptive OCR (learns from SpellFormer feedback)
        self.ocr = AdaptiveOCR(base_cnn)
        ocr_buffer_path = os.path.join(os.path.dirname(__file__), "data", "save", "ocr_buffer.pt")
        self.ocr.load_buffer(ocr_buffer_path)
        self.symbol_cnn = base_cnn  # keep reference for recognize_strokes compatibility

        # Drawing state
        self.strokes: list[list[tuple[int, int]]] = []  # list of stroke point lists
        self.current_stroke: list[tuple[int, int]] = []
        self.drawing = False
        self.circle_closed = True
        self.draw_mode = "free"  # "free" or "line"
        self.line_start: tuple[int, int] | None = None  # for line tool

        # Recognized symbols
        self.recognized: list[tuple[str, float, pygame.Rect]] = []

        # Spell state
        self.spell_graph: SpellGraph | None = None
        self.last_effect: SpellEffect | None = None
        self.last_effect_desc: EffectDescription | None = None
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
        """Group overlapping strokes into rune clusters.

        Strokes whose bounding boxes overlap (IoU > 0.3 or center distance < 80px)
        are merged into one cluster = one multi-line rune.
        """
        # Build per-stroke bboxes
        stroke_bboxes = []
        for stroke in self.strokes:
            if len(stroke) < 2:
                continue
            xs = [p[0] for p in stroke]
            ys = [p[1] for p in stroke]
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
            if w < 5 and h < 5:
                continue
            bbox = pygame.Rect(min(xs) - 5, min(ys) - 5, max(w, 10) + 10, max(h, 10) + 10)
            stroke_bboxes.append((stroke, bbox))

        if not stroke_bboxes:
            return []

        # Union-find clustering by spatial proximity
        n = len(stroke_bboxes)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            a, b = find(a), find(b)
            if a != b:
                parent[b] = a

        for i in range(n):
            for j in range(i + 1, n):
                _, bi = stroke_bboxes[i]
                _, bj = stroke_bboxes[j]
                # Center distance
                dist = math.hypot(bi.centerx - bj.centerx, bi.centery - bj.centery)
                # Bbox overlap (IoU-like)
                overlap = bi.clip(bj)
                overlap_area = max(0, overlap.w) * max(0, overlap.h)
                min_area = min(bi.w * bi.h, bj.w * bj.h)
                iou = overlap_area / max(min_area, 1)

                if dist < 80 or iou > 0.3:
                    union(i, j)

        # Group by root
        groups: dict[int, list[int]] = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(i)

        # Build clusters
        clusters = []
        for indices in groups.values():
            strokes_in_cluster = [stroke_bboxes[i][0] for i in indices]
            # Merged bbox
            all_xs = [p[0] for i in indices for p in stroke_bboxes[i][0]]
            all_ys = [p[1] for i in indices for p in stroke_bboxes[i][0]]
            merged_bbox = pygame.Rect(
                min(all_xs) - 5, min(all_ys) - 5,
                max(all_xs) - min(all_xs) + 10,
                max(all_ys) - min(all_ys) + 10,
            )
            clusters.append((strokes_in_cluster, merged_bbox))
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
        SYMBOL_INFO = {
            "fire": (SignalKind.FORCE, Element.FIRE),
            "water": (SignalKind.FORCE, Element.WATER),
            "air": (SignalKind.FORCE, Element.AIR),
            "earth": (SignalKind.FORCE, Element.EARTH),
            "light": (SignalKind.FORCE, Element.LIGHT),
            "dark": (SignalKind.FORCE, Element.DARK),
            "body": (SignalKind.DOMAIN, None),
            "mind": (SignalKind.DOMAIN, None),
            "matter": (SignalKind.DOMAIN, None),
            "space": (SignalKind.DOMAIN, None),
            "time": (SignalKind.DOMAIN, None),
            "life": (SignalKind.DOMAIN, None),
            "inward": (SignalKind.VECTOR, None),
            "outward": (SignalKind.VECTOR, None),
            "up": (SignalKind.VECTOR, None),
            "down": (SignalKind.VECTOR, None),
            "through": (SignalKind.VECTOR, None),
            "reverse": (SignalKind.VECTOR, None),
            "point": (SignalKind.SCALAR, None),
            "flow": (SignalKind.SCALAR, None),
            "wave": (SignalKind.SCALAR, None),
            "all": (SignalKind.SCALAR, None),
            "amplify": (SignalKind.MODIFIER, None),
            "weaken": (SignalKind.MODIFIER, None),
            "hold": (SignalKind.MODIFIER, None),
            "dispel": (SignalKind.MODIFIER, None),
            "here": (SignalKind.REFERENCE, None),
            "there": (SignalKind.REFERENCE, None),
            "self": (SignalKind.REFERENCE, None),
            "target": (SignalKind.REFERENCE, None),
        }
        for name, conf, bbox in self.recognized:
            sym_info = SYMBOL_INFO.get(name, (SignalKind.DOMAIN, None))

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
        """Execute a spell cast through SpellFormer."""
        self.recognize_strokes()
        graph = self.build_spell()
        if graph is None or not graph.nodes:
            self.add_log("nothing to cast")
            return

        self.spell_graph = graph

        # Algebra evaluation (for particles + comparison)
        effect = evaluate_spell(graph)
        self.last_effect = effect

        # --- SpellFormer inference ---
        tokens = self.tokenizer.tokenize(graph)
        ids = torch.tensor([tokens.token_ids])
        roles = torch.tensor([tokens.role_ids])
        degs = torch.tensor([tokens.degree])

        with torch.no_grad():
            output = self.spell_former(ids, roles, degs)
            base_raw = output["raw"]
            spell_poles = output["pole_distribution"]

            modified = self.adapter(base_raw, spell_poles)

            poles = torch.softmax(modified[:, :6], dim=-1)[0].numpy()
            power = torch.sigmoid(modified[:, 6:7])[0, 0].item()
            stability = torch.sigmoid(modified[:, 7:8])[0, 0].item()
            area = torch.nn.functional.softplus(modified[:, 8:9])[0, 0].item()
            duration = torch.nn.functional.softplus(modified[:, 9:10])[0, 0].item()
            elem_int = torch.sigmoid(modified[:, 10:16])[0].numpy()
            dom_int = torch.sigmoid(modified[:, 16:22])[0].numpy()
            sem_emb = torch.tanh(modified[:, 22:38])[0].numpy()

        # OCR clarity modifier
        if self.recognized:
            clarity = np.mean([conf for _, conf, _ in self.recognized])
            power *= clarity
            stability *= 0.5 + 0.5 * clarity

        desc = interpret_effect(poles, power, stability, area, duration, elem_int, dom_int, sem_emb)
        self.last_effect_desc = desc

        # OCR feedback from SpellFormer — magic teaches OCR what was drawn
        if self.recognized:
            ocr_images = []
            ocr_preds = []
            for cluster_strokes, bbox in self._cluster_strokes():
                img = self._render_to_image(cluster_strokes, bbox)
                ocr_images.append(img)
            ocr_preds = [name for name, _, _ in self.recognized]
            if len(ocr_images) == len(ocr_preds):
                self.ocr.feedback_from_spell(
                    ocr_images, ocr_preds,
                    {"element_intensity": elem_int, "domain_intensity": dom_int}
                )

        # Crystallization
        canon, crystallized = self.crystal.record_cast(tokens)
        if crystallized:
            with torch.no_grad():
                h = self.spell_former.encode(ids, roles, degs)
                pooled = self.spell_former.pool(h, ids)
                self.crystal.set_crystal_embedding(canon, pooled[0])
                entry = self.crystal.get_crystal_by_hash(canon)
                if entry:
                    self.spell_former.set_crystal_embedding(entry.crystal_id, pooled[0])
            self.add_log(f"** CRYSTAL: {entry.name} **")
            # Golden crystal burst
            ccx = CANVAS_X + CANVAS_W // 2
            ccy = CANVAS_Y + CANVAS_H // 2
            for _ in range(60):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(1, 5)
                self.particles.append(Particle(
                    ccx, ccy, (255, 220, 60),
                    math.cos(angle) * speed, math.sin(angle) * speed,
                    random.uniform(1.0, 2.5), gravity=0, size=5,
                    behavior=random.choice(["orbit", "scatter", "rise"])))

        self.last_cast_info = {
            "symbols": [graph.nodes[i].name for i in sorted(graph.nodes)],
            "power": power,
            "multiplier": 1.0,
            "effect": effect,
            "desc": desc,
        }

        # Log
        syms = " + ".join(self.last_cast_info["symbols"])
        self.add_log(f"cast: {syms}")
        self.add_log(f"  {desc.title}")
        if desc.tags:
            self.add_log(f"  [{', '.join(t for t, _ in desc.tags[:3])}]")
        self.add_log(f"  power={power:.2f} ({desc.power_word})")

        # Spawn particles (use both algebra effect + SpellFormer desc)
        self._spawn_cast_particles(effect, desc)

        # Experience events
        pole_aff = poles.tolist()
        weight = 0.1 + 0.3 * min(power * 3, 1.0)
        self.pending_events.append(Event(
            EventType.ACTION, pole_aff,
            Emotion(min(power, 0.9), weight, 0.5),
            weight,
        ))

        self.cast_count += 1
        self.cast_flash = 30
        self.history.append(self.last_cast_info)

        # Record in grimoire
        rune_names = [graph.nodes[i].name for i in sorted(graph.nodes)]
        edge_triples = [(graph.nodes[e.from_id].name, e.edge_type.name, graph.nodes[e.to_id].name)
                        for e in graph.edges]
        self.grimoire.record_cast(
            runes=rune_names,
            edges=edge_triples,
            circle_closed=graph.circle_closed,
            power=power,
            stability=stability,
            title=desc.title,
            tags=[t for t, _ in desc.tags[:3]],
            crystal_hash=canon if crystallized else None,
        )

    def _spawn_cast_particles(self, effect: SpellEffect, desc: EffectDescription | None = None):
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

        # --- SpellFormer-driven layers (semantic aura + tag overlays) ---
        if desc is not None:
            self._spawn_desc_layers(desc, cx, cy)

        # --- Fallback if nothing generated ---
        if not self.particles:
            color = desc.dominant_color if desc is not None else DIM_TEXT
            for _ in range(10):
                angle = random.uniform(0, 2 * math.pi)
                self.particles.append(Particle(
                    cx, cy, color,
                    math.cos(angle), math.sin(angle),
                    0.3, size=2, behavior="default"))

    def _spawn_desc_layers(self, desc: EffectDescription, cx: int, cy: int):
        """Layer particles driven by SpellFormer interpretation.

        Adds a semantic aura ring in dominant_color, palette-weighted clouds,
        and per-tag-domain overlays so domains/tags read visually, not just
        elements.
        """
        power = max(0.05, desc.power)
        lifetime = max(0.4, desc.particle_lifetime)

        # 1. Aura ring — always; tints whole cast in the blended dominant.
        aura_n = int(8 + power * 18)
        for _ in range(aura_n):
            angle = random.uniform(0, 2 * math.pi)
            r = random.uniform(20, 60)
            self.particles.append(Particle(
                cx + r * math.cos(angle), cy + r * math.sin(angle),
                desc.dominant_color,
                math.cos(angle) * random.uniform(0.3, 1.2),
                math.sin(angle) * random.uniform(0.3, 1.2),
                random.uniform(0.5, 1.0) * lifetime,
                gravity=0, size=3, behavior="orbit"))

        # 2. Palette clouds — each weighted source contributes a small puff.
        total_w = sum(w for _, w in desc.palette) or 1.0
        for color, weight in desc.palette:
            n = int(4 + (weight / total_w) * 18 * power)
            if n <= 0:
                continue
            for _ in range(n):
                self.particles.append(Particle(
                    cx + random.gauss(0, 35), cy + random.gauss(0, 35),
                    color,
                    random.gauss(0, 0.6), random.gauss(0, 0.6),
                    random.uniform(0.4, 0.9) * lifetime,
                    gravity=0.02, size=3, behavior="default"))

        # 3. Tag-domain overlays — top tag drives a signature flourish.
        if desc.tags:
            top_name, top_sim = desc.tags[0]
            tag_obj = TAG_BY_NAME.get(top_name)
            if tag_obj is not None and top_sim > 0.3:
                self._spawn_tag_overlay(tag_obj, top_sim, desc.aura_color, cx, cy, lifetime)

    def _spawn_tag_overlay(self, tag: SemanticTag, sim: float,
                            aura: tuple[int, int, int],
                            cx: int, cy: int, lifetime: float):
        """Per-tag-domain visual flourish."""
        n = int(8 + sim * 14)

        if tag.domain == "biology":
            # Healing/blessing — gentle upward sparkles, golden-green
            for _ in range(n):
                self.particles.append(Particle(
                    cx + random.gauss(0, 30), cy + random.uniform(0, 40),
                    aura,
                    random.gauss(0, 0.3), random.uniform(-1.5, -0.5),
                    random.uniform(0.8, 1.5) * lifetime,
                    gravity=-0.02, size=4, behavior="rise"))

        elif tag.domain == "mental":
            # Mind — slow swirling violet motes
            for _ in range(n):
                angle = random.uniform(0, 2 * math.pi)
                r = random.uniform(15, 45)
                self.particles.append(Particle(
                    cx + r * math.cos(angle), cy + r * math.sin(angle),
                    aura, 0, 0,
                    random.uniform(0.7, 1.4) * lifetime,
                    gravity=0, size=2, behavior="orbit"))

        elif tag.domain == "combo" or tag.domain == "cross":
            # Hybrid — dual-color burst (aura + secondary palette color)
            secondary = aura
            d = self.last_effect_desc
            if d is not None and len(d.palette) > 1:
                secondary = d.palette[1][0]
            for _ in range(n):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(1.5, 4)
                color = aura if random.random() < 0.5 else secondary
                self.particles.append(Particle(
                    cx, cy, color,
                    math.cos(angle) * speed, math.sin(angle) * speed,
                    random.uniform(0.5, 1.0) * lifetime,
                    gravity=0, size=4, behavior="orbit"))

        elif tag.domain == "time":
            # Time — slow rewinding scatter
            for _ in range(n):
                self.particles.append(Particle(
                    cx + random.gauss(0, 20), cy + random.gauss(0, 20),
                    aura,
                    random.gauss(0, 0.4), random.gauss(0, 0.4),
                    random.uniform(1.0, 2.0) * lifetime,
                    gravity=0, size=2, behavior="pulse"))

        elif tag.domain == "space":
            # Space — converging implode
            for _ in range(n):
                angle = random.uniform(0, 2 * math.pi)
                r = random.uniform(50, 90)
                self.particles.append(Particle(
                    cx + r * math.cos(angle), cy + r * math.sin(angle),
                    aura, 0, 0,
                    random.uniform(0.6, 1.0) * lifetime,
                    gravity=0, size=3, behavior="implode"))

        elif tag.domain == "transform":
            # Transform — color-shifting scatter
            for _ in range(n):
                self.particles.append(Particle(
                    cx + random.gauss(0, 20), cy + random.gauss(0, 20),
                    aura,
                    random.gauss(0, 0.7), random.gauss(0, 0.7),
                    random.uniform(0.5, 1.0) * lifetime,
                    gravity=0, size=3, behavior="scatter"))

        elif tag.domain == "combat":
            # Combat — sharp outward shards
            for _ in range(n):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(3, 6)
                self.particles.append(Particle(
                    cx, cy, aura,
                    math.cos(angle) * speed, math.sin(angle) * speed,
                    random.uniform(0.3, 0.6) * lifetime,
                    gravity=0, size=3, behavior="default"))

        # Other domains (household/cooking/crafting/nature/...) — palette
        # clouds already cover them, no extra flourish needed.

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
        dummy = torch.zeros(1, self.adapter.effect_dim)
        self.trainer.train_events(self.pending_events, dummy)
        self.sleep_count += 1
        self.pending_events = []
        self.add_log(f"...dreamed of {n} moments...")

    def _save_state(self):
        """Save wand adapter, hexagon, crystals, and OCR buffer to disk."""
        save_dir = os.path.join(os.path.dirname(__file__), "data", "save")
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.adapter.state_dict(), os.path.join(save_dir, "adapter.pt"))
        self.hexagon.save(os.path.join(save_dir, "hexagon.json"))
        self.crystal.save_dir = Path(os.path.join(save_dir, "crystals"))
        self.crystal.save()
        # Save adapted OCR state + replay buffer
        torch.save(self.symbol_cnn.state_dict(), os.path.join(save_dir, "ocr_adapted.pt"))
        self.ocr.save_buffer(os.path.join(save_dir, "ocr_buffer.pt"))

    def _load_state(self):
        """Load saved state if exists."""
        save_dir = os.path.join(os.path.dirname(__file__), "data", "save")
        adapter_path = os.path.join(save_dir, "adapter.pt")
        if os.path.exists(adapter_path):
            self.adapter.load_state_dict(torch.load(adapter_path, weights_only=True))
            print("Wand adapter loaded.")
        hex_path = os.path.join(save_dir, "hexagon.json")
        self.hexagon.load(hex_path)
        crystal_dir = os.path.join(save_dir, "crystals")
        self.crystal.save_dir = Path(crystal_dir)
        self.crystal.load()
        # Restore cast count from grimoire
        self.cast_count = self.grimoire.total_casts()

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

        # Line tool preview
        if self.line_start is not None:
            mx, my = pygame.mouse.get_pos()
            pygame.draw.circle(self.screen, (100, 200, 100), self.line_start, 4)
            pygame.draw.line(self.screen, (80, 160, 80), self.line_start, (mx, my), 1)

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
        profile = self.adapter.affinity_profile().numpy()
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
        y += 16

        # XP per pole (compact bars)
        hex_levels = self.hexagon.levels()
        hex_profile = self.hexagon.profile()
        for i, pole in enumerate(Pole):
            lv = hex_levels[i]
            xp_frac = hex_profile[i]
            bar_w = int(xp_frac * 100)
            c = POLE_COLORS[pole]
            # Dim bar background
            pygame.draw.rect(self.screen, (30, 28, 42), (PANEL_X, y, 100, 8))
            if bar_w > 0:
                pygame.draw.rect(self.screen, c, (PANEL_X, y, bar_w, 8))
            lbl = self.font_small.render(f"{pole.name[:3]} lv{lv}", True, DIM_TEXT)
            self.screen.blit(lbl, (PANEL_X + 105, y - 2))
            y += 11
        y += 4

        # Crystal count
        n_crystals = self.crystal.num_crystals
        if n_crystals > 0:
            cr_text = self.font_small.render(f"crystals: {n_crystals}", True, (255, 200, 60))
            self.screen.blit(cr_text, (PANEL_X, y))
            y += 14

        # Last cast info
        if self.last_cast_info:
            y += 5
            desc = self.last_cast_info.get("desc")

            # SpellFormer title
            if desc:
                title_text = self.font.render(desc.title[:35], True, ACCENT)
                self.screen.blit(title_text, (PANEL_X, y))
                y += 18
                pw_text = self.font_small.render(
                    f"{desc.power_word}  stab={desc.stability:.2f}", True, TEXT_COLOR)
                self.screen.blit(pw_text, (PANEL_X, y))
                y += 16

                # Element intensity bars
                elem_colors_map = {
                    "fire": (232, 93, 58), "water": (33, 150, 243),
                    "air": (144, 202, 249), "earth": (139, 105, 20),
                    "light": (255, 236, 179), "dark": (80, 50, 120),
                }
                for name, intensity in desc.active_elements[:4]:
                    bar_w = int(intensity * 150)
                    c = elem_colors_map.get(name, ACCENT)
                    pygame.draw.rect(self.screen, c, (PANEL_X, y, bar_w, 12))
                    label = self.font_small.render(f"{name} {intensity:.2f}", True, TEXT_COLOR)
                    self.screen.blit(label, (PANEL_X + bar_w + 5, y - 1))
                    y += 16

                for name, intensity in desc.active_domains[:3]:
                    bar_w = int(intensity * 150)
                    pygame.draw.rect(self.screen, (76, 175, 80), (PANEL_X, y, bar_w, 12))
                    label = self.font_small.render(f"{name} {intensity:.2f}", True, TEXT_COLOR)
                    self.screen.blit(label, (PANEL_X + bar_w + 5, y - 1))
                    y += 16

                # Tags
                if desc.tags:
                    tag_str = ", ".join(t for t, _ in desc.tags[:3])
                    tag_text = self.font_small.render(tag_str[:40], True, DIM_TEXT)
                    self.screen.blit(tag_text, (PANEL_X, y))
                    y += 16
            else:
                line = self.font.render("last cast:", True, DIM_TEXT)
                self.screen.blit(line, (PANEL_X, y))
                y += 18

            pwr_text = self.font.render(
                f"power: {self.last_cast_info['power']:.2f}", True, ACCENT)
            self.screen.blit(pwr_text, (PANEL_X, y))
            y += 20

            # Crystal progress bar
            crystal_prog = self.crystal.approaching_crystal(
                self.tokenizer.tokenize(self.spell_graph)
            ) if self.spell_graph else 0
            if crystal_prog > 0.02:
                bar_w = int(crystal_prog * 120)
                bar_color = (255, 200, 60) if crystal_prog < 1.0 else (255, 255, 100)
                pygame.draw.rect(self.screen, (30, 28, 42), (PANEL_X, y, 120, 8))
                pygame.draw.rect(self.screen, bar_color, (PANEL_X, y, bar_w, 8))
                label = self.font_small.render(
                    f"crystal {crystal_prog:.0%}" if crystal_prog < 1.0 else "CRYSTALLIZED",
                    True, bar_color)
                self.screen.blit(label, (PANEL_X + 125, y - 2))
                y += 14

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
        hints = "draw | Shift+click: line | Ctrl+click: circle | RMB: cast | Space: clear | S: sleep | C: circle"
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
                            mods = pygame.key.get_mods()
                            if mods & pygame.KMOD_SHIFT:
                                # Shift+click = line tool
                                if self.line_start is None:
                                    self.line_start = (mx, my)
                                else:
                                    # Draw straight line between two points
                                    x0, y0 = self.line_start
                                    steps = max(abs(mx - x0), abs(my - y0), 1)
                                    line_stroke = [(int(x0 + (mx - x0) * t / steps),
                                                    int(y0 + (my - y0) * t / steps))
                                                   for t in range(steps + 1)]
                                    self.strokes.append(line_stroke)
                                    self.line_start = (mx, my)  # chain lines
                            elif mods & pygame.KMOD_CTRL:
                                # Ctrl+click = circle tool (draw arc from center)
                                if self.line_start is None:
                                    self.line_start = (mx, my)
                                else:
                                    cx, cy = self.line_start
                                    r = int(math.hypot(mx - cx, my - cy))
                                    arc_stroke = [(int(cx + r * math.cos(a * math.pi / 180)),
                                                   int(cy + r * math.sin(a * math.pi / 180)))
                                                  for a in range(0, 361, 5)]
                                    self.strokes.append(arc_stroke)
                                    self.line_start = None
                            else:
                                # Free draw
                                self.line_start = None
                                self.drawing = True
                                self.current_stroke = [(mx, my)]
                    elif event.button == 3:  # Right click — cast
                        self.line_start = None
                        self.cast()
                    elif event.button == 2:  # Middle click — clear
                        self.clear_canvas()
                        self.line_start = None

                elif event.type == pygame.MOUSEMOTION:
                    if self.drawing:
                        mx, my = event.pos
                        canvas_rect = pygame.Rect(CANVAS_X, CANVAS_Y, CANVAS_W, CANVAS_H)
                        if canvas_rect.collidepoint(mx, my):
                            self.current_stroke.append((mx, my))

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1 and self.drawing:
                        self.drawing = False
                        if len(self.current_stroke) >= 2:
                            self.strokes.append(self.current_stroke)
                        self.current_stroke = []

            self.draw_ui()
            self.clock.tick(60)

        # Save state on exit
        self._save_state()
        self.grimoire.close()
        pygame.quit()


if __name__ == "__main__":
    Game().run()
