"""Reference VFX renderer for SpellProgram (pygame-ce).

This file is deliberately an *engine adapter*: it knows nothing about the
algebra, the transformer or OCR. It only reads a SpellProgram (the same JSON
the HTTP API returns) plus the player's stroke geometry, so it doubles as
an example of what a Godot/Unity side would do with the API.

Look & feel — "ink of light":
  - the player's own strokes are the fuse: they ignite during the charge,
    burn away into sparks on release;
  - additive glow sprites (pre-rendered radial gradients) instead of flat dots;
  - a timeline from the program: charge → release → travel → impact → linger;
  - motion language per channel (rise/fall/orbit/pulse/scatter/implode);
  - signatures (steam, storm, lava, crystal, overload, leak …) as flourishes;
  - instability is visible: turbulence jitters particles, an open circle leaks
    sparks through its gap, an overloaded circle cracks; backfire implodes.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import pygame

Color = tuple[int, int, int]


# ═══════════════════════════════════════════
# Glow sprites
# ═══════════════════════════════════════════

class GlowCache:
    """Radial-gradient sprites for additive blending, cached by (colour, radius)."""

    def __init__(self):
        self._cache: dict[tuple, pygame.Surface] = {}

    def get(self, color: Color, radius: int) -> pygame.Surface:
        radius = max(2, min(int(radius), 96))
        key = (color[0] >> 3, color[1] >> 3, color[2] >> 3, radius)
        surf = self._cache.get(key)
        if surf is None:
            size = radius * 2
            y, x = np.mgrid[0:size, 0:size].astype(np.float32)
            d = np.sqrt((x - radius + 0.5) ** 2 + (y - radius + 0.5) ** 2) / radius
            fall = np.clip(1.0 - d, 0.0, 1.0) ** 2.2
            core = np.clip(1.0 - d * 3.0, 0.0, 1.0) ** 2 * 0.6  # white-hot centre
            rgb = np.zeros((size, size, 3), dtype=np.float32)
            for i in range(3):
                rgb[..., i] = fall * color[i] + core * 255
            arr = np.clip(rgb, 0, 255).astype(np.uint8).transpose(1, 0, 2)
            surf = pygame.surfarray.make_surface(arr).convert()
            self._cache[key] = surf
        return surf


def _mix(a: Color, b: Color, t: float) -> Color:
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _scale(c: Color, k: float) -> Color:
    return tuple(max(0, min(255, int(v * k))) for v in c)


# ═══════════════════════════════════════════
# Particles
# ═══════════════════════════════════════════

class Particle:
    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "color", "size", "drag",
                 "motion", "ox", "oy", "phase", "grow")

    def __init__(self, x, y, vx, vy, life, color, size, motion="drift", ox=0.0, oy=0.0,
                 drag=0.98, grow=0.0):
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
        self.life = self.max_life = life
        self.color, self.size, self.motion = color, size, motion
        self.ox, self.oy = ox, oy
        self.drag = drag
        self.phase = random.random() * math.tau
        self.grow = grow

    def update(self, dt: float, turbulence: float) -> bool:
        m = self.motion
        if m == "rise":
            self.vy -= 70 * dt
            self.vx += math.sin(self.phase + self.life * 9) * 30 * dt
        elif m == "fall":
            self.vy += 160 * dt
        elif m == "orbit":
            dx, dy = self.x - self.ox, self.y - self.oy
            r = math.hypot(dx, dy) + 1e-3
            self.vx += (-dy / r * 140 - dx / r * 10) * dt
            self.vy += (dx / r * 140 - dy / r * 10) * dt
        elif m == "implode":
            self.vx += (self.ox - self.x) * 3.0 * dt
            self.vy += (self.oy - self.y) * 3.0 * dt
        elif m == "scatter":
            self.vx += random.gauss(0, 120) * dt
            self.vy += random.gauss(0, 120) * dt
        elif m == "pulse":
            self.size *= 1.0 + 0.6 * math.sin(self.phase + self.life * 12) * dt
        if turbulence > 0.05:
            j = 260 * turbulence * dt
            self.vx += random.gauss(0, j)
            self.vy += random.gauss(0, j)
        self.vx *= self.drag
        self.vy *= self.drag
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.size = max(0.5, self.size + self.grow * dt)
        self.life -= dt
        return self.life > 0

    def draw(self, surf: pygame.Surface, glow: GlowCache):
        k = max(self.life / self.max_life, 0.0)
        r = int(self.size * (0.8 + 1.4 * k)) + 2
        sprite = glow.get(_scale(self.color, 0.35 + 0.65 * k), r)
        surf.blit(sprite, (self.x - r, self.y - r), special_flags=pygame.BLEND_ADD)


# ═══════════════════════════════════════════
# Spell timeline player
# ═══════════════════════════════════════════

@dataclass
class Stage:
    origin: tuple[float, float]
    target: tuple[float, float]
    radius_px: float


class SpellPlayer:
    """Plays one SpellProgram (as dict) over time."""

    def __init__(self, program: dict, origin, target_pos, glyph_strokes, vfx: "VFX",
                 backfired: bool = False, dream: bool = False):
        self.p = program
        self.vfx = vfx
        self.t = 0.0
        self.backfired = backfired
        self.dream = dream
        v = program["visual"]
        self.palette: list[Color] = [tuple(c) for c, _ in v["palette"]] or [tuple(v["core_color"])]
        if dream:
            self.palette = [_mix(c, (140, 170, 255), 0.45) for c in self.palette]
        self.core: Color = tuple(v["core_color"]) if not dream else self.palette[0]
        self.glow: Color = tuple(v["glow_color"]) if not dream else self.palette[-1]
        self.motions: list[str] = v["motions"] or ["pulse"]
        self.signatures: set[str] = set(v["signatures"])
        self.intensity = max(0.15, float(v["intensity"]))
        self.turbulence = float(v["turbulence"])
        sel = program["target"]["selector"]
        self.stage = Stage(
            origin=origin,
            target=origin if sel in ("self", "caster_area") else target_pos,
            radius_px=max(24.0, min(program["shape"]["radius"] * 30.0, 210.0)),
        )
        self.glyph_strokes = glyph_strokes  # list[list[(x, y)]]
        self.phases = {ph["name"]: (ph["start"], ph["start"] + ph["duration"])
                       for ph in program["phases"]}
        self.end = max(e for _, e in self.phases.values()) + 0.8
        self._done_events: set[str] = set()
        self.proj = None  # (x, y) while travelling

    # ── timeline helpers ──

    def _in(self, name: str) -> float | None:
        if name not in self.phases:
            return None
        a, b = self.phases[name]
        if a <= self.t < b:
            return (self.t - a) / max(b - a, 1e-6)
        return None

    def _once(self, name: str) -> bool:
        if name in self.phases and self.t >= self.phases[name][0] and name not in self._done_events:
            self._done_events.add(name)
            return True
        return False

    @property
    def charge_level(self) -> float:
        """0..1 during charge, 1 after release (strokes burn), used by the game."""
        if "charge" not in self.phases:
            return 1.0
        a, b = self.phases["charge"]
        return max(0.0, min((self.t - a) / max(b - a, 1e-6), 1.0))

    @property
    def released(self) -> bool:
        return self.t >= self.phases.get("release", (0, 0))[0]

    def pcolor(self) -> Color:
        return random.choice(self.palette)

    def motion(self) -> str:
        return random.choice(self.motions)

    # ── update ──

    def update(self, dt: float) -> bool:
        self.t += dt
        vfx, st = self.vfx, self.stage
        ox, oy = st.origin

        c = self._in("charge")
        if c is not None:
            vfx.flare(st.origin, self.core, 30 + 50 * c * self.intensity, 0.05)
            # sparks crawl along the strokes, motes are drawn into the circle
            for stroke in self.glyph_strokes:
                if stroke and random.random() < 0.5 + c:
                    x, y = random.choice(stroke)
                    vfx.add(Particle(x, y, random.gauss(0, 12), random.gauss(-10, 10),
                                     0.5, self.pcolor(), 3 + 3 * c, "drift"))
            for _ in range(int(3 + 10 * c * self.intensity)):
                a = random.uniform(0, math.tau)
                r = random.uniform(120, 220)
                vfx.add(Particle(ox + math.cos(a) * r, oy + math.sin(a) * r, 0, 0,
                                 random.uniform(0.4, 0.8), self.pcolor(), 3, "implode", ox, oy,
                                 drag=0.9))

        if self._once("release"):
            vfx.flash(self.core, 0.35 + 0.4 * self.intensity)
            vfx.flare(st.origin, self.glow, 70 + 60 * self.intensity, 0.5)
            vfx.shake(4 + 10 * self.intensity)
            for stroke in self.glyph_strokes:   # the drawing burns away
                for (x, y) in stroke[::2]:
                    a = math.atan2(y - oy, x - ox)
                    sp = random.uniform(40, 160)
                    vfx.add(Particle(x, y, math.cos(a) * sp, math.sin(a) * sp,
                                     random.uniform(0.4, 0.9), self.pcolor(), 4, self.motion(),
                                     ox, oy, drag=0.95))
            vfx.ring(st.origin, self.core, 18, 140 + 120 * self.intensity, 0.45)
            if self.backfired:
                self._backfire()
            if "leak" in self.signatures:
                vfx.leak_burst = 1.2
            if "overload" in self.signatures:
                vfx.crack(st.origin)

        tr = self._in("travel")
        kind = self.p["delivery"]["kind"]
        if tr is not None and kind == "projectile":
            tx, ty = st.target
            arc = math.sin(tr * math.pi) * -30
            x, y = ox + (tx - ox) * tr, oy + (ty - oy) * tr + arc
            self.proj = (x, y)
            for _ in range(int(4 + 10 * self.intensity)):
                vfx.add(Particle(x + random.gauss(0, 4), y + random.gauss(0, 4),
                                 random.gauss(0, 30), random.gauss(0, 30),
                                 random.uniform(0.25, 0.6), self.pcolor(),
                                 5 + 6 * self.intensity, self.motion(), x, y, drag=0.9))
            vfx.flare((x, y), self.glow, 26 + 26 * self.intensity, 0.12)
            vfx.flare((x, y), (255, 250, 235), 9 + 6 * self.intensity, 0.06)
        else:
            self.proj = None

        if self._once("impact"):
            self._impact(st.target)

        lg = self._in("linger")
        if lg is not None:
            self._linger(lg, kind)
        if kind == "beam" and (self._in("impact") is not None or lg is not None):
            vfx.beam(st.origin, st.target, self.core, self.glow, self.intensity)
        return self.t < self.end

    def _impact(self, pos):
        vfx, st = self.vfx, self.stage
        x, y = pos
        vfx.shake(3 + 12 * self.intensity)
        vfx.flare(pos, self.glow, 60 + st.radius_px * 0.8, 0.7)
        vfx.flare(pos, (255, 250, 235), 22 + 20 * self.intensity, 0.25)
        vfx.ring(pos, self.glow, 10, st.radius_px * 1.2, 0.6)
        n = int(30 + 90 * self.intensity)
        for _ in range(n):
            a = random.uniform(0, math.tau)
            sp = random.uniform(60, 260) * (0.6 + self.intensity)
            vfx.add(Particle(x, y, math.cos(a) * sp, math.sin(a) * sp,
                             random.uniform(0.5, 1.3), self.pcolor(), random.uniform(3, 8),
                             self.motion(), x, y, drag=0.93))
        self._flourishes(pos)

    def _flourishes(self, pos):
        vfx, sig = self.vfx, self.signatures
        x, y = pos
        r = self.stage.radius_px
        if sig & {"steam", "mist", "smoke"}:
            for _ in range(40):
                vfx.add(Particle(x + random.gauss(0, r * 0.4), y + random.gauss(0, 10),
                                 random.gauss(0, 15), random.uniform(-60, -20), random.uniform(1.0, 2.0),
                                 (150, 160, 175), random.uniform(10, 18), "rise", drag=0.97, grow=6))
        if sig & {"storm", "void_wind", "dust"}:
            for _ in range(70):
                a = random.uniform(0, math.tau)
                rr = random.uniform(10, r)
                vfx.add(Particle(x + math.cos(a) * rr, y + math.sin(a) * rr, 0, 0,
                                 random.uniform(0.8, 1.6), random.choice([self.core, (220, 235, 255)]),
                                 4, "orbit", x, y, drag=0.97))
        if sig & {"lava", "forge", "volcano"}:
            for _ in range(40):
                vfx.add(Particle(x + random.gauss(0, 12), y, random.gauss(0, 90), random.uniform(-260, -120),
                                 random.uniform(0.8, 1.4), (255, random.randint(90, 160), 30),
                                 random.uniform(4, 7), "fall", drag=0.99))
        if sig & {"crystal", "gem", "prism", "rainbow"}:
            for k in range(6):
                a = k / 6 * math.tau
                for s in range(8):
                    d = 12 + s * r / 10
                    vfx.add(Particle(x + math.cos(a) * d, y + math.sin(a) * d, 0, 0,
                                     0.5 + s * 0.06, _mix((255, 255, 255), self.pcolor(), s / 8),
                                     6 - s * 0.5, "pulse", drag=1.0))
        if sig & {"flash", "radiance"}:
            vfx.flash((255, 250, 220), 0.6)
        if sig & {"twilight", "eclipse", "hellfire"}:
            vfx.ring(pos, (90, 40, 140), r * 0.3, r * 1.4, 0.9)
        if "fizzle" in sig:
            for _ in range(25):
                vfx.add(Particle(x + random.gauss(0, 10), y + random.gauss(0, 10),
                                 random.gauss(0, 30), random.uniform(10, 50), random.uniform(0.3, 0.7),
                                 (110, 100, 100), 3, "fall", drag=0.95))
        if "dispel" in sig or "dispelled" in sig:
            vfx.ring(pos, (210, 230, 255), r * 1.5, 6, 0.6)  # collapsing ring
        if "blast" in sig:
            vfx.ring(pos, (255, 230, 180), 8, r * 1.8, 0.4)
            vfx.shake(10)

    def _linger(self, k: float, kind: str):
        vfx, st = self.vfx, self.stage
        x, y = st.target
        rate = int(2 + 6 * self.intensity)
        if kind == "aura":
            x, y = st.origin
            for _ in range(rate):
                a = random.uniform(0, math.tau)
                rr = st.radius_px * random.uniform(0.5, 0.9)
                vfx.add(Particle(x + math.cos(a) * rr, y + math.sin(a) * rr, 0, 0,
                                 random.uniform(0.6, 1.0), self.pcolor(), 4, "orbit", x, y))
        else:
            for _ in range(rate):
                a = random.uniform(0, math.tau)
                rr = st.radius_px * math.sqrt(random.random())
                vfx.add(Particle(x + math.cos(a) * rr, y + math.sin(a) * rr,
                                 random.gauss(0, 10), random.gauss(0, 10),
                                 random.uniform(0.5, 1.1), self.pcolor(), random.uniform(3, 6),
                                 self.motion(), x, y))
            if random.random() < 0.08:
                vfx.ring((x, y), self.glow, st.radius_px * 0.6, st.radius_px, 0.7)

    def _backfire(self):
        vfx = self.vfx
        x, y = self.stage.origin
        vfx.shake(18)
        vfx.flash((255, 60, 40), 0.5)
        for _ in range(120):
            a = random.uniform(0, math.tau)
            rr = random.uniform(80, 180)
            vfx.add(Particle(x + math.cos(a) * rr, y + math.sin(a) * rr, 0, 0,
                             random.uniform(0.4, 0.8), random.choice([(200, 30, 30), (60, 10, 40)]),
                             5, "implode", x, y, drag=0.88))


# ═══════════════════════════════════════════
# Scene-level effects
# ═══════════════════════════════════════════

@dataclass
class Flare:
    """A big soft light that fades — muzzle flash, impact bloom, orb core."""
    pos: tuple[float, float]
    color: Color
    radius: float
    life: float
    t: float = 0.0


@dataclass
class Ring:
    pos: tuple[float, float]
    color: Color
    r0: float
    r1: float
    life: float
    t: float = 0.0


class VFX:
    MAX_PARTICLES = 2600

    def __init__(self, size: tuple[int, int]):
        self.glow = GlowCache()
        self.particles: list[Particle] = []
        self.players: list[SpellPlayer] = []
        self.rings: list[Ring] = []
        self.flares: list[Flare] = []
        self.beams: list[tuple] = []
        self.cracks: list[tuple[list, float]] = []
        self.size = size
        self._shake = 0.0
        self._flash: tuple[Color, float] = ((0, 0, 0), 0.0)
        self.leak_burst = 0.0
        self.layer = pygame.Surface(size).convert()

    # ── API used by players / the game ──

    def add(self, p: Particle):
        if len(self.particles) < self.MAX_PARTICLES:
            self.particles.append(p)

    def play(self, player: SpellPlayer):
        self.players.append(player)

    def shake(self, amount: float):
        self._shake = max(self._shake, amount)

    def flash(self, color: Color, strength: float):
        self._flash = (color, max(self._flash[1], strength))

    def flare(self, pos, color, radius, life):
        self.flares.append(Flare(pos, color, radius, life))

    def ring(self, pos, color, r0, r1, life):
        self.rings.append(Ring(pos, color, r0, r1, life))

    def beam(self, a, b, core, glow, intensity):
        self.beams.append((a, b, core, glow, intensity))

    def crack(self, center):
        x, y = center
        lines = []
        for _ in range(6):
            a = random.uniform(0, math.tau)
            pts = [(x + math.cos(a) * 180, y + math.sin(a) * 180)]
            for k in range(5):
                a2 = a + random.gauss(0, 0.35)
                d = 200 + k * 14
                pts.append((x + math.cos(a2) * d, y + math.sin(a2) * d))
            lines.append(pts)
        self.cracks.append((lines, 1.4))

    def ambient(self, center, radius, color: Color, circle_closed: bool, gap_angle: float):
        """Idle motes inside the circle; sparks leak through the gap if open."""
        x, y = center
        if random.random() < 0.35:
            a = random.uniform(0, math.tau)
            rr = radius * math.sqrt(random.random())
            self.add(Particle(x + math.cos(a) * rr, y + math.sin(a) * rr,
                              random.gauss(0, 6), random.uniform(-14, -4),
                              random.uniform(1.5, 3.0), _scale(color, 0.5), 2.5, "drift"))
        if not circle_closed:
            n = 1 + int(self.leak_burst * 6)
            for _ in range(n):
                a = gap_angle + random.gauss(0, 0.08)
                px, py = x + math.cos(a) * radius, y + math.sin(a) * radius
                sp = random.uniform(40, 120) * (1 + self.leak_burst)
                self.add(Particle(px, py, math.cos(a) * sp, math.sin(a) * sp,
                                  random.uniform(0.3, 0.7), (255, 170, 90), 3, "drift", drag=0.97))

    # ── frame ──

    def update(self, dt: float):
        self.players = [p for p in self.players if p.update(dt)]
        turb = max((p.turbulence for p in self.players), default=0.0)
        self.particles = [p for p in self.particles if p.update(dt, turb * 0.6)]
        for r in self.rings:
            r.t += dt
        self.rings = [r for r in self.rings if r.t < r.life]
        for f in self.flares:
            f.t += dt
        self.flares = [f for f in self.flares if f.t < f.life]
        self.cracks = [(l, t - dt) for l, t in self.cracks if t - dt > 0]
        self._shake *= 0.86
        c, s = self._flash
        self._flash = (c, s * 0.88)
        self.leak_burst = max(0.0, self.leak_burst - dt)

    def offset(self) -> tuple[int, int]:
        if self._shake < 0.5:
            return 0, 0
        return int(random.gauss(0, self._shake)), int(random.gauss(0, self._shake))

    def draw(self, screen: pygame.Surface, clip: pygame.Rect | None = None):
        layer = self.layer
        layer.fill((0, 0, 0))
        for f in self.flares:
            k = 1.0 - f.t / f.life
            r = int(f.radius * (0.7 + 0.3 * k))
            sprite = self.glow.get(_scale(f.color, k), r)
            layer.blit(sprite, (f.pos[0] - r, f.pos[1] - r), special_flags=pygame.BLEND_ADD)
        for p in self.particles:
            p.draw(layer, self.glow)
        for (a, b, core, glow, inten) in self.beams:
            w = int(4 + 10 * inten)
            for k in range(3):
                jitter = [(a[0] + (b[0] - a[0]) * t + random.gauss(0, 2 + k),
                           a[1] + (b[1] - a[1]) * t + random.gauss(0, 2 + k))
                          for t in np.linspace(0, 1, 12)]
                pygame.draw.lines(layer, _scale(glow, 0.5 - k * 0.12), False, jitter, max(1, w - k * 3))
            pygame.draw.line(layer, core, a, b, max(1, w // 3))
        self.beams.clear()
        for r in self.rings:
            k = r.t / r.life
            rad = r.r0 + (r.r1 - r.r0) * (1 - (1 - k) ** 2)
            col = _scale(r.color, (1 - k) * 0.9)
            pygame.draw.circle(layer, col, (int(r.pos[0]), int(r.pos[1])), max(2, int(rad)),
                               max(1, int(6 * (1 - k)) + 1))
        for lines, t in self.cracks:
            col = _scale((255, 120, 80), min(t, 1.0))
            for pts in lines:
                pygame.draw.lines(layer, col, False, pts, 2)
        c, s = self._flash
        if s > 0.02:
            layer.fill(_scale(c, s * 0.6), special_flags=pygame.BLEND_ADD)
        screen.blit(layer, (0, 0), special_flags=pygame.BLEND_ADD)
