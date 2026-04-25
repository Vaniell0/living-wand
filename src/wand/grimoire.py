"""Grimoire — spell journal with SQLite persistence.

Records every cast and tracks discovered recipes.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime


@dataclass
class CastRecord:
    id: int
    timestamp: str
    runes: list[str]
    edges: list[tuple[str, str, str]]
    circle_closed: bool
    power: float
    stability: float
    title: str
    tags: list[str]
    crystal_hash: str | None


@dataclass
class Recipe:
    id: int
    name: str
    runes: list[str]
    edges: list[tuple[str, str, str]]
    times_cast: int
    best_power: float
    discovered: str


class Grimoire:
    """SQLite-backed spell journal with discovery tracking."""

    def __init__(self, db_path: str = "data/grimoire.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS casts (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                runes TEXT,
                edges TEXT,
                circle_closed INTEGER,
                power REAL,
                stability REAL,
                title TEXT,
                tags TEXT,
                crystal_hash TEXT
            );
            CREATE TABLE IF NOT EXISTS recipes (
                id INTEGER PRIMARY KEY,
                name TEXT,
                runes TEXT,
                edges TEXT,
                times_cast INTEGER DEFAULT 1,
                best_power REAL,
                discovered TEXT
            );
            CREATE TABLE IF NOT EXISTS discoveries (
                id INTEGER PRIMARY KEY,
                discovery_key TEXT UNIQUE,
                name TEXT,
                name_ru TEXT,
                category TEXT,
                discovered_at TEXT,
                cast_id INTEGER
            );
        """)
        self.conn.commit()

    def record_cast(
        self,
        runes: list[str],
        edges: list[tuple[str, str, str]],
        circle_closed: bool,
        power: float,
        stability: float,
        title: str,
        tags: list[str],
        crystal_hash: str | None = None,
    ) -> list:
        """Record a spell cast. Returns list of new Discoverable finds."""
        from .combo_data import match_cast, Discoverable

        now = datetime.now().isoformat(timespec="seconds")
        self.conn.execute(
            "INSERT INTO casts (timestamp, runes, edges, circle_closed, power, stability, title, tags, crystal_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (now, json.dumps(runes), json.dumps(edges), int(circle_closed),
             power, stability, title, json.dumps(tags), crystal_hash),
        )
        cast_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        # Update or create recipe
        rune_key = json.dumps(sorted(runes))
        row = self.conn.execute(
            "SELECT id, times_cast, best_power FROM recipes WHERE runes = ?",
            (rune_key,),
        ).fetchone()
        if row:
            new_best = max(row["best_power"], power)
            self.conn.execute(
                "UPDATE recipes SET times_cast = times_cast + 1, best_power = ? WHERE id = ?",
                (new_best, row["id"]),
            )
        else:
            self.conn.execute(
                "INSERT INTO recipes (name, runes, edges, times_cast, best_power, discovered) "
                "VALUES (?, ?, ?, 1, ?, ?)",
                (title, rune_key, json.dumps(edges), power, now),
            )

        # Check discoveries
        new_discoveries: list[Discoverable] = []
        matches = match_cast(runes)
        for disc in matches:
            if not self.is_discovered(disc.key):
                self._record_discovery(disc.key, disc.name, disc.name_ru, disc.category, cast_id, now)
                new_discoveries.append(disc)

        self.conn.commit()
        return new_discoveries

    def recent_casts(self, limit: int = 20) -> list[CastRecord]:
        rows = self.conn.execute(
            "SELECT * FROM casts ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_cast(r) for r in rows]

    def total_casts(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM casts").fetchone()
        return row[0]

    def recipes(self, limit: int = 50) -> list[Recipe]:
        rows = self.conn.execute(
            "SELECT * FROM recipes ORDER BY times_cast DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_recipe(r) for r in rows]

    def top_recipes(self, limit: int = 10) -> list[Recipe]:
        rows = self.conn.execute(
            "SELECT * FROM recipes ORDER BY best_power DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_recipe(r) for r in rows]

    def _row_to_cast(self, row) -> CastRecord:
        return CastRecord(
            id=row["id"],
            timestamp=row["timestamp"],
            runes=json.loads(row["runes"]),
            edges=json.loads(row["edges"]),
            circle_closed=bool(row["circle_closed"]),
            power=row["power"],
            stability=row["stability"],
            title=row["title"],
            tags=json.loads(row["tags"]),
            crystal_hash=row["crystal_hash"],
        )

    def _row_to_recipe(self, row) -> Recipe:
        return Recipe(
            id=row["id"],
            name=row["name"],
            runes=json.loads(row["runes"]),
            edges=json.loads(row["edges"]),
            times_cast=row["times_cast"],
            best_power=row["best_power"],
            discovered=row["discovered"],
        )

    # ═══ Discovery tracking ═══

    def _record_discovery(self, key: str, name: str, name_ru: str,
                          category: str, cast_id: int, timestamp: str):
        self.conn.execute(
            "INSERT OR IGNORE INTO discoveries (discovery_key, name, name_ru, category, discovered_at, cast_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (key, name, name_ru, category, timestamp, cast_id),
        )

    def is_discovered(self, key: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM discoveries WHERE discovery_key = ?", (key,)
        ).fetchone()
        return row is not None

    def discovered_keys(self) -> set[str]:
        rows = self.conn.execute("SELECT discovery_key FROM discoveries").fetchall()
        return {r[0] for r in rows}

    def all_discoveries(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT discovery_key, name, name_ru, category, discovered_at FROM discoveries ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]

    def discovery_progress(self) -> tuple[int, int]:
        from .combo_data import DISCOVERY_CATALOG
        found = self.conn.execute("SELECT COUNT(*) FROM discoveries").fetchone()[0]
        return found, len(DISCOVERY_CATALOG)

    def close(self):
        self.conn.close()
