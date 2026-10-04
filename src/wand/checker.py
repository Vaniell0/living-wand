"""Syntax checker — validates spell structure before it reaches the wand.

Checks:
- Circle closed
- Graph connectivity (no isolated symbols)
- No hanging symbols (unconnected)
- Contradiction detection (antagonistic primitives in direct connection)
- Returns verdict: OK / WARNING / ERROR / CRITICAL
"""

from dataclasses import dataclass
from enum import IntEnum
from .spells import Spell, Primitive, Connection, Operator, SPELL_DIM

# Known antagonisms — pairs that cause instability when directly connected
ANTAGONISMS = {
    frozenset({"fire", "water"}),
    frozenset({"light", "dark"}),
    frozenset({"amplify", "weaken"}),
    frozenset({"life", "dark"}),      # necromancy
}

# Pairs that cause CRITICAL (collapse risk) when connected via MUTUAL operator
COLLAPSE_PAIRS = {
    frozenset({"time", "time"}),      # temporal paradox (double time)
    frozenset({"amplify", "weaken"}), # direct contradiction
}


class Verdict(IntEnum):
    OK = 2
    WARNING = 1
    ERROR = 0
    CRITICAL = -1


@dataclass
class CheckResult:
    verdict: Verdict
    issues: list[str]
    circle_ok: bool
    connected: bool
    has_hanging: bool
    has_contradiction: bool
    has_collapse_risk: bool

    @property
    def can_cast(self) -> bool:
        return self.verdict >= Verdict.ERROR  # even ERROR can cast, just badly

    @property
    def stability_penalty(self) -> float:
        """Stability multiplier based on issues."""
        if self.verdict == Verdict.OK:
            return 1.0
        elif self.verdict == Verdict.WARNING:
            return 0.7
        elif self.verdict == Verdict.ERROR:
            return 0.3
        else:  # CRITICAL
            return 0.1


def check_spell(spell: Spell) -> CheckResult:
    """Run all syntax checks on a spell."""
    issues = []
    circle_ok = True
    connected = True
    has_hanging = False
    has_contradiction = False
    has_collapse = False

    # 1. Circle check
    if not spell.circle_closed:
        issues.append("Circle is not closed — energy will leak")
        circle_ok = False

    # 2. Empty spell
    if not spell.primitives:
        return CheckResult(Verdict.ERROR, ["No primitives in spell"],
                           circle_ok, False, True, False, False)

    # 3. Connectivity — build adjacency from connections
    n = len(spell.primitives)
    adj = [set() for _ in range(n)]
    for conn in spell.connections:
        if conn.from_idx < n and conn.to_idx < n:
            adj[conn.from_idx].add(conn.to_idx)
            adj[conn.to_idx].add(conn.from_idx)

    # BFS to check connectivity
    if n > 1 and spell.connections:
        visited = set()
        queue = [0]
        visited.add(0)
        while queue:
            node = queue.pop(0)
            for nb in adj[node]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        if len(visited) < n:
            connected = False
            isolated = [spell.primitives[i].name for i in range(n) if i not in visited]
            issues.append(f"Disconnected symbols: {', '.join(isolated)}")

    # 4. Hanging symbols (no connections at all)
    connected_nodes = set()
    for conn in spell.connections:
        connected_nodes.add(conn.from_idx)
        connected_nodes.add(conn.to_idx)

    hanging = [spell.primitives[i].name for i in range(n)
               if i not in connected_nodes and n > 1]
    if hanging:
        has_hanging = True
        issues.append(f"Hanging symbols (no connections): {', '.join(hanging)}")

    # 5. Antagonism check
    for conn in spell.connections:
        if conn.from_idx < n and conn.to_idx < n:
            pair = frozenset({
                spell.primitives[conn.from_idx].name,
                spell.primitives[conn.to_idx].name,
            })
            if pair in ANTAGONISMS:
                has_contradiction = True
                names = sorted(pair)
                issues.append(f"Antagonism: {names[0]} ↔ {names[-1]} — unstable")

            # Collapse check (MUTUAL connection of collapse pairs)
            if pair in COLLAPSE_PAIRS and conn.operator == Operator.MUTUAL:
                has_collapse = True
                names = sorted(pair)  # frozenset({"time","time"}) has one element
                issues.append(f"COLLAPSE RISK: {names[0]} ═══ {names[-1]} — paradox!")

    # 6. Determine verdict
    if has_collapse:
        verdict = Verdict.CRITICAL
    elif has_contradiction or (not circle_ok and has_hanging):
        verdict = Verdict.ERROR
    elif not circle_ok or has_hanging or not connected:
        verdict = Verdict.WARNING
    else:
        verdict = Verdict.OK

    return CheckResult(
        verdict=verdict,
        issues=issues,
        circle_ok=circle_ok,
        connected=connected,
        has_hanging=has_hanging,
        has_contradiction=has_contradiction,
        has_collapse_risk=has_collapse,
    )


def spell_from_graph(graph) -> Spell:
    """Convert an algebra SpellGraph into the checker's Spell.

    Edge types are mapped by NAME: EdgeType and Operator enums have different
    numbering (Operator has REVERSE at 1), so mapping by value turned MUTUAL
    into REVERSE and hid every collapse check.
    """
    from .spells import PRIMITIVE_BY_NAME

    nid_to_idx: dict[int, int] = {}
    primitives: list[Primitive] = []
    for nid, node in graph.nodes.items():
        prim = PRIMITIVE_BY_NAME.get(node.name)
        if prim is not None:
            nid_to_idx[nid] = len(primitives)
            primitives.append(prim)
    connections = []
    for edge in graph.edges:
        if edge.from_id in nid_to_idx and edge.to_id in nid_to_idx:
            connections.append(Connection(nid_to_idx[edge.from_id], nid_to_idx[edge.to_id],
                                          Operator[edge.edge_type.name]))
    return Spell(primitives, connections, graph.circle_closed, graph.circle_radius)


def check_graph(graph) -> CheckResult:
    """Syntax check straight from a SpellGraph."""
    return check_spell(spell_from_graph(graph))
