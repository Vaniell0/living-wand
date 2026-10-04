"""Spell graph tokenizer — converts SpellGraph to token sequences for SpellFormer.

Serialization format (max 64 tokens):
  [BOS] [CIRCLE_x] [rune_0] ... [rune_N] [SEP] [EDGE_START] [src] [op] [tgt] ... [EOS] [PAD...]

Section 1 (before SEP): which runes are present (node list).
Section 2 (after SEP): how they connect (edge triples: source, operator, target).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from .algebra import EdgeType, Element, SignalKind, SpellGraph


# ═══════════════════════════════════════════
# Vocabulary
# ═══════════════════════════════════════════

# Rune tokens 0-29 (same order as spells.py primitives)
RUNE_NAMES = [
    # Elements (6)
    "fire", "water", "air", "earth", "light", "dark",
    # Domains (6)
    "body", "mind", "matter", "space", "time", "life",
    # Directions (6)
    "inward", "outward", "up", "down", "through", "reverse",
    # Quantity (4)
    "point", "flow", "wave", "all",
    # Modifiers (4)
    "amplify", "weaken", "hold", "dispel",
    # Meta (4)
    "here", "there", "self", "target",
]
NUM_RUNES = len(RUNE_NAMES)  # 30
RUNE_TO_ID = {name: i for i, name in enumerate(RUNE_NAMES)}

# Operator tokens 30-36
OPERATOR_OFFSET = NUM_RUNES  # 30
OPERATOR_NAMES = ["DIRECT", "MUTUAL", "SEQUENCE", "CONDITION", "NEGATE", "ITERATE", "LIMIT"]
NUM_OPERATORS = len(OPERATOR_NAMES)  # 7
OPERATOR_TO_ID = {name: OPERATOR_OFFSET + i for i, name in enumerate(OPERATOR_NAMES)}

# EdgeType enum → token id
EDGE_TO_TOKEN = {
    EdgeType.DIRECT: OPERATOR_OFFSET + 0,
    EdgeType.MUTUAL: OPERATOR_OFFSET + 1,
    EdgeType.SEQUENCE: OPERATOR_OFFSET + 2,
    EdgeType.CONDITION: OPERATOR_OFFSET + 3,
    EdgeType.NEGATE: OPERATOR_OFFSET + 4,
    EdgeType.ITERATE: OPERATOR_OFFSET + 5,
    EdgeType.LIMIT: OPERATOR_OFFSET + 6,
}


class SpecialToken(IntEnum):
    BOS = 37
    EOS = 38
    PAD = 39
    CIRCLE_OPEN = 40
    CIRCLE_CLOSED = 41
    SEP = 42
    EDGE_START = 43


# Base vocab size (before crystals)
BASE_VOCAB_SIZE = 44
# Crystal tokens start here
CRYSTAL_OFFSET = 48
# Max crystal slots pre-allocated
MAX_CRYSTALS = 64
# Total vocab capacity
VOCAB_SIZE = CRYSTAL_OFFSET + MAX_CRYSTALS  # 112

# Sequence limits
MAX_SEQ_LEN = 64


# ═══════════════════════════════════════════
# Structural roles (for role embedding)
# ═══════════════════════════════════════════

class StructRole(IntEnum):
    META = 0       # BOS, EOS, PAD, CIRCLE_x
    NODE = 1       # rune tokens in node section
    SEPARATOR = 2  # SEP, EDGE_START
    EDGE = 3       # tokens within edge triples


# ═══════════════════════════════════════════
# Tokenizer
# ═══════════════════════════════════════════

@dataclass
class TokenizedSpell:
    """Result of tokenization."""
    token_ids: list[int]        # length = MAX_SEQ_LEN (padded)
    role_ids: list[int]         # structural role per position
    degree: list[int]           # graph degree per rune (0 for non-rune tokens)
    length: int                 # actual length before padding
    node_count: int             # number of runes
    edge_count: int             # number of edges


class SpellTokenizer:
    """Converts SpellGraph ↔ token sequences."""

    def tokenize(self, graph: SpellGraph) -> TokenizedSpell:
        """Serialize a SpellGraph to a fixed-length token sequence."""
        tokens: list[int] = []
        roles: list[int] = []
        degrees: list[int] = []

        # Compute per-node degree
        node_degrees: dict[int, int] = {nid: 0 for nid in graph.nodes}
        for edge in graph.edges:
            node_degrees[edge.from_id] = node_degrees.get(edge.from_id, 0) + 1
            node_degrees[edge.to_id] = node_degrees.get(edge.to_id, 0) + 1

        # Map node ids to their rune token ids
        node_id_to_token: dict[int, int] = {}
        for nid, node in graph.nodes.items():
            name = node.name.lower()
            if name in RUNE_TO_ID:
                node_id_to_token[nid] = RUNE_TO_ID[name]

        # --- BOS ---
        tokens.append(SpecialToken.BOS)
        roles.append(StructRole.META)
        degrees.append(0)

        # --- Circle state ---
        circle_tok = SpecialToken.CIRCLE_CLOSED if graph.circle_closed else SpecialToken.CIRCLE_OPEN
        tokens.append(circle_tok)
        roles.append(StructRole.META)
        degrees.append(0)

        # --- Node section: list runes ---
        # Canonical order (rune id, degree) — tokens must not depend on the
        # order in which the player happened to draw the runes.
        node_order = sorted(
            graph.nodes.keys(),
            key=lambda nid: (node_id_to_token.get(nid, 999), node_degrees.get(nid, 0)),
        )
        for nid in node_order:
            tok = node_id_to_token.get(nid)
            if tok is None:
                continue  # skip unknown nodes
            tokens.append(tok)
            roles.append(StructRole.NODE)
            degrees.append(min(node_degrees.get(nid, 0), 8))

        node_count = sum(1 for r in roles if r == StructRole.NODE)

        # --- SEP ---
        tokens.append(SpecialToken.SEP)
        roles.append(StructRole.SEPARATOR)
        degrees.append(0)

        # --- Edge section: triples ---
        edge_count = 0
        for src_tok, op_tok, tgt_tok, src_deg, tgt_deg in self._canonical_edges(
                graph, node_id_to_token, node_degrees):

            # Check space: need 4 tokens (EDGE_START + src + op + tgt) + EOS
            if len(tokens) + 5 > MAX_SEQ_LEN:
                break

            tokens.append(SpecialToken.EDGE_START)
            roles.append(StructRole.SEPARATOR)
            degrees.append(0)

            tokens.append(src_tok)
            roles.append(StructRole.EDGE)
            degrees.append(src_deg)

            tokens.append(op_tok)
            roles.append(StructRole.EDGE)
            degrees.append(0)

            tokens.append(tgt_tok)
            roles.append(StructRole.EDGE)
            degrees.append(tgt_deg)

            edge_count += 1

        # --- EOS ---
        tokens.append(SpecialToken.EOS)
        roles.append(StructRole.META)
        degrees.append(0)

        actual_len = len(tokens)

        # --- PAD to MAX_SEQ_LEN ---
        while len(tokens) < MAX_SEQ_LEN:
            tokens.append(SpecialToken.PAD)
            roles.append(StructRole.META)
            degrees.append(0)

        # Truncate if somehow over (shouldn't happen with edge check above)
        tokens = tokens[:MAX_SEQ_LEN]
        roles = roles[:MAX_SEQ_LEN]
        degrees = degrees[:MAX_SEQ_LEN]

        return TokenizedSpell(
            token_ids=tokens,
            role_ids=roles,
            degree=degrees,
            length=actual_len,
            node_count=node_count,
            edge_count=edge_count,
        )

    @staticmethod
    def _canonical_edges(graph: SpellGraph, node_id_to_token: dict[int, int],
                         node_degrees: dict[int, int]) -> list[tuple[int, int, int, int, int]]:
        """Edge triples (src, op, tgt, src_deg, tgt_deg) in canonical order.

        MUTUAL is symmetric, so its endpoints are sorted too.
        """
        triples = []
        for edge in graph.edges:
            src_tok = node_id_to_token.get(edge.from_id)
            tgt_tok = node_id_to_token.get(edge.to_id)
            op_tok = EDGE_TO_TOKEN.get(edge.edge_type)
            if src_tok is None or tgt_tok is None or op_tok is None:
                continue
            src_deg = min(node_degrees.get(edge.from_id, 0), 8)
            tgt_deg = min(node_degrees.get(edge.to_id, 0), 8)
            if edge.edge_type == EdgeType.MUTUAL and (tgt_tok, tgt_deg) < (src_tok, src_deg):
                src_tok, tgt_tok, src_deg, tgt_deg = tgt_tok, src_tok, tgt_deg, src_deg
            triples.append((src_tok, op_tok, tgt_tok, src_deg, tgt_deg))
        triples.sort()
        return triples

    def detokenize(self, token_ids: list[int]) -> tuple[list[str], list[tuple[str, str, str]]]:
        """Convert token sequence back to (rune_names, edge_triples).

        Returns:
            (runes, edges) where edges are (source_name, operator_name, target_name)
        """
        id_to_rune = {i: name for name, i in RUNE_TO_ID.items()}
        id_to_op = {OPERATOR_OFFSET + i: name for i, name in enumerate(OPERATOR_NAMES)}

        runes: list[str] = []
        edges: list[tuple[str, str, str]] = []

        in_edges = False
        i = 0
        while i < len(token_ids):
            tid = token_ids[i]

            if tid == SpecialToken.EOS or tid == SpecialToken.PAD:
                break
            elif tid == SpecialToken.SEP:
                in_edges = True
            elif not in_edges and tid in id_to_rune:
                runes.append(id_to_rune[tid])
            elif tid == SpecialToken.EDGE_START and i + 3 < len(token_ids):
                src = id_to_rune.get(token_ids[i + 1], "?")
                op = id_to_op.get(token_ids[i + 2], "?")
                tgt = id_to_rune.get(token_ids[i + 3], "?")
                edges.append((src, op, tgt))
                i += 3  # skip triple (loop will +1)

            i += 1

        return runes, edges
