"""Default configuration for the hidden machine-world SCM prototype."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Dict

VARS: Tuple[str, ...] = ("A", "B", "C", "D", "E")
VAR_INDEX: Dict[str, int] = {v: i for i, v in enumerate(VARS)}
STAGES: Dict[str, int] = {"A": 0, "B": 1, "C": 1, "D": 2, "E": 3}

LEGAL_EDGES: Tuple[Tuple[str, str], ...] = (
    ("A", "B"),
    ("A", "C"),
    ("A", "D"),
    ("A", "E"),
    ("B", "D"),
    ("B", "E"),
    ("C", "D"),
    ("C", "E"),
    ("D", "E"),
)

TRUE_GRAPH_EDGES = frozenset({("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", "E")})
DIRECT_COMPRESSION_EDGES = frozenset({("A", "B"), ("A", "C"), ("A", "D"), ("D", "E")})
ROOT_FANOUT_EDGES = frozenset({("A", "B"), ("A", "C"), ("A", "D"), ("A", "E")})
SPARSE_GRAPH_EDGES = frozenset({("A", "B"), ("A", "C"), ("D", "E")})
DENSE_LEGAL_EDGES = frozenset(LEGAL_EDGES)

CHECKPOINTS_DEFAULT: Tuple[int, ...] = (12, 30, 48)
CONDITIONS: Tuple[str, ...] = ("compression_first", "diagnostic_first", "mixed")

SPECIAL_GRAPH_NAMES = {
    "TRUE_GRAPH": TRUE_GRAPH_EDGES,
    "DIRECT_COMPRESSION_GRAPH": DIRECT_COMPRESSION_EDGES,
    "ROOT_FANOUT_GRAPH": ROOT_FANOUT_EDGES,
    "SPARSE_GRAPH": SPARSE_GRAPH_EDGES,
    "DENSE_LEGAL_GRAPH": DENSE_LEGAL_EDGES,
}

FAMILY_WEIGHTS_DEFAULT = {
    "true_direct_edge": 1.00,
    "false_direct_compression": 1.00,
    "false_same_stage_edge": 1.00,
    "false_reverse_edge": 0.90,
    "false_reverse_path": 0.90,
    "true_path": 0.85,
    "true_indirect_path": 0.65,
    "true_common_cause": 0.70,
    "true_common_effect": 0.65,
    "true_mediation": 0.55,
    "true_compound_mediation": 0.75,
    "true_compound_direction": 0.75,
    "false_compound": 0.75,
    "true_compound": 0.75,
    "false_no_connection": 0.70,
}

FAMILY_BIASES_DEFAULT = {k: 0.0 for k in FAMILY_WEIGHTS_DEFAULT}


@dataclass(frozen=True)
class RunConfig:
    seed: int = 2026
    n_episodes: int = 48
    bag_search_limit: int = 5000
    beam_width: int = 20
    alpha: float = 1.0
    lambda_complexity: float = 1.0
    association_decay: float = 0.95
    checkpoints: Tuple[int, ...] = CHECKPOINTS_DEFAULT
    outdir: str = "outputs"
    family_weights: Dict[str, float] = field(default_factory=lambda: dict(FAMILY_WEIGHTS_DEFAULT))
    family_biases: Dict[str, float] = field(default_factory=lambda: dict(FAMILY_BIASES_DEFAULT))
