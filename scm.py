"""Hidden five-variable binary SCM and fitted CPD row likelihoods."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from config import VAR_INDEX, VARS
from graphs import Graph


def clip01(p: np.ndarray | float) -> np.ndarray | float:
    return np.clip(p, 0.0, 1.0)


def simulate_true_scm(n: int, seed: int) -> pd.DataFrame:
    """Simulate n emitted rows from the fixed hidden machine-world SCM."""
    rng = np.random.default_rng(seed)
    a = rng.binomial(1, 0.50, size=n)
    b = rng.binomial(1, clip01(0.15 + 0.70 * a))
    c = rng.binomial(1, clip01(0.15 + 0.65 * a))
    d_prob = clip01(0.05 + 0.35 * b + 0.35 * c + 0.20 * b * c)
    d = rng.binomial(1, d_prob)
    e = rng.binomial(1, clip01(0.10 + 0.75 * d))
    return pd.DataFrame({"A": a, "B": b, "C": c, "D": d, "E": e})


def ensure_row_ids(df: pd.DataFrame, candidate_seed: int | None = None) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out.insert(0, "row_id", np.arange(1, len(out) + 1, dtype=int))
    if candidate_seed is not None:
        out["candidate_seed"] = int(candidate_seed)
    return out


def data_matrix(df: pd.DataFrame) -> np.ndarray:
    return df.loc[:, list(VARS)].to_numpy(dtype=np.int8)


def parent_indices_from_mask(parent_mask: int) -> Tuple[int, ...]:
    return tuple(i for i in range(len(VARS)) if parent_mask & (1 << i))


def parent_config_int(row: np.ndarray, parent_indices: Sequence[int]) -> int:
    cfg = 0
    for bit_pos, col_idx in enumerate(parent_indices):
        cfg |= int(row[col_idx]) << bit_pos
    return cfg


@dataclass(frozen=True)
class FittedCPDs:
    graph_id: str
    alpha: float
    probs: Tuple[Dict[int, float], ...]
    parent_indices: Tuple[Tuple[int, ...], ...]


def fit_cpds_laplace(df: pd.DataFrame, graph: Graph, alpha: float = 1.0) -> FittedCPDs:
    x = data_matrix(df)
    probs: List[Dict[int, float]] = []
    pidxs: List[Tuple[int, ...]] = []
    for j in range(len(VARS)):
        parent_indices = parent_indices_from_mask(graph.parent_masks[j])
        pidxs.append(parent_indices)
        counts: Dict[int, List[float]] = {}
        for row in x:
            cfg = parent_config_int(row, parent_indices)
            if cfg not in counts:
                counts[cfg] = [0.0, 0.0]  # n0, n1
            counts[cfg][int(row[j])] += 1.0
        var_probs: Dict[int, float] = {}
        for cfg, (n0, n1) in counts.items():
            var_probs[cfg] = float((n1 + alpha) / (n0 + n1 + 2.0 * alpha))
        probs.append(var_probs)
    return FittedCPDs(graph_id=graph.graph_id, alpha=alpha, probs=tuple(probs), parent_indices=tuple(pidxs))


def log_likelihood_rows(df: pd.DataFrame, cpds: FittedCPDs) -> np.ndarray:
    x = data_matrix(df)
    out = np.zeros(len(x), dtype=float)
    eps = 1e-12
    for i, row in enumerate(x):
        ll = 0.0
        for j in range(len(VARS)):
            cfg = parent_config_int(row, cpds.parent_indices[j])
            p1 = cpds.probs[j].get(cfg, 0.5)
            p1 = min(max(p1, eps), 1.0 - eps)
            ll += np.log(p1 if row[j] == 1 else 1.0 - p1)
        out[i] = ll
    return out


def log_likelihood_row(row: Mapping[str, int], cpds: FittedCPDs) -> float:
    arr = np.asarray([row[v] for v in VARS], dtype=np.int8)
    ll = 0.0
    eps = 1e-12
    for j in range(len(VARS)):
        cfg = parent_config_int(arr, cpds.parent_indices[j])
        p1 = cpds.probs[j].get(cfg, 0.5)
        p1 = min(max(p1, eps), 1.0 - eps)
        ll += np.log(p1 if arr[j] == 1 else 1.0 - p1)
    return float(ll)
