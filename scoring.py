"""Beta-Bernoulli graph marginal likelihoods and posterior utilities."""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp

from config import VARS
from graphs import Graph, graph_index, rank_graph
from scm import data_matrix, parent_indices_from_mask, parent_config_int


def log_beta_bernoulli_contribution(n1: float, n0: float, alpha: float) -> float:
    return float(
        gammaln(alpha + n1)
        + gammaln(alpha + n0)
        - gammaln(2.0 * alpha + n1 + n0)
        - (2.0 * gammaln(alpha) - gammaln(2.0 * alpha))
    )


def local_marginal_score(x: np.ndarray, var_idx: int, parent_mask: int, alpha: float) -> float:
    parent_indices = parent_indices_from_mask(parent_mask)
    counts: Dict[int, List[float]] = {}
    for row in x:
        cfg = parent_config_int(row, parent_indices)
        if cfg not in counts:
            counts[cfg] = [0.0, 0.0]  # n0, n1
        counts[cfg][int(row[var_idx])] += 1.0
    score = 0.0
    for n0, n1 in counts.values():
        score += log_beta_bernoulli_contribution(n1=n1, n0=n0, alpha=alpha)
    return float(score)


def compute_local_score_cache(x: np.ndarray, graphs: Sequence[Graph], alpha: float) -> Dict[Tuple[int, int], float]:
    needed = sorted({(j, g.parent_masks[j]) for g in graphs for j in range(len(VARS))})
    return {(j, mask): local_marginal_score(x, j, mask, alpha) for j, mask in needed}


def graph_log_marginal_likelihoods_from_matrix(x: np.ndarray, graphs: Sequence[Graph], alpha: float) -> np.ndarray:
    cache = compute_local_score_cache(x, graphs, alpha)
    out = np.zeros(len(graphs), dtype=float)
    for i, g in enumerate(graphs):
        out[i] = sum(cache[(j, g.parent_masks[j])] for j in range(len(VARS)))
    return out


def graph_log_scores_from_matrix(
    x: np.ndarray,
    graphs: Sequence[Graph],
    alpha: float,
    lambda_edges: float,
) -> np.ndarray:
    log_ml = graph_log_marginal_likelihoods_from_matrix(x, graphs, alpha)
    edge_penalty = lambda_edges * np.log(2.0) * np.asarray([g.num_edges for g in graphs], dtype=float)
    return log_ml - edge_penalty


def graph_log_scores(df: pd.DataFrame, graphs: Sequence[Graph], alpha: float, lambda_edges: float) -> np.ndarray:
    return graph_log_scores_from_matrix(data_matrix(df), graphs, alpha, lambda_edges)


def posterior_from_log_scores(log_scores: np.ndarray) -> np.ndarray:
    norm = logsumexp(log_scores)
    return np.exp(log_scores - norm)


def posterior_summary(
    log_scores: np.ndarray,
    posterior: np.ndarray,
    graphs: Sequence[Graph],
    truth_mat: np.ndarray,
    true_graph_id: str,
    direct_compression_graph_id: str,
    proposition_ids: Sequence[str],
) -> Dict[str, float | int]:
    idx = graph_index(graphs)
    prop_idx = {pid: i for i, pid in enumerate(proposition_ids)}
    return {
        "rank_true_graph": rank_graph(log_scores, graphs, true_graph_id),
        "rank_direct_compression_graph": rank_graph(log_scores, graphs, direct_compression_graph_id),
        "posterior_true_graph": float(posterior[idx[true_graph_id]]),
        "posterior_direct_compression_graph": float(posterior[idx[direct_compression_graph_id]]),
        "posterior_P06_Direct_A_D": float(posterior @ truth_mat[:, prop_idx["P06"]]),
        "posterior_P12_Indirect_A_D": float(posterior @ truth_mat[:, prop_idx["P12"]]),
        "posterior_P18_Mediated_A_D": float(posterior @ truth_mat[:, prop_idx["P18"]]),
        "posterior_P19_A_affects_D_not_direct": float(posterior @ truth_mat[:, prop_idx["P19"]]),
    }


def sorted_graph_indices(log_scores: np.ndarray, graphs: Sequence[Graph]) -> np.ndarray:
    # Stable deterministic ranking: highest score first, graph_id/mask as tie breaker.
    masks = np.asarray([g.mask for g in graphs], dtype=int)
    return np.lexsort((masks, -log_scores))
