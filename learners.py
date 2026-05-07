"""Learner models for the hidden machine-world SCM task."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from config import FAMILY_BIASES_DEFAULT, FAMILY_WEIGHTS_DEFAULT, LEGAL_EDGES, STAGES, VARS
from graphs import Graph, graph_index, rank_graph
from propositions import Prop, PropositionItem, probability, truth
from scm import data_matrix
from scoring import graph_log_scores_from_matrix, posterior_from_log_scores, posterior_summary, sorted_graph_indices


@dataclass(frozen=True)
class GraphPosteriorResult:
    log_scores: np.ndarray
    posterior: np.ndarray
    prop_probs: np.ndarray
    summary: Dict[str, float | int]


@dataclass(frozen=True)
class MDLResult:
    log_scores: np.ndarray
    selected_idx: int
    selected_graph_id: str
    selected_score: float
    prop_probs: np.ndarray


def sigmoid(z: float | np.ndarray) -> float | np.ndarray:
    z_arr = np.asarray(z, dtype=float)
    out = np.empty_like(z_arr, dtype=float)
    pos = z_arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z_arr[pos]))
    ez = np.exp(z_arr[~pos])
    out[~pos] = ez / (1.0 + ez)
    if np.isscalar(z):
        return float(out)
    return out


def logit(p: float | np.ndarray) -> float | np.ndarray:
    p_arr = np.asarray(p, dtype=float)
    out = np.log(p_arr / (1.0 - p_arr))
    if np.isscalar(p):
        return float(out)
    return out


def graph_posterior_learner(
    prefix_df: pd.DataFrame,
    graphs: Sequence[Graph],
    bank: Sequence[PropositionItem],
    truth_mat: np.ndarray,
    alpha: float,
    lambda_edges: float,
    true_graph_id: str,
    direct_compression_graph_id: str,
) -> GraphPosteriorResult:
    x = data_matrix(prefix_df)
    log_scores = graph_log_scores_from_matrix(x, graphs, alpha=alpha, lambda_edges=lambda_edges)
    posterior = posterior_from_log_scores(log_scores)
    prop_probs = posterior @ truth_mat
    summary = posterior_summary(
        log_scores,
        posterior,
        graphs,
        truth_mat,
        true_graph_id,
        direct_compression_graph_id,
        [item.proposition_id for item in bank],
    )
    return GraphPosteriorResult(log_scores=log_scores, posterior=posterior, prop_probs=prop_probs, summary=summary)


def mdl_map_learner(
    prefix_df: pd.DataFrame,
    graphs: Sequence[Graph],
    truth_mat: np.ndarray,
    alpha: float,
    lambda_edges: float,
) -> MDLResult:
    x = data_matrix(prefix_df)
    log_scores = graph_log_scores_from_matrix(x, graphs, alpha=alpha, lambda_edges=lambda_edges)
    order = sorted_graph_indices(log_scores, graphs)
    selected_idx = int(order[0])
    selected_truths = truth_mat[selected_idx, :]
    prop_probs = np.where(selected_truths >= 0.5, 0.98, 0.02).astype(float)
    return MDLResult(
        log_scores=log_scores,
        selected_idx=selected_idx,
        selected_graph_id=graphs[selected_idx].graph_id,
        selected_score=float(log_scores[selected_idx]),
        prop_probs=prop_probs,
    )


def _weighted_deltas(prefix_df: pd.DataFrame, mode: str, alpha: float, decay: float) -> Dict[Tuple[str, str], float]:
    x = prefix_df.loc[:, list(VARS)].to_numpy(dtype=float)
    n = len(prefix_df)
    if mode == "association_full_memory":
        weights = np.ones(n, dtype=float)
    elif mode == "association_recency_weighted":
        positions = np.arange(1, n + 1, dtype=float)
        weights = decay ** (n - positions)
    else:
        raise ValueError(f"Unknown association mode: {mode}")
    deltas: Dict[Tuple[str, str], float] = {}
    for xi, xname in enumerate(VARS):
        xcol = x[:, xi]
        for yi, yname in enumerate(VARS):
            if xi == yi:
                deltas[(xname, yname)] = 0.0
                continue
            ycol = x[:, yi]
            mask1 = xcol == 1.0
            mask0 = xcol == 0.0
            w1 = float(weights[mask1].sum())
            w0 = float(weights[mask0].sum())
            y1_given_x1 = float((weights[mask1] * ycol[mask1]).sum()) if w1 > 0 else 0.0
            y1_given_x0 = float((weights[mask0] * ycol[mask0]).sum()) if w0 > 0 else 0.0
            p1 = (y1_given_x1 + alpha) / (w1 + 2.0 * alpha)
            p0 = (y1_given_x0 + alpha) / (w0 + 2.0 * alpha)
            deltas[(xname, yname)] = float(p1 - p0)
    return deltas


LEGAL_CHILDREN: Dict[str, Tuple[str, ...]] = {
    v: tuple(y for x, y in LEGAL_EDGES if x == v) for v in VARS
}


def _legal_chains(x: str, y: str, min_length: int = 1) -> List[Tuple[str, ...]]:
    if x == y:
        return []
    paths: List[Tuple[str, ...]] = []

    def dfs(node: str, visited: set[str], path: List[str]) -> None:
        for child in LEGAL_CHILDREN.get(node, ()):  # legal edges already increase stage
            if child in visited:
                continue
            new_path = path + [child]
            if child == y:
                if len(new_path) - 1 >= min_length:
                    paths.append(tuple(new_path))
            else:
                dfs(child, visited | {child}, new_path)

    dfs(x, {x}, [x])
    return paths


def _chain_strength(chains: Sequence[Tuple[str, ...]], deltas: Dict[Tuple[str, str], float]) -> float:
    best = 0.0
    for chain in chains:
        prod = 1.0
        for a, b in zip(chain[:-1], chain[1:]):
            prod *= max(deltas.get((a, b), 0.0), 0.0)
        best = max(best, prod)
    return float(best)


def association_atomic_probability(phi: Prop, deltas: Dict[Tuple[str, str], float]) -> float:
    op = phi.op
    a = phi.args
    if op == "Direct":
        x, y = a
        s = deltas.get((x, y), 0.0)
        temporal_bonus = 0.8 if STAGES[x] < STAGES[y] else -2.0
        return float(sigmoid(-1.5 + 6.0 * s + temporal_bonus))
    if op == "Path":
        x, y = a
        strength = _chain_strength(_legal_chains(x, y, min_length=1), deltas)
        return float(sigmoid(-1.2 + 7.0 * strength))
    if op == "Indirect":
        x, y = a
        strength = _chain_strength(_legal_chains(x, y, min_length=2), deltas)
        return float(sigmoid(-1.2 + 7.0 * strength))
    if op == "CommonCause":
        z, x, y = a
        s1 = max(deltas.get((z, x), 0.0), 0.0)
        s2 = max(deltas.get((z, y), 0.0), 0.0)
        return float(sigmoid(-1.2 + 4.0 * (s1 + s2)))
    if op == "CommonEffect":
        z, x, y = a
        s1 = max(deltas.get((x, z), 0.0), 0.0)
        s2 = max(deltas.get((y, z), 0.0), 0.0)
        return float(sigmoid(-1.2 + 4.0 * (s1 + s2)))
    if op == "Mediated":
        x, y, via_tuple = a
        via = set(via_tuple)
        chains = [chain for chain in _legal_chains(x, y, min_length=2) if set(chain[1:-1]).intersection(via)]
        strength = _chain_strength(chains, deltas)
        return float(sigmoid(-1.2 + 7.0 * strength))
    if op == "NoConnection":
        x, y = a
        assoc = abs(deltas.get((x, y), 0.0))
        return float(sigmoid(1.0 - 8.0 * assoc))
    raise ValueError(f"association_atomic_probability expected an atomic proposition, got {op}")


def association_learner(
    prefix_df: pd.DataFrame,
    bank: Sequence[PropositionItem],
    mode: str,
    alpha: float,
    decay: float,
) -> np.ndarray:
    deltas = _weighted_deltas(prefix_df, mode=mode, alpha=alpha, decay=decay)
    probs = []
    for item in bank:
        probs.append(probability(item.formal, lambda p: association_atomic_probability(p, deltas)))
    return np.asarray(probs, dtype=float)


def feature_weighted_causal_learner(
    raw_prop_probs: np.ndarray,
    bank: Sequence[PropositionItem],
    family_weights: Dict[str, float] | None = None,
    family_biases: Dict[str, float] | None = None,
    clip_range: Tuple[float, float] = (1e-5, 1.0 - 1e-5),
) -> Tuple[np.ndarray, List[Dict[str, float | str]]]:
    weights = dict(FAMILY_WEIGHTS_DEFAULT)
    if family_weights:
        weights.update(family_weights)
    biases = dict(FAMILY_BIASES_DEFAULT)
    if family_biases:
        biases.update(family_biases)

    weighted = np.zeros_like(raw_prop_probs, dtype=float)
    details: List[Dict[str, float | str]] = []
    for i, item in enumerate(bank):
        p_raw = float(np.clip(raw_prop_probs[i], clip_range[0], clip_range[1]))
        weight = float(weights.get(item.family, 1.0))
        bias = float(biases.get(item.family, 0.0))
        p_weighted = float(sigmoid(weight * logit(p_raw) + bias))
        weighted[i] = p_weighted
        details.append(
            {
                "proposition_id": item.proposition_id,
                "proposition_family": item.family,
                "p_raw": float(raw_prop_probs[i]),
                "p_weighted": p_weighted,
                "family_weight": weight,
                "family_bias": bias,
            }
        )
    return weighted, details


def beam_search_rmdl(
    ordered_condition_df: pd.DataFrame,
    graphs: Sequence[Graph],
    bank: Sequence[PropositionItem],
    truth_mat: np.ndarray,
    checkpoints: Sequence[int],
    alpha: float,
    lambda_edges: float,
    beam_width: int,
    true_graph_id: str,
    direct_compression_graph_id: str,
) -> Dict[int, Dict[str, object]]:
    ordered = ordered_condition_df.sort_values("position").reset_index(drop=True)
    checkpoints_set = set(int(c) for c in checkpoints)
    max_checkpoint = max(checkpoints)
    idx_by_graph = graph_index(graphs)
    true_idx = idx_by_graph[true_graph_id]
    direct_idx = idx_by_graph[direct_compression_graph_id]

    live = np.arange(len(graphs), dtype=int)
    true_pruned_at: int | None = None
    direct_pruned_at: int | None = None
    results: Dict[int, Dict[str, object]] = {}

    graph_masks = np.asarray([g.mask for g in graphs], dtype=int)

    for t in range(1, max_checkpoint + 1):
        prefix = ordered.iloc[:t]
        scores_all = graph_log_scores_from_matrix(data_matrix(prefix), graphs, alpha=alpha, lambda_edges=lambda_edges)
        live_scores = scores_all[live]
        # Deterministic top-k among the live set only.
        order = np.lexsort((graph_masks[live], -live_scores))
        live = live[order[:beam_width]]
        if true_pruned_at is None and true_idx not in set(live.tolist()):
            true_pruned_at = t
        if direct_pruned_at is None and direct_idx not in set(live.tolist()):
            direct_pruned_at = t

        if t in checkpoints_set:
            live_scores = scores_all[live]
            posterior = posterior_from_log_scores(live_scores)
            prop_probs = posterior @ truth_mat[live, :]
            order_live = np.lexsort((graph_masks[live], -live_scores))
            top_live = live[order_live]
            top_idx = int(top_live[0])
            beam_map_probs = np.where(truth_mat[top_idx, :] >= 0.5, 0.98, 0.02).astype(float)
            entropy = float(-np.sum(posterior * np.log(np.clip(posterior, 1e-300, 1.0))))
            results[t] = {
                "live_indices": live.copy(),
                "live_scores": live_scores.copy(),
                "posterior": posterior.copy(),
                "prop_probs": prop_probs.copy(),
                "beam_map_probs": beam_map_probs,
                "top_idx": top_idx,
                "top_graph_id": graphs[top_idx].graph_id,
                "top_score": float(scores_all[top_idx]),
                "beam_entropy": entropy,
                "true_in_beam": bool(true_idx in set(live.tolist())),
                "true_pruned_at": true_pruned_at,
                "direct_compression_in_beam": bool(direct_idx in set(live.tolist())),
                "direct_compression_pruned_at": direct_pruned_at,
            }
    return results


def graph_metric_for_graph(
    log_scores: np.ndarray,
    posterior: np.ndarray | None,
    graphs: Sequence[Graph],
    graph_id: str,
) -> Dict[str, float | int]:
    idx = graph_index(graphs)[graph_id]
    rank = rank_graph(log_scores, graphs, graph_id)
    rec: Dict[str, float | int] = {
        "rank": rank,
        "log_score": float(log_scores[idx]),
    }
    if posterior is not None:
        rec["posterior"] = float(posterior[idx])
    else:
        rec["posterior"] = np.nan
    return rec
