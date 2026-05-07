"""Evidence-bag search and order-condition construction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from config import DIRECT_COMPRESSION_EDGES, TRUE_GRAPH_EDGES, VARS
from graphs import Graph, find_graph_by_edges
from propositions import PropositionItem
from scm import ensure_row_ids, fit_cpds_laplace, log_likelihood_rows, simulate_true_scm
from scoring import graph_log_scores_from_matrix, posterior_from_log_scores, posterior_summary


@dataclass(frozen=True)
class EvidenceSelection:
    evidence_bag: pd.DataFrame
    selected_seed: int
    accepted_by_thresholds: bool
    search_records: pd.DataFrame
    final_summary: Dict[str, float | int]
    warning: str | None = None


def search_evidence_bag(
    seed: int,
    n_episodes: int,
    bag_search_limit: int,
    graphs: Sequence[Graph],
    bank: Sequence[PropositionItem],
    truth_mat: np.ndarray,
    alpha: float,
) -> EvidenceSelection:
    proposition_ids = [item.proposition_id for item in bank]
    true_graph = find_graph_by_edges(graphs, TRUE_GRAPH_EDGES)
    direct_graph = find_graph_by_edges(graphs, DIRECT_COMPRESSION_EDGES)
    records: List[Dict[str, float | int | bool]] = []
    best_score = -np.inf
    best_seed = seed
    best_df: pd.DataFrame | None = None
    best_summary: Dict[str, float | int] | None = None

    for candidate_seed in range(seed, seed + bag_search_limit):
        raw = simulate_true_scm(n_episodes, candidate_seed)
        x = raw.loc[:, list(VARS)].to_numpy(dtype=np.int8)
        log_scores = graph_log_scores_from_matrix(x, graphs, alpha=alpha, lambda_edges=0.0)
        posterior = posterior_from_log_scores(log_scores)
        summary = posterior_summary(
            log_scores,
            posterior,
            graphs,
            truth_mat,
            true_graph.graph_id,
            direct_graph.graph_id,
            proposition_ids,
        )
        accepted = bool(
            summary["rank_true_graph"] <= 5
            and summary["posterior_P06_Direct_A_D"] <= 0.40
            and summary["posterior_P12_Indirect_A_D"] >= 0.50
            and summary["posterior_P18_Mediated_A_D"] >= 0.40
        )
        score = (
            float(summary["posterior_true_graph"])
            - float(summary["posterior_P06_Direct_A_D"])
            + float(summary["posterior_P12_Indirect_A_D"])
            + float(summary["posterior_P18_Mediated_A_D"])
            - 0.05 * float(summary["rank_true_graph"])
        )
        rec = {
            "candidate_seed": candidate_seed,
            "accepted_by_thresholds": accepted,
            "search_score": score,
            **summary,
        }
        records.append(rec)
        if score > best_score:
            best_score = score
            best_seed = candidate_seed
            best_df = raw
            best_summary = summary
        if accepted:
            return EvidenceSelection(
                evidence_bag=ensure_row_ids(raw, candidate_seed),
                selected_seed=candidate_seed,
                accepted_by_thresholds=True,
                search_records=pd.DataFrame.from_records(records),
                final_summary=summary,
                warning=None,
            )

    assert best_df is not None and best_summary is not None
    warning = "No 48-row evidence bag satisfied all sufficiency thresholds; selected the highest-scoring bag by the declared fallback rule."
    return EvidenceSelection(
        evidence_bag=ensure_row_ids(best_df, best_seed),
        selected_seed=best_seed,
        accepted_by_thresholds=False,
        search_records=pd.DataFrame.from_records(records),
        final_summary=best_summary,
        warning=warning,
    )


def build_order_conditions(
    evidence_bag: pd.DataFrame,
    graphs: Sequence[Graph],
    seed: int,
    alpha: float,
) -> pd.DataFrame:
    true_graph = find_graph_by_edges(graphs, TRUE_GRAPH_EDGES)
    direct_graph = find_graph_by_edges(graphs, DIRECT_COMPRESSION_EDGES)
    true_cpds = fit_cpds_laplace(evidence_bag, true_graph, alpha=alpha)
    direct_cpds = fit_cpds_laplace(evidence_bag, direct_graph, alpha=alpha)
    llr = log_likelihood_rows(evidence_bag, true_cpds) - log_likelihood_rows(evidence_bag, direct_cpds)

    base = evidence_bag.copy().reset_index(drop=True)
    base["llr_true_minus_compression"] = llr

    low_ids = base.sort_values(["llr_true_minus_compression", "row_id"], ascending=[True, True]).head(16)["row_id"].tolist()
    high_ids = base.sort_values(["llr_true_minus_compression", "row_id"], ascending=[False, True]).head(16)["row_id"].tolist()

    def ordered_rows(first_ids: List[int], condition: str) -> pd.DataFrame:
        first = base.set_index("row_id").loc[first_ids].reset_index()
        remaining = base[~base["row_id"].isin(first_ids)].copy()
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(remaining))
        remaining = remaining.iloc[perm].reset_index(drop=True)
        out = pd.concat([first, remaining], ignore_index=True)
        out.insert(0, "position", np.arange(1, len(out) + 1, dtype=int))
        out.insert(0, "condition", condition)
        return out

    compression = ordered_rows(low_ids, "compression_first")
    diagnostic = ordered_rows(high_ids, "diagnostic_first")

    rng = np.random.default_rng(seed)
    mixed = base.iloc[rng.permutation(len(base))].reset_index(drop=True)
    mixed.insert(0, "position", np.arange(1, len(mixed) + 1, dtype=int))
    mixed.insert(0, "condition", "mixed")

    order_df = pd.concat([compression, diagnostic, mixed], ignore_index=True)
    required_cols = [
        "condition",
        "position",
        "row_id",
        "A",
        "B",
        "C",
        "D",
        "E",
        "llr_true_minus_compression",
    ]
    # Preserve candidate_seed too if present; required columns are first.
    extra_cols = [c for c in order_df.columns if c not in required_cols]
    order_df = order_df[required_cols + extra_cols]

    row_sets = {cond: set(order_df.loc[order_df["condition"] == cond, "row_id"]) for cond in order_df["condition"].unique()}
    if len({frozenset(v) for v in row_sets.values()}) != 1:
        raise AssertionError("Order conditions do not contain the same row IDs.")
    return order_df
