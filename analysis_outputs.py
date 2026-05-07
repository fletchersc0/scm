"""Output-table builders, markdown summaries, plots, and runtime checks."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CONDITIONS, VARS
from graphs import Graph, graph_index, rank_graph
from propositions import PropositionItem


def ensure_output_dirs(outdir: str | Path) -> Path:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    return out


def model_label(model: str, variant: str) -> str:
    return f"{model}\n{variant}" if variant else model


def make_prediction_rows(
    condition: str,
    checkpoint: int,
    model: str,
    model_variant: str,
    probs: Sequence[float],
    bank: Sequence[PropositionItem],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for p, item in zip(probs, bank):
        pf = float(p)
        truth = int(item.truth_in_true_graph)
        rows.append(
            {
                "condition": condition,
                "checkpoint": int(checkpoint),
                "model": model,
                "model_variant": model_variant,
                "proposition_id": item.proposition_id,
                "proposition_text": item.text,
                "proposition_family": item.family,
                "truth": truth,
                "p": pf,
                "binary_prediction": bool(pf >= 0.5),
                "absolute_error": abs(pf - truth),
                "signed_error": pf - truth,
            }
        )
    return rows


def build_key_target_values(predictions_long: pd.DataFrame) -> pd.DataFrame:
    target_map = {
        "P06": "p_P06_Direct_A_D",
        "P12": "p_P12_Indirect_A_D",
        "P18": "p_P18_Mediated_A_D",
        "P19": "p_P19_A_affects_D_not_direct",
    }
    sub = predictions_long[predictions_long["proposition_id"].isin(target_map)].copy()
    wide = sub.pivot_table(
        index=["condition", "checkpoint", "model", "model_variant"],
        columns="proposition_id",
        values="p",
        aggfunc="first",
    ).reset_index()
    wide = wide.rename(columns=target_map)
    for col in target_map.values():
        if col not in wide.columns:
            wide[col] = np.nan
    return wide[["condition", "checkpoint", "model", "model_variant", *target_map.values()]]


def build_order_effects(predictions_long: pd.DataFrame) -> pd.DataFrame:
    piv = predictions_long.pivot_table(
        index=["checkpoint", "model", "model_variant", "proposition_id", "proposition_text"],
        columns="condition",
        values="p",
        aggfunc="first",
    ).reset_index()
    for cond in CONDITIONS:
        if cond not in piv.columns:
            piv[cond] = np.nan
    piv = piv.rename(
        columns={
            "compression_first": "p_compression_first",
            "diagnostic_first": "p_diagnostic_first",
            "mixed": "p_mixed",
        }
    )
    piv["delta_compression_minus_diagnostic"] = piv["p_compression_first"] - piv["p_diagnostic_first"]
    piv["delta_compression_minus_mixed"] = piv["p_compression_first"] - piv["p_mixed"]
    piv["delta_diagnostic_minus_mixed"] = piv["p_diagnostic_first"] - piv["p_mixed"]
    cols = [
        "checkpoint",
        "model",
        "model_variant",
        "proposition_id",
        "proposition_text",
        "p_compression_first",
        "p_diagnostic_first",
        "p_mixed",
        "delta_compression_minus_diagnostic",
        "delta_compression_minus_mixed",
        "delta_diagnostic_minus_mixed",
    ]
    return piv[cols].sort_values(["checkpoint", "model", "model_variant", "proposition_id"]).reset_index(drop=True)


def graph_posterior_record_base(
    condition: str,
    checkpoint: int,
    model: str,
    model_variant: str,
    graph: Graph,
    log_score: float,
    posterior: float | None,
    special_ids: Dict[str, str],
) -> Dict[str, object]:
    return {
        "condition": condition,
        "checkpoint": int(checkpoint),
        "model": model,
        "model_variant": model_variant,
        "graph_id": graph.graph_id,
        "edge_string": graph.edge_string,
        "num_edges": graph.num_edges,
        "log_score": float(log_score),
        "posterior": np.nan if posterior is None else float(posterior),
        "is_true_graph": graph.graph_id == special_ids["TRUE_GRAPH"],
        "is_direct_compression_graph": graph.graph_id == special_ids["DIRECT_COMPRESSION_GRAPH"],
        "is_root_fanout_graph": graph.graph_id == special_ids["ROOT_FANOUT_GRAPH"],
        "is_sparse_graph": graph.graph_id == special_ids["SPARSE_GRAPH"],
        "is_dense_legal_graph": graph.graph_id == special_ids["DENSE_LEGAL_GRAPH"],
    }


def build_sufficiency_report(
    n_episodes: int,
    selected_seed: int,
    bag_search_limit: int,
    accepted_by_thresholds: bool,
    final_summary: Dict[str, float | int],
    warning: str | None,
) -> Dict[str, object]:
    p06 = float(final_summary["posterior_P06_Direct_A_D"])
    p12 = float(final_summary["posterior_P12_Indirect_A_D"])
    p18 = float(final_summary["posterior_P18_Mediated_A_D"])
    partially_sufficient = p06 <= 0.40 and p12 >= 0.50 and p18 >= 0.40
    if partially_sufficient:
        interpretation = "The evidence bag is at least partially sufficient in principle for the target distinction."
    else:
        interpretation = (
            "The evidence bag is not sufficient for the target distinction; model errors on this bag "
            "should be interpreted as possible evidence insufficiency."
        )
    report = {
        "n_episodes": int(n_episodes),
        "selected_seed": int(selected_seed),
        "bag_search_limit": int(bag_search_limit),
        "accepted_by_thresholds": bool(accepted_by_thresholds),
        "rank_true_graph_exact_final": int(final_summary["rank_true_graph"]),
        "posterior_true_graph_exact_final": float(final_summary["posterior_true_graph"]),
        "posterior_direct_A_D_exact_final": p06,
        "posterior_indirect_A_D_exact_final": p12,
        "posterior_mediated_A_D_exact_final": p18,
        "interpretation": interpretation,
    }
    if warning:
        report["warning"] = warning
    return report


def save_json(path: str | Path, obj: object) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _grid_shape(n: int) -> tuple[int, int]:
    cols = min(3, n)
    rows = int(math.ceil(n / cols))
    return rows, cols


def plot_target_p06_trajectory(key_targets: pd.DataFrame, outpath: str | Path) -> None:
    labels = key_targets[["model", "model_variant"]].drop_duplicates().apply(lambda r: model_label(r["model"], r["model_variant"]), axis=1).tolist()
    key_targets = key_targets.copy()
    key_targets["label"] = key_targets.apply(lambda r: model_label(r["model"], r["model_variant"]), axis=1)
    rows, cols = _grid_shape(len(labels))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.8 * rows), squeeze=False)
    for ax, label in zip(axes.flat, labels):
        sub = key_targets[key_targets["label"] == label]
        for cond in sorted(sub["condition"].unique()):
            csub = sub[sub["condition"] == cond].sort_values("checkpoint")
            ax.plot(csub["checkpoint"], csub["p_P06_Direct_A_D"], marker="o", label=cond)
        ax.axhline(0.5, linestyle="--", linewidth=1)
        ax.set_title(label)
        ax.set_xlabel("checkpoint")
        ax.set_ylabel("p(P06 Direct(A,D))")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    for ax in axes.flat[len(labels):]:
        ax.axis("off")
    fig.suptitle("False direct-cause compression target over checkpoints", y=1.01)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_mediation_contrast_trajectory(key_targets: pd.DataFrame, outpath: str | Path) -> None:
    cols_to_plot = {
        "p_P06_Direct_A_D": "P06 Direct(A,D)",
        "p_P12_Indirect_A_D": "P12 Indirect(A,D)",
        "p_P18_Mediated_A_D": "P18 Mediated(A,D; B,C)",
    }
    labels = key_targets[["model", "model_variant"]].drop_duplicates().apply(lambda r: model_label(r["model"], r["model_variant"]), axis=1).tolist()
    key_targets = key_targets.copy()
    key_targets["label"] = key_targets.apply(lambda r: model_label(r["model"], r["model_variant"]), axis=1)
    rows, cols = _grid_shape(len(labels))
    fig, axes = plt.subplots(rows, cols, figsize=(6.8 * cols, 4.4 * rows), squeeze=False)
    for ax, label in zip(axes.flat, labels):
        sub = key_targets[key_targets["label"] == label]
        for cond in sorted(sub["condition"].unique()):
            csub = sub[sub["condition"] == cond].sort_values("checkpoint")
            for col, name in cols_to_plot.items():
                ax.plot(csub["checkpoint"], csub[col], marker="o", linewidth=1.3, label=f"{cond} | {name}")
        ax.axhline(0.5, linestyle="--", linewidth=1)
        ax.set_title(label)
        ax.set_xlabel("checkpoint")
        ax.set_ylabel("probability")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=6, ncol=1)
    for ax in axes.flat[len(labels):]:
        ax.axis("off")
    fig.suptitle("Direct-vs-indirect mediation contrast", y=1.01)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_graph_metric_trajectory(graph_metrics: pd.DataFrame, graph_name: str, outpath: str | Path) -> None:
    sub = graph_metrics[graph_metrics["target_graph"] == graph_name].copy()
    if sub.empty:
        return
    sub["label"] = sub.apply(lambda r: model_label(r["model"], r["model_variant"]), axis=1)
    labels = sub["label"].drop_duplicates().tolist()
    rows, cols = _grid_shape(len(labels))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.8 * rows), squeeze=False)
    for ax, label in zip(axes.flat, labels):
        lsub = sub[sub["label"] == label]
        for cond in sorted(lsub["condition"].unique()):
            csub = lsub[lsub["condition"] == cond].sort_values("checkpoint")
            ax.plot(csub["checkpoint"], csub["metric_value"], marker="o", label=cond)
        ax.set_title(label)
        ax.set_xlabel("checkpoint")
        ax.set_ylabel(sub["metric_name"].iloc[0])
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    for ax in axes.flat[len(labels):]:
        ax.axis("off")
    fig.suptitle(f"{graph_name} trajectory", y=1.01)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_final_prediction_heatmap(predictions_long: pd.DataFrame, outpath: str | Path) -> None:
    final = predictions_long[predictions_long["checkpoint"] == predictions_long["checkpoint"].max()].copy()
    final["label"] = final.apply(lambda r: model_label(r["model"], r["model_variant"]), axis=1)
    labels = final["label"].drop_duplicates().tolist()
    prop_order = final[["proposition_id", "proposition_text"]].drop_duplicates().sort_values("proposition_id")
    prop_ids = prop_order["proposition_id"].tolist()
    ylabels = [f"{pid}: {txt}" for pid, txt in zip(prop_order["proposition_id"], prop_order["proposition_text"])]

    conditions = [c for c in CONDITIONS if c in final["condition"].unique()]
    fig, axes = plt.subplots(1, len(conditions), figsize=(max(8, 2.2 * len(labels)) * len(conditions), 12), squeeze=False)
    for ax, cond in zip(axes.flat, conditions):
        csub = final[final["condition"] == cond]
        mat = np.full((len(prop_ids), len(labels)), np.nan)
        for i, pid in enumerate(prop_ids):
            for j, lab in enumerate(labels):
                vals = csub[(csub["proposition_id"] == pid) & (csub["label"] == lab)]["p"].values
                if len(vals):
                    mat[i, j] = vals[0]
        im = ax.imshow(mat, vmin=0, vmax=1, aspect="auto")
        ax.set_title(cond)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels if ax is axes.flat[0] else [""] * len(ylabels), fontsize=7)
        for i in range(len(prop_ids)):
            for j in range(len(labels)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=5)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.02, label="p(phi)")
    fig.suptitle("Final checkpoint Boolean proposition predictions", y=0.99)
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def runtime_checks(
    predictions_long: pd.DataFrame,
    order_sequences: pd.DataFrame,
    tolerance: float = 1e-10,
) -> None:
    # Order conditions contain the same row IDs.
    row_sets = {
        cond: frozenset(order_sequences.loc[order_sequences["condition"] == cond, "row_id"].tolist())
        for cond in order_sequences["condition"].unique()
    }
    if len(set(row_sets.values())) != 1:
        raise AssertionError("Order-condition row ID multisets are not identical.")

    if not predictions_long["p"].between(0, 1).all():
        bad = predictions_long.loc[~predictions_long["p"].between(0, 1)].head()
        raise AssertionError(f"Model prediction outside [0,1]: {bad}")

    final_cp = int(predictions_long["checkpoint"].max())
    for model, variant in [("exact_oracle", "lambda_edges=0"), ("complexity_bayes", "lambda_edges=1")]:
        sub = predictions_long[
            (predictions_long["checkpoint"] == final_cp)
            & (predictions_long["model"] == model)
            & (predictions_long["model_variant"] == variant)
        ]
        piv = sub.pivot_table(index="proposition_id", columns="condition", values="p", aggfunc="first")
        if not np.allclose(piv["compression_first"], piv["diagnostic_first"], atol=tolerance, rtol=0):
            raise AssertionError(f"{model} final predictions differ between compression_first and diagnostic_first.")
        if not np.allclose(piv["compression_first"], piv["mixed"], atol=tolerance, rtol=0):
            raise AssertionError(f"{model} final predictions differ between compression_first and mixed.")
