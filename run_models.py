#!/usr/bin/env python3
"""Command-line entry point for the hidden machine-world SCM prototype."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from analysis_outputs import (
    build_key_target_values,
    build_order_effects,
    build_sufficiency_report,
    ensure_output_dirs,
    graph_posterior_record_base,
    make_prediction_rows,
    plot_final_prediction_heatmap,
    plot_graph_metric_trajectory,
    plot_mediation_contrast_trajectory,
    plot_target_p06_trajectory,
    runtime_checks,
    save_json,
)
from config import (
    CHECKPOINTS_DEFAULT,
    CONDITIONS,
    DIRECT_COMPRESSION_EDGES,
    FAMILY_BIASES_DEFAULT,
    FAMILY_WEIGHTS_DEFAULT,
    ROOT_FANOUT_EDGES,
    SPARSE_GRAPH_EDGES,
    TRUE_GRAPH_EDGES,
    VARS,
)
from evidence import build_order_conditions, search_evidence_bag
from graphs import Graph, enumerate_legal_graphs, find_graph_by_edges, rank_graph, special_graph_ids
from learners import (
    association_learner,
    beam_search_rmdl,
    feature_weighted_causal_learner,
    graph_metric_for_graph,
    graph_posterior_learner,
    mdl_map_learner,
)
from propositions import bank_records, build_proposition_bank, truth_matrix
from scoring import posterior_from_log_scores, sorted_graph_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run learner models on a hidden machine-world Boolean SCM task.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n_episodes", type=int, default=48)
    parser.add_argument("--bag_search_limit", type=int, default=5000)
    parser.add_argument("--beam_width", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--lambda_complexity", type=float, default=1.0)
    parser.add_argument("--association_decay", type=float, default=0.95)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=list(CHECKPOINTS_DEFAULT))
    parser.add_argument("--outdir", type=str, default="outputs")
    return parser.parse_args()


def _append_graph_records_all(
    rows: List[Dict[str, object]],
    condition: str,
    checkpoint: int,
    model: str,
    model_variant: str,
    graphs: Sequence[Graph],
    log_scores: np.ndarray,
    posterior: np.ndarray | None,
    special_ids: Dict[str, str],
    selected_idx: int | None = None,
) -> None:
    for i, graph in enumerate(graphs):
        rec = graph_posterior_record_base(
            condition,
            checkpoint,
            model,
            model_variant,
            graph,
            log_scores[i],
            None if posterior is None else posterior[i],
            special_ids,
        )
        rec["in_beam"] = False
        rec["is_selected_map"] = selected_idx is not None and i == selected_idx
        rows.append(rec)


def _append_graph_records_beam(
    rows: List[Dict[str, object]],
    condition: str,
    checkpoint: int,
    graphs: Sequence[Graph],
    live_indices: np.ndarray,
    live_scores: np.ndarray,
    posterior: np.ndarray,
    special_ids: Dict[str, str],
    beam_result: Dict[str, object],
) -> None:
    for local_i, gi in enumerate(live_indices):
        graph = graphs[int(gi)]
        rec = graph_posterior_record_base(
            condition,
            checkpoint,
            "beam_rmdl",
            "beam_model_average",
            graph,
            float(live_scores[local_i]),
            float(posterior[local_i]),
            special_ids,
        )
        rec["in_beam"] = True
        rec["is_selected_map"] = False
        rec["beam_entropy"] = beam_result["beam_entropy"]
        rec["true_graph_in_beam"] = beam_result["true_in_beam"]
        rec["true_graph_pruned_at"] = beam_result["true_pruned_at"]
        rec["direct_compression_graph_in_beam"] = beam_result["direct_compression_in_beam"]
        rec["direct_compression_graph_pruned_at"] = beam_result["direct_compression_pruned_at"]
        rows.append(rec)


def _top10_lines_all(
    graphs: Sequence[Graph], log_scores: np.ndarray, posterior: np.ndarray | None, top_n: int = 10) -> List[str]:
    order = sorted_graph_indices(log_scores, graphs)[:top_n]
    lines = ["| rank | graph_id | posterior | log_score | edges |", "|---:|---|---:|---:|---|"]
    for rank, gi in enumerate(order, start=1):
        post = "NA" if posterior is None else f"{posterior[gi]:.6f}"
        lines.append(f"| {rank} | {graphs[gi].graph_id} | {post} | {log_scores[gi]:.6f} | {graphs[gi].edge_string} |")
    return lines


def _top10_lines_beam(
    graphs: Sequence[Graph], live_indices: np.ndarray, live_scores: np.ndarray, posterior: np.ndarray, top_n: int = 10) -> List[str]:
    masks = np.asarray([graphs[int(i)].mask for i in live_indices], dtype=int)
    order = np.lexsort((masks, -live_scores))[:top_n]
    lines = ["| live_rank | graph_id | posterior_in_beam | log_score | edges |", "|---:|---|---:|---:|---|"]
    for rank, local_i in enumerate(order, start=1):
        gi = int(live_indices[local_i])
        lines.append(
            f"| {rank} | {graphs[gi].graph_id} | {posterior[local_i]:.6f} | {live_scores[local_i]:.6f} | {graphs[gi].edge_string} |"
        )
    return lines


def _append_graph_metrics(
    rows: List[Dict[str, object]],
    condition: str,
    checkpoint: int,
    model: str,
    model_variant: str,
    target_graph: str,
    metric_value: float,
    rank: int | None,
    posterior: float | None,
) -> None:
    rows.append(
        {
            "condition": condition,
            "checkpoint": int(checkpoint),
            "model": model,
            "model_variant": model_variant,
            "target_graph": target_graph,
            "metric_name": "posterior_or_inverse_rank",
            "metric_value": float(metric_value),
            "rank": np.nan if rank is None else int(rank),
            "posterior": np.nan if posterior is None else float(posterior),
        }
    )


def _graph_library_records(graphs: Sequence[Graph], special_ids: Dict[str, str]) -> List[Dict[str, object]]:
    records = []
    for g in graphs:
        records.append(
            {
                "graph_id": g.graph_id,
                "mask": g.mask,
                "edge_string": g.edge_string,
                "num_edges": g.num_edges,
                "adjacency_matrix": ";".join("".join(str(x) for x in row) for row in g.adjacency),
                "is_TRUE_GRAPH": g.graph_id == special_ids["TRUE_GRAPH"],
                "is_DIRECT_COMPRESSION_GRAPH": g.graph_id == special_ids["DIRECT_COMPRESSION_GRAPH"],
                "is_ROOT_FANOUT_GRAPH": g.graph_id == special_ids["ROOT_FANOUT_GRAPH"],
                "is_SPARSE_GRAPH": g.graph_id == special_ids["SPARSE_GRAPH"],
                "is_DENSE_LEGAL_GRAPH": g.graph_id == special_ids["DENSE_LEGAL_GRAPH"],
            }
        )
    return records


def main() -> None:
    args = parse_args()
    checkpoints = tuple(int(c) for c in args.checkpoints)
    if max(checkpoints) > args.n_episodes:
        raise ValueError("The largest checkpoint cannot exceed --n_episodes.")

    outdir = ensure_output_dirs(args.outdir)
    plots_dir = outdir / "plots"

    graphs = enumerate_legal_graphs()
    special_ids = special_graph_ids(graphs)
    true_graph_id = special_ids["TRUE_GRAPH"]
    direct_graph_id = special_ids["DIRECT_COMPRESSION_GRAPH"]
    bank = build_proposition_bank(graphs)
    tmat = truth_matrix(graphs, bank)

    pd.DataFrame(_graph_library_records(graphs, special_ids)).to_csv(outdir / "graph_library.csv", index=False)
    pd.DataFrame(bank_records(bank)).to_csv(outdir / "proposition_bank.csv", index=False)

    selection = search_evidence_bag(
        seed=args.seed,
        n_episodes=args.n_episodes,
        bag_search_limit=args.bag_search_limit,
        graphs=graphs,
        bank=bank,
        truth_mat=tmat,
        alpha=args.alpha,
    )
    selection.evidence_bag.to_csv(outdir / "evidence_bag.csv", index=False)
    selection.search_records.to_csv(outdir / "evidence_search_records.csv", index=False)

    order_sequences = build_order_conditions(selection.evidence_bag, graphs, seed=args.seed, alpha=args.alpha)
    order_sequences.to_csv(outdir / "order_sequences.csv", index=False)

    prediction_rows: List[Dict[str, object]] = []
    graph_rows: List[Dict[str, object]] = []
    graph_metric_rows: List[Dict[str, object]] = []
    feature_detail_rows: List[Dict[str, object]] = []
    beam_diag_rows: List[Dict[str, object]] = []
    top_md: List[str] = [
        "# Top graph diagnostics",
        "",
        "This file summarizes latent causal structure rankings for graph-based learners.",
        "",
    ]
    exact_final_summaries: Dict[str, Dict[str, float | int]] = {}

    for condition in CONDITIONS:
        cond_df = order_sequences[order_sequences["condition"] == condition].sort_values("position").reset_index(drop=True)

        # Beam-search rMDL is online and should be run once per order so that pruning carries forward.
        beam_results = beam_search_rmdl(
            cond_df,
            graphs,
            bank,
            tmat,
            checkpoints=checkpoints,
            alpha=args.alpha,
            lambda_edges=args.lambda_complexity,
            beam_width=args.beam_width,
            true_graph_id=true_graph_id,
            direct_compression_graph_id=direct_graph_id,
        )

        for checkpoint in checkpoints:
            prefix = cond_df.iloc[:checkpoint].copy()

            # 1. Exact SCM oracle.
            exact = graph_posterior_learner(
                prefix,
                graphs,
                bank,
                tmat,
                alpha=args.alpha,
                lambda_edges=0.0,
                true_graph_id=true_graph_id,
                direct_compression_graph_id=direct_graph_id,
            )
            prediction_rows.extend(
                make_prediction_rows(condition, checkpoint, "exact_oracle", "lambda_edges=0", exact.prop_probs, bank)
            )
            _append_graph_records_all(
                graph_rows,
                condition,
                checkpoint,
                "exact_oracle",
                "lambda_edges=0",
                graphs,
                exact.log_scores,
                exact.posterior,
                special_ids,
            )
            for graph_name, graph_id in [("TRUE_GRAPH", true_graph_id), ("DIRECT_COMPRESSION_GRAPH", direct_graph_id)]:
                gi = next(i for i, g in enumerate(graphs) if g.graph_id == graph_id)
                _append_graph_metrics(
                    graph_metric_rows,
                    condition,
                    checkpoint,
                    "exact_oracle",
                    "lambda_edges=0",
                    graph_name,
                    exact.posterior[gi],
                    rank_graph(exact.log_scores, graphs, graph_id),
                    exact.posterior[gi],
                )
            if checkpoint == max(checkpoints):
                exact_final_summaries[condition] = exact.summary
            top_md.extend(
                [
                    f"## {condition} | checkpoint {checkpoint} | exact_oracle",
                    "",
                    f"TRUE_GRAPH rank: {exact.summary['rank_true_graph']} | P(TRUE_GRAPH): {exact.summary['posterior_true_graph']:.6f}",
                    f"DIRECT_COMPRESSION_GRAPH rank: {exact.summary['rank_direct_compression_graph']} | P(DIRECT_COMPRESSION_GRAPH): {exact.summary['posterior_direct_compression_graph']:.6f}",
                    f"P06 Direct(A,D): {exact.summary['posterior_P06_Direct_A_D']:.6f} | P12 Indirect(A,D): {exact.summary['posterior_P12_Indirect_A_D']:.6f} | P18 Mediated(A,D; B,C): {exact.summary['posterior_P18_Mediated_A_D']:.6f}",
                    "",
                    *_top10_lines_all(graphs, exact.log_scores, exact.posterior),
                    "",
                ]
            )

            # 2. Complexity-biased Bayes.
            comp = graph_posterior_learner(
                prefix,
                graphs,
                bank,
                tmat,
                alpha=args.alpha,
                lambda_edges=args.lambda_complexity,
                true_graph_id=true_graph_id,
                direct_compression_graph_id=direct_graph_id,
            )
            prediction_rows.extend(
                make_prediction_rows(condition, checkpoint, "complexity_bayes", "lambda_edges=1", comp.prop_probs, bank)
            )
            _append_graph_records_all(
                graph_rows,
                condition,
                checkpoint,
                "complexity_bayes",
                "lambda_edges=1",
                graphs,
                comp.log_scores,
                comp.posterior,
                special_ids,
            )
            for graph_name, graph_id in [("TRUE_GRAPH", true_graph_id), ("DIRECT_COMPRESSION_GRAPH", direct_graph_id)]:
                gi = next(i for i, g in enumerate(graphs) if g.graph_id == graph_id)
                _append_graph_metrics(
                    graph_metric_rows,
                    condition,
                    checkpoint,
                    "complexity_bayes",
                    "lambda_edges=1",
                    graph_name,
                    comp.posterior[gi],
                    rank_graph(comp.log_scores, graphs, graph_id),
                    comp.posterior[gi],
                )
            top_md.extend(
                [
                    f"## {condition} | checkpoint {checkpoint} | complexity_bayes",
                    "",
                    f"TRUE_GRAPH rank: {comp.summary['rank_true_graph']} | P(TRUE_GRAPH): {comp.summary['posterior_true_graph']:.6f}",
                    f"DIRECT_COMPRESSION_GRAPH rank: {comp.summary['rank_direct_compression_graph']} | P(DIRECT_COMPRESSION_GRAPH): {comp.summary['posterior_direct_compression_graph']:.6f}",
                    f"P06 Direct(A,D): {comp.summary['posterior_P06_Direct_A_D']:.6f} | P12 Indirect(A,D): {comp.summary['posterior_P12_Indirect_A_D']:.6f} | P18 Mediated(A,D; B,C): {comp.summary['posterior_P18_Mediated_A_D']:.6f}",
                    "",
                    *_top10_lines_all(graphs, comp.log_scores, comp.posterior),
                    "",
                ]
            )

            # 6. Feature-weighted causal learner, based on complexity-biased Bayes posterior.
            weighted_probs, details = feature_weighted_causal_learner(
                comp.prop_probs,
                bank,
                family_weights=FAMILY_WEIGHTS_DEFAULT,
                family_biases=FAMILY_BIASES_DEFAULT,
            )
            prediction_rows.extend(
                make_prediction_rows(
                    condition,
                    checkpoint,
                    "feature_weighted_causal",
                    "weighted_salience",
                    weighted_probs,
                    bank,
                )
            )
            for detail in details:
                feature_detail_rows.append(
                    {
                        "condition": condition,
                        "checkpoint": checkpoint,
                        "model": "feature_weighted_causal",
                        "model_variant": "weighted_salience",
                        **detail,
                    }
                )

            # 3. MDL / MAP learner.
            mdl = mdl_map_learner(
                prefix,
                graphs,
                tmat,
                alpha=args.alpha,
                lambda_edges=args.lambda_complexity,
            )
            prediction_rows.extend(
                make_prediction_rows(condition, checkpoint, "mdl_map", "smoothed_MAP_lambda_edges=1", mdl.prop_probs, bank)
            )
            _append_graph_records_all(
                graph_rows,
                condition,
                checkpoint,
                "mdl_map",
                "smoothed_MAP_lambda_edges=1",
                graphs,
                mdl.log_scores,
                None,
                special_ids,
                selected_idx=mdl.selected_idx,
            )
            for graph_name, graph_id in [("TRUE_GRAPH", true_graph_id), ("DIRECT_COMPRESSION_GRAPH", direct_graph_id)]:
                r = rank_graph(mdl.log_scores, graphs, graph_id)
                _append_graph_metrics(
                    graph_metric_rows,
                    condition,
                    checkpoint,
                    "mdl_map",
                    "smoothed_MAP_lambda_edges=1",
                    graph_name,
                    1.0 / r,
                    r,
                    None,
                )
            prop_idx = {item.proposition_id: i for i, item in enumerate(bank)}
            true_rank_mdl = rank_graph(mdl.log_scores, graphs, true_graph_id)
            direct_rank_mdl = rank_graph(mdl.log_scores, graphs, direct_graph_id)
            top_md.extend(
                [
                    f"## {condition} | checkpoint {checkpoint} | mdl_map",
                    "",
                    f"Selected MAP graph: {mdl.selected_graph_id} | score: {mdl.selected_score:.6f} | edges: {graphs[mdl.selected_idx].edge_string}",
                    f"TRUE_GRAPH rank: {true_rank_mdl} | DIRECT_COMPRESSION_GRAPH rank: {direct_rank_mdl}",
                    f"Smoothed P06 Direct(A,D): {mdl.prop_probs[prop_idx['P06']]:.6f} | P12 Indirect(A,D): {mdl.prop_probs[prop_idx['P12']]:.6f} | P18 Mediated(A,D; B,C): {mdl.prop_probs[prop_idx['P18']]:.6f}",
                    "",
                    *_top10_lines_all(graphs, mdl.log_scores, None),
                    "",
                ]
            )

            # 4. Beam-search rMDL outputs for this checkpoint.
            beam = beam_results[checkpoint]
            live_indices = beam["live_indices"]
            live_scores = beam["live_scores"]
            beam_posterior = beam["posterior"]
            prediction_rows.extend(
                make_prediction_rows(
                    condition,
                    checkpoint,
                    "beam_rmdl",
                    "beam_model_average",
                    beam["prop_probs"],
                    bank,
                )
            )
            prediction_rows.extend(
                make_prediction_rows(
                    condition,
                    checkpoint,
                    "beam_rmdl",
                    "beam_MAP",
                    beam["beam_map_probs"],
                    bank,
                )
            )
            _append_graph_records_beam(
                graph_rows,
                condition,
                checkpoint,
                graphs,
                live_indices,
                live_scores,
                beam_posterior,
                special_ids,
                beam,
            )
            beam_diag_rows.append(
                {
                    "condition": condition,
                    "checkpoint": checkpoint,
                    "model": "beam_rmdl",
                    "beam_width": args.beam_width,
                    "top_graph_id": beam["top_graph_id"],
                    "top_score": beam["top_score"],
                    "beam_entropy": beam["beam_entropy"],
                    "true_graph_in_beam": beam["true_in_beam"],
                    "true_graph_pruned_at": beam["true_pruned_at"],
                    "direct_compression_graph_in_beam": beam["direct_compression_in_beam"],
                    "direct_compression_graph_pruned_at": beam["direct_compression_pruned_at"],
                }
            )
            live_map = {int(gi): i for i, gi in enumerate(live_indices)}
            idx_by_id = {g.graph_id: i for i, g in enumerate(graphs)}
            for graph_name, graph_id in [("TRUE_GRAPH", true_graph_id), ("DIRECT_COMPRESSION_GRAPH", direct_graph_id)]:
                gi = idx_by_id[graph_id]
                if gi in live_map:
                    local_i = live_map[gi]
                    post = float(beam_posterior[local_i])
                    # rank within the live beam
                    masks = np.asarray([graphs[int(i)].mask for i in live_indices], dtype=int)
                    live_order = np.lexsort((masks, -live_scores))
                    live_rank = int(np.where(live_order == local_i)[0][0] + 1)
                    metric_value = post
                else:
                    post = None
                    live_rank = None
                    metric_value = 0.0
                _append_graph_metrics(
                    graph_metric_rows,
                    condition,
                    checkpoint,
                    "beam_rmdl",
                    "beam_model_average",
                    graph_name,
                    metric_value,
                    live_rank,
                    post,
                )
            prop_idx = {item.proposition_id: i for i, item in enumerate(bank)}
            true_status = "in beam" if beam["true_in_beam"] else f"pruned at t={beam['true_pruned_at']}"
            direct_status = "in beam" if beam["direct_compression_in_beam"] else f"pruned at t={beam['direct_compression_pruned_at']}"
            top_md.extend(
                [
                    f"## {condition} | checkpoint {checkpoint} | beam_rmdl",
                    "",
                    f"TRUE_GRAPH status: {true_status} | DIRECT_COMPRESSION_GRAPH status: {direct_status}",
                    f"Top graph: {beam['top_graph_id']} | beam entropy: {beam['beam_entropy']:.6f}",
                    f"Beam-average P06 Direct(A,D): {beam['prop_probs'][prop_idx['P06']]:.6f} | P12 Indirect(A,D): {beam['prop_probs'][prop_idx['P12']]:.6f} | P18 Mediated(A,D; B,C): {beam['prop_probs'][prop_idx['P18']]:.6f}",
                    "",
                    *_top10_lines_beam(graphs, live_indices, live_scores, beam_posterior),
                    "",
                ]
            )

            # 5. Association-only learner.
            for mode, variant in [
                ("association_full_memory", "full_memory"),
                ("association_recency_weighted", "recency_weighted"),
            ]:
                assoc_probs = association_learner(
                    prefix,
                    bank,
                    mode=mode,
                    alpha=args.alpha,
                    decay=args.association_decay,
                )
                prediction_rows.extend(
                    make_prediction_rows(condition, checkpoint, "association_only", variant, assoc_probs, bank)
                )

    predictions_long = pd.DataFrame.from_records(prediction_rows)
    graph_posteriors_long = pd.DataFrame.from_records(graph_rows)
    feature_details = pd.DataFrame.from_records(feature_detail_rows)
    beam_diagnostics = pd.DataFrame.from_records(beam_diag_rows)
    graph_metrics = pd.DataFrame.from_records(graph_metric_rows)

    predictions_long.to_csv(outdir / "predictions_long.csv", index=False)
    key_targets = build_key_target_values(predictions_long)
    key_targets.to_csv(outdir / "key_target_values.csv", index=False)
    order_effects = build_order_effects(predictions_long)
    order_effects.to_csv(outdir / "order_effects.csv", index=False)
    graph_posteriors_long.to_csv(outdir / "graph_posteriors_long.csv", index=False)
    feature_details.to_csv(outdir / "feature_weighted_details.csv", index=False)
    beam_diagnostics.to_csv(outdir / "beam_diagnostics.csv", index=False)
    graph_metrics.to_csv(outdir / "graph_target_metrics.csv", index=False)

    sufficiency_report = build_sufficiency_report(
        n_episodes=args.n_episodes,
        selected_seed=selection.selected_seed,
        bag_search_limit=args.bag_search_limit,
        accepted_by_thresholds=selection.accepted_by_thresholds,
        final_summary=selection.final_summary,
        warning=selection.warning,
    )
    save_json(outdir / "sufficiency_report.json", sufficiency_report)

    (outdir / "top_graphs.md").write_text("\n".join(top_md), encoding="utf-8")

    plot_target_p06_trajectory(key_targets, plots_dir / "target_P06_DirectAD_trajectory.png")
    plot_mediation_contrast_trajectory(key_targets, plots_dir / "mediation_contrast_trajectory.png")
    plot_graph_metric_trajectory(graph_metrics, "TRUE_GRAPH", plots_dir / "true_graph_posterior_trajectory.png")
    plot_graph_metric_trajectory(graph_metrics, "DIRECT_COMPRESSION_GRAPH", plots_dir / "direct_compression_graph_trajectory.png")
    plot_final_prediction_heatmap(predictions_long, plots_dir / "final_prediction_heatmap.png")

    runtime_checks(predictions_long, order_sequences)

    # Concise console summary.
    exact_final = exact_final_summaries.get("mixed") or next(iter(exact_final_summaries.values()))
    print("\n=== Hidden machine-world SCM run summary ===")
    print(f"Selected evidence seed: {selection.selected_seed}")
    print(f"Sufficiency thresholds met: {selection.accepted_by_thresholds}")
    if selection.warning:
        print(f"Sufficiency warning: {selection.warning}")
    print("\nExact oracle final:")
    print(f"  TRUE_GRAPH rank: {exact_final['rank_true_graph']}")
    print(f"  P(TRUE_GRAPH): {exact_final['posterior_true_graph']:.6f}")
    print(f"  P(P06 Direct(A,D)): {exact_final['posterior_P06_Direct_A_D']:.6f}")
    print(f"  P(P12 Indirect(A,D)): {exact_final['posterior_P12_Indirect_A_D']:.6f}")
    print(f"  P(P18 Mediated(A,D)): {exact_final['posterior_P18_Mediated_A_D']:.6f}")

    print("\nFinal P06 Direct(A,D) by model and order:")
    p06_final = order_effects[(order_effects["checkpoint"] == max(checkpoints)) & (order_effects["proposition_id"] == "P06")]
    for _, row in p06_final.sort_values(["model", "model_variant"]).iterrows():
        print(
            f"  {row['model']} / {row['model_variant']}: "
            f"compression={row['p_compression_first']:.4f}, "
            f"diagnostic={row['p_diagnostic_first']:.4f}, "
            f"mixed={row['p_mixed']:.4f}, "
            f"compression-diagnostic={row['delta_compression_minus_diagnostic']:.4f}"
        )

    print("\nBeam-rMDL final survival and P06:")
    beam_final = beam_diagnostics[beam_diagnostics["checkpoint"] == max(checkpoints)]
    beam_p06 = p06_final[(p06_final["model"] == "beam_rmdl") & (p06_final["model_variant"] == "beam_model_average")]
    beam_p06_map = beam_p06.set_index("model") if not beam_p06.empty else pd.DataFrame()
    p06_values = predictions_long[
        (predictions_long["checkpoint"] == max(checkpoints))
        & (predictions_long["model"] == "beam_rmdl")
        & (predictions_long["model_variant"] == "beam_model_average")
        & (predictions_long["proposition_id"] == "P06")
    ].set_index("condition")["p"].to_dict()
    for _, row in beam_final.sort_values("condition").iterrows():
        print(
            f"  {row['condition']}: TRUE_GRAPH survived={row['true_graph_in_beam']}, "
            f"DIRECT_COMPRESSION_GRAPH survived={row['direct_compression_graph_in_beam']}, "
            f"final P06={p06_values.get(row['condition'], float('nan')):.4f}"
        )

    print("\nAssociation recency-weighted final P06:")
    assoc_p06 = predictions_long[
        (predictions_long["checkpoint"] == max(checkpoints))
        & (predictions_long["model"] == "association_only")
        & (predictions_long["model_variant"] == "recency_weighted")
        & (predictions_long["proposition_id"] == "P06")
    ].sort_values("condition")
    for _, row in assoc_p06.iterrows():
        print(f"  {row['condition']}: {row['p']:.4f}")

    print(f"\nFiles written to {outdir}/")


if __name__ == "__main__":
    main()
