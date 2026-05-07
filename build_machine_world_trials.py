#!/usr/bin/env python3
"""Build the browser corpus for the Hidden Machine experiment.

Place this file next to the modelling package files (config.py, evidence.py,
graphs.py, propositions.py, scm.py, scoring.py, etc.) and run:

    python build_machine_world_trials.py --out machine_world_trials.json

The generated JSON is the static file consumed by index.html. The browser never
runs the modelling code; this script is an offline export bridge.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd

from config import CHECKPOINTS_DEFAULT, CONDITIONS, VARS
from evidence import build_order_conditions, search_evidence_bag
from graphs import enumerate_legal_graphs
from propositions import PropositionItem, build_proposition_bank, formal_string, truth_matrix

TASK_VERSION = "HM-SCM-v1"
CHECKPOINT_PROPOSITION_IDS = ["P01", "P05", "P06", "P12", "P18", "P19"]

VARIABLE_SPECS = [
    {"id": "A", "label": "A", "shape": "circle", "stage": 0},
    {"id": "B", "label": "B", "shape": "square", "stage": 1},
    {"id": "C", "label": "C", "shape": "triangle", "stage": 1},
    {"id": "D", "label": "D", "shape": "diamond", "stage": 2},
    {"id": "E", "label": "E", "shape": "hexagon", "stage": 3},
]

# Participant-facing wording used by the browser task.
DISPLAY_TEXT: Dict[str, str] = {
    "P01": "A directly makes B more likely to light up.",
    "P02": "A directly makes C more likely to light up.",
    "P03": "B directly makes D more likely to light up.",
    "P04": "C directly makes D more likely to light up.",
    "P05": "D directly makes E more likely to light up.",
    "P06": "A directly makes D more likely to light up.",
    "P07": "A directly makes E more likely to light up.",
    "P08": "B directly makes C more likely to light up.",
    "P09": "C directly makes B more likely to light up.",
    "P10": "D directly makes B more likely to light up.",
    "P11": "E directly makes D more likely to light up.",
    "P12": "A can make D more likely to light up through other symbols.",
    "P13": "A can make E more likely to light up through a chain of symbols.",
    "P14": "B can make E more likely to light up through D.",
    "P15": "C can make E more likely to light up through D.",
    "P16": "A can directly make both B and C more likely to light up.",
    "P17": "B and C can each directly make D more likely to light up.",
    "P18": "A’s influence on D goes through B or C.",
    "P19": "A can make D more likely to light up, but not directly.",
    "P20": "D can directly make E more likely to light up, and E does not directly make D more likely to light up.",
    "P21": "Either A directly makes D more likely to light up, or B directly makes C more likely to light up.",
    "P22": "A directly makes B more likely to light up, and D directly makes E more likely to light up.",
    "P23": "B and C have no connection in the machine.",
    "P24": "E is upstream of A in the machine.",
}

# Alternative formal strings matching the browser-spec wording for compound items.
FORMAL_OVERRIDE: Dict[str, str] = {
    "P16": "CommonCause(A;B,C)",
    "P17": "CommonEffect(D;B,C)",
    "P18": "Mediated(A,D;{B,C})",
    "P19": "Path(A,D) AND NOT Direct(A,D)",
    "P20": "Direct(D,E) AND NOT Direct(E,D)",
    "P21": "Direct(A,D) OR Direct(B,C)",
    "P22": "Direct(A,B) AND Direct(D,E)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate machine_world_trials.json for the Hidden Machine browser task.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n_episodes", type=int, default=48)
    parser.add_argument("--bag_search_limit", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--out", type=str, default="machine_world_trials.json")
    return parser.parse_args()


def variables_in_text(text: str) -> List[str]:
    seen: List[str] = []
    for match in re.findall(r"\b[A-E]\b", text):
        if match not in seen:
            seen.append(match)
    return seen


def build_proposition_records(bank: Sequence[PropositionItem]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for item in bank:
        text = DISPLAY_TEXT.get(item.proposition_id, item.text)
        formal = FORMAL_OVERRIDE.get(item.proposition_id, formal_string(item.formal))
        records.append(
            {
                "proposition_id": item.proposition_id,
                "family": item.family,
                "formal": formal,
                "truth_value": int(item.truth_in_true_graph),
                "primary_target": bool(item.primary_target),
                "statement_template": text,
                "display_text": text,
                "variables_in_statement": variables_in_text(text + " " + formal),
            }
        )
    return records


def row_type_map_from_orders(order_df: pd.DataFrame) -> Dict[Any, str]:
    """Label the rows used to create the order manipulation.

    build_order_conditions places the 16 rows most supportive of the direct
    compression graph first in compression_first and the 16 rows most supportive
    of the true graph first in diagnostic_first. The browser does not use these
    labels operationally, but logging them is useful for later checks.
    """
    compression_ids = set(
        order_df.loc[
            (order_df["condition"] == "compression_first") & (order_df["position"] <= 16), "row_id"
        ].tolist()
    )
    diagnostic_ids = set(
        order_df.loc[
            (order_df["condition"] == "diagnostic_first") & (order_df["position"] <= 16), "row_id"
        ].tolist()
    )
    out: Dict[Any, str] = {}
    for row_id in sorted(set(order_df["row_id"].tolist())):
        if row_id in compression_ids:
            out[row_id] = "compression_supporting"
        elif row_id in diagnostic_ids:
            out[row_id] = "diagnostic_supporting"
        else:
            out[row_id] = "neutral"
    return out


def build_condition_rows(order_df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    row_types = row_type_map_from_orders(order_df)
    condition_rows: Dict[str, List[Dict[str, Any]]] = {}
    for condition in CONDITIONS:
        sub = order_df.loc[order_df["condition"] == condition].sort_values("position")
        rows: List[Dict[str, Any]] = []
        for _, r in sub.iterrows():
            row_id = int(r["row_id"]) if str(r["row_id"]).isdigit() else r["row_id"]
            row: Dict[str, Any] = {
                "position": int(r["position"]),
                "row_id": row_id,
                "A": int(r["A"]),
                "B": int(r["B"]),
                "C": int(r["C"]),
                "D": int(r["D"]),
                "E": int(r["E"]),
                "row_type": row_types.get(r["row_id"], "neutral"),
            }
            if "llr_true_minus_compression" in r.index:
                row["llr_true_minus_compression"] = round(float(r["llr_true_minus_compression"]), 6)
            rows.append(row)
        condition_rows[condition] = rows
    return condition_rows


def graph_visible_control() -> List[Dict[str, Any]]:
    edges = [["X", "Y"], ["Y", "Z"]]
    return [
        {
            "control_id": "GVC01",
            "display_graph_edges": edges,
            "statement_text": "X directly makes Y more likely to light up.",
            "truth_value": 1,
            "family": "visible_direct",
            "formal": "Direct(X,Y)",
        },
        {
            "control_id": "GVC02",
            "display_graph_edges": edges,
            "statement_text": "Y directly makes Z more likely to light up.",
            "truth_value": 1,
            "family": "visible_direct",
            "formal": "Direct(Y,Z)",
        },
        {
            "control_id": "GVC03",
            "display_graph_edges": edges,
            "statement_text": "X directly makes Z more likely to light up.",
            "truth_value": 0,
            "family": "visible_false_direct",
            "formal": "Direct(X,Z)",
        },
        {
            "control_id": "GVC04",
            "display_graph_edges": edges,
            "statement_text": "X can make Z more likely to light up through another symbol.",
            "truth_value": 1,
            "family": "visible_indirect",
            "formal": "Indirect(X,Z)",
        },
        {
            "control_id": "GVC05",
            "display_graph_edges": edges,
            "statement_text": "Z directly makes X more likely to light up.",
            "truth_value": 0,
            "family": "visible_reverse",
            "formal": "Direct(Z,X)",
        },
        {
            "control_id": "GVC06",
            "display_graph_edges": edges,
            "statement_text": "Y is between X and Z in the visible machine.",
            "truth_value": 1,
            "family": "visible_mediation",
            "formal": "Mediated(X,Z;{Y})",
        },
        {
            "control_id": "GVC07",
            "display_graph_edges": edges,
            "statement_text": "X and Z have no connection in the visible machine.",
            "truth_value": 0,
            "family": "visible_false_no_connection",
            "formal": "NoConnection(X,Z)",
        },
        {
            "control_id": "GVC08",
            "display_graph_edges": edges,
            "statement_text": "Y directly makes X more likely to light up.",
            "truth_value": 0,
            "family": "visible_reverse",
            "formal": "Direct(Y,X)",
        },
    ]


def practice_block() -> Dict[str, Any]:
    return {
        "variables": [
            {"id": "X", "label": "X", "shape": "circle", "stage": 0},
            {"id": "Y", "label": "Y", "shape": "square", "stage": 1},
            {"id": "Z", "label": "Z", "shape": "diamond", "stage": 2},
        ],
        "observation_rows": [
            {"row_id": "practice_obs_1", "X": 1, "Y": 1, "Z": 1},
            {"row_id": "practice_obs_2", "X": 0, "Y": 0, "Z": 0},
            {"row_id": "practice_obs_3", "X": 1, "Y": 1, "Z": 0},
            {"row_id": "practice_obs_4", "X": 1, "Y": 0, "Z": 0},
        ],
        "propositions": [
            {
                "practice_id": "practice_prop_1",
                "statement_text": "X directly makes Y more likely to light up.",
                "truth_value": 1,
                "feedback_true": "Correct. In this practice machine, X can directly affect Y.",
                "feedback_false": "Not quite. In this practice machine, X can directly affect Y.",
            },
            {
                "practice_id": "practice_prop_2",
                "statement_text": "Z directly makes X more likely to light up.",
                "truth_value": 0,
                "feedback_true": "Not quite. In this practice machine, later symbols do not affect earlier symbols.",
                "feedback_false": "Correct. In this practice machine, Z does not directly affect X.",
            },
        ],
    }


def validate_browser_corpus(corpus: Dict[str, Any]) -> None:
    required_vars = {"A": 0, "B": 1, "C": 1, "D": 2, "E": 3}
    variable_map = {v.get("id"): v for v in corpus.get("variables", [])}
    for var, stage in required_vars.items():
        if var not in variable_map or int(variable_map[var].get("stage")) != stage:
            raise ValueError(f"Missing or invalid variable stage for {var}.")
    for condition in CONDITIONS:
        rows = corpus["conditions"].get(condition)
        if not isinstance(rows, list) or len(rows) != 48:
            raise ValueError(f"{condition} must contain exactly 48 rows.")
        for row in rows:
            for var in VARS:
                if row.get(var) not in (0, 1):
                    raise ValueError(f"Non-binary {var} value in {condition}, row {row.get('row_id')}.")
    if sorted(corpus.get("checkpoint_proposition_ids", [])) != sorted(CHECKPOINT_PROPOSITION_IDS):
        raise ValueError("checkpoint_proposition_ids do not match the required six IDs.")
    if len(corpus.get("proposition_bank", [])) < 24:
        raise ValueError("proposition_bank must contain at least 24 items.")
    if len(corpus.get("graph_visible_control", [])) < 6:
        raise ValueError("graph_visible_control must contain at least 6 items.")
    if not corpus.get("practice"):
        raise ValueError("practice block is missing.")


def main() -> None:
    args = parse_args()
    graphs = enumerate_legal_graphs()
    bank = build_proposition_bank(graphs)
    tmat = truth_matrix(graphs, bank)

    selection = search_evidence_bag(
        seed=args.seed,
        n_episodes=args.n_episodes,
        bag_search_limit=args.bag_search_limit,
        graphs=graphs,
        bank=bank,
        truth_mat=tmat,
        alpha=args.alpha,
    )
    order_df = build_order_conditions(selection.evidence_bag, graphs, seed=args.seed, alpha=args.alpha)

    summary = selection.final_summary
    corpus: Dict[str, Any] = {
        "metadata": {
            "task_name": "Hidden Machine",
            "version": TASK_VERSION,
            "n_observations": int(args.n_episodes),
            "checkpoints": list(CHECKPOINTS_DEFAULT),
            "conditions": list(CONDITIONS),
            "generation": {
                "generator": "build_machine_world_trials.py",
                "seed_start": int(args.seed),
                "selected_seed": int(selection.selected_seed),
                "bag_search_limit": int(args.bag_search_limit),
                "accepted_by_thresholds": bool(selection.accepted_by_thresholds),
                "rank_true_graph_exact_final": int(summary["rank_true_graph"]),
                "posterior_true_graph_exact_final": round(float(summary["posterior_true_graph"]), 8),
                "posterior_direct_A_D_exact_final": round(float(summary["posterior_P06_Direct_A_D"]), 8),
                "posterior_indirect_A_D_exact_final": round(float(summary["posterior_P12_Indirect_A_D"]), 8),
                "posterior_mediated_A_D_exact_final": round(float(summary["posterior_P18_Mediated_A_D"]), 8),
                "posterior_compound_mediation_A_D_exact_final": round(
                    float(summary["posterior_P19_A_affects_D_not_direct"]), 8
                ),
                "warning": selection.warning or "",
                "true_scm": {
                    "A": "Bernoulli(0.50)",
                    "B": "Bernoulli(0.15 + 0.70*A)",
                    "C": "Bernoulli(0.15 + 0.65*A)",
                    "D": "Bernoulli(0.05 + 0.35*B + 0.35*C + 0.20*B*C)",
                    "E": "Bernoulli(0.10 + 0.75*D)",
                },
            },
        },
        "variables": VARIABLE_SPECS,
        "conditions": build_condition_rows(order_df),
        "checkpoint_proposition_ids": CHECKPOINT_PROPOSITION_IDS,
        "proposition_bank": build_proposition_records(bank),
        "graph_visible_control": graph_visible_control(),
        "practice": practice_block(),
    }
    validate_browser_corpus(corpus)

    out = Path(args.out)
    out.write_text(json.dumps(corpus, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    print(f"Selected evidence seed: {selection.selected_seed}")
    print(f"Sufficiency thresholds met: {selection.accepted_by_thresholds}")
    print(f"P06 Direct(A,D) posterior: {float(summary['posterior_P06_Direct_A_D']):.6f}")
    print(f"P12 Indirect(A,D) posterior: {float(summary['posterior_P12_Indirect_A_D']):.6f}")
    print(f"P18 Mediated(A,D) posterior: {float(summary['posterior_P18_Mediated_A_D']):.6f}")


if __name__ == "__main__":
    main()
