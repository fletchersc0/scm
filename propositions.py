"""Boolean proposition language and proposition bank."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from config import TRUE_GRAPH_EDGES
from graphs import Graph, all_directed_paths, ancestors, descendants, enumerate_legal_graphs, find_graph_by_edges, has_directed_path


@dataclass(frozen=True)
class Prop:
    op: str
    args: Tuple[object, ...]

    def __str__(self) -> str:
        return formal_string(self)


@dataclass(frozen=True)
class PropositionItem:
    proposition_id: str
    text: str
    family: str
    formal: Prop
    truth_in_true_graph: int
    primary_target: bool = False


def Direct(x: str, y: str) -> Prop:
    return Prop("Direct", (x, y))


def Path(x: str, y: str) -> Prop:
    return Prop("Path", (x, y))


def Indirect(x: str, y: str) -> Prop:
    return Prop("Indirect", (x, y))


def CommonCause(z: str, x: str, y: str) -> Prop:
    return Prop("CommonCause", (z, x, y))


def CommonEffect(z: str, x: str, y: str) -> Prop:
    return Prop("CommonEffect", (z, x, y))


def Mediated(x: str, y: str, via_set: Iterable[str]) -> Prop:
    return Prop("Mediated", (x, y, tuple(sorted(via_set))))


def NoConnection(x: str, y: str) -> Prop:
    return Prop("NoConnection", (x, y))


def NOT(phi: Prop) -> Prop:
    return Prop("NOT", (phi,))


def AND(phi: Prop, psi: Prop) -> Prop:
    return Prop("AND", (phi, psi))


def OR(phi: Prop, psi: Prop) -> Prop:
    return Prop("OR", (phi, psi))


def formal_string(phi: Prop) -> str:
    op = phi.op
    a = phi.args
    if op in {"Direct", "Path", "Indirect", "NoConnection"}:
        return f"{op}({a[0]},{a[1]})"
    if op in {"CommonCause", "CommonEffect"}:
        return f"{op}({a[0]}; {a[1]},{a[2]})"
    if op == "Mediated":
        via = ",".join(a[2])
        return f"Mediated({a[0]},{a[1]}; {{{via}}})"
    if op == "NOT":
        return f"NOT({formal_string(a[0])})"
    if op == "AND":
        return f"AND({formal_string(a[0])}, {formal_string(a[1])})"
    if op == "OR":
        return f"OR({formal_string(a[0])}, {formal_string(a[1])})"
    raise ValueError(f"Unknown proposition op: {op}")


def truth(phi: Prop, graph: Graph) -> int:
    op = phi.op
    a = phi.args
    if op == "Direct":
        return int(graph.has_edge(a[0], a[1]))
    if op == "Path":
        return int(has_directed_path(graph, a[0], a[1], min_length=1))
    if op == "Indirect":
        x, y = a
        return int(has_directed_path(graph, x, y, min_length=2) and not graph.has_edge(x, y))
    if op == "CommonCause":
        z, x, y = a
        return int(z in ancestors(graph, x) and z in ancestors(graph, y))
    if op == "CommonEffect":
        z, x, y = a
        return int(graph.has_edge(x, z) and graph.has_edge(y, z))
    if op == "Mediated":
        x, y, via_tuple = a
        via = set(via_tuple)
        if not has_directed_path(graph, x, y, min_length=1):
            return 0
        if graph.has_edge(x, y):
            return 0
        paths = all_directed_paths(graph, x, y)
        if not paths:
            return 0
        for path in paths:
            internal = set(path[1:-1])
            if not internal.intersection(via):
                return 0
        return 1
    if op == "NoConnection":
        x, y = a
        if has_directed_path(graph, x, y) or has_directed_path(graph, y, x):
            return 0
        if ancestors(graph, x).intersection(ancestors(graph, y)):
            return 0
        if descendants(graph, x).intersection(descendants(graph, y)):
            return 0
        return 1
    if op == "NOT":
        return 1 - truth(a[0], graph)
    if op == "AND":
        return min(truth(a[0], graph), truth(a[1], graph))
    if op == "OR":
        return max(truth(a[0], graph), truth(a[1], graph))
    raise ValueError(f"Unknown proposition op: {op}")


def probability(phi: Prop, atomic_probability: Callable[[Prop], float]) -> float:
    op = phi.op
    a = phi.args
    if op == "NOT":
        return 1.0 - probability(a[0], atomic_probability)
    if op == "AND":
        return probability(a[0], atomic_probability) * probability(a[1], atomic_probability)
    if op == "OR":
        p = probability(a[0], atomic_probability)
        q = probability(a[1], atomic_probability)
        return 1.0 - (1.0 - p) * (1.0 - q)
    return float(atomic_probability(phi))


def build_proposition_bank(graphs: Sequence[Graph] | None = None) -> List[PropositionItem]:
    if graphs is None:
        graphs = enumerate_legal_graphs()
    true_graph = find_graph_by_edges(graphs, TRUE_GRAPH_EDGES)
    specs = [
        ("P01", "A directly causes B.", "true_direct_edge", Direct("A", "B"), 1, False),
        ("P02", "A directly causes C.", "true_direct_edge", Direct("A", "C"), 1, False),
        ("P03", "B directly causes D.", "true_direct_edge", Direct("B", "D"), 1, False),
        ("P04", "C directly causes D.", "true_direct_edge", Direct("C", "D"), 1, False),
        ("P05", "D directly causes E.", "true_direct_edge", Direct("D", "E"), 1, False),
        ("P06", "A directly causes D.", "false_direct_compression", Direct("A", "D"), 0, True),
        ("P07", "A directly causes E.", "false_direct_compression", Direct("A", "E"), 0, False),
        ("P08", "B directly causes C.", "false_same_stage_edge", Direct("B", "C"), 0, False),
        ("P09", "C directly causes B.", "false_same_stage_edge", Direct("C", "B"), 0, False),
        ("P10", "D directly causes B.", "false_reverse_edge", Direct("D", "B"), 0, False),
        ("P11", "E directly causes D.", "false_reverse_edge", Direct("E", "D"), 0, False),
        ("P12", "A affects D indirectly through other symbols.", "true_indirect_path", Indirect("A", "D"), 1, False),
        ("P13", "A can affect E through a chain of symbols.", "true_path", Path("A", "E"), 1, False),
        ("P14", "B can affect E through D.", "true_path", Path("B", "E"), 1, False),
        ("P15", "C can affect E through D.", "true_path", Path("C", "E"), 1, False),
        ("P16", "A is a common cause of B and C.", "true_common_cause", CommonCause("A", "B", "C"), 1, False),
        ("P17", "B and C can each directly contribute to D.", "true_common_effect", CommonEffect("D", "B", "C"), 1, False),
        ("P18", "A's influence on D is mediated by B or C.", "true_mediation", Mediated("A", "D", {"B", "C"}), 1, False),
        ("P19", "A affects D, but not directly.", "true_compound_mediation", AND(Path("A", "D"), NOT(Direct("A", "D"))), 1, False),
        ("P20", "D directly causes E, and E does not directly cause D.", "true_compound_direction", AND(Direct("D", "E"), NOT(Direct("E", "D"))), 1, False),
        ("P21", "Either A directly causes D or B directly causes C.", "false_compound", OR(Direct("A", "D"), Direct("B", "C")), 0, False),
        ("P22", "A directly causes B and D directly causes E.", "true_compound", AND(Direct("A", "B"), Direct("D", "E")), 1, False),
        ("P23", "B and C have no connection in the machine.", "false_no_connection", NoConnection("B", "C"), 0, False),
        ("P24", "E is upstream of A.", "false_reverse_path", Path("E", "A"), 0, False),
    ]
    bank = [PropositionItem(*s) for s in specs]
    for item in bank:
        observed = truth(item.formal, true_graph)
        if observed != item.truth_in_true_graph:
            raise AssertionError(
                f"Truth validation failed for {item.proposition_id}: expected {item.truth_in_true_graph}, got {observed}"
            )
    return bank


def truth_matrix(graphs: Sequence[Graph], bank: Sequence[PropositionItem]) -> np.ndarray:
    mat = np.zeros((len(graphs), len(bank)), dtype=float)
    for gi, graph in enumerate(graphs):
        for pi, item in enumerate(bank):
            mat[gi, pi] = truth(item.formal, graph)
    return mat


def bank_records(bank: Sequence[PropositionItem]) -> List[Dict[str, object]]:
    return [
        {
            "proposition_id": item.proposition_id,
            "text": item.text,
            "family": item.family,
            "formal": formal_string(item.formal),
            "truth_in_true_graph": item.truth_in_true_graph,
            "primary_target": item.primary_target,
        }
        for item in bank
    ]
