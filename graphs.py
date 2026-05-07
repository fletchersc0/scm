"""Candidate graph enumeration and graph-level utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Sequence, Set, Tuple

import numpy as np

from config import LEGAL_EDGES, SPECIAL_GRAPH_NAMES, VAR_INDEX, VARS

Edge = Tuple[str, str]


@dataclass(frozen=True)
class Graph:
    graph_id: str
    mask: int
    edges: FrozenSet[Edge]
    num_edges: int
    parent_masks: Tuple[int, ...]
    edge_string: str
    adjacency: Tuple[Tuple[int, ...], ...]

    def has_edge(self, x: str, y: str) -> bool:
        return (x, y) in self.edges

    def parents_of(self, y: str) -> Tuple[str, ...]:
        return tuple(x for x in VARS if (x, y) in self.edges)

    def children_of(self, x: str) -> Tuple[str, ...]:
        return tuple(y for y in VARS if (x, y) in self.edges)

    def adjacency_matrix(self) -> np.ndarray:
        return np.asarray(self.adjacency, dtype=int)


def _edge_string(edges: Iterable[Edge]) -> str:
    ordered = [e for e in LEGAL_EDGES if e in set(edges)]
    return ", ".join(f"{a}->{b}" for a, b in ordered) if ordered else "<empty>"


def _parent_masks(edges: FrozenSet[Edge]) -> Tuple[int, ...]:
    masks: List[int] = []
    for y in VARS:
        m = 0
        for x in VARS:
            if (x, y) in edges:
                m |= 1 << VAR_INDEX[x]
        masks.append(m)
    return tuple(masks)


def _adjacency_tuple(edges: FrozenSet[Edge]) -> Tuple[Tuple[int, ...], ...]:
    mat = [[0 for _ in VARS] for _ in VARS]
    for x, y in edges:
        mat[VAR_INDEX[x]][VAR_INDEX[y]] = 1
    return tuple(tuple(row) for row in mat)


def enumerate_legal_graphs() -> List[Graph]:
    graphs: List[Graph] = []
    for mask in range(1 << len(LEGAL_EDGES)):
        edges = frozenset(edge for i, edge in enumerate(LEGAL_EDGES) if mask & (1 << i))
        graphs.append(
            Graph(
                graph_id=f"G{mask:03d}",
                mask=mask,
                edges=edges,
                num_edges=len(edges),
                parent_masks=_parent_masks(edges),
                edge_string=_edge_string(edges),
                adjacency=_adjacency_tuple(edges),
            )
        )
    return graphs


def find_graph_by_edges(graphs: Sequence[Graph], edges: Iterable[Edge]) -> Graph:
    target = frozenset(edges)
    for graph in graphs:
        if graph.edges == target:
            return graph
    raise KeyError(f"No candidate graph has exactly these edges: {sorted(target)}")


def special_graph_ids(graphs: Sequence[Graph]) -> Dict[str, str]:
    return {name: find_graph_by_edges(graphs, edges).graph_id for name, edges in SPECIAL_GRAPH_NAMES.items()}


def graph_to_record(graph: Graph) -> Dict[str, object]:
    return {
        "graph_id": graph.graph_id,
        "edge_string": graph.edge_string,
        "num_edges": graph.num_edges,
        "adjacency_matrix": graph.adjacency,
    }


def has_directed_path(graph: Graph, x: str, y: str, min_length: int = 1) -> bool:
    return any(len(path) - 1 >= min_length for path in all_directed_paths(graph, x, y))


def all_directed_paths(graph: Graph, x: str, y: str) -> List[Tuple[str, ...]]:
    if x == y:
        return []
    paths: List[Tuple[str, ...]] = []

    def dfs(node: str, target: str, visited: Set[str], path: List[str]) -> None:
        for child in graph.children_of(node):
            if child in visited:
                continue
            new_path = path + [child]
            if child == target:
                paths.append(tuple(new_path))
            else:
                dfs(child, target, visited | {child}, new_path)

    dfs(x, y, {x}, [x])
    return paths


def ancestors(graph: Graph, x: str) -> Set[str]:
    return {v for v in VARS if v != x and has_directed_path(graph, v, x)}


def descendants(graph: Graph, x: str) -> Set[str]:
    return {v for v in VARS if v != x and has_directed_path(graph, x, v)}


def rank_graph(scores: np.ndarray, graphs: Sequence[Graph], graph_id: str) -> int:
    idx = next(i for i, g in enumerate(graphs) if g.graph_id == graph_id)
    return int(1 + np.sum(scores > scores[idx]))


def graph_index(graphs: Sequence[Graph]) -> Dict[str, int]:
    return {g.graph_id: i for i, g in enumerate(graphs)}
