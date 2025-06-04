"""
Windy Postman Vehicle Routing Problem with Zig-Zag and Time Windows
===================================================================

This script models and solves a variant of the vehicle routing problem on a directed graph,
with required arcs, zig-zag arcs, and time windows, using Google OR-Tools and NetworkX.

Features:
- Computes all-pairs shortest paths between required vertices.
- Builds a metric closure and solves a MIP for the optimal route.
- Handles zig-zag arcs and time window constraints.
- Prints solution details.

Author: Andreas Feldmann
Date: 2025-05-30

Requirements:
- networkx
- ortools

Usage:
    python zigzag-n-time.py

"""

import logging
from typing import List, Tuple, Dict, Set, Any
import networkx as nx
from ortools.linear_solver import pywraplp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Example data from paper
V: List[int] = [0, 1, 2, 3, 4]  # vertices
A: List[Tuple[int, int]] = [
    (0, 1), (1, 0), (1, 2), (2, 1),
    (2, 3), (3, 2), (3, 4), (4, 3),
    (4, 0), (0, 4), (4, 1)
]  # all arcs
R: List[Tuple[int, int]] = [(0, 1), (1, 0), (1, 2), (2, 3), (3, 2), (3, 4), (4, 3)]  # required arcs
Z: List[Tuple[int, int]] = [(0, 1), (1, 0), (2, 3), (3, 2), (3, 4), (4, 3)]  # zig-zag arcs (subset of R)
W: List[Tuple[int, int]] = [(3, 4), (4, 3)]  # arcs with time windows (subset of R)

# Travel and service times
t: Dict[Tuple[int, int], int] = {a: 10 for a in A}
t.update({(2, 1): 5, (4, 1): 8, (4, 0): 12})  # travel time
s1: Dict[Tuple[int, int], int] = {a: 5 for a in R} # service time for all required arcs
s2: Dict[Tuple[int, int], int] = {a: 12 for a in Z}  # zig-zag service time

# Time windows
T: Dict[Tuple[int, int], int] = {a: 35 for a in W}

depot: int = 0

def create_digraph(
    V: List[int],
    A: List[Tuple[int, int]],
    orig_depot: int,
    t: Dict[Tuple[int, int], int]
) -> Tuple[nx.DiGraph, int]:
    """
    Creates a networkx.DiGraph from the given vertices, arcs, and travel times.
    Adds an artificial depot node so that all incident arcs are non-required.

    Returns:
        G: networkx.DiGraph with 'weight' attribute for travel time
        art_depot: artificial depot node id
    """
    G = nx.DiGraph()
    G.add_nodes_from(V)
    for (u, v) in A:
        G.add_edge(u, v, weight=t[(u, v)])

    # Add artificial depot node
    art_depot = min(V) - 1  # use a new node that is not in V
    G.add_node(art_depot)
    G.add_edge(art_depot, orig_depot, weight=0)
    G.add_edge(orig_depot, art_depot, weight=0)
    t[(art_depot, orig_depot)] = 0
    t[(orig_depot, art_depot)] = 0
    logger.info(f"Created digraph with artificial depot {art_depot}")
    return G, art_depot

def compute_required_shortest_paths(
    G: nx.DiGraph,
    R: List[Tuple[int, int]],
    orig_depot: int,
    art_depot: int
) -> Tuple[Set[int], Dict[Tuple[int, int], List[int]], Dict[Tuple[int, int], float], Dict[Tuple[int, int], List[Tuple[int, int]]]]:
    """
    Computes all shortest paths and distances between required vertices and depot.
    Additionally, for each arc pq of the input graph, stores a list of arcs uv such that pq lies on the shortest_path from u to v.

    Returns:
        required_vertices: set of vertices incident to R
        shortest_paths: dict (u, v) -> list of nodes (the shortest path)
        shortest_distances: dict (u, v) -> distance (sum of t)
        arc_to_uv: dict (p, q) -> list of (u, v) such that (p, q) is on the shortest path from u to v
    """
    required_vertices = set([art_depot, orig_depot])
    for (u, v) in R:
        required_vertices.add(u)
        required_vertices.add(v)

    shortest_paths = {}
    shortest_distances = {}
    arc_to_uv = {edge: [] for edge in G.edges}
    for u in required_vertices:
        for v in required_vertices:
            if u != v:
                try:
                    dist, path = nx.single_source_dijkstra(G, u, v, weight='weight')
                    shortest_paths[(u, v)] = path
                    shortest_distances[(u, v)] = dist
                    for i in range(len(path) - 1):
                        pq = (path[i], path[i+1])
                        if pq in arc_to_uv:
                            arc_to_uv[pq].append((u, v))
                        else:
                            arc_to_uv[pq] = [(u, v)]
                except nx.NetworkXNoPath:
                    shortest_paths[(u, v)] = None
                    shortest_distances[(u, v)] = float('inf')
    logger.info(f"Computed shortest paths between {len(required_vertices)} required vertices.")
    return required_vertices, shortest_paths, shortest_distances, arc_to_uv

def solve_metric_closure_mip(
    required_vertices: Set[int],
    shortest_paths: Dict[Tuple[int, int], List[int]],
    shortest_distances: Dict[Tuple[int, int], float],
    arc_to_uv: Dict[Tuple[int, int], List[Tuple[int, int]]],
    R: List[Tuple[int, int]],
    Z: List[Tuple[int, int]],
    s1: Dict[Tuple[int, int], int],
    s2: Dict[Tuple[int, int], int],
    orig_depot: int,
    art_depot: int,
    W: List[Tuple[int, int]],
    T: Dict[Tuple[int, int], int],
    t: Dict[Tuple[int, int], int],
    verbose: bool = True
) -> None:
    """
    Solves the metric closure MIP using Google OR-Tools, including time window variables and constraints.
    Prints the solution if found.
    """
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        raise Exception("Could not create solver.")

    C = [(u, v) for u in required_vertices for v in required_vertices if u != v]

    # Decision variables
    f = { (u, v): solver.BoolVar(f'f_{u}_{v}') for (u, v) in C }
    r = { (p, q): solver.BoolVar(f'r_{p}_{q}') for (p, q) in R }
    z = { (p, q): solver.BoolVar(f'z_{p}_{q}') for (p, q) in Z }
    w = { (u, v): solver.NumVar(0, solver.infinity(), f'w_{u}_{v}') for (u, v) in C }

    # Objective: minimize arrival time at original depot
    solver.Minimize(w[(orig_depot, art_depot)])

    # 1. Flow conservation in closure
    for u in required_vertices:
        solver.Add(
            solver.Sum(f[(u, v)] for v in required_vertices if v != u) ==
            solver.Sum(f[(v, u)] for v in required_vertices if v != u)
        )
    solver.Add(f[(orig_depot, art_depot)] == 1)
    for u in required_vertices:
        if u != orig_depot and u != art_depot:
            solver.Add(f[(u, art_depot)] == 0)

    # 2. Demand satisfaction for required arcs
    for (p, q) in R:
        if (p, q) in Z:
            solver.Add(
                solver.Sum(f[(u, v)] for (u, v) in arc_to_uv[(p, q)]) >= r[(p, q)] + z[(p, q)]
            )
        else:
            solver.Add(r[(p, q)] == 1)
            solver.Add(
                solver.Sum(f[(u, v)] for (u, v) in arc_to_uv[(p, q)]) >= 1
            )
    for (p, q) in Z:
        pq = (p, q)
        qp = (q, p)
        solver.Add(r[pq] + r[qp] + 2 * z[pq] + 2 * z[qp] == 2)

    # --- Time window constraints ---
    for (p, q) in W:
        for (u, v) in arc_to_uv.get((p, q), []):
            path = shortest_paths.get((u, v))
            idx = path.index(p)
            sum_before = 0
            for j in range(idx):
                arc = (path[j], path[j + 1])
                sum_before += t[arc]
                if arc in s1:
                    sum_before += s1[arc] * r[arc]
                if arc in s2:
                    sum_before += s2[arc] * z[arc]
            solver.Add(w[(u, v)] <= T[(p, q)] - sum_before)

    # Compute big-M
    M = 0
    for (u, v) in C:
        path = shortest_paths.get((u, v))
        if path is None:
            continue
        for i in range(len(path) - 1):
            arc = (path[i], path[i+1])
            M += t[arc]
            if arc in s1:
                M += s1[arc]
            if arc in s2:
                M += s2[arc]

    # Time propagation constraints
    for (u, v) in C:
        if u == art_depot:
            solver.Add(w[(u, v)] == 0)
            continue
        for v_prime in required_vertices:
            if v_prime == u:
                continue
            path_prev = shortest_paths.get((v_prime, u))
            sum_prev = 0
            for i in range(len(path_prev) - 1):
                arc = (path_prev[i], path_prev[i+1])
                sum_prev += t[arc]
                if arc in s1:
                    sum_prev += s1[arc] * r[arc]
                if arc in s2:
                    sum_prev += s2[arc] * z[arc]
            solver.Add(w[(u, v)] >= w[(v_prime, u)] + sum_prev - M * (1 - f[(u, v)]))

    # Solve
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        logger.info("Optimal solution found:")
        arcs_with_times = [((u, v), w[(u, v)].solution_value()) for (u, v) in C if f[(u, v)].solution_value() > 0.5]
        arcs_with_times.sort(key=lambda x: x[1])
        for (u, v), w_val in arcs_with_times:
            path = shortest_paths.get((u, v))
            logger.info(f"f[{u},{v}] = 1, w[{u},{v}] = {round(w_val)}, path: {path}")
        for (p, q) in R:
            if r[(p, q)].solution_value() > 0.5:
                logger.info(f"r[{p},{q}] = 1")
        for (p, q) in Z:
            if z[(p, q)].solution_value() > 0.5:
                logger.info(f"z[{p},{q}] = 1")
        logger.info(f"Objective value: {round(solver.Objective().Value())}")
    else:
        logger.error("No optimal solution found.")

def main() -> None:
    """
    Main function to build the graph, compute shortest paths, and solve the MIP.
    """
    logger.info("Building graph and data...")
    G, art_depot = create_digraph(V, A, depot, t)
    required_vertices, shortest_paths, shortest_distances, arc_to_uv = compute_required_shortest_paths(G, R, depot, art_depot)
    
    logger.info("Solving metric closure MIP...")
    solve_metric_closure_mip(required_vertices, shortest_paths, shortest_distances, arc_to_uv, R, Z, s1, s2, depot, art_depot, W, T, t)

if __name__ == "__main__":
    main()
