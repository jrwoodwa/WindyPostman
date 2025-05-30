import networkx as nx
from ortools.linear_solver import pywraplp


# Example data from paper
V = [0, 1, 2, 3, 4]  # vertices
A = [
    (0, 1), (1, 0), (1, 2), (2, 1),
    (2, 3), (3, 2), (3, 4), (4, 3),
    (4, 0), (0, 4), (4, 1)
]  # all arcs
R = [(0, 1), (1, 0), (1, 2), (2, 3), (3, 2), (3, 4), (4, 3)]  # required arcs
Z = [(0, 1), (1, 0), (2, 3), (3, 2), (3, 4), (4, 3)]  # zig-zag arcs (subset of R)
W = [(3, 4), (4, 3)]  # arcs with time windows (subset of R)

# Travel and service times
t = {a: 10 for a in A}
t.update({(2, 1): 5, (4, 1): 8, (4, 0): 12})  # travel time
s1 = {a: 5 for a in R} # service time for all required arcs
s2 = {}
for a in Z:
    s2[a] = 12  # zig-zag service time

# Time windows
T = {}
for a in W:
    T[a] = 35

depot = 0


def compute_required_shortest_paths(G, R, orig_depot, art_depot):
    """
    Computes all shortest paths and distances between required vertices and depot.
    Additionally, for each arc pq of the input graph, stores a list of arcs uv such that pq lies on the shortest_path from u to v.

    Parameters:
        G: digraph
        R: set of required arcs (subset of A)
        depot: vertex representing the depot

    Returns:
        required_vertices: set of vertices incident to R
        shortest_paths: dict (u, v) -> list of nodes (the shortest path)
        shortest_distances: dict (u, v) -> distance (sum of t)
        arc_to_uv: dict (p, q) -> list of (u, v) such that (p, q) is on the shortest path from u to v
    """

    # Find required vertices (incident to R and depots)
    required_vertices = set([art_depot, orig_depot])  # start with depots
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
                    dist, path = nx.single_source_dijkstra(G, u, v, weight='weight') # TODO: this can probably be improved by avoiding recomputing paths
                    shortest_paths[(u, v)] = path
                    shortest_distances[(u, v)] = dist
                    # For each arc (p, q) on this path, record that (u, v) uses it
                    for i in range(len(path) - 1):
                        pq = (path[i], path[i+1])
                        if pq in arc_to_uv:
                            arc_to_uv[pq].append((u, v))
                        else:
                            arc_to_uv[pq] = [(u, v)]
                except nx.NetworkXNoPath:
                    shortest_paths[(u, v)] = None
                    shortest_distances[(u, v)] = float('inf')
    return required_vertices, shortest_paths, shortest_distances, arc_to_uv

def create_digraph(V, A, orig_depot, t):
    """
    Creates a networkx.DiGraph from the given vertices, arcs, and travel times.
    We add a artificial depot node so that all incident arcs are non-required. This is needed for the objective function to work correctly.

    Parameters:
        V: list of vertices
        A: list of arcs (tuples: (u, v))
        t: dict mapping (u, v) to travel time

    Returns:
        G: networkx.DiGraph with 'weight' attribute for travel time
    """
    G = nx.DiGraph()
    G.add_nodes_from(V)
    for (u, v) in A:
        G.add_edge(u, v, weight=t[(u, v)])

    # Add depot node
    art_depot = min(V) - 1  # use a new node that is not in V
    G.add_node(art_depot)
    G.add_edge(art_depot, orig_depot, weight=0)  # zero travel time from artificial depot to original depot
    G.add_edge(orig_depot, art_depot, weight=0)

    t[(art_depot, orig_depot)] = 0  # add travel time for the artificial depot to original depot
    t[(orig_depot, art_depot)] = 0  # add travel time for the original depot to artificial depot
            
    return G, art_depot

def solve_metric_closure_mip(required_vertices, shortest_paths, shortest_distances, arc_to_uv, R, Z, s1, s2, orig_depot, art_depot, W, T, t):
    """
    Solves the metric closure MIP using Google OR-Tools, including time window variables and constraints.

    Args:
        required_vertices: set of required vertices (including depot)
        shortest_paths: dict (u, v) -> list of nodes (the shortest path)
        shortest_distances: dict (u, v) -> shortest path distance between required vertices
        arc_to_uv: dict (p, q) -> list of (u, v) such that (p, q) is on the shortest path from u to v
        R: list of required arcs (tuples)
        Z: list of zig-zag arcs (subset of R)
        s1: dict of service times for required arcs
        s2: dict of service times for zig-zag arcs
        depot: depot node
        W: list of required arcs with time windows
        T: dict mapping arcs in W to their time window upper bounds
        t: dict mapping (u, v) to travel time
    """
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        raise Exception("Could not create solver.")

    C = [(u, v) for u in required_vertices for v in required_vertices if u != v]

    # Decision variables
    f = { (u, v): solver.BoolVar(f'f_{u}_{v}') for (u, v) in C }
    r = { (p, q): solver.BoolVar(f'r_{p}_{q}') for (p, q) in R }
    z = { (p, q): solver.BoolVar(f'z_{p}_{q}') for (p, q) in Z }
    # TODO: test if solver is faster if w is a continuous variable
    w = { (u, v): solver.NumVar(0, solver.infinity(), f'w_{u}_{v}') for (u, v) in C }

    # Objective
    # travel_cost = solver.Sum(shortest_distances[(u, v)] * f[(u, v)] for (u, v) in C)
    # service_cost = solver.Sum(s1[a] * r[a] for a in R)
    # zigzag_cost = solver.Sum(s2[a] * z[a] for a in Z)
    # obj = travel_cost + service_cost + zigzag_cost
    solver.Minimize(w[(orig_depot, art_depot)])  # minimize last time window variable, which is the arrival time at the original depot

    # 1. Flow conservation in closure
    for u in required_vertices:
        solver.Add(
            solver.Sum(f[(u, v)] for v in required_vertices if v != u) ==
            solver.Sum(f[(v, u)] for v in required_vertices if v != u)
        )
    # flow to airtificial depot:
    # this means that the flow goes through the original depot and
    # it let's us define the objective correctly
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

    # Bounding the time window variables w_{uv} for arcs in W
    for (p, q) in W:
        for (u, v) in arc_to_uv.get((p, q), []):
            path = shortest_paths.get((u, v))
            idx = path.index(p)
            # Compute sum over arcs before (p, q) in path
            # TODO: we could also try to explicitly store the edges of the path in the metric closure instead of just the shortcut edges
            sum_before = 0
            for j in range(idx):
                arc = (path[j], path[j + 1])
                sum_before += t[arc]
                if arc in s1:
                    sum_before += s1[arc] * r[arc]
                if arc in s2:
                    sum_before += s2[arc] * z[arc]
            # w_{uv} <= T(pq) - sum_before
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
        if u == art_depot: # the arrival time of the first arc is 0
            solver.Add(w[(u, v)] == 0)
            continue
        for v_prime in required_vertices:
            if v_prime == u:
                continue
            path_prev = shortest_paths.get((v_prime, u))
            # sum over arcs in path_prev
            sum_prev = 0
            for i in range(len(path_prev) - 1):
                arc = (path_prev[i], path_prev[i+1])
                sum_prev += t[arc]
                if arc in s1:
                    sum_prev += s1[arc] * r[arc]
                if arc in s2:
                    sum_prev += s2[arc] * z[arc]
            # w_{uv} <= w_{v'u} + sum_prev + M*(1-f_{uv})
            solver.Add(w[(u, v)] >= w[(v_prime, u)] + sum_prev - M * (1 - f[(u, v)]))

    # Solve
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        print("Optimal solution found:")
        # Sort arcs in C by their w_{uv} value
        arcs_with_times = [((u, v), w[(u, v)].solution_value()) for (u, v) in C if f[(u, v)].solution_value() > 0.5]
        arcs_with_times.sort(key=lambda x: x[1])
        for (u, v), w_val in arcs_with_times:
            path = shortest_paths.get((u, v))
            print(f"f[{u},{v}] = 1, w[{u},{v}] = {round(w_val)}, path: {path}")
        for (p, q) in R:
            if r[(p, q)].solution_value() > 0.5:
                print(f"r[{p},{q}] = 1")
        for (p, q) in Z:
            if z[(p, q)].solution_value() > 0.5:
                print(f"z[{p},{q}] = 1")
        print("Objective value:", round(solver.Objective().Value()))
    else:
        print("No optimal solution found.")

def main():
    # Build graph and data
    G, art_depot = create_digraph(V, A, depot, t)
    required_vertices, shortest_paths, shortest_distances, arc_to_uv = compute_required_shortest_paths(G, R, depot, art_depot)
    solve_metric_closure_mip(required_vertices, shortest_paths, shortest_distances, arc_to_uv, R, Z, s1, s2, depot, art_depot, W, T, t)

if __name__ == "__main__":
    main()
