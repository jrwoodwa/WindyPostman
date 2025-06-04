import networkx as nx
import matplotlib.pyplot as plt

def build_nx_graph(graph_dict):
    G = nx.DiGraph()
    G.add_nodes_from(graph_dict["nodes"])
    G.add_edges_from([(u, v, d) for (u, v), d in graph_dict["edges"].items()])
    return G

def visualize_graph(graph, title="Graph Visualization", pos=None):
    g = nx.MultiDiGraph()

    # Add nodes
    for node in graph["nodes"]:
        g.add_node(node)

    # Add edges
    for (u, v), attrs in graph["edges"].items():
        label = attrs.get("travel_time", "")
        g.add_edge(u, v, label=label, **attrs)

    # Use provided layout or fallback to automatic
    if pos is None:
        try:
            pos = nx.kamada_kawai_layout(g)
        except:
            pos = nx.spring_layout(g)

    # Draw nodes and labels
    nx.draw_networkx_nodes(g, pos, node_size=500, node_color='lightblue')
    nx.draw_networkx_labels(g, pos, font_size=10)

    # Draw edges with appropriate styles
    edge_styles = []
    for u, v, k, data in g.edges(keys=True, data=True):
        style = 'solid' if data.get("required", False) else 'dashed'
        edge_styles.append((u, v, k, style))

    for u, v, k, style in edge_styles:
        nx.draw_networkx_edges(
            g, pos, edgelist=[(u, v)],
            style=style,
            connectionstyle=f"arc3,rad={(0.2 if k == 0 else -0.2)}"
        )

    # Draw edge labels
    for u, v, k, data in g.edges(keys=True, data=True):
        edge_label = {(u, v): data.get("label", "")}
        label_pos = 0.2 if k == 0 else -0.2
        nx.draw_networkx_edge_labels(
            g, pos, edge_labels=edge_label,
            font_color='red', label_pos=0.3,
            rotate=False,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.6),
            connectionstyle=f"arc3,rad={label_pos}"
        )

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def find_required_nodes(graph, debug=False):
    """
    Identify all nodes involved in 'required' edges, including the depot.

    Parameters:
    - graph (dict): A dictionary with keys 'nodes', 'edges', and 'depot'.
                    Each edge has attributes like 'travel_time' and 'required'.
    - debug (bool): If True, print step-by-step debug info for understanding.

    Returns:
    - List[int]: List of nodes that are required (including depot).
    """

    # initialize sets for required nodes
    required_nodes = set()

    # always include depot
    depot = graph["depot"]
    required_nodes.add(depot)
    if debug: print(f"Depot node added: {depot}")

    for (u, v), attrs in graph["edges"].items():
        if attrs.get("required", False):
            required_nodes.add(u)
            required_nodes.add(v)
            if debug: print(f"Required edge found: ({u} -> {v}), adding nodes {u} and {v}")
        else:
            if debug: print(f"Non-required edge: ({u} -> {v}), ignored")

    result = list(required_nodes)
    if debug: print(f"Final required nodes list: {result}")

    return result

def compute_all_pairs_sp(G, return_paths=True, weight="travel_time"):
    if return_paths:
        lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight=weight))
        paths = dict(nx.all_pairs_dijkstra_path(G, weight=weight))
        return paths, lengths
    else:
        return dict(nx.all_pairs_dijkstra_path_length(G, weight=weight))

def build_complete_edges(required_nodes, sp_lengths):
    """
    Construct a complete-edge dict connecting every pair of required nodes,
    with the edge weight taken from precomputed shortest-path lengths.

    Args:
        required_nodes (list): List of required node IDs.
        sp_lengths (dict): {u: {v: distance}} from compute_all_pairs_sp.

    Returns:
        dict: {(u, v): {"travel_time": distance, "required": False}}
    """
    complete = {}
    for u in required_nodes:
        for v in required_nodes:
            if u == v:
                continue
            complete[(u, v)] = {
                "travel_time": sp_lengths[u][v],
                "required": False
            }
    return complete

def prune_shortcut_edges(complete_edges, sp_paths, required_set):
    """
    From a complete-edge dict, remove any (u,v) whose SP path
    goes through another required node.

    Args:
        complete_edges (dict): {(u,v): attrs}
        sp_paths (dict): {u: {v: [u,...,v]}}
        required_set (set): set of required node IDs

    Returns:
        dict: pruned_edges
    """
    pruned = {}
    for (u, v), attrs in complete_edges.items():
        path = sp_paths[u][v]  # e.g. [u, x, y, v]
        # drop this edge if any intermediate node is required
        if any(w in required_set for w in path[1:-1]): # exclusive, excluding the endpoints
            continue
        pruned[(u, v)] = attrs
    return pruned