import networkx as nx
from shapely.geometry import MultiPoint, Point
import osmnx as ox

# Load the graph from OpenStreetMap
place = "Saugus, Massachusetts, USA"
G = ox.graph_from_place(place, network_type="drive")

# Convert to GeoDataFrames
nodes, edges = ox.graph_to_gdfs(G)

# seed polygon
seed = [
    ["42.45671", "-71.01679"],
    ["42.45668", "-71.01198"],
    ["42.45747", "-71.01001"],
    ["42.45332", "-71.00825"],
    ["42.45187", "-71.01168"],
    ["42.45104", "-71.01485"],
    ["42.45446", "-71.01649"],
    ["42.45263", "-71.01616"],
    ["42.45080", "-71.01545"],
    ["42.45081", "-71.01438"],
    ["42.45178", "-71.01232"],
    ["42.45241", "-71.00919"],
    ["42.45312", "-71.00782"],
    ["42.45336", "-71.00732"],
    ["42.45410", "-71.00766"],
    ["42.45747", "-71.00934"],
    ["42.45822", "-71.00945"],
    ["42.45804", "-71.01024"],
    ["42.45882", "-71.01078"],
    ["42.45755", "-71.01140"],
    ["42.45720", "-71.01157"]
]
#hull_poly = MultiPoint([(lon, lat) for lat, lon in seed]).convex_hull
hull_poly = MultiPoint([(float(lon), float(lat)) for lat, lon in seed]).convex_hull

# only mark edge as required if all its points are within the convex hull
for u, v, data in G.edges(data=True):
    geom = data.get('geometry')
    if geom:
        # check all points in geometry
        if all(hull_poly.contains(Point(x, y)) for x, y in geom.coords):
            data['required'] = True  # edge is fully inside
        else:
            data['required'] = False  # some part is outside
    else:
        data['required'] = False  # no geometry, can't verify

# edges that are required
required_edges = [(u, v, k) for u, v, k, data in G.edges(keys=True, data=True) if data.get('required')]


import matplotlib.pyplot as plt
# mark the depot node ID
depot_node = 74944114

# build edge colors
edge_colors = ['red' if data.get('required') else 'gray' for _, _, _, data in G.edges(keys=True, data=True)]

# build node colors: red for depot, gray otherwise
node_colors = ['red' if n == depot_node else 'gray' for n in G.nodes]

# build node sizes: 100 for depot, 0 for others
node_sizes = [100 if n == depot_node else 0 for n in G.nodes]

# # plot full graph with colored edges and highlighted depot node
# fig, ax = ox.plot_graph(
#     G,
#     edge_color=edge_colors,
#     node_color=node_colors,
#     node_size=node_sizes,
#     show=False,
#     close=False
# )

# plt.show()

import networkx as nx
from typing import Tuple, List, Dict
import numpy as np

# check if edge is zig-zag
def is_zigzag_edge(data):
    highway = data.get("highway")
    if isinstance(highway, list):
        highway = highway[0]

    width = data.get("width")
    if isinstance(width, list):
        try:
            width = float(width[0])
        except:
            width = None
    else:
        try:
            width = float(width)
        except:
            width = None

    return highway in {"residential", "service", "living_street", "unclassified"} or (
        width is not None and width < 3.5
    )

# estimate travel time
def estimate_travel_time(length: float, data: dict) -> int:
    maxspeed = data.get("maxspeed", None)
    if isinstance(maxspeed, list):
        maxspeed = maxspeed[0]
    try:
        speed_kmh = float(maxspeed)
    except:
        speed_kmh = 30.0  # fallback
    speed_mps = speed_kmh * 1000 / 3600
    return max(int(length / speed_mps), 1)

# estimate service time
def estimate_service_time(length: float, is_zigzag: bool, 
                          house_density: float = 0.05  # houses per meter 
                         ) -> int:
    base_time = length * house_density * 0.5 # 0.50 minutes or 30 seconds per house
    if is_zigzag:
        return np.round(3 * base_time, 0).astype(int)
    else:
        return np.round(base_time,0).astype(int)

# main conversion
def graph_to_windy_inputs(G: nx.MultiDiGraph, depot_node: int) -> Tuple:
    V = list(G.nodes)
    A = []
    R = []
    Z = []
    W = []
    t = {}
    s1 = {}
    s2 = {}
    T = {}

    for u, v, key, data in G.edges(keys=True, data=True):
        arc = (u, v)
        A.append(arc)
        length = data.get("length", 1.0)
        t[arc] = estimate_travel_time(length, data)

        is_required = data.get("required", False)
        is_zigzag = is_zigzag_edge(data)

        if is_required:
            R.append(arc)
            s1[arc] = estimate_service_time(length, is_zigzag=False)

        if is_zigzag:
            Z.append(arc)
            s2[arc] = estimate_service_time(length, is_zigzag=True)

    return V, A, R, Z, W, t, s1, s2, T, depot_node

V, A, R, Z, W, t, s1, s2, T, depot_node = graph_to_windy_inputs(G, depot_node)

# import solver
from windy_postman_timewindows_zigzags import solve_windy_postman
solve_windy_postman(V, A, R, Z, s1, s2, depot_node, W, T, t)