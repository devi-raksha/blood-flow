import xml.etree.ElementTree as ET
import networkx as nx
import numpy as np
import math
from scipy.optimize import brentq


# ---- file paths ----
XML_PATH = "graphExport_4500.xml"         
OUT_VTK  = "liver_vascular_tree_4500.vtk"    # output VTK file

# ---- unit conversion ----
# Assume XML positions & the edge "radius" attribute are given in millimetres.
# Everything downstream (points, radius, length) is converted to metres so it is
# SI-consistent with E [Pa], h_wall [m], pressures [Pa], R1/R2 [Pa*s/m^3], C [m^3/Pa].
LENGTH_SCALE = 0.01

# ---- wall material properties: ARBITRARY, deliberately very stiff, same for every vessel ----
# Only scalability is being tested here, so there is no root/daughter split — just crank E up
# far enough that the vessel barely distends (a0 ~ a_d), approximating a rigid Poiseuille pipe.
E_STIFF  = 1e4 #5e7     # Young's modulus [Pa]  (very stiff; raise further to approach rigid limit)
HW_STIFF = 1e-3     # wall thickness [m]    (arbitrary, unused as a tuning knob here)
RADIUS_RATIO_CAP = 8 
# ---- reference (zero-transmural) pressure for the elastic tube law ----
P0 = 0.0

# ---- constant boundary condition: same working pressure prescribed on every vessel ----
# (replaces the depth-interpolated pPerf -> pTerm pressure drop; arbitrary, just needs to be
# a single fixed value so the only thing varying along the tree is geometry/resistance)
P_CONST = 0   #1e3  # working pressure p_d for every vessel [Pa]

# ---- single-resistor boundary condition, applied at terminal (leaf) nodes only ----
# IMPORTANT — this matches BloodFlowSystem's actual convention, not a generic guess:
#   - rcr_map is only populated where R2 > 0 (see mesh-read code, "if (rcr.R2 > 0.0)"),
#     so R2 must be the nonzero one or terminals never get registered at all.
#   - assemble_trace_boundary_equations branches on C: C > 0 -> true 3-element case
#     using R1 in series before the capacitor; C <= 0 -> "single R: P = R2*Q + P_out"
#     (their comment, verbatim). So for a single resistor, R2 carries the resistance
#     and R1 is unused/irrelevant.
R2 = 5e9             # unused whenever C <= 0 (single-resistor mode)
R1 = 0.0            # lumped terminal resistance [Pa*s/m^3] — the value that's actually used
C  = 0.0            # <= 0 selects the single-resistor formula (P = R2*Q + P_out)
P_OUT = 0.0         # distal/venous reference pressure [Pa]


# ## 2. Parse the GXL/XML graph
# 
# Reads every `node` (id, `nodeType`, 3D `position`) and every `edge` (`from`, `to`, and any numeric `attr` such as `flow`, `resistance`, `radius`). Also reads the graph-level `pPerf` / `pTerm` attributes.

# In[3]:


tree = ET.parse(XML_PATH)
xroot = tree.getroot()

node_type, node_pos = {}, {}
for node in xroot.iter("node"):
    nid = node.attrib["id"]
    for attr in node.findall("attr"):
        nm = attr.attrib["name"].strip()
        if nm == "nodeType":
            node_type[nid] = attr.find("string").text.strip()
        elif nm == "position":
            vals = [float(f.text) for f in attr.find("tup").findall("float")]
            node_pos[nid] = tuple(vals)

# graph-level info (perfusion / terminal pressure)
info = {}
for attr in xroot.find(".//info_graph").findall("attr"):
    info[attr.attrib["name"].strip()] = float(attr.find("float").text)
p_perf, p_term = info["pPerf"], info["pTerm"]

edge_radius = {}
edges = []
G = nx.DiGraph()
for edge in xroot.iter("edge"):
    u, v = edge.attrib["from"], edge.attrib["to"]
    for attr in edge.findall("attr"):
        if attr.attrib["name"].strip() == "radius":
            edge_radius[(u, v)] = float(attr.find("float").text)
    edges.append((u, v))
    G.add_edge(u, v)
G.add_nodes_from(node_pos.keys())

print(f"nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
print(f"pPerf = {p_perf} Pa, pTerm = {p_term} Pa")


# ## 3. Tree topology — root, generations, boundary classification
# 
# The root is the unique node with in-degree 0. Generation depth (shortest path length from the root) drives both the Murray's-law radius recursion and the linear pressure interpolation. Boundary classification: root = 0, terminal (leaf) points get a unique running id 1, 2, 3, ..., and internal bifurcation ("junction") points all get one fixed sentinel value (25555) so they're easy to mask out downstream.

# In[4]:


root_node = [n for n in G.nodes if G.in_degree(n) == 0][0]
assert nx.is_weakly_connected(G) and nx.is_tree(G.to_undirected()), "graph is not a single tree"

gen = nx.single_source_shortest_path_length(G, root_node)
max_gen = max(gen.values())

JUNCTION_ID = 25555   # sentinel for every internal bifurcation point (arbitrary, easy to spot/mask)

boundary_id = {}
terminal_counter = 0
for n in node_pos.keys():        # iterate in the same order the points array is built (Sec. 6)
    if n == root_node:
        boundary_id[n] = 0
    elif G.out_degree(n) == 0:
        terminal_counter += 1
        boundary_id[n] = terminal_counter    # 1, 2, 3, ... per terminal, in point order
    else:
        boundary_id[n] = JUNCTION_ID

print(f"root node: {root_node} ({node_type.get(root_node)})")
print(f"max generation depth: {max_gen}")
print(f"terminals: {terminal_counter}, "
      f"internal bifurcations (id={JUNCTION_ID}): "
      f"{sum(1 for b in boundary_id.values() if b == JUNCTION_ID)}")


# ## 4. Radius — Murray's law
# 
# The measured radius of the true root vessel (first edge out of the root) seeds the recursion. At every bifurcation (2 children) both daughters get `r_parent / 2**(1/3)`; a node with a single child (only the initial root stub here) passes its radius through unchanged.

# In[5]:


root_edge = list(G.out_edges(root_node))[0]
ROOT_RADIUS = edge_radius[root_edge] * LENGTH_SCALE   # metres
R_FLOOR     = ROOT_RADIUS / RADIUS_RATIO_CAP
node_radius_out = {root_node: ROOT_RADIUS}   # radius of the vessel(s) leaving each node
for parent in nx.topological_sort(G):
    children = list(G.successors(parent))
    if not children:
        continue
    rp = node_radius_out[parent]
    rc = rp if len(children) == 1 else max(rp / (2.0 ** (1.0 / 3.0)), R_FLOOR)
    for ch in children:
        node_radius_out[ch] = rc

vessel_radius = {(u, v): node_radius_out[u] for (u, v) in edges}

r_mm = np.array(list(vessel_radius.values())) * 1e3
print(f"ROOT_RADIUS = {ROOT_RADIUS*1e3:.4f} mm")
print(f"vessel radius range: {r_mm.min():.5f} - {r_mm.max():.5f} mm")


# ## 5. Elastic tube law — solve for the zero-pressure reference area `a0`
# 
# Standard nonlinear pressure-area relation used in 1D arterial/structured-tree models:
# 
# $$p - p_0 = \frac{\beta}{A_0}\left(\sqrt{A} - \sqrt{A_0}\right), \qquad \beta = \frac{4}{3}\sqrt{\pi}\,E\,h$$
# 
# Given the vessel's distended area `A_d = pi r²` (from the Murray-law radius) and its working pressure `p_d`, we solve numerically for `A0`.
# With `E` set very stiff (see config), `A0` ends up nearly equal to `A_d` — the tube behaves like a rigid pipe, which is the point of this run (Poiseuille-like check).

# In[6]:


def solve_A0(A_d, E, h, p_d, p0):
    if p_d - p0 == 0:
        return A_d
    beta = (4.0 / 3.0) * math.sqrt(math.pi) * E * h
    sqrtAd = math.sqrt(A_d)

    def f(A0):
        return (beta / A0) * (sqrtAd - math.sqrt(A0)) - (p_d - p0)

    return brentq(f, A_d * 1e-6, A_d * 10)


# ## 6. Assemble per-vessel (cell) and per-node (point) fields

# In[7]:


node_ids = list(node_pos.keys())
node_index = {nid: i for i, nid in enumerate(node_ids)}

points = [node_pos[nid] for nid in node_ids]
cells_conn = [(node_index[u], node_index[v]) for (u, v) in edges]

vessel_id, a0_list, a_d_list = [], [], []
E_list, h_list, p0_list, pd_list, L_list = [], [], [], [], []
rd_list = []

for i, (u, v) in enumerate(edges):
    g = gen[v]  # generation of the vessel = generation of its downstream node

    # arbitrary, deliberately stiff material — same for every vessel (no root/daughter split)
    E_ = E_STIFF
    h_ = HW_STIFF

    # constant boundary condition: same working pressure on every vessel, regardless of depth
    p_d = P_CONST

    r = vessel_radius[(u, v)]
    A_d = math.pi * r * r
    A0 = A_d

    p0v = np.array(node_pos[u]) * LENGTH_SCALE
    p1v = np.array(node_pos[v]) * LENGTH_SCALE
    L = float(np.linalg.norm(p1v - p0v))

    vessel_id.append(i)
    a0_list.append(A0)
    a_d_list.append(A_d)
    E_list.append(E_)
    h_list.append(h_)
    p0_list.append(P0)
    pd_list.append(p_d)
    L_list.append(L)
    rd_list.append(r) 

print(f"vessels: {len(edges)}, points: {len(points)}")
print(f"a0 range: {min(a0_list):.4e} - {max(a0_list):.4e} m^2")
print(f"p_d range: {min(pd_list):.1f} - {max(pd_list):.1f} Pa")
print(f"rd range: {min(rd_list):.5f} - {max(rd_list):.5f} m")

# ---------- ADD THIS BLOCK ----------
print("\n========== Network Statistics ==========")
print(f"Number of points      : {len(points)}")
print(f"Number of vessels     : {len(edges)}")
print(f"Number of terminals   : {terminal_counter}")

print(f"\nRadius:")
print(f"  min = {min(rd_list):.6e} m")
print(f"  max = {max(rd_list):.6e} m")

print(f"\nLength:")
print(f"  min = {min(L_list):.6e} m")
print(f"  max = {max(L_list):.6e} m")

print(f"\nArea:")
print(f"  min(A_d) = {min(a_d_list):.6e}")
print(f"  max(A_d) = {max(a_d_list):.6e}")
print("========================================\n")
# --------------------------------------


# ## 7. Write the VTK file
# 
# Legacy VTK **3.0** `UNSTRUCTURED_GRID` layout matching the simple reference format: plain
# `CELLS`/`CELL_TYPES` (no `OFFSETS`/`CONNECTIVITY`), one scalar value per line (no 9-per-line
# blocks), `int` type for `vessel_id`/`boundary_id`, and `POINT_DATA` with `boundary_id, R1, R2,
# C, P_out`. R1 is zeroed everywhere (unused in single-resistor mode); R2 carries the resistance
# only at terminal points (`boundary_id` not in {0, JUNCTION_ID}); C and P_out are 0.0 everywhere
# (C <= 0 is what tells the solver to use the single-resistor formula); root + junction points
# get all four BC fields at 0.0.

# In[8]:


def write_vtk(fname):
    n_pts = len(points)
    n_cells = len(cells_conn)
    bvals = [boundary_id[nid] for nid in node_ids]

    with open(fname, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("vtk output\nASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {n_pts} double\n")
        for (x, y, z) in points:
            f.write(f"{x*LENGTH_SCALE:.10f} {y*LENGTH_SCALE:.10f} {z*LENGTH_SCALE:.10f}\n")

        f.write(f"CELLS {n_cells} {3*n_cells}\n")
        for (a, b) in cells_conn:
            f.write(f"2 {a} {b}\n")

        f.write(f"CELL_TYPES {n_cells}\n")
        f.write(("3\n") * n_cells)

        f.write(f"CELL_DATA {n_cells}\n")
        f.write("SCALARS vessel_id int 1\nLOOKUP_TABLE default\n")
        for v in vessel_id:
            f.write(f"{v}\n")
        for name, arr in [("a0", a0_list), ("a_d", a_d_list), ("E", E_list),
                           ("h_wall", h_list), ("p0", p0_list), ("p_d", pd_list),
                           ("L", L_list), ("r_d", rd_list)]:
            f.write(f"SCALARS {name} double 1\nLOOKUP_TABLE default\n")
            for v in arr:
                f.write(f"{v}\n")

        f.write(f"POINT_DATA {n_pts}\n")
        f.write("SCALARS boundary_id int 1\nLOOKUP_TABLE default\n")
        for b in bvals:
            f.write(f"{b}\n")
        is_terminal = [b not in (0, JUNCTION_ID) for b in bvals]
        for name, val in [("R1", R1), ("R2", R2), ("C", C), ("P_out", P_OUT)]:
            f.write(f"SCALARS {name} double 1\nLOOKUP_TABLE default\n")
            for term in is_terminal:
                f.write(f"{val if term else 0.0}\n")

write_vtk(OUT_VTK)
print("wrote", OUT_VTK)


# ## 8. Quick sanity check
# 
# Re-parse the written file to confirm it loads cleanly and the field names/shapes match the reference format.

# In[9]:

try:
    import meshio
    m = meshio.read(OUT_VTK)
    print(m)
    print("point_data:", list(m.point_data.keys()))
    print("cell_data:", list(m.cell_data.keys()))
except ImportError:
    print("(optional) install 'meshio' to auto-validate the written VTK file")