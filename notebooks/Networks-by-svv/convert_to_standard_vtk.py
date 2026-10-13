"""
Convert an svVascularize .tree(.npz) network into a legacy ASCII VTK file

Geometry, length and radius come straight from svVascularize.
Everything else is derived with standard 1-D hemodynamics relations so the
file carries the same fields as the reference:

    CELL_DATA  : vessel_id, a0, a_d, E, h_wall, p_d, p0, L, r_d
    POINT_DATA : boundary_id, R1, R2, C, P_out

Only the physiological constants in the CONFIG block below are assumptions;
the geometry, L, r_d, areas and IDs are exact.

Outlet model:
    USE_RCR = True  -> RCR Windkessel (R1 = R1_FRACTION*R_total, R2 = rest, C = TAU/R_total)
    USE_RCR = False -> single resistor (R1 = 0, R2 = R_total, C = 0)
"""

import numpy as np
from svv.tree.tree import Tree
import matplotlib.pyplot as plt

# ==================================================================
# Files
# ==================================================================

input_file  = "./trees/network_50000.tree.npz"
output_file = "./trees/network_100k_adnr.vtk"
dataset_title = "svVascularize vessel network"


# ==================================================================
# CONFIG  -  physiological model parameters
# ==================================================================

# --- wall material -------------------------------------------------
E_YOUNG = 2.25e4          # Young's modulus  [Pa]   -> field "E"
P_D     = 0.0        # diastolic pressure [Pa] -> field "p_d"
P_0     = 0.0             # reference pressure [Pa] -> field "p0"

# --- wall-thickness law:  E*h/r = k1*exp(k2*r) + k3 ----------------
#     => h = (r / E) * (k1*exp(k2*r) + k3)
WALL_K1 = 6.538312e4
WALL_K2 = -4.615404e2
WALL_K3 = 2.632577e4

# --- outlet boundary model ----------------------------------------
#     R_total = DELTA_P / Q          (Q = svVascularize terminal flow)
#
#     DELTA_P is the driving/perfusion pressure used ONLY to scale the
#     outlet resistance. It is kept independent of P_D / P_0 on purpose,
#     so setting p_d = 0 and p0 = 0 does NOT force the resistances to 0.
USE_RCR     = False      # False -> single resistor (R1 = 0, C = 0)
DELTA_P     = 1.25e5      # perfusion pressure across the outlet bed [Pa]
R1_FRACTION = 0.2          # only used when USE_RCR is True
TAU         = 0.283        # RC time constant [s]  (only used when USE_RCR)
P_OUT       = 0.0          # outlet pressure  [Pa] -> field "P_out"

# Diagnostic only: the steady inflow you drive in the solver, so the script
# can report the mean pressure drop this network will actually produce.
# Set this to match your parameter-file "Inflow function" plateau.
REFERENCE_INFLOW = 1.0e-3  # [m^3/s]

# ##########

# R_PARALLEL     = 4.0e4                      # from your diagnostics [Pa.s/m^3]
# DELTA_P_TARGET = 100.0 * 133.322           # peak pressure drop you want [Pa] (100 mmHg here)
# Q_PEAK         = DELTA_P_TARGET / R_PARALLEL   # -> ~0.333 m^3/s

# T_PEAK = 0.2      # where the pulse peaks
# ALPHA  = 4.0      # sharpness: larger = narrower pulse
# Q_BASE = 1e-4     # set >0 (e.g. 0.05*Q_PEAK) if you want a nonzero diastolic baseline

# def inflow(t):
#     x = t / T_PEAK
#     shape = (x**ALPHA) * np.exp(ALPHA * (1.0 - x))   # = 1 at t=T_PEAK, 0 at t=0
#     return Q_BASE + (Q_PEAK - Q_BASE) * shape

# # Print the expression
# print(
#     f"Q(t) = {Q_BASE:.6f} + "
#     f"({Q_PEAK:.6f} - {Q_BASE:.6f}) * "
#     f"(t/{T_PEAK:.3f})^{ALPHA:.1f} * "
#     f"exp({ALPHA:.1f} * (1 - t/{T_PEAK:.3f}))"
# )

# # Time points for plotting
# t_values = np.linspace(0, 1.0, 500)

# # Compute Q(t)
# Q_values = inflow(t_values)

# # Plot
# plt.figure(figsize=(8, 5))
# plt.plot(t_values, Q_values, linewidth=2)

# plt.xlabel("Time t")
# plt.ylabel("Inflow Q(t)")
# plt.title("Inflow profile Q(t)")
# plt.grid(True)

# plt.show()
# boundary_id codes
#   root      -> 0
#   outlets   -> 1, 2, 3, ...  (one per terminal, in node-id order)
#   junctions -> 255           (matches 56_adnr_new.vtk convention; not a
#                                real outlet, so it must sit outside the
#                                1..n_outlets range used above)
BID_ROOT     = 0
BID_INTERIOR = 300655

# ==================================================================
# svVascularize data-column layout
# ==================================================================

COL_PROXIMAL      = slice(0, 3)
COL_DISTAL        = slice(3, 6)
COL_PARENT        = 17
COL_PROXIMAL_NODE = 18
COL_DISTAL_NODE   = 19
COL_LENGTH        = 20
COL_RADIUS        = 21
COL_FLOW          = 22


# ==================================================================
# Load tree
# ==================================================================

tree = Tree.load(input_file)
data = np.asarray(tree.data)

print("Tree loaded successfully!")
print("Number of vessel segments:", tree.segment_count)
print("Data shape:", data.shape)

proximal      = data[:, COL_PROXIMAL]
distal        = data[:, COL_DISTAL]
proximal_node = data[:, COL_PROXIMAL_NODE].astype(int)
distal_node   = data[:, COL_DISTAL_NODE].astype(int)
length        = data[:, COL_LENGTH].astype(float)
radius        = data[:, COL_RADIUS].astype(float)
flow          = np.abs(data[:, COL_FLOW].astype(float))

n_vessels = len(data)


# ==================================================================
# Unique VTK points (vessels sharing a junction share a point)
# ==================================================================

node_ids     = np.concatenate([proximal_node, distal_node])
unique_nodes = np.unique(node_ids)
node_to_point = {node_id: i for i, node_id in enumerate(unique_nodes)}

node_coordinates = {}
for i in range(n_vessels):
    node_coordinates[int(proximal_node[i])] = proximal[i]
    node_coordinates[int(distal_node[i])]   = distal[i]

points = np.array([node_coordinates[n] for n in unique_nodes])
n_points = len(points)

# line cells, remapped to point indices
cells = np.array([
    (node_to_point[int(proximal_node[i])], node_to_point[int(distal_node[i])])
    for i in range(n_vessels)
])


# ==================================================================
# CELL_DATA  (per vessel)
# ==================================================================

vessel_id = np.arange(n_vessels, dtype=float)

r_d = radius.copy()
L   = length.copy()

# a0 = a_d = pi * r_d^2
a_d = np.pi * r_d ** 2
a0  = a_d.copy()

E   = np.full(n_vessels, E_YOUNG)
p_d = np.full(n_vessels, P_D)
p0  = np.full(n_vessels, P_0)

# wall thickness:  h = (r/E) * (k1*exp(k2*r) + k3)
h_wall = (r_d / E_YOUNG) * (WALL_K1 * np.exp(WALL_K2 * r_d) + WALL_K3)


# ==================================================================
# POINT_DATA  (per node)  -  boundary_id + outlet model
# ==================================================================

# classify nodes by topology
is_proximal = np.isin(unique_nodes, proximal_node)   # acts as a parent
is_distal   = np.isin(unique_nodes, distal_node)     # acts as a child

boundary_id = np.full(n_points, BID_INTERIOR, dtype=int)   # interior by default

# root: never appears as a child
boundary_id[~is_distal] = BID_ROOT

# outlets: never appear as a parent -> number them 1, 2, 3, ...
outlet_point_idx = np.where(~is_proximal)[0]
outlet_point_idx = outlet_point_idx[np.argsort(unique_nodes[outlet_point_idx])]

n_outlets = len(outlet_point_idx)
if n_outlets >= BID_INTERIOR:
    raise ValueError(
        f"{n_outlets} outlets found, but BID_INTERIOR={BID_INTERIOR} would "
        f"collide with an outlet id. Raise BID_INTERIOR above {n_outlets}."
    )

for k, pidx in enumerate(outlet_point_idx, start=1):
    boundary_id[pidx] = k

# arrays are zero everywhere; only outlet nodes get filled
R1    = np.zeros(n_points)
R2    = np.zeros(n_points)
C     = np.zeros(n_points)
P_out = np.full(n_points, P_OUT)

# map each outlet node -> the vessel that terminates there (distal node)
vessel_by_distal = {int(distal_node[i]): i for i in range(n_vessels)}

skipped_outlets = []

for pidx in outlet_point_idx:
    node = int(unique_nodes[pidx])
    vi   = vessel_by_distal.get(node)
    if vi is None:
        skipped_outlets.append(node)
        continue
    q = flow[vi]
    if q <= 0.0 or not np.isfinite(q):
        skipped_outlets.append(node)
        continue
    R_total = DELTA_P / q

    if USE_RCR:
        # full RCR Windkessel
        R1[pidx] = R1_FRACTION * R_total
        R2[pidx] = R_total - R1[pidx]
        C[pidx]  = TAU / R_total
    else:
        # single-resistor model: all resistance in R2, R1 = 0, C = 0
        R1[pidx] = 0.0
        R2[pidx] = R_total
        C[pidx]  = 0.0

if skipped_outlets:
    print(
        f"WARNING: {len(skipped_outlets)} outlet node(s) had zero/invalid "
        f"flow and were left with R1=R2=C=0 (not tagged as RCR/pressure-"
        f"capacitor DOFs): {skipped_outlets}"
    )

# Sanity check: junctions and root must never carry R1/C (RCR-DOF markers)
non_outlet_mask = np.ones(n_points, dtype=bool)
non_outlet_mask[outlet_point_idx] = False
assert np.all(R1[non_outlet_mask] == 0.0) and np.all(C[non_outlet_mask] == 0.0), \
    "Internal error: a non-outlet node was assigned R1/C > 0."


# ==================================================================
# Write legacy ASCII VTK  (byte-format matched to 56_adnr_new.vtk)
# ==================================================================

def write_scalar(f, name, values, fmt):
    f.write(f"SCALARS {name} double 1\n")
    f.write("LOOKUP_TABLE default\n")
    for v in values:
        f.write(fmt.format(v) + "\n")

with open(output_file, "w") as f:

    # ---- header ----
    f.write("# vtk DataFile Version 3.0\n")
    f.write(f"{dataset_title}\n")
    f.write("ASCII\n")
    f.write("DATASET UNSTRUCTURED_GRID\n")

    # ---- points ----
    f.write(f"POINTS {n_points} double\n")
    for p in points:
        f.write(f"{p[0]:.12e} {p[1]:.12e} {p[2]:.12e}\n")

    # ---- cells ----
    f.write(f"CELLS {n_vessels} {n_vessels * 3}\n")
    for a, b in cells:
        f.write(f"2 {a} {b}\n")

    # ---- cell types (VTK_LINE = 3) ----
    f.write(f"CELL_TYPES {n_vessels}\n")
    for _ in range(n_vessels):
        f.write("3\n")

    # ---- cell data ----
    f.write(f"CELL_DATA {n_vessels}\n")
    write_scalar(f, "vessel_id", vessel_id, "{:.15e}")
    write_scalar(f, "a0",        a0,        "{:.15e}")
    write_scalar(f, "a_d",       a_d,       "{:.6e}")
    write_scalar(f, "E",         E,         "{:.15e}")
    write_scalar(f, "h_wall",    h_wall,    "{:.6e}")
    write_scalar(f, "p_d",       p_d,       "{:.15e}")
    write_scalar(f, "p0",        p0,        "{:.15e}")
    write_scalar(f, "L",         L,         "{:.15e}")
    write_scalar(f, "r_d",       r_d,       "{:.15e}")

    # ---- point data ----
    f.write(f"POINT_DATA {n_points}\n")
    f.write("SCALARS boundary_id double 1\n")
    f.write("LOOKUP_TABLE default\n")
    for v in boundary_id:
        f.write(f"{int(v)}\n")
    write_scalar(f, "R1",    R1,    "{:.12e}")
    write_scalar(f, "R2",    R2,    "{:.12e}")
    write_scalar(f, "C",     C,     "{:.12e}")
    write_scalar(f, "P_out", P_out, "{:.12e}")


print()
print("VTK file created successfully!")
print(output_file)
print(f"  points   : {n_points}")
print(f"  vessels  : {n_vessels}")
print(f"  outlets  : {len(outlet_point_idx)}")
print(f"  outlet model : {'RCR' if USE_RCR else 'single resistor (R1=0, C=0)'}")

# ==================================================================
# Pressure-drop diagnostics
#   The steady spatial pressure drop the network will produce is
#       dP_mean = Q_in * R_parallel = DELTA_P * (Q_in / Q_design)
#   where Q_design is the tree's total terminal (design) flow.
# ==================================================================
terminal_flows = np.array([
    flow[vessel_by_distal[int(unique_nodes[p])]]
    for p in outlet_point_idx
    if int(unique_nodes[p]) in vessel_by_distal
])
q_design = float(terminal_flows.sum())

outlet_R_total = R1[outlet_point_idx] + R2[outlet_point_idx]
valid_R = outlet_R_total[outlet_R_total > 0]
R_parallel = 1.0 / np.sum(1.0 / valid_R) if valid_R.size else float("inf")

print()
print("  --- pressure diagnostics -------------------------------")
print(f"  total design flow  Q_design   : {q_design:.3e} m^3/s")
print(f"  network resistance R_parallel : {R_parallel:.3e} Pa.s/m^3")
if valid_R.size:
    print(f"  outlet R_total range          : "
          f"{valid_R.min():.3e} .. {valid_R.max():.3e} Pa.s/m^3")
    dp_design = q_design * R_parallel
    dp_actual = REFERENCE_INFLOW * R_parallel
    print(f"  dP at design flow  ({q_design:.2e}) : {dp_design:.3e} Pa "
          f"({dp_design/133.322:.1f} mmHg)")
    print(f"  dP at your inflow  ({REFERENCE_INFLOW:.2e}) : {dp_actual:.3e} Pa "
          f"({dp_actual/133.322:.1f} mmHg)")
    if q_design > 0 and REFERENCE_INFLOW < 0.1 * q_design:
        print("  NOTE: your inflow << design flow -> small pressure drop. "
              "Raise the inflow toward Q_design or raise DELTA_P.")
    if not USE_RCR:
        print("  NOTE: USE_RCR is False -> R1=C=0 at every outlet. If your "
              "solver picks RCR/PC-DOFs by (R1>0 and C>0), no outlet will "
              "qualify. Set USE_RCR=True for an RCR run.")
print("  --------------------------------------------------------")