from pathlib import Path
import numpy as np
import pyvista as pv
from pyvista import CellType

# -----------------------------
# Geometry
# -----------------------------
points = np.array(
    [
        [0.0, 0.0, 0.0],   # inlet
        [1.0, 0.0, 0.0],   # junction
        [2.0, 0.8, 0.0],   # outlet 1
        [2.0, -0.8, 0.0],  # outlet 2
    ],
    dtype=float,
)

cells = np.array(
    [
        2, 0, 1,   # vessel 0
        2, 1, 2,   # vessel 1
        2, 1, 3,   # vessel 2
    ],
    dtype=np.int64,
)

celltypes = np.array([CellType.LINE]*3, dtype=np.uint8)
network = pv.UnstructuredGrid(cells, celltypes, points)

# -----------------------------
# Vessel IDs
# -----------------------------
network.cell_data["vessel_id"] = np.array([0, 1, 2])

# -----------------------------
# Vessel parameters (CELL DATA)
# -----------------------------
network.cell_data["a0"] = np.array([3.0605e-4, 9.4787e-5, 9.4787e-5])
network.cell_data["a_d"] = np.array([2.3235e-4, 1.1310e-4, 1.1310e-4])
network.cell_data["E"] = np.array([5e5, 7e5, 7e5])
network.cell_data["h_wall"] = np.array([1.032e-3, 0.72e-3, 0.72e-3])
network.cell_data["p0"] = np.array([0.0, 0.0, 0.0])
network.cell_data["p_d"] = np.array([9460.0, 9460.0, 9460.0])
network.cell_data["L"] = np.array([1.0, 1.0, 1.0])

#  derived radius 
network.cell_data["r_d"] = np.sqrt(network.cell_data["a_d"] / np.pi)

# -----------------------------
# Boundary IDs (POINT DATA)
# -----------------------------
boundary_id = np.array([
    0,  # inlet
    1,  # junction (internal)
    2,  # Boundary 1
    3   # Boundary 2
])
network.point_data["boundary_id"] = boundary_id

# -----------------------------
# Windkessel parameters (POINT DATA)
# -----------------------------
# Default zero everywhere
R1 = np.zeros(4)
R2 = np.zeros(4)
C  = np.zeros(4)
P_out = np.zeros(4)

# Apply only to terminal points
R1[2] = 6.8123e7
R2[2] = 3.1013e9
C[2]  = 3.6664e-10
P_out[2] = 0.0

R1[3] = 6.8123e7
R2[3] = 3.1013e9
C[3]  = 3.6664e-10
P_out[3] = 0.0

network.point_data["R1"] = R1
network.point_data["R2"] = R2
network.point_data["C"] = C
network.point_data["P_out"] = P_out

network.field_data.clear()
# -----------------------------
# Save VTK
# -----------------------------
output_path = Path("bifurcation_physics.vtk")
network.save(output_path, binary=False)

print(f"Saved VTK file to: {output_path.resolve()}")
print(network)
print(f"Number of cells in mesh: {network.n_cells}")

