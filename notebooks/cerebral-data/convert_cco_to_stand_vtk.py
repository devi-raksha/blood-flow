#!/usr/bin/env python3
"""
Convert a CCO arteriolar tree (Blanco et al. 2017, Stroke Vasc Neurol 2:108-117)
stored as VTK POLYDATA with per-segment FIELD data
    scaledRadius [cm], length [cm], lod [-], pressure [mmHg], flow [cm^3/s],
    area [cm^2], vel [cm/s]
into the "37-vessel network" VTK layout read by BloodFlowSystem::create_triangulation:

  UNSTRUCTURED_GRID, one line cell (type 3) per vessel, SI units
  CELL_DATA : vessel_id a0 E h_wall p_d p0 L a_d r_d
  POINT_DATA: boundary_id R1 R2 C P_out

Modelling choices (all overridable on the command line):
  * Geometry and lengths in m (default) or cm with --units cgs.  vessel_id == cell index (the solver
    indexes the per-vessel arrays by vessel_id).
  * Reference (diastolic) state = the CCO mean state of each segment:
        a_d = a0 = CCO area,  r_d = sqrt(a_d/pi),  p_d = CCO mean pressure.
    With A = a_d the tube law gives P = p_d, i.e. the initial condition is the
    CCO pressure field (at zero velocity).
  * Wall stiffness: the CCO data obey sqrt(A/(pi r^2)) = s0 + k P (fitted), which
    in the tube law P = p_d + beta/A_d (sqrt(A) - sqrt(A_d)) means
        beta = A_d / (k sqrt(pi) r_cco),   beta = 4/3 sqrt(pi) E h.
    h is taken from Avolio's formula (as in ADAN), E = 3 beta / (4 sqrt(pi) h).
    Only the product E*h enters the tube law.
  * Terminals: purely resistive (paper), R1 = 0, C = 0, R2 = (P_outlet - P_out)/Q.
    CCO pressures are segment *midpoint* values (verified from the data), so the
    outlet pressure is P_mid - 0.5 * R_seg * Q with R_seg from Poiseuille + Pries
    viscosity (plasma viscosity fitted to the CCO pressure drops).
  * Boundary ids: 0 = inlet, terminals 1..N (255 skipped), 255 = interior point.
"""
import argparse
import csv
import numpy as np

MMHG = 133.322387415  # Pa


def read_cco(fn):
    t = open(fn).read().split()
    k = t.index("POINTS")
    n = int(t[k + 1])
    pts = np.array(t[k + 3:k + 3 + 3 * n], float).reshape(n, 3)
    k = t.index("LINES")
    nl = int(t[k + 1])
    lines = np.array(t[k + 3:k + 3 + 3 * nl], int).reshape(nl, 3)
    assert np.all(lines[:, 0] == 2), "only 2-point line cells supported"
    cells = lines[:, 1:]
    k = t.index("FIELD")
    nf = int(t[k + 2])
    k += 3
    data = {}
    for _ in range(nf):
        name, nc, nt = t[k], int(t[k + 1]), int(t[k + 2])
        k += 4
        data[name] = np.array(t[k:k + nc * nt], float)
        k += nc * nt
    return pts, cells, data


def pries_rel_visc(r_cm):
    """Pries et al. (1992) in-vitro relative viscosity at Hd = 0.45."""
    D = 2.0 * r_cm * 1e4  # um
    mu45 = 220 * np.exp(-1.3 * D) + 3.2 - 2.44 * np.exp(-0.06 * D ** 0.645)
    f = (D / (D - 1.1)) ** 2
    return (1 + (mu45 - 1) * f) * f


def avolio_h_cm(r_cm):
    """Wall thickness h = r (a e^{b r} + c e^{d r}), r in cm (Avolio 1980)."""
    return r_cm * (0.2802 * np.exp(-5.053 * r_cm) + 0.1324 * np.exp(-0.1114 * r_cm))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cco_vtk")
    ap.add_argument("out_vtk")
    ap.add_argument("--p-out", type=float, default=0.0,
                    help="pressure downstream of terminal resistances [mmHg] (default 0)")
    ap.add_argument("--terminal-pressure", choices=["outlet", "midpoint"], default="outlet",
                    help="pressure used for R2: extrapolated outlet (default) or the CCO "
                         "segment value as is")
    ap.add_argument("--terminal-csv", default=None, help="write terminal table here")
    ap.add_argument("--title", default="CCO parietal network")
    ap.add_argument("--units", choices=["si", "cgs"], default="si",
                    help="units of the written VTK: si (m, m^2, Pa, Pa s/m^3) or "
                         "cgs (cm, cm^2, dyn/cm^2, dyn s/cm^5)")
    a = ap.parse_args()

    pts, cells, d = read_cco(a.cco_vtk)
    npts, nseg = len(pts), len(cells)
    r = d["scaledRadius"]; L = d["length"]; P = d["pressure"]; Q = d["flow"]; A = d["area"]

    # ---- topology --------------------------------------------------------------
    seg_of_distal = {cells[i, 1]: i for i in range(nseg)}
    parent = np.array([seg_of_distal.get(cells[i, 0], -1) for i in range(nseg)])
    outdeg = np.bincount(cells[:, 0], minlength=npts)
    indeg = np.bincount(cells[:, 1], minlength=npts)
    roots = np.where(parent < 0)[0]
    assert len(roots) == 1, "expected a single root segment"
    root = roots[0]
    inlet_pt = cells[root, 0]
    terminal_segs = np.where(outdeg[cells[:, 1]] == 0)[0]

    # ---- stiffness from the CCO pressure-area relation -------------------------
    s = np.sqrt(A / (np.pi * r ** 2))
    k_mmHg, s0 = np.polyfit(P, s, 1)           # s = s0 + k P
    k_Pa = k_mmHg / MMHG
    r_m = r * 1e-2
    a_d = A * 1e-4                              # m^2
    r_d = np.sqrt(a_d / np.pi)
    beta = a_d / (k_Pa * np.sqrt(np.pi) * r_m)  # Pa m
    h = avolio_h_cm(r) * 1e-2                   # m (based on CCO radius)
    E = 3.0 * beta / (4.0 * np.sqrt(np.pi) * h)

    # ---- plasma viscosity fitted to CCO midpoint pressure drops ----------------
    i = np.where(parent >= 0)[0]; j = parent[i]
    Rg = lambda idx: 8 * pries_rel_visc(r[idx]) * L[idx] / (np.pi * r[idx] ** 4)  # cgs / mu
    mu_plasma = np.median((P[j] - P[i]) * MMHG * 10 /
                          (0.5 * Rg(i) * Q[i] + 0.5 * Rg(j) * Q[j]))  # Poise

    # ---- terminal resistances --------------------------------------------------
    R_seg_cgs = mu_plasma * Rg(terminal_segs)                    # dyn s / cm^5
    P_mid_Pa = P[terminal_segs] * MMHG
    if a.terminal_pressure == "outlet":
        P_term_Pa = P_mid_Pa - 0.5 * R_seg_cgs * Q[terminal_segs] * 0.1  # dyn/cm^2 -> Pa
    else:
        P_term_Pa = P_mid_Pa
    Q_term_SI = Q[terminal_segs] * 1e-6
    P_out_Pa = a.p_out * MMHG
    R2_term = (P_term_Pa - P_out_Pa) / Q_term_SI
    assert np.all(R2_term > 0)

    # ---- point data ------------------------------------------------------------
    bid = np.full(npts, 255, dtype=int)
    bid[inlet_pt] = 0
    R1 = np.zeros(npts); R2 = np.zeros(npts); C = np.zeros(npts)
    Pout = np.full(npts, P_out_Pa)
    next_id = 1
    term_ids = []
    for n_, seg in enumerate(terminal_segs):
        if next_id == 255:
            next_id += 1
        v = cells[seg, 1]
        bid[v] = next_id
        R2[v] = R2_term[n_]
        term_ids.append(next_id)
        next_id += 1

    # ---- unit scaling (everything above is SI) ---------------------------------
    if a.units == "cgs":
        fL, fA, fP, fR, fQ = 1e2, 1e4, 10.0, 1e-5, 1e6
        uL, uA, uP, uR, uQ = "cm", "cm2", "dyncm2", "dyn_s_per_cm5", "cm3ps"
    else:
        fL, fA, fP, fR, fQ = 1.0, 1.0, 1.0, 1.0, 1.0
        uL, uA, uP, uR, uQ = "m", "m2", "Pa", "Pa_s_per_m3", "m3ps"

    # ---- write -----------------------------------------------------------------
    def block(f, name, vals, fmt, typ="double"):
        f.write(f"SCALARS {name} {typ} 1\nLOOKUP_TABLE default\n")
        f.write("\n".join(fmt % v for v in vals) + "\n")

    with open(a.out_vtk, "w") as f:
        f.write("# vtk DataFile Version 3.0\n%s\nASCII\nDATASET UNSTRUCTURED_GRID\n" % a.title)
        f.write(f"POINTS {npts} double\n")
        for p in pts * 1e-2 * fL:
            f.write("%.9e %.9e %.9e\n" % tuple(p))
        f.write(f"CELLS {nseg} {3 * nseg}\n")
        for c in cells:
            f.write(f"2 {c[0]} {c[1]}\n")
        f.write(f"CELL_TYPES {nseg}\n" + "3\n" * nseg)
        f.write(f"CELL_DATA {nseg}\n")
        block(f, "vessel_id", range(nseg), "%d", "int")
        block(f, "a0", a_d * fA, "%.9e")
        block(f, "E", E * fP, "%.9e")
        block(f, "h_wall", h * fL, "%.9e")
        block(f, "p_d", P * MMHG * fP, "%.9e")
        block(f, "p0", np.zeros(nseg), "%.1f")
        block(f, "L", L * 1e-2 * fL, "%.9e")
        block(f, "a_d", a_d * fA, "%.9e")
        block(f, "r_d", r_d * fL, "%.9e")
        f.write(f"POINT_DATA {npts}\n")
        block(f, "boundary_id", bid, "%d", "int")
        block(f, "R1", R1, "%.9e")
        block(f, "R2", R2 * fR, "%.9e")
        block(f, "C", C, "%.9e")
        block(f, "P_out", Pout * fP, "%.9e")

    if a.terminal_csv:
        with open(a.terminal_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["boundary_id", "point", "vessel_id", "Q_cco_" + uQ, "P_mid_cco_" + uP,
                        "P_outlet_" + uP, "R2_" + uR])
            for n_, seg in enumerate(terminal_segs):
                w.writerow([term_ids[n_], cells[seg, 1], seg, "%.9e" % (Q_term_SI[n_] * fQ),
                            "%.6f" % (P_mid_Pa[n_] * fP), "%.6f" % (P_term_Pa[n_] * fP),
                            "%.9e" % (R2_term[n_] * fR)])

    print(f"points {npts}, vessels {nseg}, terminals {len(terminal_segs)}")
    print(f"inlet point {inlet_pt}, root vessel {root}")
    print(f"tube-law fit: sqrt(A/pi r^2) = {s0:.5f} + {k_mmHg:.6f} P[mmHg]  ->  "
          f"E*h/r = {np.median(E * h / r_m):.4e} Pa")
    print(f"h/r range {np.min(h / r_m):.3f}-{np.max(h / r_m):.3f},  "
          f"E range {E.min():.4e}-{E.max():.4e} Pa")
    print(f"fitted plasma viscosity {mu_plasma * 0.1 * 1e3:.3f} mPa s")
    print(f"inlet flow (root vessel)     {Q[root] * 1e-6:.9e} m^3/s")
    print(f"sum of terminal flows        {Q[terminal_segs].sum() * 1e-6:.9e} m^3/s")
    print(f"R2 range {R2_term.min():.4e}-{R2_term.max():.4e} Pa s/m^3, P_out {P_out_Pa} Pa")


if __name__ == "__main__":
    main()