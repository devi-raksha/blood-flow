#!/usr/bin/env python3
"""
Publication figure for the global mass-conservation check written by
BloodFlowSystem::check_mass_conservation()  ->  mass_conservation.csv

Columns expected (SI):  t, total_volume, dVdt, net_flux, imbalance, rel_imbalance
Optional columns (if you add them to the C++):  Q_in, Q_out

Two single-column figures, each as PDF + PNG:
  <out>_flux  : net boundary flux and dV/dt [mL/s] - they must coincide
      (inflow and outflow separately if Q_in is logged or --q-in is given)
  <out>_error : relative mass error (log scale):
        - instantaneous: the code's rel_imbalance = |dV/dt - (Q_in - sum Q_out)| / max|Q_net|
        - cumulative:    |V(t) - V(0) - int_0^t (Q_in - sum Q_out) dt| / V(t)
          (trapezoidal integral; free of the first-order error of the
           finite-difference dV/dt, so it measures the scheme itself)

Usage (nothing else needed if mass_conservation.csv is in the current folder):
  python3 plot_mass_conservation.py
  python3 plot_mass_conservation.py path/to/mass_conservation.csv --out fig_mass
  python3 plot_mass_conservation.py --units cgs          # for a CGS run
If the file holds several runs (it is opened in append mode), the last run is used.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# reference categorical palette (fixed order) + text inks
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_last_run(path):
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    hdr_idx = [i for i, l in enumerate(lines) if l[0].isalpha()]
    if not hdr_idx:
        raise SystemExit("no header line found in " + path)
    start = hdr_idx[-1]
    header = lines[start].split(",")
    rows = [list(map(float, l.split(","))) for l in lines[start + 1:]]
    data = np.array(rows)
    if len(hdr_idx) > 1:
        print(f"note: {len(hdr_idx)} runs found in file, using the last one ({len(rows)} rows)")
    # restart detection without a repeated header: time going backwards
    back = np.where(np.diff(data[:, header.index("t")]) <= 0)[0]
    if len(back):
        data = data[back[-1] + 1:]
        print(f"note: time restarts inside the file, using the last {len(data)} rows")
    return {h: data[:, k] for k, h in enumerate(header)}


def cumtrapz(y, x):
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", default="mass_conservation.csv",
                    help="CSV written by the solver (default: ./mass_conservation.csv)")
    ap.add_argument("--q-in", default=None,
                    help="OPTIONAL inflow Q_in(t) as a numpy expression of t, to also "
                         "plot inflow and outflow separately (not needed)")
    ap.add_argument("--units", choices=["si", "cgs"], default="si",
                    help="units of the simulation (si: m^3/s, cgs: cm^3/s)")
    ap.add_argument("--out", default="fig_mass_conservation")
    ap.add_argument("--t-min", type=float, default=0.0, help="hide t < t_min in the plots")
    ap.add_argument("--width", type=float, default=3.5, help="figure width in inches")
    a = ap.parse_args()

    d = load_last_run(a.csv)
    t, V, dVdt, net = d["t"], d["total_volume"], d["dVdt"], d["net_flux"]

    if "Q_in" in d:
        Qin = d["Q_in"]
    elif a.q_in is not None:
        Qin = np.broadcast_to(eval(a.q_in, {"np": np, "t": t}), t.shape).astype(float)
    else:
        Qin = None                                 # only the CSV columns are used
    Qout = (d["Q_out"] if "Q_out" in d else Qin - net) if Qin is not None else None

    # cumulative (integral) balance, independent of the finite-difference dV/dt,
    # relative to the total blood volume in the network
    cum_err = np.abs((V - V[0]) - cumtrapz(net, t))
    cum_rel = cum_err / V
    inst_rel = d["rel_imbalance"]

    m = t >= a.t_min
    m_inst = m & (np.arange(len(t)) > 0)          # first row has dV/dt = 0 by construction
    to_mLs = 1e6 if a.units == "si" else 1.0       # -> mL/s (= cm^3/s)

    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "mathtext.fontset": "cm", "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "axes.linewidth": 0.6, "lines.linewidth": 1.4,
    })
    figsize = (a.width, a.width * 0.68)
    leg_kw = dict(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                  ncol=2, borderaxespad=0.2, handlelength=2.2)

    def finish(fig, ax, name):
        ax.set_xlabel("Time [s]")
        ax.grid(True, color=GRID, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(**leg_kw)
        fig.savefig(name + ".pdf", bbox_inches="tight")
        fig.savefig(name + ".png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {name}.pdf and {name}.png")

    # ---- figure 1: volume balance ------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    if Qin is not None:
        ax.plot(t[m], Qin[m] * to_mLs, color=BLUE, label=r"$Q_\mathrm{in}$")
        ax.plot(t[m], Qout[m] * to_mLs, color=ORANGE, label=r"$\sum_k Q_{\mathrm{out},k}$")
        ax.plot(t[m_inst], dVdt[m_inst] * to_mLs, color=AQUA,
                label=r"$\mathrm{d}V/\mathrm{d}t$")
        leg_kw["ncol"] = 3
    else:
        ax.plot(t[m], net[m] * to_mLs, color=BLUE,
                label=r"net flux $Q_\mathrm{in}-\sum_k Q_{\mathrm{out},k}$")
        ax.plot(t[m_inst], dVdt[m_inst] * to_mLs, color=ORANGE, linestyle=(0, (4, 2)),
                label=r"$\mathrm{d}V/\mathrm{d}t$")
    ax.set_ylabel("Flow rate [mL/s]")
    finish(fig, ax, a.out + "_flux")
    leg_kw["ncol"] = 2

    # ---- figure 2: relative mass error -------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    ax.semilogy(t[m_inst], np.maximum(inst_rel[m_inst], 1e-16), color=BLUE, linewidth=1.0,
                label=r"instantaneous $\varepsilon_\mathrm{inst}$")
    ax.semilogy(t[m_inst], np.maximum(cum_rel[m_inst], 1e-16), color=ORANGE,
                label=r"cumulative $\varepsilon_\mathrm{cum}$")
    ax.set_ylabel("Relative mass error [-]")
    finish(fig, ax, a.out + "_error")

    late = m_inst & (t >= t[-1] - 0.25 * (t[-1] - t[0]))
    print(f"rows {len(t)}, t = {t[0]:.4g} .. {t[-1]:.4g} s")
    if Qin is not None:
        print(f"final: Q_in {Qin[-1]:.6e}, sum Q_out {Qout[-1]:.6e}, "
              f"outflow/inflow = {Qout[-1] / Qin[-1]:.6f}")
    print(f"final net flux {net[-1]:.6e} (peak {np.max(np.abs(net)):.6e}), "
          f"network volume {V[-1]:.6e}")
    print(f"max instantaneous error (all t>t_min): {np.nanmax(inst_rel[m_inst]):.3e}; "
          f"last quarter of run: {np.nanmax(inst_rel[late]):.3e}")
    print(f"cumulative error at end: {cum_rel[-1]:.3e} of network volume "
          f"(absolute {cum_err[-1]:.3e}); max over run {np.max(cum_rel[m_inst]):.3e}")


if __name__ == "__main__":
    main()