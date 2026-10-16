"""
Plot IDA (SUNDIALS) step-history CSV in the same pgfplots-like style as
aorta_mid_pressure.pdf: Computer Modern fonts, small square-ish axes,
centred title, thick black frame, light major grid.

Produces
  Single-axis plots
    ida_step_size          IDA internal time step h (log)
    ida_cumulative         cumulative steps & Newton iterations
    ida_nonlin_per_step    Newton iterations per step
  Dual-axis plots (left / right y-axis)
    ida_dual_h_cumsteps    time step h  |  cumulative steps
    ida_dual_h_newton      time step h  |  Newton iterations per step
  Overview
    ida_overview           the three single-axis plots side by side

Usage:
    python plot_ida_history.py ida_step_history.csv [output_dir]
"""
import io
import sys
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator


# --------------------------------------------------------------------------
# Style (mimics pgfplots defaults used in the reference figure)
# --------------------------------------------------------------------------
def _latex_works():
    """True if matplotlib can render with a full LaTeX install (needs cm-super)."""
    if shutil.which("latex") is None:
        return False
    try:
        with plt.rc_context({"text.usetex": True}):
            f = plt.figure()
            f.text(0.5, 0.5, r"$x$ Test")
            f.savefig(io.BytesIO(), format="pdf")
            plt.close(f)
        return True
    except Exception:
        plt.close("all")
        return False


USE_TEX = _latex_works()

plt.rcParams.update({
    "text.usetex": USE_TEX,
    "font.family": "serif",
    # With LaTeX: real Computer Modern. Without: matplotlib's bundled cmr10.
    "font.serif": ["Computer Modern Roman"] if USE_TEX else ["cmr10"],
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "mathtext.fontset": "cm",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 1.0,          # thick black frame
    "axes.edgecolor": "black",
    "axes.grid": True,
    "grid.color": "#b3b3b3",        # pgfplots "major grid" grey
    "grid.linewidth": 0.4,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": False,
    "legend.handlelength": 2.2,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# pgfplots colours used in the reference (blue / orange)
BLUE = (0.0, 0.0, 1.0)
ORANGE = (1.0, 0.5, 0.0)

FIGSIZE = (2.12, 2.02)        # inches, ~ same as the reference PDF (153 x 146 pt)
FIGSIZE_DUAL = (2.55, 2.02)   # a bit wider to make room for the right-hand axis

XLABEL = "Time [s]"
T_RANGE = None   # set from the data in main()


def sci_formatter():
    """Puts the common power of ten above the axis, like the '·10^5' in pgfplots."""
    f = ScalarFormatter(useMathText=True)
    f.set_powerlimits((-2, 3))
    return f


def style_axes(ax, title=None, ylabel=None):
    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xlabel(XLABEL)
    ax.margins(x=0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))


def save(fig, out, name):
    # twinx() resets x-margins, so pin the x-range to the data on every axes
    for ax in fig.axes:
        ax.set_xlim(T_RANGE)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=300)
    plt.close(fig)


def dual_axes(title, figsize=FIGSIZE_DUAL):
    """Left/right y-axes sharing x. Each y-axis is coloured like its curve."""
    fig, ax_l = plt.subplots(figsize=figsize)
    ax_r = ax_l.twinx()
    ax_l.tick_params(axis="y", right=False)          # right side belongs to ax_r
    ax_r.grid(False)                                  # one grid only (from left axis)
    ax_r.tick_params(axis="y", direction="in", right=True, which="both")
    style_axes(ax_l, title)
    for ax, c in ((ax_l, BLUE), (ax_r, ORANGE)):
        ax.yaxis.label.set_color(c)
        ax.tick_params(axis="y", colors=c, which="both")
    # draw the right axis curve behind? keep left on top for readability
    ax_l.set_zorder(ax_r.get_zorder() + 1)
    ax_l.patch.set_visible(False)
    return fig, ax_l, ax_r


def joint_legend(ax_l, ax_r, loc):
    h1, l1 = ax_l.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    ax_l.legend(h1 + h2, l1 + l2, loc=loc)


# --------------------------------------------------------------------------
def main(csv_path, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(csv_path)
    t = d["ida_time"].to_numpy()
    global T_RANGE
    T_RANGE = (t.min(), t.max())

    # skip the initial-condition row (t = 0) where step size is a placeholder
    m = (d["steps_cumulative"] > 0).to_numpy()
    h = d["last_step_size"].to_numpy()

    # Newton iterations per step (undefined when no step was taken)
    nps = d["nonlin_iters_per_step"].to_numpy().astype(float)
    nps[d["steps_interval"].to_numpy() == 0] = np.nan

    # ---------------- single-axis plots ----------------
    def single(title, ylabel):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        style_axes(ax, title, ylabel)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        return fig, ax

    # 1) IDA internal time step h
    fig, ax = single("IDA Internal Time Step", r"Time step $h$ [s]")
    ax.semilogy(t[m], h[m], color=ORANGE, lw=1.2)
    save(fig, out, "ida_step_size")

    # 2) Cumulative steps & Newton iterations (same units -> one axis)
    fig, ax = single("Cumulative Work", "Count")
    ax.plot(t, d["steps_cumulative"], color=BLUE, lw=1.2, ls=":", label="Steps")
    ax.plot(t, d["nonlin_iters_cumulative"], color=ORANGE, lw=1.2, label="Newton it.")
    ax.yaxis.set_major_formatter(sci_formatter())
    ax.legend(loc="lower right")
    save(fig, out, "ida_cumulative")

    # 3) Newton iterations per step
    fig, ax = single("Newton Iterations per Step", "Iterations / step")
    ax.plot(t, nps, color=ORANGE, lw=0, marker=".", ms=2)
    ax.set_ylim(0, 2.5)
    save(fig, out, "ida_nonlin_per_step")

    # ---------------- dual-axis plots ----------------
    # A) step size (left, log)  |  cumulative steps (right)
    fig, ax_l, ax_r = dual_axes("Time Step and Cumulative Steps")
    ax_l.semilogy(t[m], h[m], color=BLUE, lw=1.2, label=r"$h$")
    ax_r.plot(t, d["steps_cumulative"], color=ORANGE, lw=1.2, ls="--", label="Steps")
    ax_l.set_ylabel(r"Time step $h$ [s]")
    ax_r.set_ylabel("Cumulative steps")
    ax_r.yaxis.set_major_formatter(sci_formatter())
    joint_legend(ax_l, ax_r, "lower right")
    save(fig, out, "ida_dual_h_cumsteps")

    # B) time step h (left, log)  |  Newton iterations per step (right)
    fig, ax_l, ax_r = dual_axes("Time Step and Newton Iterations")
    ax_l.semilogy(t[m], h[m], color=BLUE, lw=1.2, label=r"$h$")
    ax_r.plot(t, nps, color=ORANGE, lw=0, marker=".", ms=2, label="Newton it. / step")
    ax_l.set_ylabel(r"Time step $h$ [s]")
    ax_r.set_ylabel("Newton iterations / step")
    ax_r.set_ylim(0, 3)
    joint_legend(ax_l, ax_r, "lower right")
    save(fig, out, "ida_dual_h_newton")

    # ---------------- overview (1 x 3) ----------------
    fig, axs = plt.subplots(1, 3, figsize=(3 * FIGSIZE[0] + 0.6, FIGSIZE[1] + 0.2),
                            constrained_layout=True)
    a = axs.ravel()
    a[0].semilogy(t[m], h[m], color=ORANGE, lw=1.0)
    style_axes(a[0], "IDA Internal Time Step", r"$h$ [s]")
    a[1].plot(t, d["steps_cumulative"], color=BLUE, ls=":", lw=1.0, label="Steps")
    a[1].plot(t, d["nonlin_iters_cumulative"], color=ORANGE, lw=1.0, label="Newton it.")
    style_axes(a[1], "Cumulative Work", "Count")
    a[1].yaxis.set_major_formatter(sci_formatter())
    a[1].legend(loc="lower right")
    a[2].plot(t, nps, color=ORANGE, lw=0, marker=".", ms=1.5)
    style_axes(a[2], "Newton Iterations per Step", "Iterations / step")
    a[2].set_ylim(0, 2.5)
    save(fig, out, "ida_overview")

    print(f"Saved figures to {out.resolve()}  (usetex={USE_TEX})")


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else "ida_step_history.csv"
    outdir = sys.argv[2] if len(sys.argv) > 2 else "figures"
    main(csv, outdir)