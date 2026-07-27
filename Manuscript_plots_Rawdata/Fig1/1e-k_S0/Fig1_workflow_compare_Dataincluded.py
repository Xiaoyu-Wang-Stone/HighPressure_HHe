"""Reproduce Fig1 workflow comparison from a standalone CSV dataset only."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce Fig1_workflow_compare exactly"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="/Users/xiaoyuwang/Project/H-He/H-He-demixing/Figs/Fig1_workflow_compare.pdf",
        help="Output PDF path",
    )
    parser.add_argument(
        "-d",
        "--data",
        default=str(Path(__file__).with_name("Fig1_workflow_compare_data.csv")),
        help="Standalone CSV data path",
    )
    parser.add_argument("--show", action="store_true", help="Display figure window")
    return parser.parse_args()


def ebar(ax, *args, **kwargs):
    kwargs.setdefault("elinewidth", 0.9)
    kwargs.setdefault("capsize", 3.0)
    kwargs.setdefault("capthick", 0.9)
    return ax.errorbar(*args, **kwargs)


def get_series(df: pd.DataFrame, condition: str, quantity: str, model: str):
    sub = df[
        (df["condition"] == condition)
        & (df["quantity"] == quantity)
        & (df["model"] == model)
    ]
    if sub.empty:
        raise ValueError(f"No data found for {condition=} {quantity=} {model=}")
    order = np.argsort(np.asarray(sub["index"], dtype=float))
    sub = sub.iloc[order]
    x = sub["x_he"].to_numpy(dtype=float)
    y = sub["y"].to_numpy(dtype=float)
    yerr = np.abs(sub["yerr"].to_numpy(dtype=float))
    return x, y, yerr


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"condition", "quantity", "model", "index", "x_he", "y", "yerr"}
    missing = required.difference(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required CSV columns: {missing_str}")
    return df


def gmix_from_mu(x, mu_he, mu_h, err_he, err_h):
    gmix = x * mu_he + (1.0 - x) * mu_h
    gmix_err = np.sqrt((x * err_he) ** 2 + ((1.0 - x) * err_h) ** 2)
    return gmix, gmix_err


def main() -> None:
    args = parse_args()
    df = load_data(args.data)

    n2p2_color = "#e12729"
    cace_color = "#1092ee"
    mace_color = "#fead10"

    font = {"family": "serif", "size": 8}
    plt.rc("font", **font)
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    matplotlib.rcParams["axes.titlesize"] = 8
    matplotlib.rcParams["axes.labelsize"] = 8
    matplotlib.rcParams["xtick.labelsize"] = 7
    matplotlib.rcParams["ytick.labelsize"] = 7
    matplotlib.rcParams["legend.fontsize"] = 7
    legend_font_size = matplotlib.rcParams["legend.fontsize"]

    fep_style = {
        "color": "#ff7eb6",
        "fmt": "D--",
        "dashes": (2, 1),
        "markersize": 4.5,
        "zorder": 10,
        "clip_on": False,
    }

    a4_width_mm = 210
    fig_width_mm = a4_width_mm * 0.25
    fig_height_mm = fig_width_mm / 1.618 * 1.2
    fig_width_inches = fig_width_mm / 25.4
    fig_height_inches = fig_height_mm / 25.4

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(fig_width_inches * 2.5, fig_height_inches * 3),
        dpi=500,
        gridspec_kw={"hspace": 0.2, "wspace": 0.15},
        sharex=True,
        sharey="row",
    )
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]
    ax5, ax6 = axes[2]

    cond_left = "150_5000"
    cond_right = "800_7000"

    x, y, yerr = get_series(df, cond_left, "integrand_he", "CACE")
    ebar(ax1, x, y, yerr, color=cace_color, label="CACE", marker="o", clip_on=False)
    x, y, yerr = get_series(df, cond_left, "integrand_he", "N2P2")
    ebar(
        ax1,
        x,
        y,
        yerr,
        color=n2p2_color,
        label="N2P2",
        ls="--",
        marker="x",
        clip_on=False,
    )
    x, y, yerr = get_series(df, cond_left, "integrand_he", "MACE")
    ebar(
        ax1,
        x,
        y,
        yerr,
        color=mace_color,
        label="MACE",
        ls=":",
        marker="^",
        clip_on=False,
    )
    ax1.set_ylabel(
        r"${\partial \mu_{\rm He}}/{\partial \ln x_{\rm He}} \ [k_{\rm B}T]$"
    )
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax1.legend(
        loc="upper center",
        title="vdW-DF MLPs",
        frameon=False,
        fontsize=legend_font_size,
    )

    x_cace, mu_he_cace, mu_he_cace_err = get_series(df, cond_left, "mu_he", "CACE")
    _, mu_h_cace, mu_h_cace_err = get_series(df, cond_left, "mu_h", "CACE")
    x_n2p2, mu_he_n2p2, mu_he_n2p2_err = get_series(df, cond_left, "mu_he", "N2P2")
    _, mu_h_n2p2, mu_h_n2p2_err = get_series(df, cond_left, "mu_h", "N2P2")
    x_mace, mu_he_mace, mu_he_mace_err = get_series(df, cond_left, "mu_he", "MACE")
    _, mu_h_mace, mu_h_mace_err = get_series(df, cond_left, "mu_h", "MACE")

    ebar(
        ax3,
        x_cace,
        mu_he_cace,
        mu_he_cace_err,
        color=cace_color,
        marker="p",
        fillstyle="none",
        clip_on=False,
    )
    ebar(
        ax3,
        x_cace,
        mu_h_cace,
        mu_h_cace_err,
        color=cace_color,
        marker="<",
        fillstyle="none",
        clip_on=False,
    )
    ebar(
        ax3,
        x_n2p2,
        mu_he_n2p2,
        mu_he_n2p2_err,
        color=n2p2_color,
        ls="--",
        marker="p",
        fillstyle="none",
        clip_on=False,
    )
    ebar(
        ax3,
        x_n2p2,
        mu_h_n2p2,
        mu_h_n2p2_err,
        color=n2p2_color,
        ls="--",
        marker="<",
        fillstyle="none",
        clip_on=False,
    )
    ebar(
        ax3,
        x_mace,
        mu_he_mace,
        mu_he_mace_err,
        color=mace_color,
        ls=":",
        marker="p",
        fillstyle="none",
        clip_on=False,
    )
    ebar(
        ax3,
        x_mace,
        mu_h_mace,
        mu_h_mace_err,
        color=mace_color,
        ls=":",
        marker="<",
        fillstyle="none",
        clip_on=False,
    )
    x_line = np.linspace(0.0, 1.0, 200)
    kb_t = 5000 * 0.00008661733
    ax3.plot(
        x_line,
        kb_t * np.log(x_line),
        color="gray",
        ls=(0, (1, 3)),
        alpha=0.5,
        linewidth=1,
    )
    ax3.plot(
        x_line,
        kb_t * np.log(1.0 - x_line),
        color="gray",
        ls=(0, (1, 3)),
        alpha=0.5,
        linewidth=1,
    )
    ax3.set_ylim(bottom=-1.03)
    ax3.set_ylabel(r"$\mu$ [eV/atom]")
    ax3.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    gmix_cace, gmix_cace_err = gmix_from_mu(
        x_cace, mu_he_cace, mu_h_cace, mu_he_cace_err, mu_h_cace_err
    )
    gmix_n2p2, gmix_n2p2_err = gmix_from_mu(
        x_n2p2, mu_he_n2p2, mu_h_n2p2, mu_he_n2p2_err, mu_h_n2p2_err
    )
    gmix_mace, gmix_mace_err = gmix_from_mu(
        x_mace, mu_he_mace, mu_h_mace, mu_he_mace_err, mu_h_mace_err
    )
    x_fep_l, y_fep_l, yerr_fep_l = get_series(df, cond_left, "gmix_fep", "FEP_vdW-DF")

    ebar(
        ax5,
        np.r_[1.0, x_cace, 0.0],
        np.r_[0.0, gmix_cace, 0.0],
        np.r_[0.0, gmix_cace_err, 0.0],
        color=cace_color,
        marker="o",
        clip_on=False,
        label="_nolegend_",
    )
    ebar(ax5, x_fep_l, y_fep_l, yerr_fep_l, label="vdW-DF DFT (FEP)", **fep_style)
    ebar(
        ax5,
        np.r_[1.0, x_n2p2, 0.0],
        np.r_[0.0, gmix_n2p2, 0.0],
        np.r_[0.0, gmix_n2p2_err, 0.0],
        color=n2p2_color,
        ls="--",
        marker="x",
        clip_on=False,
        label="_nolegend_",
    )
    ebar(
        ax5,
        np.r_[1.0, x_mace, 0.0],
        np.r_[0.0, gmix_mace, 0.0],
        np.r_[0.0, gmix_mace_err, 0.0],
        color=mace_color,
        ls=":",
        marker="^",
        clip_on=False,
        label="_nolegend_",
    )
    ax5.set_xlabel(r"$x_{\rm He}$")
    ax5.set_ylabel(r"$\Delta G^{\rm {mix}}$ [eV/atom]")
    ax5.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax5.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax5.legend(
        loc="upper center", frameon=False, handlelength=1.2, fontsize=legend_font_size
    )

    x, y, yerr = get_series(df, cond_right, "integrand_he", "CACE")
    ebar(
        ax2, x, y, yerr, color=cace_color, marker="o", clip_on=False, label="_nolegend_"
    )
    x, y, yerr = get_series(df, cond_right, "integrand_he", "N2P2")
    ebar(
        ax2,
        x,
        y,
        yerr,
        color=n2p2_color,
        ls="--",
        marker="x",
        clip_on=False,
        label="_nolegend_",
    )
    x, y, yerr = get_series(df, cond_right, "integrand_he", "MACE")
    ebar(
        ax2,
        x,
        y,
        yerr,
        color=mace_color,
        ls=":",
        marker="^",
        clip_on=False,
        label="_nolegend_",
    )
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax2.axvline(x=0.19, c="k", ls="--", linewidth=1)
    ax2.axvline(x=0.77, c="k", ls="--", linewidth=1)

    x_cace, mu_he_cace, mu_he_cace_err = get_series(df, cond_right, "mu_he", "CACE")
    _, mu_h_cace, mu_h_cace_err = get_series(df, cond_right, "mu_h", "CACE")
    x_n2p2, mu_he_n2p2, mu_he_n2p2_err = get_series(df, cond_right, "mu_he", "N2P2")
    _, mu_h_n2p2, mu_h_n2p2_err = get_series(df, cond_right, "mu_h", "N2P2")
    x_mace, mu_he_mace, mu_he_mace_err = get_series(df, cond_right, "mu_he", "MACE")
    _, mu_h_mace, mu_h_mace_err = get_series(df, cond_right, "mu_h", "MACE")

    ebar(
        ax4,
        x_cace,
        mu_he_cace,
        mu_he_cace_err,
        color=cace_color,
        marker="p",
        fillstyle="none",
        clip_on=False,
    )
    ebar(
        ax4,
        x_cace,
        mu_h_cace,
        mu_h_cace_err,
        color=cace_color,
        marker="<",
        markersize=5,
        fillstyle="none",
        clip_on=False,
    )
    ebar(
        ax4,
        x_n2p2,
        mu_he_n2p2,
        mu_he_n2p2_err,
        color=n2p2_color,
        ls="--",
        marker="p",
        fillstyle="none",
        clip_on=False,
    )
    ebar(
        ax4,
        x_n2p2,
        mu_h_n2p2,
        mu_h_n2p2_err,
        color=n2p2_color,
        ls="--",
        marker="<",
        fillstyle="none",
        clip_on=False,
    )
    ebar(
        ax4,
        x_mace,
        mu_he_mace,
        mu_he_mace_err,
        color=mace_color,
        ls=":",
        marker="p",
        fillstyle="none",
        clip_on=False,
    )
    ebar(
        ax4,
        x_mace,
        mu_h_mace,
        mu_h_mace_err,
        color=mace_color,
        ls=":",
        marker="<",
        fillstyle="none",
        clip_on=False,
    )
    kb_t = 7000 * 0.00008661733
    ax4.plot(
        x_line,
        kb_t * np.log(x_line),
        color="gray",
        ls=(0, (1, 3)),
        alpha=0.5,
        linewidth=1,
    )
    ax4.plot(
        x_line,
        kb_t * np.log(1.0 - x_line),
        color="gray",
        ls=(0, (1, 3)),
        alpha=0.5,
        linewidth=1,
    )
    ax4.set_ylim(bottom=-1.03)
    ax4.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax4.axvline(x=0.19, c="k", ls="--", linewidth=1)
    ax4.axvline(x=0.77, c="k", ls="--", linewidth=1)

    gmix_cace, gmix_cace_err = gmix_from_mu(
        x_cace, mu_he_cace, mu_h_cace, mu_he_cace_err, mu_h_cace_err
    )
    gmix_n2p2, gmix_n2p2_err = gmix_from_mu(
        x_n2p2, mu_he_n2p2, mu_h_n2p2, mu_he_n2p2_err, mu_h_n2p2_err
    )
    gmix_mace, gmix_mace_err = gmix_from_mu(
        x_mace, mu_he_mace, mu_h_mace, mu_he_mace_err, mu_h_mace_err
    )
    x_fep_r, y_fep_r, yerr_fep_r = get_series(df, cond_right, "gmix_fep", "FEP_vdW-DF")

    ebar(
        ax6,
        np.r_[1.0, x_cace, 0.0],
        np.r_[0.0, gmix_cace, 0.0],
        np.r_[0.0, gmix_cace_err, 0.0],
        color=cace_color,
        marker="o",
        clip_on=False,
        label="CACE",
    )
    ebar(ax6, x_fep_r, y_fep_r, yerr_fep_r, label="_nolegend_", **fep_style)
    ebar(
        ax6,
        np.r_[1.0, x_n2p2, 0.0],
        np.r_[0.0, gmix_n2p2, 0.0],
        np.r_[0.0, gmix_n2p2_err, 0.0],
        color=n2p2_color,
        ls="--",
        marker="x",
        clip_on=False,
        label="_nolegend_",
    )
    ebar(
        ax6,
        np.r_[1.0, x_mace, 0.0],
        np.r_[0.0, gmix_mace, 0.0],
        np.r_[0.0, gmix_mace_err, 0.0],
        color=mace_color,
        ls=":",
        marker="^",
        clip_on=False,
        label="_nolegend_",
    )
    ax6.set_xlabel(r"$x_{\rm He}$")
    ax6.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax6.axvline(x=0.19, c="k", ls="--", linewidth=1)
    ax6.axvline(x=0.77, c="k", ls="--", linewidth=1)

    for ax in axes.flatten():
        ax.set_xlim(0.0, 1.0)

    ax1.yaxis.set_label_coords(-0.20, 0.5)
    ax3.yaxis.set_label_coords(-0.20, 0.5)
    ax5.yaxis.set_label_coords(-0.20, 0.5)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(args.output, dpi=500, bbox_inches="tight")
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
