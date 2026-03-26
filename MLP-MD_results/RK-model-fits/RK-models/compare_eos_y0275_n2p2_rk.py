#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from compare_rk_vs_tab_vex import compute_rk_vex_cm3g, load_tp, y_to_xhe


NA = 6.02214076e23
M_H = 1.008
M_HE = 4.002602


def load_n2p2_p_list(path: Path) -> pd.DataFrame:
    arr = np.genfromtxt(path, dtype=str)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError(f"Unexpected format: {path}")
    d = pd.DataFrame(arr[:, :5], columns=["x", "T_K", "V_A3", "P_GPa", "U"])
    d["x"] = d["x"].astype(float)
    d["T_K"] = d["T_K"].astype(float)
    d["V_A3"] = d["V_A3"].astype(float)
    d["P_GPa"] = d["P_GPa"].astype(float)

    # Follow notebook convention exactly.
    d["N_He"] = 32.0 * d["x"]
    d["N_H"] = 64.0 - d["N_He"]
    d["mass_g"] = d["N_H"] * M_H / NA + d["N_He"] * M_HE / NA
    d["rho_gcc"] = d["mass_g"] / d["V_A3"] * 1.0e24
    return d


def interp_rho_vs_p(data: pd.DataFrame, p_target: np.ndarray) -> np.ndarray:
    s = data.sort_values("P_GPa")
    p = s["P_GPa"].to_numpy()
    rho = s["rho_gcc"].to_numpy()
    idx = np.argsort(p)
    p = p[idx]
    rho = rho[idx]
    p_unique, uidx = np.unique(p, return_index=True)
    rho_unique = rho[uidx]
    return np.interp(np.log10(p_target), np.log10(p_unique), rho_unique)


def interp_tab_rho_2d(
    mix_all: pd.DataFrame, t_k: float, p_target: np.ndarray
) -> np.ndarray:
    lt = np.log10(t_k)
    lp_target = np.log10(p_target)

    t_grid = np.sort(mix_all["logT"].unique())
    t_lo = t_grid[t_grid <= lt].max() if np.any(t_grid <= lt) else t_grid.min()
    t_hi = t_grid[t_grid >= lt].min() if np.any(t_grid >= lt) else t_grid.max()

    def interp_at_logt(logt: float) -> np.ndarray:
        s = mix_all[np.isclose(mix_all["logT"], logt)].sort_values("logP")
        lp = s["logP"].to_numpy()
        rho = s["rho_tab_gcc"].to_numpy()
        return np.interp(lp_target, lp, rho)

    rho_lo = interp_at_logt(float(t_lo))
    if np.isclose(t_lo, t_hi):
        return rho_lo
    rho_hi = interp_at_logt(float(t_hi))
    wt = (lt - t_lo) / (t_hi - t_lo)
    return (1.0 - wt) * rho_lo + wt * rho_hi


def build_distinct_colors(n: int, cmap_file: Path | None) -> list:
    if n <= 0:
        return []
    if cmap_file is not None and cmap_file.exists():
        cm_data = np.loadtxt(cmap_file)
        cmap = LinearSegmentedColormap.from_list("lipari", cm_data)
        new_colors = cmap(np.linspace(0.0, 0.9, max(n, 10)))
        truncated_cmap = ListedColormap(new_colors)
        return [truncated_cmap(i / max(n - 1, 1)) for i in range(n)]
    tab = plt.get_cmap("tab10")
    return [tab(i % 10) for i in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare EOS at Y=0.275 using N2P2 pure H/He + RK excess volume."
    )
    parser.add_argument("--workdir", type=Path, default=Path("."))
    parser.add_argument("--y", type=float, default=0.275)
    parser.add_argument("--mix-file", type=str, default="TABLEEOS_2021_TP_Y0275_v1")
    parser.add_argument(
        "--n2p2-vdw-file",
        type=Path,
        default=Path(
            "/Users/xiaoyuwang/Project/H-He/Test/test-n2p2-gen6-VDW/nvt-rdf-r-1/P.list"
        ),
    )
    parser.add_argument(
        "--n2p2-pbe-file",
        type=Path,
        default=Path(
            "/Users/xiaoyuwang/Project/H-He/Test/test-n2p2-gen6-PBE/nvt-rdf-r-4/P.list"
        ),
    )
    parser.add_argument(
        "--n2p2-hse-file",
        type=Path,
        default=Path("/Volumes/Extreme Pro/Hhe_revision_rerun/EOS-HSE-HHe/P.list"),
    )
    parser.add_argument("--x-h", type=float, default=0.0)
    parser.add_argument("--x-he", type=float, default=1.0)
    parser.add_argument("--p-min", type=float, default=100.0)
    parser.add_argument("--p-max", type=float, default=1000.0)
    parser.add_argument("--n-p", type=int, default=120)
    parser.add_argument(
        "--t-lines",
        type=float,
        nargs="+",
        default=[2000.0, 4000.0, 6000.0, 8000.0, 10000.0],
    )
    parser.add_argument("--t-scale", type=float, default=1000.0)
    parser.add_argument("--p-scale", type=float, default=100.0)
    parser.add_argument(
        "--cmap-file",
        type=Path,
        default=Path(
            "/Users/xiaoyuwang/Software/ScientificColourMaps8/lipari/lipari.txt"
        ),
    )
    parser.add_argument(
        "--omega-unit",
        type=str,
        default="ev",
        choices=["dimensionless_eq3", "kb_1e3k", "kelvin", "ev", "joule"],
    )
    args = parser.parse_args()

    wd = args.workdir
    mix_all = load_tp(wd / args.mix_file).rename(columns={"logrho": "logrho_mix"})
    mix_all["T_K"] = 10.0 ** mix_all["logT"]
    mix_all["P_GPa"] = 10.0 ** mix_all["logP"]
    mix_all["rho_tab_gcc"] = 10.0 ** mix_all["logrho_mix"]

    n2p2 = {
        "vdW-DF": load_n2p2_p_list(args.n2p2_vdw_file),
        "PBE": load_n2p2_p_list(args.n2p2_pbe_file),
        "HSE": load_n2p2_p_list(args.n2p2_hse_file),
    }

    x_he_mix = y_to_xhe(args.y)
    x_mass_h = 1.0 - args.y

    outdir = wd / "plots" / "eos_y0275_n2p2_rk"
    tag = f"Y{args.y:.3f}_xh{args.x_h:g}_xhe{args.x_he:g}_{args.omega_unit}"
    outdir = outdir / tag
    outdir.mkdir(parents=True, exist_ok=True)

    fig_width_inches = 210.0 / 25.4
    fig_height_inches = fig_width_inches / 2.15
    axis_label_fs = 11
    fig, axes = plt.subplots(
        1, 3, figsize=(fig_width_inches, fig_height_inches), dpi=220, sharey=True
    )

    for ax, func in zip(axes, ["vdW-DF", "PBE", "HSE"]):
        src = n2p2[func]
        rows = []
        colors = build_distinct_colors(len(args.t_lines), args.cmap_file)

        t_xh = set(src[np.isclose(src["x"], args.x_h)]["T_K"].to_numpy())
        t_xhe = set(src[np.isclose(src["x"], args.x_he)]["T_K"].to_numpy())
        t_avail = sorted(t_xh.intersection(t_xhe))
        t_req = [float(t) for t in args.t_lines]
        t_missing = [t for t in t_req if t not in t_avail]
        if t_missing:
            print(
                f"[WARN] {func}: missing requested temperatures {t_missing}; "
                f"available at x={args.x_h} & x={args.x_he}: {t_avail}"
            )

        for i, t in enumerate(args.t_lines):
            c = colors[i % len(colors)]

            sh = src[np.isclose(src["x"], args.x_h) & np.isclose(src["T_K"], t)].copy()
            she = src[
                np.isclose(src["x"], args.x_he) & np.isclose(src["T_K"], t)
            ].copy()
            if sh.empty or she.empty:
                continue

            p_lo = max(args.p_min, sh["P_GPa"].min(), she["P_GPa"].min())
            p_hi = min(args.p_max, sh["P_GPa"].max(), she["P_GPa"].max())
            if p_hi <= p_lo:
                continue

            p_grid = np.logspace(np.log10(p_lo), np.log10(p_hi), args.n_p)
            rho_h = interp_rho_vs_p(sh, p_grid)
            rho_he = interp_rho_vs_p(she, p_grid)

            v_id = x_mass_h / rho_h + args.y / rho_he
            dtmp = pd.DataFrame(
                {"logT": np.full(args.n_p, np.log10(t)), "logP": np.log10(p_grid)}
            )
            v_ex = compute_rk_vex_cm3g(
                dtmp,
                func_name=func,
                x_he=x_he_mix,
                t_scale_k=args.t_scale,
                p_scale_gpa=args.p_scale,
                omega_unit=args.omega_unit,
            )
            rho_rk = 1.0 / (v_id + v_ex)

            rho_tab = interp_tab_rho_2d(mix_all, t, p_grid)

            model_label = "PBE N2P2 MLP" if func == "PBE" else "vdW-DF N2P2 MLP"
            ax.plot(
                p_grid,
                rho_rk,
                color=c,
                lw=1.8,
                ls="-",
                label="_nolegend_",
            )
            ax.plot(
                p_grid,
                rho_tab,
                color=c,
                lw=2.2,
                ls=":",
                label="_nolegend_",
            )

            delta = 100.0 * (rho_rk / rho_tab - 1.0)
            for p, rt, rr, rh, rhe, vx, de in zip(
                p_grid, rho_tab, rho_rk, rho_h, rho_he, v_ex, delta
            ):
                rows.append(
                    {
                        "functional": func,
                        "T_K": t,
                        "P_GPa": p,
                        "rho_tab_gcc": rt,
                        "rho_rk_n2p2_gcc": rr,
                        "rho_h_n2p2_gcc": rh,
                        "rho_he_n2p2_gcc": rhe,
                        "v_ex_rk_cm3g": vx,
                        "delta_rho_pct": de,
                    }
                )

        ax.set_xscale("log")
        ax.set_xlabel("Pressure [GPa]", fontsize=axis_label_fs)
        ax.text(
            0.03,
            0.96,
            rf"{func} at $x_{{\rm He}}={x_he_mix:.2f}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )

        pd.DataFrame(rows).to_csv(
            outdir / f"{func.replace('-', '').lower()}_rk_n2p2_vs_tab.csv", index=False
        )

    colors = build_distinct_colors(len(args.t_lines), args.cmap_file)
    temp_handles = [
        Patch(facecolor=colors[i], edgecolor="none", label=f"{t:.0f} K")
        for i, t in enumerate(args.t_lines)
    ]
    style_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            lw=1.8,
            ls="-",
            label=r"N2P2 MLP $\rho^{\rm mix}_{\rm RK}$",
        ),
        Line2D(
            [0], [0], color="black", lw=2.2, ls=":", label="Chabrier and Debras (2021)"
        ),
    ]
    fig.legend(
        handles=temp_handles,
        loc="upper right",
        bbox_to_anchor=(0.85, 0.97),
        ncol=5,
        frameon=False,
        fontsize=8,
        handlelength=1.0,
        columnspacing=0.8,
    )
    fig.legend(
        handles=style_handles,
        loc="upper left",
        bbox_to_anchor=(0.15, 1.0),
        ncol=1,
        frameon=False,
        fontsize=8,
        handlelength=3.0,
        columnspacing=1.0,
    )
    fig.supylabel(r"Density [g/cm$^3$]", fontsize=axis_label_fs)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(outdir / "EOS_Y0275.png", bbox_inches="tight")
    plt.close(fig)

    (outdir / "run_config.txt").write_text(
        "\n".join(
            [
                f"x_He={x_he_mix}",
                f"mix_file={args.mix_file}",
                f"n2p2_vdw_file={args.n2p2_vdw_file}",
                f"n2p2_pbe_file={args.n2p2_pbe_file}",
                f"n2p2_hse_file={args.n2p2_hse_file}",
                f"x_h_key={args.x_h}",
                f"x_he_key={args.x_he}",
                f"P_range=[{args.p_min}, {args.p_max}] GPa",
                f"t_lines={args.t_lines}",
                f"t_scale={args.t_scale}",
                f"p_scale={args.p_scale}",
                f"omega_unit={args.omega_unit}",
            ]
        )
        + "\n"
    )

    print(f"Saved outputs to: {outdir}")


if __name__ == "__main__":
    main()
