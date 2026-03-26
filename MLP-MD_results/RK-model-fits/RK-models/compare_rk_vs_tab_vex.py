#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


K_B = 1.380649e-23  # J/K
E_CHARGE = 1.602176634e-19  # J/eV
AMU = 1.66053906660e-27  # kg
M_H = 1.00784 * AMU
M_HE = 4.002602 * AMU


COEFFS = {
    "PBE": {
        "w1": {
            "type": "tp_quad",
            "a0": -0.0461952,
            "b1": 0.233846,
            "b2": -0.00699237,
            "c1": 0.056833,
            "c2": -0.00247029,
        },
        "w2": {
            "type": "tp_quad",
            "a0": 0.200476,
            "b1": 0.0157427,
            "b2": -0.00661924,
            "c1": -0.0241811,
            "c2": 0.00226865,
        },
        "w3": {
            "type": "p_quad",
            "a0": 0.0514824,
            "a1": 0.0988138,
            "a2": -0.00725365,
        },
    },
    "vdW-DF": {
        "w1": {
            "type": "tp_quad",
            "a0": -0.230428,
            "b1": 0.272684,
            "b2": -0.0120829,
            "c1": 0.116921,
            "c2": -0.00575461,
        },
        "w2": {
            "type": "tp_quad",
            "a0": 0.111668,
            "b1": 0.0180989,
            "b2": -0.00436409,
            "c1": -0.0212551,
            "c2": 0.000976806,
        },
        "w3": {"type": "p_lin", "a0": 0.244669, "c1": 0.0052039},
    },
    "HSE": {
        "w1": {
            "type": "tp_quad",
            "a0": -0.468495,
            "b1": 0.324966,
            "b2": -0.0168621,
            "c1": 0.126185,
            "c2": -0.00618012,
        },
        "w2": {
            "type": "tp_quad",
            "a0": 0.239973,
            "b1": -0.0283817,
            "b2": -0.00139613,
            "c1": -0.00963777,
            "c2": 0.000350994,
        },
        "w3": {"type": "p_quad", "a0": 0.165216, "a1": 0.0498199, "a2": -0.00370128},
    },
}


def load_tp(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            a = s.split()
            rows.append((float(a[0]), float(a[1]), float(a[2])))
    return pd.DataFrame(rows, columns=["logT", "logP", "logrho"])


def y_to_xhe(y: float) -> float:
    num = y / 4.0
    den = (1.0 - y) / 1.0 + y / 4.0
    return num / den


def omega_energy_factor_j(omega_unit: str, temperature_k: np.ndarray) -> np.ndarray:
    if omega_unit == "dimensionless_eq3":
        return K_B * temperature_k
    if omega_unit == "kb_1e3k":
        return np.full_like(temperature_k, K_B * 1.0e3)
    if omega_unit == "kelvin":
        return np.full_like(temperature_k, K_B)
    if omega_unit == "ev":
        return np.full_like(temperature_k, E_CHARGE)
    if omega_unit == "joule":
        return np.ones_like(temperature_k)
    raise ValueError(f"Unknown omega unit: {omega_unit}")


def dwdp_var(model: dict, p_var: np.ndarray) -> np.ndarray:
    if model["type"] == "tp_quad":
        return model["c1"] + 2.0 * model["c2"] * p_var
    if model["type"] == "p_quad":
        return model["a1"] + 2.0 * model["a2"] * p_var
    if model["type"] == "p_lin":
        return np.full_like(p_var, model["c1"])
    raise ValueError(f"Unsupported model type: {model['type']}")


def compute_rk_vex_cm3g(
    df: pd.DataFrame,
    func_name: str,
    x_he: float,
    t_scale_k: float,
    p_scale_gpa: float,
    omega_unit: str,
) -> np.ndarray:
    models = COEFFS[func_name]
    t_k = 10.0 ** df["logT"].to_numpy()
    p_gpa = 10.0 ** df["logP"].to_numpy()
    p_var = p_gpa / p_scale_gpa

    dwdp1 = dwdp_var(models["w1"], p_var)
    dwdp2 = dwdp_var(models["w2"], p_var)
    dwdp3 = dwdp_var(models["w3"], p_var)

    # Coefficients are fitted versus normalized pressure P_var=P/p_scale_gpa (P in GPa).
    # Convert d/dP_var -> d/dP[GPa] -> d/dP[Pa] for physical volume units.
    dwdp_pa_1 = (dwdp1 / p_scale_gpa) / 1.0e9
    dwdp_pa_2 = (dwdp2 / p_scale_gpa) / 1.0e9
    dwdp_pa_3 = (dwdp3 / p_scale_gpa) / 1.0e9

    f = 1.0 - 2.0 * x_he
    poly = dwdp_pa_1 + dwdp_pa_2 * f + dwdp_pa_3 * (f**2)

    pref_j = omega_energy_factor_j(omega_unit, t_k)
    v_particle_m3 = x_he * (1.0 - x_he) * pref_j * poly

    m_particle_kg = x_he * M_HE + (1.0 - x_he) * M_H
    v_specific_m3kg = v_particle_m3 / m_particle_kg
    return v_specific_m3kg * 1000.0  # cm^3/g


def make_plots(
    df: pd.DataFrame, outdir: Path, functional: str, y: float, omega_unit: str
) -> None:
    d = df.copy()
    d["T_K"] = 10.0 ** d["logT"]
    d["P_GPa"] = 10.0 ** d["logP"]

    t_lines = [1000.0, 3162.27766, 10000.0]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # (1) Vex comparison curves
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=180)
    for t, c in zip(t_lines, colors):
        s = d[np.isclose(d["T_K"], t, rtol=0, atol=1e-8)].sort_values("P_GPa")
        if s.empty:
            continue
        ax.plot(
            s["P_GPa"], s["v_ex_tab_cm3g"], color=c, lw=2.2, label=f"T={t:.0f} K tab"
        )
        ax.plot(
            s["P_GPa"],
            s["v_ex_rk_cm3g"],
            color=c,
            lw=1.8,
            ls="--",
            label=f"T={t:.0f} K RK",
        )
    ax.axhline(0.0, color="black", lw=1.0, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("Pressure [GPa]")
    ax.set_ylabel(r"$V^{ex}$ [cm$^3$/g]")
    ax.set_title(f"{functional}: RK vs tabulated excess volume (Y={y:.3f})")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / f"{functional}_vex_vs_P.png", bbox_inches="tight")
    plt.close(fig)

    return


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare RK fitted Vex against tabulated EOS Vex."
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("."),
        help="Directory containing EOS tables",
    )
    parser.add_argument(
        "--y",
        type=float,
        default=0.275,
        help="Helium mass fraction Y used for the mixture table",
    )
    parser.add_argument(
        "--mix-file",
        type=str,
        default="TABLEEOS_2021_TP_Y0275_v1",
        help="Mixture TP table file",
    )
    parser.add_argument(
        "--h-file",
        type=str,
        default="TABLE_H_TP_v1",
        help="Pure hydrogen TP table file",
    )
    parser.add_argument(
        "--he-file",
        type=str,
        default="TABLE_HE_TP_v1",
        help="Pure helium TP table file",
    )
    parser.add_argument(
        "--t-min", type=float, default=1000.0, help="Min temperature [K]"
    )
    parser.add_argument(
        "--t-max", type=float, default=10000.0, help="Max temperature [K]"
    )
    parser.add_argument("--p-min", type=float, default=100.0, help="Min pressure [GPa]")
    parser.add_argument(
        "--p-max", type=float, default=1000.0, help="Max pressure [GPa]"
    )
    parser.add_argument(
        "--t-scale",
        type=float,
        default=1.0e3,
        help="Scaling used in fit variable T_var=T/t_scale",
    )
    parser.add_argument(
        "--p-scale",
        type=float,
        default=100.0,
        help="Scaling used in fit variable P_var=P/p_scale",
    )
    parser.add_argument(
        "--omega-unit",
        type=str,
        default="ev",
        choices=["dimensionless_eq3", "kb_1e3k", "kelvin", "ev", "joule"],
        help=(
            "Interpretation of fitted omega: "
            "dimensionless_eq3 => Eq.3 form with k_B*T prefactor; "
            "kb_1e3k => omega/k_B tabulated in units of 10^3 K (multiply by k_B*1e3); "
            "kelvin => omega in K and interaction term k_B*omega; "
            "ev => omega in eV; "
            "joule => omega in J"
        ),
    )
    args = parser.parse_args()

    wd = args.workdir
    h = load_tp(wd / args.h_file).rename(columns={"logrho": "logrho_H"})
    he = load_tp(wd / args.he_file).rename(columns={"logrho": "logrho_He"})
    mix = load_tp(wd / args.mix_file).rename(columns={"logrho": "logrho_mix"})

    df = mix.merge(h, on=["logT", "logP"], how="inner").merge(
        he, on=["logT", "logP"], how="inner"
    )
    df["T_K"] = 10.0 ** df["logT"]
    df["P_GPa"] = 10.0 ** df["logP"]
    df = df[
        (df["T_K"] >= args.t_min)
        & (df["T_K"] <= args.t_max)
        & (df["P_GPa"] >= args.p_min)
        & (df["P_GPa"] <= args.p_max)
    ].copy()

    x_he = y_to_xhe(args.y)
    x_h = 1.0 - x_he
    x_mass = 1.0 - args.y

    df["rho_mix"] = 10.0 ** df["logrho_mix"]
    df["rho_H"] = 10.0 ** df["logrho_H"]
    df["rho_He"] = 10.0 ** df["logrho_He"]
    df["v_mix_cm3g"] = 1.0 / df["rho_mix"]
    df["v_id_cm3g"] = x_mass / df["rho_H"] + args.y / df["rho_He"]
    df["v_ex_tab_cm3g"] = df["v_mix_cm3g"] - df["v_id_cm3g"]

    root = wd / "plots" / "rk_vs_tab_vex"
    tag = f"ts{args.t_scale:g}_ps{args.p_scale:g}_{args.omega_unit}"
    out_root = root / tag
    out_root.mkdir(parents=True, exist_ok=True)

    for func in COEFFS:
        d = df.copy()
        d["v_ex_rk_cm3g"] = compute_rk_vex_cm3g(
            d,
            func_name=func,
            x_he=x_he,
            t_scale_k=args.t_scale,
            p_scale_gpa=args.p_scale,
            omega_unit=args.omega_unit,
        )
        d["v_ex_res_cm3g"] = d["v_ex_rk_cm3g"] - d["v_ex_tab_cm3g"]

        out_csv = out_root / f"{func}_comparison.csv"
        d.sort_values(["logT", "logP"]).to_csv(out_csv, index=False)
        make_plots(d, out_root, func, args.y, args.omega_unit)

    cfg = out_root / "run_config.txt"
    cfg.write_text(
        "\n".join(
            [
                f"Y={args.y}",
                f"x_He(from Y)={x_he}",
                f"T range [K]=[{args.t_min}, {args.t_max}]",
                f"P range [GPa]=[{args.p_min}, {args.p_max}]",
                f"T_scale={args.t_scale}",
                f"P_scale={args.p_scale}",
                f"omega_unit={args.omega_unit}",
                f"mix_file={args.mix_file}",
                f"fit_variables: T_var=T/{args.t_scale} K, P_var=P/{args.p_scale} GPa",
            ]
        )
        + "\n"
    )

    print(f"Saved all outputs to: {out_root}")
    print("Generated files per functional:")
    print("- <functional>_comparison.csv")
    print("- <functional>_vex_vs_P.png")


if __name__ == "__main__":
    main()
