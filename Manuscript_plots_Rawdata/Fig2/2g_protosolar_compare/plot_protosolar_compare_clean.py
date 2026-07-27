from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "all_plot_data.csv"


def load_data() -> pd.DataFrame:
    required = {"series", "pressure_gpa", "temperature_k", "yerr_k", "xerr_gpa"}
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing data file: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{DATA_FILE.name} missing columns: {missing}")

    for col in ["pressure_gpa", "temperature_k", "yerr_k", "xerr_gpa"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["series"] = df["series"].astype(str)
    return df


def get_series(
    df: pd.DataFrame, name: str, min_pressure: float | None = None
) -> pd.DataFrame:
    sub = df.loc[
        df["series"] == name,
        ["series", "pressure_gpa", "temperature_k", "yerr_k", "xerr_gpa"],
    ].copy()
    sub = sub[sub["pressure_gpa"].notna() & sub["temperature_k"].notna()]
    if min_pressure is not None:
        sub = sub[sub["pressure_gpa"] >= min_pressure]
    return sub.sort_values(by=["pressure_gpa"])


def plot_line(ax, s: pd.DataFrame, **kwargs) -> None:
    if s.empty:
        return
    ax.plot(s["pressure_gpa"], s["temperature_k"] / 1000.0, **kwargs)


def plot_errorbar(
    ax, s: pd.DataFrame, use_yerr: bool, use_xerr: bool, **kwargs
) -> None:
    if s.empty:
        return
    yerr = s["yerr_k"] / 1000.0 if use_yerr else None
    xerr = s["xerr_gpa"] if use_xerr else None
    if use_yerr and bool(s["yerr_k"].isna().all()):
        yerr = None
    if use_xerr and bool(s["xerr_gpa"].isna().all()):
        xerr = None
    ax.errorbar(
        s["pressure_gpa"], s["temperature_k"] / 1000.0, yerr=yerr, xerr=xerr, **kwargs
    )


def main() -> None:
    df = load_data()
    plt.rc("font", family="serif", size=9)

    a4_width_mm = 210
    fig_w = (a4_width_mm * 0.85) / 25.4
    fig_h = (a4_width_mm * 0.65 / 1.62) / 25.4
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=500)

    demix_colors = {"PBE": "#3A86FF", "VDW": "#FF006E", "HSE": "#FFBE0B"}
    for label in ["PBE", "VDW", "HSE"]:
        s = get_series(df, f"demix_{label}")
        plot_line(ax, s, color=demix_colors[label], lw=2.5, zorder=10)
        if label == "VDW" and not s.empty:
            t150_k = float(np.interp(150.0, s["pressure_gpa"], s["temperature_k"]))
            ax.errorbar(
                [150.0],
                [t150_k / 1000.0],
                yerr=[1.0 / 1000.0],
                color=demix_colors["VDW"],
                markersize=8,
                capsize=3,
                elinewidth=1.3,
                fmt="s",
                zorder=15,
            )

    ax.annotate(
        r"$x_{\mathrm{He}}$=0.089",
        (900, 3.45),
        xytext=(5, 10),
        textcoords="offset points",
        ha="center",
        va="center",
        fontsize=8,
        color="black",
        zorder=6,
        bbox={"facecolor": "white", "edgecolor": "none"},
    )

    plot_line(
        ax,
        get_series(df, "lorenzen_2011"),
        label="PBE, Lorenzen 2011",
        ls=(0, (5, 10)),
        marker="X",
        color="#8ECAE6",
        markersize=6,
        clip_on=False,
    )

    plot_errorbar(
        ax,
        get_series(df, "morales_2013_points", min_pressure=200),
        use_yerr=True,
        use_xerr=False,
        label="PBE, Morales 2010 & 2013",
        ls="",
        marker="h",
        color="#3596b5",
        alpha=0.5,
        markersize=6,
        clip_on=False,
    )
    plot_errorbar(
        ax,
        get_series(df, "morales_2013_points").query(
            "pressure_gpa >= 100 and pressure_gpa <= 200"
        ),
        use_yerr=False,
        use_xerr=True,
        ls="",
        marker="h",
        color="#3596b5",
        alpha=0.5,
        markersize=6,
        clip_on=False,
    )
    plot_line(
        ax,
        get_series(df, "morales_2013_line", min_pressure=100),
        ls=(0, (5, 10)),
        color="#3596b5",
        alpha=0.3,
        clip_on=False,
    )

    plot_errorbar(
        ax,
        get_series(df, "schottler_2018", min_pressure=100),
        use_yerr=True,
        use_xerr=False,
        label="vdW-DF, Schottler 2018",
        ls="dashed",
        marker="d",
        color="#cdb4db",
        alpha=1.0,
        markersize=6,
        zorder=5,
        clip_on=False,
    )

    plot_line(
        ax,
        get_series(df, "chang_2024_vdw", min_pressure=100),
        label="vdW-DF, Chang 2024",
        marker=">",
        color="#CDB4DB",
        ls=(0, (3, 1, 1, 1, 1, 1)),
        alpha=1.0,
        markersize=6,
        clip_on=False,
    )
    plot_line(
        ax,
        get_series(df, "chang_2024_scan_rvv10", min_pressure=100),
        label="SCAN+rVV10, Chang 2024",
        marker="<",
        color="#74a892",
        ls=(0, (3, 1, 1, 1, 1, 1)),
        alpha=0.7,
        markersize=6,
        clip_on=False,
    )

    color_j = "#ff5e5e"
    color_s = "#c7522a"
    rgba_j = mcolors.to_rgba(color_j, alpha=0.2)
    rgba_j_line = mcolors.to_rgba(color_j, alpha=0.5)
    rgba_j_text = mcolors.to_rgba(color_j, alpha=1.0)
    rgba_s = mcolors.to_rgba(color_s, alpha=1.0)
    rgba_s_line = mcolors.to_rgba(color_s, alpha=0.8)

    j_n = get_series(df, "jupiter_nettelmann", min_pressure=100)
    j_m = get_series(df, "jupiter_militzer", min_pressure=100)
    j_r = get_series(df, "jupiter_redmer_2018", min_pressure=100)
    if not j_n.empty and not j_m.empty and not j_r.empty:
        p_grid = np.union1d(
            np.union1d(j_n["pressure_gpa"], j_m["pressure_gpa"]), j_r["pressure_gpa"]
        )
        tn = np.interp(p_grid, j_n["pressure_gpa"], j_n["temperature_k"]) / 1000.0
        tm = np.interp(p_grid, j_m["pressure_gpa"], j_m["temperature_k"]) / 1000.0
        tr = np.interp(p_grid, j_r["pressure_gpa"], j_r["temperature_k"]) / 1000.0
        ax.fill_between(
            p_grid,
            np.minimum(np.minimum(tn, tm), tr),
            np.maximum(np.maximum(tn, tm), tr),
            color=rgba_j,
            zorder=1,
            clip_on=True,
        )

    plot_line(
        ax,
        get_series(df, "jupiter_today_hmh24", min_pressure=100),
        ls="dashed",
        color=rgba_j_line,
        linewidth=1,
        markersize=6,
        clip_on=True,
    )
    plot_line(
        ax,
        get_series(df, "saturn_today_hmh24", min_pressure=100),
        ls="dashed",
        color=rgba_s,
        linewidth=1,
        markersize=6,
        clip_on=True,
    )
    plot_line(
        ax,
        get_series(df, "saturn_today_mf20", min_pressure=100),
        ls="dotted",
        color=rgba_s,
        linewidth=1,
        markersize=6,
        zorder=15,
        clip_on=True,
    )
    plot_line(
        ax,
        get_series(df, "saturn_isentrope", min_pressure=100),
        ls="-",
        color=rgba_s_line,
        linewidth=1,
        markersize=6,
        clip_on=True,
    )

    ax.annotate(
        "Jupiter isentrope",
        xy=(500, 8.2),
        xytext=(5, 10),
        textcoords="offset points",
        rotation=12,
        ha="center",
        va="center",
        color=rgba_j_text,
        fontsize=8,
    )
    ax.annotate(
        "Jupiter today",
        xy=(900, 8.2),
        xytext=(5, 21),
        textcoords="offset points",
        rotation=8,
        ha="center",
        va="center",
        color=rgba_j_text,
        fontsize=8,
    )
    ax.annotate(
        "Saturn isentrope",
        xy=(720, 7.3),
        xytext=(5, 16),
        textcoords="offset points",
        rotation=8,
        ha="center",
        va="center",
        color=rgba_s,
        fontsize=8,
    )
    ax.annotate(
        "Saturn today [HMH24]",
        xy=(800, 5.7),
        xytext=(5, 13),
        textcoords="offset points",
        rotation=7,
        ha="center",
        va="center",
        color=rgba_s,
        fontsize=8,
    )
    ax.annotate(
        "Saturn today [MF20]",
        xy=(880, 7.0),
        xytext=(5, 15),
        textcoords="offset points",
        rotation=7,
        ha="center",
        va="center",
        color=rgba_s,
        fontsize=8,
        zorder=10,
    )
    ax.annotate(
        "HSE",
        xy=(800, 3.75),
        xytext=(5, 18),
        textcoords="offset points",
        ha="center",
        va="center",
        color="#FFBE0B",
        fontsize=8,
    )
    ax.annotate(
        "vdW-DF",
        xy=(650, 2.9),
        xytext=(5, 19),
        textcoords="offset points",
        ha="center",
        va="center",
        color="#FF006E",
        fontsize=8,
    )
    ax.annotate(
        "PBE",
        xy=(950, 4.5),
        xytext=(5, 15),
        textcoords="offset points",
        ha="center",
        va="center",
        color="#3A86FF",
        fontsize=8,
    )

    plot_line(
        ax,
        get_series(df, "exp_brygoo_line", min_pressure=100),
        ls="dashed",
        color="black",
        alpha=0.6,
        clip_on=False,
    )
    plot_errorbar(
        ax,
        get_series(df, "exp_brygoo_points"),
        use_yerr=True,
        use_xerr=False,
        color="black",
        marker="*",
        label="Exp, Brygoo 2021",
        markersize=6,
        ls="",
    )

    ax.set_axisbelow(True)
    ax.set_zorder(1)
    ax.set_ylabel(r"$T$ [$10^3$ K]")
    ax.set_xlabel(r"$P$ [GPa]")
    ax.grid(False)
    ax.legend(ncol=2, fontsize=8)
    ax.set_xlim(100, 1000)
    ax.set_ylim(1, 12)
    if matplotlib.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()
