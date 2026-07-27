from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PHASE_BOUNDARIES = {
    "PBE": (
        np.array([150.0, 200.0, 400.0, 600.0, 800.0, 1000.0]),
        np.array([5000.0, 6000.0, 7000.0, 8000.0, 8000.0, 8000.0]),
    ),
    "VDW": (
        np.array([150.0, 200.0, 400.0, 600.0, 800.0, 1000.0]),
        np.array([4000.0, 5000.0, 7000.0, 8000.0, 8000.0, 9000.0]),
    ),
    "HSE": (
        np.array([200.0, 400.0, 600.0, 800.0, 1000.0]),
        np.array([4000.0, 6000.0, 7000.0, 8000.0, 8000.0]),
    ),
}

COLORS = {
    "PBE": "#3A86FF",
    "VDW": "#FF006E",
    "HSE": "#FFBE0B",
}


def load_data(raw_dir: Path) -> dict:
    data = {}

    data["df_vdw"] = pd.read_csv(raw_dir / "all_models_phase_boundaries_VDW.csv")
    data["df_hse"] = pd.read_csv(raw_dir / "all_models_phase_boundaries_HSE.csv")
    data["df_pbe"] = pd.read_csv(raw_dir / "all_models_phase_boundaries_PBE.csv")

    data["df_vdw_model"] = data["df_vdw"][
        data["df_vdw"]["model_name"] == "Model 2: Interaction"
    ].copy()
    data["df_hse_model"] = data["df_hse"][
        data["df_hse"]["model_name"] == "Model 5: Full-Quadratic"
    ].copy()
    data["df_pbe_model"] = data["df_pbe"][
        data["df_pbe"]["model_name"] == "Model 5: Full-Quadratic"
    ].copy()

    data["VDW_contour"] = pd.read_csv(raw_dir / "VDW_contour_data.csv")
    data["PBE_contour"] = pd.read_csv(raw_dir / "PBE_contour_data.csv")
    data["HSE_contour"] = pd.read_csv(raw_dir / "HSE_contour_data.csv")

    data["Lorenzen_2011"] = pd.read_csv(raw_dir / "Lorenzen2011-Ideal-S-PBE.csv").values

    morales_points = pd.read_csv(raw_dir / "Morales_2013_nonIdeal_S.csv").values
    data["morales_high_pressure"] = morales_points[morales_points[:, 0] >= 200]
    data["morales_low_pressure"] = morales_points[
        (morales_points[:, 0] <= 200) & (morales_points[:, 0] >= 100)
    ]

    morales_line = pd.read_csv(raw_dir / "Morales_line.csv").values
    data["Morales_2013_line"] = morales_line[morales_line[:, 0] >= 1.0]

    schottler = pd.read_csv(raw_dir / "RR2018_points.csv").values
    data["Schottler_2018"] = schottler[schottler[:, 0] >= 1.0]

    return data


def backup_plot_inputs(data: dict, backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)

    data["df_vdw_model"].to_csv(
        backup_dir / "vdw_model_interaction_filtered.csv", index=False
    )
    data["df_pbe_model"].to_csv(
        backup_dir / "pbe_model_full_quadratic_filtered.csv", index=False
    )
    data["df_hse_model"].to_csv(
        backup_dir / "hse_model_full_quadratic_filtered.csv", index=False
    )

    data["VDW_contour"][data["VDW_contour"]["x_He"] == 0.089].to_csv(
        backup_dir / "vdw_contour_xHe_0.089.csv", index=False
    )
    data["PBE_contour"][data["PBE_contour"]["x_He"] == 0.089].to_csv(
        backup_dir / "pbe_contour_xHe_0.089.csv", index=False
    )
    data["HSE_contour"][data["HSE_contour"]["x_He"] == 0.089].to_csv(
        backup_dir / "hse_contour_xHe_0.089.csv", index=False
    )

    pd.DataFrame(data["Lorenzen_2011"]).to_csv(
        backup_dir / "lorenzen_2011_values.csv", index=False, header=False
    )
    pd.DataFrame(data["morales_high_pressure"]).to_csv(
        backup_dir / "morales_high_pressure_points.csv", index=False, header=False
    )
    pd.DataFrame(data["morales_low_pressure"]).to_csv(
        backup_dir / "morales_low_pressure_points.csv", index=False, header=False
    )
    pd.DataFrame(data["Morales_2013_line"]).to_csv(
        backup_dir / "morales_2013_line_filtered.csv", index=False, header=False
    )
    pd.DataFrame(data["Schottler_2018"]).to_csv(
        backup_dir / "schottler_2018_filtered.csv", index=False, header=False
    )


def make_plot(data: dict, show: bool = True) -> None:
    font = {"family": "serif", "size": 10}
    plt.rc("font", **font)

    a4_width_mm = 210
    fig_width_mm = a4_width_mm * 0.5
    fig_height_mm = a4_width_mm * 0.5 / 1.62

    fig_width_inches = fig_width_mm / 25.4
    fig_height_inches = fig_height_mm / 25.4

    fig, ax = plt.subplots(figsize=(fig_width_inches, fig_height_inches), dpi=500)

    colors_thermodynamic_model = ["#FF006E", "#3A86FF", "#FFBE0B"]
    labels = ["vdW-DF", "PBE", "HSE"]
    markers = ["s", "o", "^"]

    lines_for_legend1 = []

    for df_model, color, label, marker in zip(
        [data["df_vdw_model"], data["df_pbe_model"], data["df_hse_model"]],
        colors_thermodynamic_model,
        labels,
        markers,
    ):
        ax.fill_between(
            df_model["pressure_gpa"],
            df_model["upper_bound_k"] / 1e3,
            df_model["lower_bound_k"] / 1e3,
            color=color,
            alpha=0.1,
            edgecolor=color,
            linewidth=0,
            zorder=1,
        )

        (line,) = ax.plot(
            df_model["pressure_gpa"],
            df_model["mean_temp_k"] / 1e3,
            color=color,
            linewidth=2,
            marker=marker,
            markersize=5,
            label=label,
            ls="--",
            zorder=5,
        )
        lines_for_legend1.append(line)

        if label == "vdW-DF":
            subset = ax.scatter(
                [150, 200],
                [2.98, 3.33],
                color="pink",
                marker="*",
                alpha=1,
                lw=2,
                zorder=9,
                label="vdW-DF: subset",
            )
            lines_for_legend1.append(subset)

    contour_map = {
        "VDW": data["VDW_contour"],
        "PBE": data["PBE_contour"],
        "HSE": data["HSE_contour"],
    }

    for label, _ in PHASE_BOUNDARIES.items():
        contour_data = contour_map[label]
        contour_data = contour_data[contour_data["x_He"] == 0.089]
        ax.plot(
            contour_data["Pressure"],
            contour_data["Temperature"] / 1e3,
            color=COLORS[label],
            alpha=1,
            lw=2,
            zorder=3,
        )

    first_legend = ax.legend(
        handles=lines_for_legend1,
        fontsize=8,
        ncol=2,
        frameon=True,
        loc="lower right",
        facecolor="none",
        edgecolor="none",
    )
    ax.add_artist(first_legend)

    lines_for_legend2 = []

    (lorenzen,) = ax.plot(
        data["Lorenzen_2011"][:, 0] * 100,
        data["Lorenzen_2011"][:, 1],
        label="PBE, Lorenzen 2011",
        ls=(0, (5, 10)),
        marker="X",
        color="#8ECAE6",
        markersize=6,
        clip_on=True,
    )
    lines_for_legend2.append(lorenzen)

    morales_line, _, _ = ax.errorbar(
        data["morales_high_pressure"][:, 0],
        data["morales_high_pressure"][:, 1] / 1e3,
        yerr=data["morales_high_pressure"][:, 2] / 1e3,
        label="_nolegend_",
        ls="",
        marker="h",
        color="#3596b5",
        alpha=0.5,
        markersize=6,
        clip_on=True,
    )
    morales_line.set_label("PBE, Morales 2010 & 2013")
    lines_for_legend2.append(morales_line)

    ax.errorbar(
        data["morales_low_pressure"][:, 0],
        data["morales_low_pressure"][:, 1] / 1e3,
        xerr=data["morales_low_pressure"][:, 3],
        ls="",
        marker="h",
        color="#3596b5",
        alpha=0.5,
        markersize=6,
        clip_on=True,
    )

    ax.plot(
        data["Morales_2013_line"][:, 0] * 100,
        data["Morales_2013_line"][:, 1],
        ls=(0, (5, 10)),
        color="#3596b5",
        alpha=0.3,
        clip_on=True,
    )

    schottler_line, _, _ = ax.errorbar(
        data["Schottler_2018"][:, 0] * 100,
        data["Schottler_2018"][:, 1],
        yerr=data["Schottler_2018"][:, 2] / 1000,
        label="_nolegend_",
        ls="dashed",
        marker="d",
        color="#cdb4db",
        alpha=1.0,
        markersize=6,
        zorder=3,
        clip_on=True,
    )
    schottler_line.set_label("vdW-DF, Schottler 2018")
    lines_for_legend2.append(schottler_line)

    second_legend = ax.legend(
        handles=lines_for_legend2,
        fontsize=7,
        ncol=1,
        frameon=True,
        loc="upper right",
        facecolor="none",
        edgecolor="none",
    )
    ax.add_artist(second_legend)

    ax.grid(False)
    ax.set_xlabel(r"$P$ [GPa]", fontsize=9)
    ax.set_ylabel(r"$T$ [$10^3$ K]", fontsize=9)
    ax.set_ylim(2, 9)
    ax.set_xlim(130, 1000)

    ax.annotate(
        rf"$x_{{\mathrm{{He}}}}$=0.089",
        (500, 5),
        xytext=(5, 10),
        textcoords="offset points",
        ha="center",
        va="center",
        fontsize=8.5,
        color="black",
        zorder=6,
        bbox={"facecolor": "white", "edgecolor": "none"},
    )

    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone phase-boundary plotting script with local backups."
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Generate the plot without opening a GUI window.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    raw_dir = script_dir / "data_backup" / "raw"
    processed_dir = script_dir / "data_backup" / "plot_inputs_processed"

    data = load_data(raw_dir)
    backup_plot_inputs(data, processed_dir)
    make_plot(data, show=not args.no_show)


if __name__ == "__main__":
    main()
