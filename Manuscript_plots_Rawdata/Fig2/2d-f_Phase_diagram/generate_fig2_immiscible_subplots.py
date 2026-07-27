from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator


PHASE_BOUNDARIES = {
    "PBE": (
        np.array([100.0, 150.0, 200.0, 400.0, 600.0, 800.0, 1000.0]),
        np.array([4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 8000.0, 8000.0]),
    ),
    "VDW": (
        np.array([100.0, 150.0, 200.0, 400.0, 600.0, 800.0, 1000.0]),
        np.array([1000.0, 4000.0, 5000.0, 7000.0, 8000.0, 8000.0, 9000.0]),
    ),
    "HSE": (
        np.array([100.0, 150.0, 200.0, 400.0, 600.0, 800.0, 1000.0]),
        np.array([1000.0, 3000.0, 4000.0, 6000.0, 7000.0, 8000.0, 8000.0]),
    ),
}

CUSTOM_PURPLES_PALETTE = [
    "#c6dbef",
    "#9ecae1",
    "#6baed6",
    "#4292c6",
    "#2171b5",
    "#084594",
]

COLOR_JUPITER = "#ff5e5e"
COLOR_SATURN = "#c7522a"
COLOR_SATURN_MF20 = "#c7522a"

LABEL_POS = {
    ("PBE", 0.25): (680.0, 7.60),
    ("PBE", 0.20): (880.0, 7.20),
    ("PBE", 0.15): (780.0, 6.00),
    ("PBE", 0.089): (770.0, 5.10),
    ("PBE", 0.05): (820.0, 2.45),
    ("VDW", 0.25): (910.0, 8.10),
    ("VDW", 0.20): (850.0, 7.20),
    ("VDW", 0.15): (810.0, 6.35),
    ("VDW", 0.089): (760.0, 4.50),
    ("VDW", 0.05): (820.0, 2.45),
    ("HSE", 0.25): (700.0, 7.10),
    ("HSE", 0.20): (860.0, 7.10),
    ("HSE", 0.15): (810.0, 6.15),
    ("HSE", 0.089): (760.0, 4.60),
    ("HSE", 0.05): (820.0, 3.10),
}

GAP_WIDTH = {
    ("PBE", 0.25): 80.0,
    ("PBE", 0.20): 80.0,
    ("PBE", 0.15): 80.0,
    ("PBE", 0.089): 100.0,
    ("PBE", 0.05): 80.0,
    ("VDW", 0.25): 80.0,
    ("VDW", 0.20): 80.0,
    ("VDW", 0.15): 80.0,
    ("VDW", 0.089): 100.0,
    ("VDW", 0.05): 80.0,
    ("HSE", 0.25): 80.0,
    ("HSE", 0.20): 80.0,
    ("HSE", 0.15): 80.0,
    ("HSE", 0.089): 100.0,
    ("HSE", 0.05): 80.0,
}

SOURCE_FILES = {
    "Jupiter_Militzer": "Jupiter_Militzer.csv",
    "Jupiter_Nettelmann": "Jupiter_Nettelmann.csv",
    "Jupiter_RR_2018": "Redmer_2018_Jupiter_ada.csv",
    "Jupiter_today_2024": "Jupiter_today_HMH24.csv",
    "Saturn_today_2024": "Saturn_today_HMH24.csv",
    "Saturn_present_MF20": "Saturn_CM_JF_2020.csv",
    "Saturn_isentrope_Nettelmann": "Saturn_Nettelmann.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Figure 2 subplots and export all used data to one CSV."
    )
    parser.add_argument(
        "--base-data-dir",
        default="/Users/xiaoyuwang/Project/H-He",
        help="Base directory containing analysis_check and Previous_Studies_check",
    )
    parser.add_argument(
        "--export-csv",
        default=str(Path(__file__).resolve().parent / "fig2_all_used_data.csv"),
        help="Single CSV file containing all source/config/derived series data",
    )
    parser.add_argument("--show", action="store_true", help="Display the plot window")
    return parser.parse_args()


def to_xy(df: pd.DataFrame, name: str) -> np.ndarray:
    arr = df.to_numpy(dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"{name} must contain at least two numeric columns")
    return arr[:, :2]


def load_data(base_data_dir: Path) -> dict[str, pd.DataFrame | np.ndarray]:
    analysis_dir = base_data_dir / "analysis_check"
    studies_dir = base_data_dir / "Previous_Studies_check"

    data: dict[str, pd.DataFrame | np.ndarray] = {}
    for functional in ("PBE", "VDW", "HSE"):
        data[f"{functional}_contour"] = pd.read_csv(
            analysis_dir / f"{functional}_contour_data.csv"
        )

    for key, filename in SOURCE_FILES.items():
        data[key] = to_xy(pd.read_csv(studies_dir / filename), key)

    return data


def add_rows(
    rows: list[dict[str, float | str]],
    section: str,
    panel: str,
    series: str,
    x: np.ndarray,
    y: np.ndarray,
    y2: np.ndarray | None = None,
    functional: str = "",
    x_he: float | None = None,
) -> None:
    if y2 is None:
        y2 = np.full_like(y, np.nan, dtype=float)
    for xv, yv, y2v in zip(x, y, y2):
        rows.append(
            {
                "section": section,
                "panel": panel,
                "functional": functional,
                "series": series,
                "x_he": np.nan if x_he is None else float(x_he),
                "x": float(xv),
                "y": float(yv),
                "y2": float(y2v),
            }
        )


def compute_jupiter_envelope(
    data: dict[str, pd.DataFrame | np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    j_n = np.asarray(data["Jupiter_Nettelmann"], dtype=float)
    j_m = np.asarray(data["Jupiter_Militzer"], dtype=float)
    j_r = np.asarray(data["Jupiter_RR_2018"], dtype=float)

    p = np.union1d(np.union1d(j_n[:, 0] * 100.0, j_m[:, 0] * 100.0), j_r[:, 0] * 100.0)
    t_n = np.interp(p, j_n[:, 0] * 100.0, j_n[:, 1])
    t_m = np.interp(p, j_m[:, 0] * 100.0, j_m[:, 1])
    t_r = np.interp(p, j_r[:, 0] * 100.0, j_r[:, 1])
    upper = np.maximum(np.maximum(t_n, t_m), t_r)
    lower = np.minimum(np.minimum(t_n, t_m), t_r)
    return p, lower, upper


def collect_source_and_config_rows(
    data: dict[str, pd.DataFrame | np.ndarray],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []

    for functional in ("PBE", "VDW", "HSE"):
        contour = data[f"{functional}_contour"]
        assert isinstance(contour, pd.DataFrame)
        for _, r in contour.iterrows():
            rows.append(
                {
                    "section": "source",
                    "panel": functional,
                    "functional": functional,
                    "series": "contour_raw",
                    "x_he": float(r["x_He"]),
                    "x": float(r["Pressure"]),
                    "y": float(r["Temperature"]),
                    "y2": np.nan,
                }
            )

    for key in SOURCE_FILES:
        arr = np.asarray(data[key], dtype=float)
        add_rows(rows, "source", "ALL", key, arr[:, 0], arr[:, 1])

    for functional, (p, t) in PHASE_BOUNDARIES.items():
        add_rows(
            rows, "config", functional, "phase_boundary_k", p, t, functional=functional
        )

    for (functional, x_he), (x_text, y_text) in LABEL_POS.items():
        rows.append(
            {
                "section": "config",
                "panel": functional,
                "functional": functional,
                "series": "label_position",
                "x_he": float(x_he),
                "x": float(x_text),
                "y": float(y_text),
                "y2": float(GAP_WIDTH[(functional, x_he)]),
            }
        )

    return rows


def build_plot_and_collect_rows(
    data: dict[str, pd.DataFrame | np.ndarray],
    show: bool,
) -> list[dict[str, float | str]]:
    plt.rc("font", **{"family": "serif", "size": 8})

    subplot_width_in = 200 / 72.27
    subplot_height_in = 150 / 72.27
    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(subplot_width_in, subplot_height_in * 3),
        sharex=True,
        dpi=500,
    )
    plt.subplots_adjust(hspace=0.15)

    all_x = set()
    for functional in ("PBE", "VDW", "HSE"):
        contour = data[f"{functional}_contour"]
        assert isinstance(contour, pd.DataFrame)
        all_x.update(contour["x_He"].unique())
    x_sorted = sorted(all_x)
    color_map = dict(
        zip(x_sorted, sns.color_palette(CUSTOM_PURPLES_PALETTE, n_colors=len(x_sorted)))
    )

    j_p, j_low, j_up = compute_jupiter_envelope(data)
    rows: list[dict[str, float | str]] = []

    for ax, functional in zip(axes, ("PBE", "VDW", "HSE")):
        boundary_p, boundary_t = PHASE_BOUNDARIES[functional]
        boundary_t_1k = boundary_t / 1e3

        def boundary_interp(x_vals: np.ndarray) -> np.ndarray:
            return np.interp(x_vals, boundary_p, boundary_t_1k)

        ax.plot(boundary_p, boundary_t_1k, "k-", linewidth=2.5, zorder=11)
        add_rows(
            rows,
            "derived",
            functional,
            "phase_boundary_1k",
            boundary_p,
            boundary_t_1k,
            functional=functional,
        )

        contour = data[f"{functional}_contour"]
        assert isinstance(contour, pd.DataFrame)

        contour_x = contour.copy()
        contour_x["x_He"] = contour_x["x_He"].astype(float)
        for group_float in sorted(contour_x["x_He"].unique()):
            subset = contour_x[contour_x["x_He"] == group_float]
            group_key = round(group_float, 3)
            order = np.argsort(np.asarray(subset["Pressure"], dtype=float))
            subset = (
                subset.iloc[order].drop_duplicates("Pressure").reset_index(drop=True)
            )
            if subset.empty:
                continue

            color = color_map.get(group_float, "gray")
            x_raw = subset["Pressure"].to_numpy(dtype=float)
            y_raw = subset["Temperature"].to_numpy(dtype=float) / 1e3

            if len(x_raw) >= 3:
                x_line = np.linspace(x_raw.min(), x_raw.max(), 240)
                y_line = np.interp(x_line, x_raw, y_raw)
            else:
                x_line = x_raw
                y_line = y_raw

            if np.isclose(group_float, 0.05):
                x_pad = np.linspace(100.0, x_line.min(), 120)
                x_fill = np.concatenate((x_pad, x_line))
                y_fill = np.concatenate((np.zeros_like(x_pad), y_line))
                y_boundary = boundary_interp(x_fill)
                ax.fill_between(
                    x_fill,
                    y_fill,
                    y_boundary,
                    color=color,
                    alpha=0.30,
                    edgecolor="none",
                    zorder=2,
                )
                ax.fill_between(
                    x_fill,
                    0,
                    y_fill,
                    color=color,
                    alpha=0.20,
                    edgecolor="none",
                    zorder=2,
                )
                add_rows(
                    rows,
                    "derived",
                    functional,
                    "xhe_fill_upper",
                    x_fill,
                    y_fill,
                    y_boundary,
                    functional,
                    group_float,
                )
                add_rows(
                    rows,
                    "derived",
                    functional,
                    "xhe_fill_lower",
                    x_fill,
                    np.zeros_like(y_fill),
                    y_fill,
                    functional,
                    group_float,
                )
            else:
                y_boundary = boundary_interp(x_line)
                ax.fill_between(
                    x_line,
                    y_line,
                    y_boundary,
                    color=color,
                    alpha=0.30,
                    edgecolor="none",
                    zorder=2,
                )
                add_rows(
                    rows,
                    "derived",
                    functional,
                    "xhe_fill_upper",
                    x_line,
                    y_line,
                    y_boundary,
                    functional,
                    group_float,
                )

            x_text, y_text = LABEL_POS.get(
                (functional, group_key),
                (
                    float(subset["Pressure"].iloc[-1]) * 0.92,
                    float(subset["Temperature"].iloc[-1]) / 1e3,
                ),
            )
            gap = GAP_WIDTH.get((functional, group_key), 36.0)
            mask_left = x_line < (x_text - gap / 2.0)
            mask_right = x_line > (x_text + gap / 2.0)

            ax.plot(x_line[mask_left], y_line[mask_left], color=color, lw=1.5, zorder=6)
            ax.plot(
                x_line[mask_right], y_line[mask_right], color=color, lw=1.5, zorder=6
            )

            add_rows(
                rows,
                "derived",
                functional,
                "xhe_line_left",
                x_line[mask_left],
                y_line[mask_left],
                functional=functional,
                x_he=group_float,
            )
            add_rows(
                rows,
                "derived",
                functional,
                "xhe_line_right",
                x_line[mask_right],
                y_line[mask_right],
                functional=functional,
                x_he=group_float,
            )

            if np.isclose(group_float, 0.089):
                label = "0.089"
            elif np.isclose(group_float, 0.05):
                label = "0.05"
            else:
                label = f"{group_float:.2f}"
            ax.text(
                x_text,
                y_text,
                label,
                weight="bold",
                color=color,
                fontsize=6,
                ha="center",
                va="center",
                zorder=12,
            )

        ax.fill_between(
            j_p, j_low, j_up, color=(1.0, 0.3686, 0.3686, 0.2), zorder=15, clip_on=True
        )
        add_rows(
            rows, "derived", functional, "jupiter_adiabat_envelope", j_p, j_low, j_up
        )

        jup = np.asarray(data["Jupiter_today_2024"], dtype=float)
        sat = np.asarray(data["Saturn_today_2024"], dtype=float)
        sat_mf20 = np.asarray(data["Saturn_present_MF20"], dtype=float)
        sat_isen = np.asarray(data["Saturn_isentrope_Nettelmann"], dtype=float)

        ax.plot(
            jup[:, 0] * 100,
            jup[:, 1] / 1000,
            ls="dashed",
            color=COLOR_JUPITER,
            linewidth=1,
            zorder=15,
        )
        ax.plot(
            sat[:, 0] * 100,
            sat[:, 1] / 1000,
            ls="dashed",
            color=COLOR_SATURN,
            linewidth=1,
            zorder=15,
        )
        ax.plot(
            sat_mf20[:, 0] * 100,
            sat_mf20[:, 1],
            ls="dotted",
            color=COLOR_SATURN_MF20,
            linewidth=1,
            zorder=15,
        )
        ax.plot(
            sat_isen[:, 0] * 100,
            sat_isen[:, 1],
            ls="-",
            color=COLOR_SATURN,
            linewidth=1,
            alpha=0.8,
            zorder=15,
        )

        add_rows(
            rows,
            "derived",
            functional,
            "jupiter_today",
            jup[:, 0] * 100,
            jup[:, 1] / 1000,
        )
        add_rows(
            rows,
            "derived",
            functional,
            "saturn_today",
            sat[:, 0] * 100,
            sat[:, 1] / 1000,
        )
        add_rows(
            rows,
            "derived",
            functional,
            "saturn_mf20",
            sat_mf20[:, 0] * 100,
            sat_mf20[:, 1],
        )
        add_rows(
            rows,
            "derived",
            functional,
            "saturn_isentrope",
            sat_isen[:, 0] * 100,
            sat_isen[:, 1],
        )

        if functional == "PBE":
            x1 = np.array([100.0, 150.0, 200.0, 300.0])
            y1 = np.array([2.0, 1.5, 1.1, 0.74])
            y2 = np.array([1795.91, 1437.13, 991.08, 708.80]) / 1000.0
            ax.plot(
                x1,
                y1,
                color="#3eb59d",
                marker="3",
                ls="dotted",
                markersize=6,
                fillstyle="none",
                zorder=3,
            )
            ax.plot(
                x1, y2, marker="^", markersize=3, linestyle="dotted", color="#bc5090"
            )
            add_rows(rows, "derived", functional, "atomic_h_line_a", x1, y1)
            add_rows(rows, "derived", functional, "atomic_h_line_b", x1, y2)
            ax.text(
                0.05,
                0.95,
                "PBE",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
            )

        elif functional == "HSE":
            x1 = np.array([100.0, 150.0, 200.0, 300.0])
            y1 = np.array([3.3, 2.5, 1.9, 1.5])
            y2 = np.array([2864.74, 2263.12, 1794.11, 1344.13]) / 1000.0
            ax.fill_between(
                [100, 150],
                [0, 1],
                [1, 3],
                color="#4292c6",
                alpha=0.3,
                zorder=2,
                edgecolor="none",
            )
            ax.plot(
                x1,
                y1,
                color="#3eb59d",
                marker="3",
                ls="dotted",
                markersize=6,
                fillstyle="none",
                zorder=3,
            )
            ax.plot(
                x1, y2, marker="^", markersize=3, linestyle="dotted", color="#bc5090"
            )
            add_rows(
                rows,
                "derived",
                functional,
                "lowp_fill",
                np.array([100.0, 150.0]),
                np.array([0.0, 1.0]),
                np.array([1.0, 3.0]),
            )
            add_rows(rows, "derived", functional, "atomic_h_line_a", x1, y1)
            add_rows(rows, "derived", functional, "atomic_h_line_b", x1, y2)
            ax.text(
                0.05,
                0.95,
                "HSE",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
            )

        else:
            x1 = np.array([100.0, 150.0, 200.0, 300.0])
            y1 = np.array([3.2, 2.5, 1.9, 1.5])
            y2 = np.array([2838.62, 2244.41, 1801.67, 1380.80]) / 1000.0
            sx = np.array([150.0, 300.0])
            sy = np.array([2.19, 0.99])
            ax.fill_between(
                [100, 150],
                [0, 1],
                [1, 4],
                color="#2171b5",
                alpha=0.3,
                zorder=2,
                edgecolor="none",
            )
            ax.plot(
                x1,
                y1,
                color="#3eb59d",
                marker="3",
                ls="dotted",
                markersize=6,
                fillstyle="none",
                zorder=3,
            )
            ax.plot(
                x1, y2, marker="^", markersize=3, linestyle="dotted", color="#bc5090"
            )
            ax.scatter(sx, sy, marker="*", color="cyan", s=25, zorder=11, clip_on=False)
            add_rows(
                rows,
                "derived",
                functional,
                "lowp_fill",
                np.array([100.0, 150.0]),
                np.array([0.0, 1.0]),
                np.array([1.0, 4.0]),
            )
            add_rows(rows, "derived", functional, "atomic_h_line_a", x1, y1)
            add_rows(rows, "derived", functional, "atomic_h_line_b", x1, y2)
            add_rows(rows, "derived", functional, "special_star", sx, sy)
            ax.text(
                0.05,
                0.95,
                "vdW-DF",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
            )

        ax.set(xlim=(100, 1000), ylim=(1, 9.5))
        ax.yaxis.set_major_locator(MultipleLocator(2))
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.minorticks_on()
        ax.tick_params(which="both", top=False, right=False)
        ax.set_ylabel(r"$T$ [$10^3$ K]", fontsize=8)
        ax.label_outer()

    axes[-1].set_xlabel(r"$P$ [GPa]")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return rows


def main() -> None:
    args = parse_args()
    data = load_data(Path(args.base_data_dir))

    all_rows = collect_source_and_config_rows(data)
    all_rows.extend(build_plot_and_collect_rows(data, args.show))

    export_csv = Path(args.export_csv)
    export_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(export_csv, index=False)

    print(f"Single CSV export saved: {export_csv}")
    print("Plot not saved.")


if __name__ == "__main__":
    main()
