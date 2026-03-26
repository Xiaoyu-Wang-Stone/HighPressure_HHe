#!/usr/bin/env python3

import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata


matplotlib.use("Agg")


PHASE_SEP_DATA_VDW_FULL = np.array([
    [150, 1000, 0.094, 0.546],
    [150, 2000, 0.067, 0.469],
    [150, 3000, 0.075, 0.504],
    [150, 4000, 0.121, 0.516],

    [200, 2000, 0.063, 0.562],
    [200, 3000, 0.069, 0.567],
    [200, 4000, 0.104, 0.581],
    [200, 5000, 0.139, 0.556],

    [400, 2000, 0.034, 0.805],
    [400, 3000, 0.072, 0.804],
    [400, 4000, 0.071, 0.758],
    [400, 5000, 0.107, 0.754],
    [400, 6000, 0.153, 0.698],
    [400, 7000, 0.256, 0.602],

    [600, 2000, 0.034, 0.824],
    [600, 3000, 0.067, 0.865],
    [600, 4000, 0.095, 0.808],
    [600, 5000, 0.106, 0.806],
    [600, 6000, 0.133, 0.800],
    [600, 7000, 0.189, 0.746],
    [600, 8000, 0.308, 0.635],
    [800, 2000, 0.035, 0.803],
    [800, 3000, 0.062, 0.886],
    [800, 4000, 0.072, 0.864],
    [800, 5000, 0.102, 0.861],
    [800, 6000, 0.137, 0.807],
    [800, 7000, 0.190, 0.770],
    [800, 8000, 0.256, 0.742],

    [1000, 2000, 0.040, 0.735],
    [1000, 3000, 0.072, 0.641],
    [1000, 4000, 0.076, 0.738],
    [1000, 5000, 0.106, 0.808],
    [1000, 6000, 0.135, 0.811],
    [1000, 7000, 0.153, 0.805],
    [1000, 8000, 0.218, 0.762],
    [1000, 9000, 0.375, 0.623]
], dtype=float)


UPPER_BOUND_VDW = (
    np.array([100.0, 150.0, 200.0, 400.0, 600.0, 800.0, 1000.0], dtype=float),
    np.array([1000.0, 4000.0, 5000.0, 7000.0, 8000.0, 8000.0, 9000.0], dtype=float),
)


def get_upper_bound_temperature(pressure: float) -> float:
    p_ref, t_upper_ref = UPPER_BOUND_VDW
    return float(np.interp(pressure, p_ref, t_upper_ref))


def fit_temperature_at_p150(
    data: np.ndarray, x_target: float, t_min: float = 2000.0
) -> float | None:
    t_upper = get_upper_bound_temperature(150.0)
    mask = (data[:, 0] == 150.0) & (data[:, 1] >= t_min) & (data[:, 1] <= t_upper)
    subset = data[mask]
    if subset.shape[0] < 2:
        raise ValueError(
            "Need at least 2 points at 150 GPa within [T_min, T_upper] for fitting"
        )

    temperature = subset[:, 1]
    composition = subset[:, 2]
    slope, intercept = np.polyfit(temperature, composition, 1)
    if np.isclose(slope, 0.0):
        raise ValueError("Linear fit slope is too close to zero at 150 GPa")

    t_fit = float((x_target - intercept) / slope)
    if t_fit < t_min or t_fit > t_upper:
        return None
    return t_fit


def collapse_duplicate_pressure(
    points: np.ndarray, round_decimals: int = 6
) -> np.ndarray:
    frame = pd.DataFrame(
        {
            "Pressure": points[:, 0],
            "Temperature": points[:, 1],
        }
    )
    frame["P_key"] = frame["Pressure"].round(round_decimals)
    collapsed = frame.groupby("P_key", as_index=False).agg(
        Pressure=("Pressure", "mean"),
        Temperature=("Temperature", "max"),
    )
    collapsed_points = np.asarray(collapsed[["Pressure", "Temperature"]], dtype=float)
    return collapsed_points[np.argsort(collapsed_points[:, 0])]


def sample_contour_every_50_gpa(points: np.ndarray, step: float = 50.0) -> np.ndarray:
    if points.shape[0] < 2:
        return points

    p = points[:, 0]
    t = points[:, 1]
    p_min = np.ceil(p.min() / step) * step
    p_max = np.floor(p.max() / step) * step
    if p_min > p_max:
        return points

    p_uniform = np.arange(p_min, p_max + 0.5 * step, step)
    t_uniform = np.interp(p_uniform, p, t)
    return np.column_stack((p_uniform, t_uniform))


def extract_contour_df(
    data: np.ndarray,
    x_targets: list[float],
    grid_points: int = 500,
) -> pd.DataFrame:
    pressure = data[:, 0]
    temperature = data[:, 1]
    composition = data[:, 2]

    p_grid, t_grid = np.meshgrid(
        np.linspace(pressure.min(), pressure.max(), grid_points),
        np.linspace(temperature.min(), temperature.max(), grid_points),
    )
    c_grid = griddata(
        (pressure, temperature), composition, (p_grid, t_grid), method="linear"
    )

    contour_frames = []
    fig = plt.figure(figsize=(6, 5))
    try:
        for x_target in x_targets:
            contour = plt.contour(
                p_grid, t_grid, c_grid, levels=[x_target], colors="white", linewidths=1
            )
            segs = contour.allsegs[0] if contour.allsegs else []
            if not segs:
                continue

            contour_coords = max(segs, key=lambda a: a.shape[0])
            sorted_contour = contour_coords[np.argsort(contour_coords[:, 0])]

            if x_target >= 0.089:
                t_fit = fit_temperature_at_p150(data, x_target, t_min=2000.0)
                if t_fit is not None:
                    fit_point = np.array([[150.0, t_fit]])
                    sorted_contour = np.vstack([sorted_contour, fit_point])
                    sorted_contour = sorted_contour[np.argsort(sorted_contour[:, 0])]

            if not np.all(np.diff(sorted_contour[:, 1]) >= 0):
                sorted_contour[:, 1] = np.maximum.accumulate(sorted_contour[:, 1])

            sorted_contour = collapse_duplicate_pressure(
                sorted_contour, round_decimals=6
            )
            sorted_contour = sample_contour_every_50_gpa(sorted_contour)

            frame = pd.DataFrame(
                {
                    "Pressure": sorted_contour[:, 0],
                    "Temperature": sorted_contour[:, 1],
                }
            )
            frame["x_He"] = x_target
            contour_frames.append(frame)
    finally:
        plt.close(fig)

    if not contour_frames:
        raise RuntimeError("No contour lines extracted for the requested x_He levels")

    return pd.concat(contour_frames, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate VDW_contour_data.csv from hardcoded VDW phase-separation data"
    )
    parser.add_argument(
        "--output",
        default="VDW_contour_data.csv",
        help="Output CSV path (default: VDW_contour_data.csv)",
    )
    parser.add_argument(
        "--x-targets",
        nargs="+",
        type=float,
        default=[0.05, 0.089, 0.15, 0.2, 0.25],
        help="x_He contour levels",
    )
    parser.add_argument(
        "--grid-points",
        type=int,
        default=500,
        help="Number of interpolation grid points per axis",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contour_df = extract_contour_df(
        PHASE_SEP_DATA_VDW_FULL,
        args.x_targets,
        args.grid_points,
    )
    contour_df.to_csv(args.output, index=False)

    fitted_targets = [x for x in args.x_targets if x >= 0.089]
    print(f"Saved {args.output} with {len(contour_df)} rows")
    print(f"x_He levels: {sorted(contour_df['x_He'].unique().tolist())}")
    t_upper_150 = get_upper_bound_temperature(150.0)
    print(f"150 GPa fitting range: 2000 K <= T <= {t_upper_150:.1f} K")
    for x_target in fitted_targets:
        t_fit = fit_temperature_at_p150(PHASE_SEP_DATA_VDW_FULL, x_target, t_min=2000.0)
        if t_fit is None:
            print(
                f"150 GPa fitted T for x_He={x_target:.3f}: out of [2000 K, {t_upper_150:.1f} K], skipped"
            )
        else:
            print(
                f"150 GPa fitted T for x_He={x_target:.3f} using T>=2000 K: {t_fit:.3f} K"
            )


if __name__ == "__main__":
    main()
