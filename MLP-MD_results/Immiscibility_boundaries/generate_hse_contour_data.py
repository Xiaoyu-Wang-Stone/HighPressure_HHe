#!/usr/bin/env python3

import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata


matplotlib.use("Agg")


PHASE_SEP_DATA_HSE_FULL = np.array(
    [
        # [Pressure, Temperature, First zero crossing from left, First zero crossing from right]
        [150, 2000, 0.109, 0.224],
        [150, 3000, 0.076, 0.244],
        [200, 2000, 0.064, 0.376],
        [200, 3000, 0.075, 0.494],
        [200, 4000, 0.105, 0.551],
        [200, 5000, 0.232, 0.395],
        [400, 2000, 0.039, 0.684],
        [400, 3000, 0.068, 0.803],
        [400, 4000, 0.076, 0.756],
        [400, 5000, 0.118, 0.736],
        [400, 6000, 0.187, 0.623],
        [400, 7000, 0.379, 0.459],
        [600, 2000, 0.038, 0.800],
        [600, 3000, 0.047, 0.804],
        [600, 4000, 0.071, 0.804],
        [600, 5000, 0.105, 0.798],
        [600, 6000, 0.141, 0.771],
        [600, 7000, 0.270, 0.596],
        [600, 8000, 0.476, 0.521],
        [800, 2000, 0.037, 0.669],
        [800, 3000, 0.071, 0.807],
        [800, 4000, 0.072, 0.832],
        [800, 5000, 0.101, 0.815],
        [800, 6000, 0.139, 0.801],
        [800, 7000, 0.207, 0.737],
        [800, 8000, 0.263, 0.583],
        [1000, 2000, 0.041, 0.645],
        [1000, 3000, 0.065, 0.862],
        [1000, 4000, 0.071, 0.863],
        [1000, 5000, 0.106, 0.826],
        [1000, 6000, 0.128, 0.807],
        [1000, 7000, 0.168, 0.785],
        [1000, 8000, 0.249, 0.575],
    ],
    dtype=float,
)


UPPER_BOUND_HSE = (
    np.array([200.0, 400.0, 600.0, 800.0, 1000.0], dtype=float),
    np.array([4000.0, 6000.0, 7000.0, 8000.0, 8000.0], dtype=float),
)


def get_upper_bound_temperature(pressure: float | np.ndarray) -> float | np.ndarray:
    p_ref, t_upper_ref = UPPER_BOUND_HSE
    return np.interp(pressure, p_ref, t_upper_ref)


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
    t_upper_grid = get_upper_bound_temperature(p_grid)
    c_grid = np.where(t_grid <= t_upper_grid, c_grid, np.nan)

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

            t_upper_contour = get_upper_bound_temperature(sorted_contour[:, 0])
            sorted_contour[:, 1] = np.minimum(sorted_contour[:, 1], t_upper_contour)

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
        description="Generate HSE_contour_data.csv from hardcoded HSE phase-separation data"
    )
    parser.add_argument(
        "--output",
        default="HSE_contour_data.csv",
        help="Output CSV path (default: HSE_contour_data.csv)",
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
        PHASE_SEP_DATA_HSE_FULL,
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
        t_fit = fit_temperature_at_p150(PHASE_SEP_DATA_HSE_FULL, x_target, t_min=2000.0)
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
