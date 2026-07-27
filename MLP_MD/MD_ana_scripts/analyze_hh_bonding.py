import csv
import os
import re
import glob

import numpy as np
import ovito.data
import ovito.io


def smooth_cutoff(r, r1=0.8, r2=1.1):
    if r <= r1:
        return 1.0
    if r >= r2:
        return 0.0
    return 0.5 * (1 + np.cos(np.pi * (r - r1) / (r2 - r1)))


def parse_folder_name(folder_name):
    match = re.match(r"H-(\d+)-P-(\d+)-T-(\d+)$", folder_name)
    if not match:
        return None
    h_count, pressure, temperature = match.groups()
    return int(h_count), int(pressure), int(temperature)


def analyze_hh_bonding(trajectory_file, h_atom_type=1, r1=0.8, r2=1.1):
    pipeline = ovito.io.import_file(trajectory_file, multiple_frames=True)

    frame_order_parameters = []
    total_h_atoms = None

    for frame_index in range(pipeline.source.num_frames):
        data = pipeline.compute(frame_index)
        particle_types = data.particles["Particle Type"].array
        total_h_atoms = int(np.count_nonzero(particle_types == h_atom_type))

        if total_h_atoms == 0:
            continue

        h_indices = np.where(particle_types == h_atom_type)[0]
        finder = ovito.data.CutoffNeighborFinder(r2, data)

        atoms_with_one_neighbor = 0
        for h_idx in h_indices:
            true_neighbor_count = 0
            for neigh in finder.find(h_idx):
                if (
                    particle_types[neigh.index] == h_atom_type
                    and smooth_cutoff(neigh.distance, r1=r1, r2=r2) > 0.5
                ):
                    true_neighbor_count += 1
            if true_neighbor_count == 1:
                atoms_with_one_neighbor += 1

        frame_order_parameters.append(atoms_with_one_neighbor / total_h_atoms)

    if not frame_order_parameters:
        return total_h_atoms, None, []

    last_30_percent_count = max(1, int(len(frame_order_parameters) * 0.3))
    avg_last_30 = float(np.mean(frame_order_parameters[-last_30_percent_count:]))
    return total_h_atoms, avg_last_30, frame_order_parameters


def interpolate_target_temperature(points, target=0.8):
    """
    points: list of (temperature, order_parameter), sorted by temperature.
    Returns list of interpolated temperatures where order parameter reaches target.
    """
    crossings = []
    for i in range(len(points) - 1):
        t1, y1 = points[i]
        t2, y2 = points[i + 1]

        if y1 == target:
            crossings.append(float(t1))
            continue

        if y2 == target:
            crossings.append(float(t2))
            continue

        if (y1 - target) * (y2 - target) < 0:
            frac = (target - y1) / (y2 - y1)
            t_cross = t1 + frac * (t2 - t1)
            crossings.append(float(t_cross))

    deduped = []
    for val in crossings:
        if not deduped or abs(val - deduped[-1]) > 1e-9:
            deduped.append(val)
    return deduped


def main():
    base_dir = "."
    h_atom_type = 1
    target_order_parameter = 0.8
    r1 = 0.8
    r2 = 1.1

    folders = sorted(
        [
            name
            for name in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, name)) and name.startswith("H-")
        ]
    )

    jobs = []

    for folder in folders:
        parsed = parse_folder_name(folder)
        if parsed is None:
            print(f"Skip invalid folder name: {folder}")
            continue

        _, pressure, temperature = parsed
        traj_pattern = os.path.join(base_dir, folder, "HHe_PIMD_*K_centroid_converted.xyz")
        traj_files = glob.glob(traj_pattern)
        if not traj_files:
            print(f"Skip missing file: {traj_pattern}")
            continue
        traj = traj_files[0]

        jobs.append(
            {
                "directory": folder,
                "pressure_GPa": pressure,
                "temperature_K": temperature,
                "trajectory": traj,
            }
        )

    jobs.sort(key=lambda r: (r["pressure_GPa"], r["temperature_K"], r["directory"]))

    by_pressure = {}
    all_rows = []

    print(f"\n===== Running analysis for r2 = {r2:.2f}, target = {target_order_parameter} =====")
    for job in jobs:
        print(f"Analyzing: {job['trajectory']}")
        total_h_atoms, avg_order, frame_orders = analyze_hh_bonding(
            job["trajectory"], h_atom_type=h_atom_type, r1=r1, r2=r2
        )

        row = {
            "directory": job["directory"],
            "pressure_GPa": job["pressure_GPa"],
            "temperature_K": job["temperature_K"],
            "r1": r1,
            "r2": r2,
            "total_h_atoms": total_h_atoms if total_h_atoms is not None else "",
            "avg_order_parameter_last30pct": avg_order
            if avg_order is not None
            else "",
            "num_frames_used": len(frame_orders),
        }
        all_rows.append(row)

        if avg_order is not None:
            by_pressure.setdefault(job["pressure_GPa"], []).append(
                (job["temperature_K"], avg_order)
            )

    summary_csv = os.path.join(base_dir, "hh_order_parameter_r2_1.10.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "directory",
                "pressure_GPa",
                "temperature_K",
                "r1",
                "r2",
                "total_h_atoms",
                "avg_order_parameter_last30pct",
                "num_frames_used",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved: {summary_csv}")

    interp_rows = []
    for pressure in sorted(by_pressure):
        points = sorted(by_pressure[pressure], key=lambda x: x[0])
        crossings = interpolate_target_temperature(points, target=target_order_parameter)

        if crossings:
            for cross_t in crossings:
                interp_row = {
                    "pressure_GPa": pressure,
                    "r1": r1,
                    "r2": r2,
                    "target_order_parameter": target_order_parameter,
                    "interpolated_temperature_K": round(cross_t, 2),
                    "status": "ok",
                    "num_points": len(points),
                }
                interp_rows.append(interp_row)
        else:
            y_values = [y for _, y in points]
            interp_row = {
                "pressure_GPa": pressure,
                "r1": r1,
                "r2": r2,
                "target_order_parameter": target_order_parameter,
                "interpolated_temperature_K": "",
                "status": f"no_crossing_in_range_[{min(y_values):.4f},{max(y_values):.4f}]",
                "num_points": len(points),
            }
            interp_rows.append(interp_row)

    interp_csv = os.path.join(base_dir, "hh_target_temperature_p0.8_r2_1.10.csv")
    with open(interp_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pressure_GPa",
                "r1",
                "r2",
                "target_order_parameter",
                "interpolated_temperature_K",
                "status",
                "num_points",
            ],
        )
        writer.writeheader()
        writer.writerows(interp_rows)

    print(f"Saved: {interp_csv}")
    print("\n===== Results =====")
    for row in interp_rows:
        print(f"Pressure {row['pressure_GPa']} GPa: T = {row['interpolated_temperature_K']} K ({row['status']})")


if __name__ == "__main__":
    main()
