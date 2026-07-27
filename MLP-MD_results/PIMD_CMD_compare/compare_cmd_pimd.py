import glob
import os
import re

import matplotlib
import numpy as np
from matplotlib.lines import Line2D
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rc("font", **{"family": "serif", "size": 11})


def fit_ornstein_zernike(k2, s0, xi):
    return s0 / (1.0 + (xi**2) * k2)


KB_EV = 8.617333262e-5
TARGET_N = 2560
DEFAULT_KCUT = 0.5


def parse_file_meta(path):
    name = os.path.basename(path)
    m_h = re.search(r"H-(\d+)", name)
    m_p = re.search(r"P-(\d+)", name)
    m_t = re.search(r"T-(\d+)", name)
    m_pair = re.search(r"Sk-(HeHe|HeH|HH)-", name)
    if not (m_h and m_p and m_t and m_pair):
        return None
    return int(m_h.group(1)), int(m_p.group(1)), int(m_t.group(1)), m_pair.group(1)


def scan_files_by_pair(work_dir, fixed_p, target_t):
    out = {"HeHe": {}, "HeH": {}, "HH": {}}
    for fpath in glob.glob(os.path.join(work_dir, "*.list")):
        meta = parse_file_meta(fpath)
        if meta is None:
            continue
        h, p, t, pair = meta
        if p != fixed_p or t != target_t:
            continue
        out[pair][h] = fpath
    return out


def fit_s0_from_file(filepath, kcut):
    data = np.loadtxt(filepath)
    if data.ndim != 2 or data.shape[1] < 4 or len(data) < 6:
        return None

    sliced = data[1:] if np.allclose(data[0, :3], 0.0) else data
    ksqr = sliced[:, 0] ** 2 + sliced[:, 1] ** 2 + sliced[:, 2] ** 2
    mask = ksqr < kcut
    if np.count_nonzero(mask) < 5:
        return None

    x = ksqr[mask]
    y = sliced[mask, 3]
    try:
        p0 = [max(y[0], 1e-6), 1.0]
        popt, _ = curve_fit(fit_ornstein_zernike, x, y, p0=p0, maxfev=8000)
    except Exception:
        popt, _ = curve_fit(fit_ornstein_zernike, x, y, maxfev=8000)

    if np.any(~np.isfinite(popt)):
        return None
    return float(popt[0])


def _to_xy_map(arr):
    x = np.round(arr[:, 0], 4)
    y = arr[:, 1]
    ux = np.unique(x)
    y_mean = np.array([np.mean(y[x == v]) for v in ux], dtype=float)
    return ux, y_mean


def derivative_curve(dataset):
    if not all(k in dataset for k in ("HeHe", "HeH", "HH")):
        return None, None

    x_aa, s_aa = _to_xy_map(dataset["HeHe"])
    x_ab, s_ab = _to_xy_map(dataset["HeH"])
    x_bb, s_bb = _to_xy_map(dataset["HH"])

    common_x = np.array(sorted(set(x_aa) & set(x_ab) & set(x_bb)))
    if len(common_x) < 5:
        return None, None

    f_aa = PchipInterpolator(x_aa, s_aa, extrapolate=False)
    f_ab = PchipInterpolator(x_ab, s_ab, extrapolate=False)
    f_bb = PchipInterpolator(x_bb, s_bb, extrapolate=False)
    saa = f_aa(common_x)
    sab = f_ab(common_x)
    sbb = f_bb(common_x)

    xb = 1.0 - common_x
    denom = xb * saa + common_x * sbb - 2.0 * np.sqrt(common_x * xb) * sab
    valid = np.isfinite(denom) & (np.abs(denom) > 1e-6)
    x = common_x[valid]
    if len(x) < 5:
        return None, None

    deriv_raw = 1.0 / denom[valid]
    deriv_despiked = median_filter(deriv_raw, size=3, mode="nearest")

    uniq_idx = np.unique(x, return_index=True)[1]
    x = x[uniq_idx]
    deriv_despiked = deriv_despiked[uniq_idx]
    if len(x) < 5:
        return None, None

    x_grid = np.linspace(x.min(), x.max(), 140)
    deriv_dense = PchipInterpolator(x, deriv_despiked, extrapolate=False)(x_grid)
    good = np.isfinite(deriv_dense)
    if np.count_nonzero(good) < 8:
        return None, None
    x_grid = x_grid[good]
    deriv_dense = deriv_dense[good]

    n = len(deriv_dense)
    win = min(15, n if n % 2 == 1 else n - 1)
    if win < 5:
        return x_grid, np.maximum(deriv_dense, 0.0)
    poly = min(3, win - 2)
    deriv_smooth = savgol_filter(
        deriv_dense, window_length=win, polyorder=poly, mode="interp"
    )
    return x_grid, np.maximum(deriv_smooth, 0.0)


def compute_mu_and_gmix(x, gamma, target_t):
    mask = (x > 1e-3) & (x < 1.0 - 1e-3)
    x_use = x[mask]
    g_use = gamma[mask]
    if len(x_use) < 10:
        return None

    beta_mu_he = cumulative_trapezoid(g_use / x_use, x_use, initial=0.0)
    beta_mu_he -= beta_mu_he[-1]
    mu_he = beta_mu_he * KB_EV * target_t

    beta_mu_h = cumulative_trapezoid(-g_use / (1.0 - x_use), x_use, initial=0.0)
    beta_mu_h -= beta_mu_h[0]
    mu_h = beta_mu_h * KB_EV * target_t

    g_mix = x_use * mu_he + (1.0 - x_use) * mu_h
    x_final = np.concatenate([[0.0], x_use, [1.0]])
    g_final = np.concatenate([[0.0], g_mix, [0.0]])
    return x_use, mu_he, mu_h, x_final, g_final


def largest_gmix_difference(cmd_result, pimd_result):
    x_cmd, g_cmd = cmd_result[3], cmd_result[4]
    x_pimd, g_pimd = pimd_result[3], pimd_result[4]

    lo = max(np.min(x_cmd), np.min(x_pimd))
    hi = min(np.max(x_cmd), np.max(x_pimd))
    if hi <= lo:
        return None

    f_cmd = PchipInterpolator(x_cmd, g_cmd, extrapolate=False)
    f_pimd = PchipInterpolator(x_pimd, g_pimd, extrapolate=False)
    x_eval = np.linspace(lo, hi, 600)
    g_cmd_eval = f_cmd(x_eval)
    g_pimd_eval = f_pimd(x_eval)
    valid = np.isfinite(g_cmd_eval) & np.isfinite(g_pimd_eval)
    if not np.any(valid):
        return None

    x_eval = x_eval[valid]
    g_cmd_eval = g_cmd_eval[valid]
    g_pimd_eval = g_pimd_eval[valid]
    diff = g_pimd_eval - g_cmd_eval
    idx = np.argmax(np.abs(diff))
    return {
        "x": float(x_eval[idx]),
        "cmd": float(g_cmd_eval[idx]),
        "pimd": float(g_pimd_eval[idx]),
        "delta": float(diff[idx]),
        "abs_delta": float(abs(diff[idx])),
    }


def build_dataset_one_method(cfg, fixed_p, target_t):
    indexed = scan_files_by_pair(cfg["dir"], fixed_p, target_t)
    common_h = sorted(set(indexed["HeHe"]) & set(indexed["HeH"]) & set(indexed["HH"]))

    out = {}
    for pair in ("HeHe", "HeH", "HH"):
        rows = []
        for h in common_h:
            fpath = indexed[pair][h]
            s0 = fit_s0_from_file(fpath, cfg["kcut"])
            if s0 is None:
                continue
            x_he = 1.0 - h / float(TARGET_N)
            rows.append([x_he, s0])
        if rows:
            out[pair] = np.array(rows, dtype=float)
    return out, len(common_h)


def discover_conditions(root_dir):
    conditions = []
    for d in sorted(glob.glob(os.path.join(root_dir, "P*-T*"))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d.rstrip(os.sep))
        m = re.fullmatch(r"P(\d+)-T(\d+)", name)
        if not m:
            continue
        conditions.append(
            {
                "name": name,
                "path": d,
                "P": int(m.group(1)),
                "T": int(m.group(2)),
            }
        )
    return conditions


def plot_compare_all(condition_results, out_png):
    ncol = min(3, len(condition_results))
    row3_all_values = []
    fig, axes = plt.subplots(
        3,
        ncol,
        figsize=(
            (210 / 25.4) * 1.4,
            ((210 / 25.4) * 1.4) * 0.65,
        ),
        sharex="col",
        sharey="row",
        dpi=300,
        gridspec_kw={"hspace": 0.1, "wspace": 0.1},
    )
    if ncol == 1:
        axes = axes.reshape(3, 1)

    for col in range(ncol):
        cond = condition_results[col]
        fixed_p = cond["P"]
        target_t = cond["T"]
        results = cond["results"]
        ax1, ax2, ax3 = axes[0, col], axes[1, col], axes[2, col]

        x_ideal = np.linspace(0.005, 0.995, 200)
        mu_he_ideal = KB_EV * target_t * np.log(x_ideal)
        mu_h_ideal = KB_EV * target_t * np.log(1.0 - x_ideal)
        g_mix_ideal = (
            KB_EV
            * target_t
            * (x_ideal * np.log(x_ideal) + (1.0 - x_ideal) * np.log(1.0 - x_ideal))
        )

        ax1.plot([0, 1], [0, 0], c="k", ls=":", lw=1.0, alpha=0.7)
        ax2.plot(x_ideal, mu_he_ideal, c="k", ls=":", lw=1.0, alpha=0.7)
        ax2.plot(x_ideal, mu_h_ideal, c="k", ls=":", lw=1.0, alpha=0.7)

        for r in results:
            x, gamma = r["x"], r["gamma"]
            ax1.plot(x, gamma, color=r["color"], ls=r["ls"], lw=1.8, label=r["label"])
            x_use, mu_he, mu_h, x_final, g_final = r["mu"]
            ax2.plot(x_use, mu_he, color=r["color"], ls="-", lw=1.8)
            ax2.plot(x_use, mu_h, color=r["color"], ls="--", lw=1.8)
            ax3.plot(x_final, g_final, color=r["color"], ls=r["ls"], lw=1.8)
            row3_all_values.append(g_final)

        title = f"{fixed_p} GPa, {target_t} K"
        if fixed_p == 150 and target_t == 2000:
            title = r"150 GPa, 2000 K"
        elif fixed_p == 150 and target_t == 3000:
            title = r"150 GPa, 3000 K "
        elif fixed_p == 400 and target_t == 2000:
            title = r"400 GPa, 2000 K"
        ax1.set_title(title, fontsize=11, pad=3)
        if col == 0:
            ax1.annotate(
                "molecular H",
                xy=(0.95, 0.90),
                xycoords="axes fraction",
                ha="right",
                va="top",
                fontsize=11,
            )
        else:
            ax1.annotate(
                "atomic H",
                xy=(0.95, 0.90),
                xycoords="axes fraction",
                ha="right",
                va="top",
                fontsize=11,
            )

        if col == 0:
            ax1.set_ylabel(r"$d\mu_{\rm He}/d\ln x_{\rm He}$ [$k_BT$]")
            ax2.set_ylabel(r"$\Delta\mu$ [eV]")
            ax3.set_ylabel(r"$\Delta G^{\rm {mix}}$ [eV/atom]")
            ax2.legend(
                handles=[
                    Line2D([0], [0], color="k", lw=1, ls="-", label="He"),
                    Line2D([0], [0], color="k", lw=1, ls="--", label="H"),
                    Line2D([0], [0], color="k", lw=1, ls=":", label="Ideal"),
                ],
                loc="lower left",
                frameon=False,
                fontsize=11,
            )
        ax3.set_xlabel(r"$x_{\rm He}$")

    for row in axes:
        for ax in row:
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.tick_params(axis="both", which="major", labelsize=7)

    for ax in axes[0, :]:
        ax.set_ylim(-0.05, 1.1)
    for ax in axes[1, :]:
        ax.set_ylim(-0.15, 0.01)
    if row3_all_values:
        row3_concat = np.concatenate(row3_all_values)
        y_min = float(np.min(row3_concat))
        y_max = float(np.max(row3_concat))
        y_span = y_max - y_min
        if y_span < 1e-6:
            pad = max(0.01, 0.1 * max(abs(y_min), 1.0))
        else:
            pad = 0.08 * y_span
        for ax in axes[2, :]:
            ax.set_ylim(y_min - pad, y_max + pad)

    method_legend = [
        Line2D([0], [0], color="#b2182b", lw=2, ls="-", label="Classical MD"),
        Line2D([0], [0], color="#2166ac", lw=2, ls="--", label="PIMD"),
    ]
    fig.legend(
        handles=method_legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        bbox_transform=fig.transFigure,
        ncol=2,
        fontsize=11,
        frameon=True,
        edgecolor="gray",
        fancybox=False,
    )
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.06, right=0.98)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    # plt.close(fig)


def main():
    root_dir = os.getcwd()
    conditions = discover_conditions(root_dir)
    if not conditions:
        raise RuntimeError(
            "No condition directories found, expected names like P150-T2000."
        )

    print("Building CMD/PIMD comparisons for all conditions...")
    condition_results = []
    global_max_diff = None

    for cond in conditions:
        fixed_p = cond["P"]
        target_t = cond["T"]
        base_dir = cond["path"]
        cmd_dir = os.path.join(base_dir, f"N{TARGET_N}_Sk-results_CMD")
        pimd_dir = os.path.join(base_dir, f"N{TARGET_N}_Sk-results_PIMD")
        plot_configs = [
            {
                "label": "Classical MD",
                "dir": cmd_dir,
                "color": "#b2182b",
                "marker": "s",
                "ls": "-",
                "kcut": DEFAULT_KCUT,
            },
            {
                "label": "PIMD",
                "dir": pimd_dir,
                "color": "#2166ac",
                "marker": "o",
                "ls": "--",
                "kcut": DEFAULT_KCUT,
            },
        ]

        print(f"\n[{cond['name']}] base_dir = {base_dir}")
        results = []
        for cfg in plot_configs:
            if not os.path.isdir(cfg["dir"]):
                print(f"[skip] missing dir: {cfg['dir']}")
                continue

            dataset, n_common = build_dataset_one_method(cfg, fixed_p, target_t)
            if not all(k in dataset for k in ("HeHe", "HeH", "HH")):
                print(f"[skip] incomplete pair data for {cfg['label']}")
                continue

            x, gamma = derivative_curve(dataset)
            if x is None:
                print(f"[skip] failed derivative for {cfg['label']}")
                continue

            mu = compute_mu_and_gmix(x, gamma, target_t)
            if mu is None:
                print(f"[skip] failed mu/gmix for {cfg['label']}")
                continue

            results.append(
                {
                    "label": cfg["label"],
                    "color": cfg["color"],
                    "ls": cfg["ls"],
                    "x": x,
                    "gamma": gamma,
                    "mu": mu,
                }
            )
            print(
                f"[ok] {cfg['label']}: common compositions={n_common}, curve points={len(x)}"
            )

        if len(results) >= 2:
            by_label = {r["label"]: r for r in results}
            if "Classical MD" in by_label and "PIMD" in by_label:
                local_diff = largest_gmix_difference(
                    by_label["Classical MD"]["mu"], by_label["PIMD"]["mu"]
                )
                if local_diff is not None:
                    print(
                        "[diff] max |ΔG_mix(PIMD-CMD)| = "
                        f"{local_diff['abs_delta']:.6f} eV/atom at x_He={local_diff['x']:.4f} "
                        f"(CMD={local_diff['cmd']:.6f}, PIMD={local_diff['pimd']:.6f})"
                    )
                    if (
                        global_max_diff is None
                        or local_diff["abs_delta"] > global_max_diff["abs_delta"]
                    ):
                        global_max_diff = {
                            **local_diff,
                            "P": fixed_p,
                            "T": target_t,
                        }
            condition_results.append({"P": fixed_p, "T": target_t, "results": results})
        else:
            print("[skip] this condition has no full CMD/PIMD pair")

    if not condition_results:
        raise RuntimeError("No valid condition has both CMD and PIMD results.")

    out_png = os.path.join(os.getcwd(), f"cmd_pimd_compare_all_N{TARGET_N}.png")
    plot_compare_all(condition_results, out_png)
    if global_max_diff is not None:
        print(
            "Largest Gibbs free-energy difference (PIMD-CMD): "
            f"|ΔG_mix|={global_max_diff['abs_delta']:.6f} eV/atom, "
            f"ΔG_mix={global_max_diff['delta']:.6f} eV/atom, "
            f"x_He={global_max_diff['x']:.4f}, "
            f"condition={global_max_diff['P']} GPa/{global_max_diff['T']} K"
        )
    print(f"\nSaved figure: {out_png}")


if __name__ == "__main__":
    main()
