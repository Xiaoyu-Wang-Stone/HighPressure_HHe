import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as mpatches

# Set global font properties
plt.rc('font', family='serif', size=9)

# ---------- Raw Data ----------
# Columns: [He fraction (x), Temperature (K), Volume (Å^3), Pressure (GPa), e]
hse_data = [
    [0.25, 6000, 60, 901.102, 0.711324],
    [0.25, 6000, 76, 549.615, 0.43179],
    [0.25, 6000, 88, 402.931, 0.328685],
    [0.25, 6000, 120, 209.952, 0.283764],
    [0.25, 6000, 172.51, 98.4776, 0.198683],
    [0.25, 6000, 277.83, 39.5541, 0.171426],
    [0.25, 6000, 320, 30.4304, 0.130521],
]

vdw_df_data = [
    [0.25, 6000, 60, 915.319, 0.542891],
    [0.25, 6000, 76, 555.936, 0.410888],
    [0.25, 6000, 88, 407.398, 0.336034],
    [0.25, 6000, 120, 211.896, 0.282649],
    [0.25, 6000, 172.51, 99.3976, 0.151698],
    [0.25, 6000, 277.83, 39.0062, 0.153273],
    [0.25, 6000, 320, 29.6662, 0.116594],
]

pbe_data = [
    [0.25, 6000, 60, 891.349, 0.538857],
    [0.25, 6000, 76, 534.612, 0.345124],
    [0.25, 6000, 88, 391.234, 0.392761],
    [0.25, 6000, 120, 200.847, 0.231605],
    [0.25, 6000, 172.51, 92.451, 0.176964],
    [0.25, 6000, 277.83, 36.1911, 0.11739],
    [0.25, 6000, 320, 27.2516, 0.110699],
]

columns = ['x', 'Temperature', 'Volume_A3', 'Pressure_GPa', 'e']

hse_df = pd.DataFrame(hse_data, columns=columns)
vdw_df = pd.DataFrame(vdw_df_data, columns=columns)
pbe_df = pd.DataFrame(pbe_data, columns=columns)

# ---------- Density Calculation ----------
# Physical constants
num_He = 32 * 0.25
num_H = 64 - num_He
mass_He = 4.002602  # atomic mass unit
mass_H = 1.008
NA = 6.02214076e23  # Avogadro's number

total_mass_g = num_He * mass_He / NA + num_H * mass_H / NA

def calculate_density(volume_A3):
    """Convert volume from Å^3 to g/cm³ density."""
    volume_cm3 = volume_A3 * 1e-24
    return total_mass_g / volume_cm3

# Apply density calculation to each DataFrame
for df in (hse_df, vdw_df, pbe_df):
    df['Density_g_cm3'] = df['Volume_A3'].apply(calculate_density)

# ---------- Plotting ----------
# Figure setup (inches: 1pt = 1/72.27in)
fig_width_pt = 173
fig_height_pt = 110
fig_width_in = fig_width_pt / 72.27
fig_height_in = fig_height_pt / 72.27

fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi=500)

# Style parameters for the plots
colors = ['#FF006E', '#3A86FF', '#FFBE0B']
labels = ['vdW-DF', 'PBE', 'HSE']
markers = ['s', 'o', '^']

# Main plot
ax.plot(vdw_df['Density_g_cm3'], vdw_df['Pressure_GPa'], marker=markers[0], linestyle='-', color=colors[0], label=labels[0])
ax.plot(pbe_df['Density_g_cm3'], pbe_df['Pressure_GPa'], marker=markers[1], linestyle='-', color=colors[1], label=labels[1])
ax.plot(hse_df['Density_g_cm3'], hse_df['Pressure_GPa'], marker=markers[2], linestyle='-', color=colors[2], label=labels[2])

# Annotation for composition and temperature
ax.text(
    0.6, 0.95, 
    r"$x_{\mathrm{He}}=0.25$" + "\n 6000 K",
    transform=ax.transAxes,
    ha='left', va='top', fontsize=6.5
)

# ---------- Inset (zoomed region around the 4th data point) ----------
# Index to zoom
zoom_idx = 3

# Densities and pressures at the 4th data point for all models
vdw_x, vdw_y = vdw_df['Density_g_cm3'].iloc[zoom_idx], vdw_df['Pressure_GPa'].iloc[zoom_idx]
pbe_x, pbe_y = pbe_df['Density_g_cm3'].iloc[zoom_idx], pbe_df['Pressure_GPa'].iloc[zoom_idx]
hse_x, hse_y = hse_df['Density_g_cm3'].iloc[zoom_idx], hse_df['Pressure_GPa'].iloc[zoom_idx]

# Compute zoom box limits
x_min = min(vdw_x, pbe_x, hse_x)
x_max = max(vdw_x, pbe_x, hse_x)
y_min = min(vdw_y, pbe_y, hse_y) * 0.98
y_max = max(vdw_y, pbe_y, hse_y) * 1.02

# Expand box for visibility
expand_factor = 6
x_range = 0.04
y_range = y_max - y_min
x_min_exp = x_min - x_range * expand_factor / 2
x_max_exp = x_max + x_range * expand_factor / 2
y_min_exp = y_min - y_range * expand_factor / 2
y_max_exp = y_max + y_range * expand_factor / 2

# Draw dashed rectangle showing zoom region on main plot
rect = mpatches.Rectangle(
    (x_min_exp, y_min_exp),
    x_max_exp - x_min_exp,
    y_max_exp - y_min_exp,
    linewidth=1, edgecolor='gray', linestyle='--', facecolor='none', zorder=10
)
ax.add_patch(rect)

# Create inset axes
ax_inset = inset_axes(
    ax,
    width="36%", height="36%",
    loc='upper left',
    borderpad=0.3,
    bbox_to_anchor=(0.14, -0.001, 1, 1),
    bbox_transform=ax.transAxes
)

# Plot data in inset
ax_inset.plot(vdw_df['Density_g_cm3'], vdw_df['Pressure_GPa'], marker=markers[0], linestyle='-', color=colors[0])
ax_inset.plot(pbe_df['Density_g_cm3'], pbe_df['Pressure_GPa'], marker=markers[1], linestyle='-', color=colors[1])
ax_inset.plot(hse_df['Density_g_cm3'], hse_df['Pressure_GPa'], marker=markers[2], linestyle='-', color=colors[2])

# Highlight the 4th data point in the inset
ax_inset.scatter([vdw_x], [vdw_y], color=colors[0], marker=markers[0], s=30, zorder=5)
ax_inset.scatter([pbe_x], [pbe_y], color=colors[1], marker=markers[1], s=30, zorder=5)
ax_inset.scatter([hse_x], [hse_y], color=colors[2], marker=markers[2], s=30, zorder=5)

ax_inset.set_xlim(1.2, 1.24)
ax_inset.set_ylim(y_min, y_max)

for spine in ax_inset.spines.values():
    spine.set_linewidth(1)

ax_inset.tick_params(axis='both', which='both', labelleft=True, labelbottom=True, labelsize=5)

line_width = 0.8
for side in ['top', 'right', 'bottom', 'left']:
    ax_inset.spines[side].set_linewidth(line_width)

# ---------- Main axes labels, legend, and grid ----------
ax.set_xlabel(r'$\rho$ [g/cm³]')
ax.set_ylabel('$P$ [GPa]')
ax.legend(
    loc='lower right',
    title='N2P2 MLPs',
    frameon=False,
    bbox_to_anchor=(1.025, -0.01),
    fontsize=6,
    title_fontsize=6
)
ax.grid(False)

# Save and/or display the plot
# plt.savefig('/Users/xiaoyuwang/Project/H-He/H-He-demixing/Figs/EoS_XC_compare.pdf', dpi=500, bbox_inches='tight')
plt.show()