import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# User inputs
# ============================================================

solid_csv = "solid_particle_vx_and_trajectory.csv"
fluid_csv = "dpd_bulk_velocity.csv"
output_csv = "stokes_relative_velocity_comparison.csv"

R = 0.2
rho_p = 1007.0
mu = 6.53
F0 = 0.5

use_direct_mass = False
m_p_direct = None

use_direct_dpd_mass = False
m_dpd_direct = None

rho_f = 1000.0
R_dpd = 0.02

time_tolerance = 1e-6

# ============================================================
# Load data
# ============================================================

solid = pd.read_csv(solid_csv)
solid.columns = solid.columns.str.strip()
solid = solid.sort_values("time").reset_index(drop=True)

required_solid = ["time", "vx"]
missing_solid = [c for c in required_solid if c not in solid.columns]
if missing_solid:
    raise ValueError(
        f"Missing columns in solid CSV: {missing_solid}. "
        f"Available: {solid.columns.tolist()}"
    )

fluid = pd.read_csv(fluid_csv)
fluid.columns = fluid.columns.str.strip()
fluid = fluid.sort_values("time").reset_index(drop=True)

required_fluid = ["time", "v_fluid_x", "n_points"]
missing_fluid = [c for c in required_fluid if c not in fluid.columns]
if missing_fluid:
    raise ValueError(
        f"Missing columns in fluid CSV: {missing_fluid}. "
        f"Available: {fluid.columns.tolist()}"
    )

merged = pd.merge_asof(
    solid.sort_values("time"),
    fluid[["time", "v_fluid_x", "n_points"]].sort_values("time"),
    on="time",
    direction="nearest",
    tolerance=time_tolerance,
)

merged = merged.dropna(subset=["v_fluid_x"]).reset_index(drop=True)

if merged.empty:
    raise ValueError(
        "No matching DEM/fluid times found after merge. "
        "Increase time_tolerance or check your CSV time columns."
    )

# ============================================================
# Masses
# ============================================================

if use_direct_mass:
    if m_p_direct is None or m_p_direct <= 0.0:
        raise ValueError("Set a positive m_p_direct when use_direct_mass=True.")
    m_p = float(m_p_direct)
else:
    m_p = rho_p * (4.0 / 3.0) * np.pi * R**3

if use_direct_dpd_mass:
    if m_dpd_direct is None or m_dpd_direct <= 0.0:
        raise ValueError("Set a positive m_dpd_direct when use_direct_dpd_mass=True.")
    m_dpd = float(m_dpd_direct)
else:
    m_dpd = rho_f * (4.0 / 3.0) * np.pi * R_dpd**3

N_f = float(merged.loc[0, "n_points"])
M_f = N_f * m_dpd
M_total = m_p + M_f

# ============================================================
# Stokes parameters
# ============================================================

zeta = 6.0 * np.pi * mu * R
tau_p = m_p / zeta
v_terminal_stokes = F0 / zeta

# ============================================================
# Derived quantities
# ============================================================

merged["v_relative_sim"] = merged["vx"] - merged["v_fluid_x"]

t = merged["time"].to_numpy(dtype=float)
elapsed_time = t - t[0]

v_dem0 = merged.loc[0, "vx"]
v_rel0 = merged.loc[0, "v_relative_sim"]

merged["v_stokes_lab"] = v_terminal_stokes + (v_dem0 - v_terminal_stokes) * np.exp(
    -elapsed_time / tau_p
)

merged["v_stokes_relative"] = v_terminal_stokes + (v_rel0 - v_terminal_stokes) * np.exp(
    -elapsed_time / tau_p
)

a_system = F0 / M_total
merged["a_system_prediction"] = a_system

merged["v_system_sim"] = (
    m_p * merged["vx"] + M_f * merged["v_fluid_x"]
) / M_total

merged["a_system_sim"] = np.gradient(
    merged["v_system_sim"].to_numpy(dtype=float),
    elapsed_time
)

# Optional wrapped-x reconstruction
if "x_wrapped" in merged.columns:
    Lx = 2.0
    xw = merged["x_wrapped"].to_numpy(dtype=float)
    dx = np.diff(xw)
    dx -= Lx * np.round(dx / Lx)
    x_unwrapped = np.concatenate(([0.0], np.cumsum(dx)))
    merged["x_sim_unwrapped"] = x_unwrapped

# Metadata columns
merged["m_p"] = m_p
merged["m_dpd"] = m_dpd
merged["N_f"] = N_f
merged["M_f"] = M_f
merged["M_total"] = M_total
merged["stokes_drag_coefficient"] = zeta
merged["stokes_response_time"] = tau_p
merged["stokes_terminal_velocity"] = v_terminal_stokes

merged.to_csv(output_csv, index=False)

print(f"Saved: {output_csv}")
print(f"Rows: {len(merged)}")
print(f"m_p      = {m_p:.8g}")
print(f"m_dpd    = {m_dpd:.8g}")
print(f"N_f      = {N_f:.8g}")
print(f"M_f      = {M_f:.8g}")
print(f"M_total  = {M_total:.8g}")
print(f"zeta     = {zeta:.8g}")
print(f"tau_p    = {tau_p:.8g}")
print(f"v_inf    = {v_terminal_stokes:.8g}")
print(f"a_system = {a_system:.8g}")

# ============================================================
# Figure 1: relative-velocity comparison
# ============================================================

figure, axis = plt.subplots(figsize=(10, 5.5))

axis.plot(
    elapsed_time,
    merged["vx"],
    color="tab:blue",
    linewidth=1.8,
    label=r"DEM velocity $V_{\mathrm{DEM},x}$",
)

axis.plot(
    elapsed_time,
    merged["v_fluid_x"],
    color="tab:orange",
    linewidth=1.8,
    label=r"DPD bulk velocity $\overline{V}_{\mathrm{DPD},x}$",
)

axis.plot(
    elapsed_time,
    merged["v_relative_sim"],
    color="tab:green",
    linewidth=2.3,
    label=r"Simulation relative velocity "
          r"$V_{\mathrm{DEM},x}-\overline{V}_{\mathrm{DPD},x}$",
)

axis.plot(
    elapsed_time,
    merged["v_stokes_relative"],
    color="black",
    linestyle="--",
    linewidth=2.2,
    label=r"Stationary-fluid Stokes solution",
)

axis.axhline(
    v_terminal_stokes,
    color="tab:red",
    linestyle=":",
    linewidth=2.0,
    label=r"$v_\infty = F_0/(6\pi\mu R)$",
)

axis.set_xlabel("Time since first matched output")
axis.set_ylabel("Velocity in forcing direction")
axis.set_title("DEM–DPD Relative Velocity vs. Stokes Baseline")
axis.grid(True, alpha=0.3)
axis.legend(loc="best", fontsize=10)

figure.tight_layout()
figure.savefig("stokes_relative_velocity_comparison.png", dpi=200)
plt.show()