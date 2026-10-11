import numpy as np
import pandas as pd

# ============================================================
# User inputs
# ============================================================

# DEM particle history CSV
solid_csv = "solid_particle_vx_and_trajectory.csv"

# DPD bulk-velocity history CSV
# This file should contain at least:
#   time, v_fluid_x
fluid_csv = "dpd_bulk_velocity.csv"

# Output CSV
output_csv = "stokes_relative_velocity_comparison.csv"

# Physical/model parameters from the SAME simulation case
R = 0.2
rho_p = 1007.0
mu = 6.53
F0 = 0.5

# If you know the DEM particle mass directly, set it here and
# change use_direct_mass to True.
use_direct_mass = False
m_p_direct = None

# Maximum allowed mismatch when merging DEM and fluid times
time_tolerance = 1e-6

# ============================================================
# Load solid-particle data
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

# ============================================================
# Load fluid bulk-velocity data
# ============================================================

fluid = pd.read_csv(fluid_csv)
fluid.columns = fluid.columns.str.strip()
fluid = fluid.sort_values("time").reset_index(drop=True)

required_fluid = ["time", "v_fluid_x"]
missing_fluid = [c for c in required_fluid if c not in fluid.columns]
if missing_fluid:
    raise ValueError(
        f"Missing columns in fluid CSV: {missing_fluid}. "
        f"Available: {fluid.columns.tolist()}"
    )

# ============================================================
# Merge by time
# ============================================================

merged = pd.merge_asof(
    solid.sort_values("time"),
    fluid[["time", "v_fluid_x"]].sort_values("time"),
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
# Mass and Stokes parameters
# ============================================================

if use_direct_mass:
    if m_p_direct is None or m_p_direct <= 0.0:
        raise ValueError("Set a positive m_p_direct when use_direct_mass=True.")
    m_p = float(m_p_direct)
else:
    m_p = rho_p * (4.0 / 3.0) * np.pi * R**3

zeta = 6.0 * np.pi * mu * R
tau_p = m_p / zeta
v_terminal_stokes = F0 / zeta

# ============================================================
# Relative velocity from simulation
# ============================================================

merged["v_relative_sim"] = merged["vx"] - merged["v_fluid_x"]

# ============================================================
# Analytical curves
# ============================================================
# Two useful analytical baselines:
#
# 1) Stationary-fluid Stokes:
#    m dv/dt = F0 - zeta v
#    terminal velocity = F0/zeta
#
# 2) Relative-velocity equation if fluid accelerates slowly enough
#    that drag still depends on v_rel:
#    m d(v_rel)/dt = F0 - zeta v_rel
#
# The mathematical form is the same, but the initial value should be
# taken from the simulated RELATIVE velocity, not from the DEM lab-frame
# velocity.

t = merged["time"].to_numpy(dtype=float)
elapsed_time = t - t[0]

v_dem0 = merged.loc[0, "vx"]
v_fluid0 = merged.loc[0, "v_fluid_x"]
v_rel0 = merged.loc[0, "v_relative_sim"]

# DEM laboratory-frame Stokes baseline
merged["v_stokes_lab"] = v_terminal_stokes + (v_dem0 - v_terminal_stokes) * np.exp(
    -elapsed_time / tau_p
)

# Relative-velocity Stokes baseline
merged["v_stokes_relative"] = v_terminal_stokes + (v_rel0 - v_terminal_stokes) * np.exp(
    -elapsed_time / tau_p
)

# Optional analytical positions for the DEM lab-frame baseline
# x(t) = x0 + v_inf * s + (v0 - v_inf) tau (1 - exp(-s/tau))
if "x_wrapped" in merged.columns:
    # Reconstruct continuous x displacement if wrapped coordinate exists
    Lx = 2.0  # change if your domain length in x is different
    xw = merged["x_wrapped"].to_numpy(dtype=float)
    dx = np.diff(xw)
    dx -= Lx * np.round(dx / Lx)
    x_unwrapped = np.concatenate(([0.0], np.cumsum(dx)))
    merged["x_sim_unwrapped"] = x_unwrapped

    x0 = x_unwrapped[0]
    merged["x_stokes_lab"] = (
        x0
        + v_terminal_stokes * elapsed_time
        + (v_dem0 - v_terminal_stokes) * tau_p * (1.0 - np.exp(-elapsed_time / tau_p))
    )

# ============================================================
# Save CSV
# ============================================================

merged["stokes_drag_coefficient"] = zeta
merged["stokes_response_time"] = tau_p
merged["stokes_terminal_velocity"] = v_terminal_stokes

merged.to_csv(output_csv, index=False)

print(f"Saved: {output_csv}")
print(f"Rows: {len(merged)}")
print(f"m_p   = {m_p:.8g}")
print(f"zeta  = {zeta:.8g}")
print(f"tau_p = {tau_p:.8g}")
print(f"v_inf = {v_terminal_stokes:.8g}")