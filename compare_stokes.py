import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Load data and reconstruct continuous x displacement
# ------------------------------------------------------------
df = pd.read_csv("solid_particle_vx_and_trajectory.csv")
df.columns = df.columns.str.strip()
df = df.sort_values("time").reset_index(drop=True)

required = ["time", "vx", "x_wrapped"]
missing = [name for name in required if name not in df.columns]
if missing:
    raise ValueError(
        f"Missing columns: {missing}. "
        f"Available columns: {df.columns.tolist()}"
    )

time = df["time"].to_numpy(dtype=float)
v_sim = df["vx"].to_numpy(dtype=float)
x_wrapped = df["x_wrapped"].to_numpy(dtype=float)

# IMPORTANT: use the actual periodic domain length in x.
Lx = 2.0

# Convert wrapped coordinate increments to continuous increments.
# Assumes motion between saved samples is less than Lx / 2.
dx = np.diff(x_wrapped)
dx -= Lx * np.round(dx / Lx)

# Continuous displacement relative to the first saved sample.
x_sim = np.concatenate(([0.0], np.cumsum(dx)))

# ------------------------------------------------------------
# 2. Parameters from your uploaded script
# ------------------------------------------------------------
R = 0.2
rho_p = 1007.0
mu = 6.53
F0 = 0.5

m_p = rho_p * (4.0 / 3.0) * np.pi * R**3

# ------------------------------------------------------------
# 3. Compare from the first recorded state
# ------------------------------------------------------------
# The CSV starts at t = 1.5, not at t = 0.
# This is a continuation from that measured state,
# not necessarily a release-from-rest comparison.
t0 = time[0]
v0 = v_sim[0]
x0 = x_sim[0]

t = time
elapsed_time = t - t0

# ------------------------------------------------------------
# 4. Stationary-fluid Stokes analytical solution
# ------------------------------------------------------------
# Drag force: F_drag = -zeta * v
zeta = 6.0 * np.pi * mu * R

# Particle response/relaxation time
tau_p = m_p / zeta

# Terminal velocity in an indefinitely stationary fluid
v_terminal = F0 / zeta

# Analytical velocity:
# v(t) = v_terminal + (v0 - v_terminal) exp[-(t-t0)/tau_p]
v_stokes = v_terminal + (v0 - v_terminal) * np.exp(
    -elapsed_time / tau_p
)

# Analytical position:
# x(t) = x0 + v_terminal*s
#        + (v0 - v_terminal)*tau_p*(1 - exp[-s/tau_p])
x_stokes = (
    x0
    + v_terminal * elapsed_time
    + (v0 - v_terminal) * tau_p * (
        1.0 - np.exp(-elapsed_time / tau_p)
    )
)

# ------------------------------------------------------------
# 5. Print useful values
# ------------------------------------------------------------
print(f"Particle mass m_p       = {m_p:.8g}")
print(f"Stokes drag coefficient = {zeta:.8g}")
print(f"Response time tau_p     = {tau_p:.8g}")
print(f"Terminal velocity       = {v_terminal:.8g}")

# ------------------------------------------------------------
# 6. Save comparison data
# ------------------------------------------------------------
comparison = pd.DataFrame({
    "time": t,
    "elapsed_time": elapsed_time,
    "velocity_simulation": v_sim,
    "velocity_stokes": v_stokes,
    "position_simulation": x_sim,
    "position_stokes": x_stokes,
    "velocity_error": v_sim - v_stokes,
    "position_error": x_sim - x_stokes,
})
comparison.to_csv("stokes_analytical_comparison.csv", index=False)

# ------------------------------------------------------------
# 7. Plot velocity and position overlays
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

axes[0].plot(elapsed_time, v_sim, color="tab:blue",
             linewidth=1.2, label="DEM–DPD simulation")
axes[0].plot(elapsed_time, v_stokes, "--", color="black",
             linewidth=2.0, label="Stationary-fluid Stokes solution")
axes[0].axhline(v_terminal, color="tab:red", linestyle=":",
                label=r"$v_\infty = F_0/(6\pi\mu R)$")
axes[0].set_ylabel("Velocity")
axes[0].grid(alpha=0.3)
axes[0].legend()

axes[1].plot(elapsed_time, x_sim - x0, color="tab:blue",
             linewidth=1.2, label="DEM–DPD simulation")
axes[1].plot(elapsed_time, x_stokes - x0, "--", color="black",
             linewidth=2.0, label="Stationary-fluid Stokes solution")
axes[1].set_xlabel("Time since forcing started")
axes[1].set_ylabel("Displacement")
axes[1].grid(alpha=0.3)
axes[1].legend()

fig.tight_layout()
fig.savefig("stokes_velocity_position_comparison.png", dpi=200)
plt.show()