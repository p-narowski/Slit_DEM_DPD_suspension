import glob
import os
import re
import numpy as np
import pandas as pd
import pyvista as pv

# ============================================================
# User inputs
# ============================================================

vtu_pattern = "Slit_Post_VTK_Files/*.vtu"
output_csv = "dpd_bulk_velocity.csv"

velocity_candidates = [
    "VELOCITY",
    "Velocity",
    "velocity",
]

# If filename number is just the output index, use this:
use_index_times_dt = True
dt_output = 0.5

# ============================================================
# Helpers
# ============================================================

def infer_time_from_filename(vtu_file):
    base = os.path.basename(vtu_file)
    nums = re.findall(r"\d+", base)
    if not nums:
        raise ValueError(f"Could not infer index from filename: {base}")
    idx = int(nums[-1])
    if use_index_times_dt:
        return idx * dt_output
    return float(idx)

def find_velocity_array(mesh):
    available = list(mesh.point_data.keys())

    for name in velocity_candidates:
        if name in mesh.point_data:
            arr = np.asarray(mesh.point_data[name])
            if arr.ndim == 2 and arr.shape[1] >= 3:
                return name, arr

    raise ValueError(
        f"No 3-component velocity array found. "
        f"Available point_data arrays: {available}"
    )

# ============================================================
# Main
# ============================================================

vtu_files = sorted(glob.glob(vtu_pattern))
if not vtu_files:
    raise FileNotFoundError(f"No VTU files found with pattern: {vtu_pattern}")

rows = []

for vtu_file in vtu_files:
    mesh = pv.read(vtu_file)

    vel_name, vel = find_velocity_array(mesh)

    v_fluid_x = vel[:, 0].mean()
    v_fluid_y = vel[:, 1].mean()
    v_fluid_z = vel[:, 2].mean()

    rows.append({
        "time": infer_time_from_filename(vtu_file),
        "vtu_file": os.path.basename(vtu_file),
        "n_points": mesh.n_points,
        "velocity_field": vel_name,
        "v_fluid_x": v_fluid_x,
        "v_fluid_y": v_fluid_y,
        "v_fluid_z": v_fluid_z,
    })

df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
df.to_csv(output_csv, index=False)

print(f"Saved {output_csv} with {len(df)} rows.")
print(df.head())