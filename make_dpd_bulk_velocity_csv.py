import glob
import os
import re
import numpy as np
import pandas as pd

# ============================================================
# User inputs
# ============================================================

vtk_pattern = "VTK_Output/*.vtk"   # change if your files are elsewhere
output_csv = "dpd_bulk_velocity.csv"

# Name of the vector field in the VTK files.
# Common possibilities: "VELOCITY", "Velocity"
velocity_field_name = "VELOCITY"

# If your VTK folder contains non-fluid particles too, set up filtering.
# For now this script assumes every point in these VTK files is DPD fluid.
# If not true, I can adapt it once you show one sample VTK file.
filter_only_fluid = False

# If your times are NOT stored in the file contents, this script tries to
# infer them from filenames. If filenames only contain step numbers, then
# set:
use_time_from_filename = False
dt_output = 0.5   # physical time between saved VTK outputs, used only if needed

# ============================================================
# Minimal legacy ASCII VTK reader for POINTS + VECTORS
# ============================================================

def read_legacy_vtk_points_and_vectors(vtk_file, vector_name="VELOCITY"):
    with open(vtk_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    n_points = None
    points_start = None

    for i, line in enumerate(lines):
        words = line.strip().split()
        if len(words) >= 3 and words[0].upper() == "POINTS":
            n_points = int(words[1])
            points_start = i + 1
            break

    if n_points is None:
        raise ValueError(f"No POINTS block found in {vtk_file}")

    all_text_after_points = " ".join(lines[points_start:])
    point_values = np.fromstring(all_text_after_points, sep=" ", count=3 * n_points)

    if point_values.size != 3 * n_points:
        raise ValueError(f"Could not read all point coordinates from {vtk_file}")

    points = point_values.reshape((n_points, 3))

    vectors_start = None
    for i, line in enumerate(lines):
        words = line.strip().split()
        if (
            len(words) >= 3
            and words[0].upper() == "VECTORS"
            and words[1].upper() == vector_name.upper()
        ):
            vectors_start = i + 1
            break

    if vectors_start is None:
        available_vector_lines = [
            line.strip() for line in lines if line.strip().upper().startswith("VECTORS ")
        ]
        raise ValueError(
            f"Vector field '{vector_name}' not found in {vtk_file}. "
            f"Available VECTORS lines: {available_vector_lines}"
        )

    all_text_after_vectors = " ".join(lines[vectors_start:])
    vector_values = np.fromstring(all_text_after_vectors, sep=" ", count=3 * n_points)

    if vector_values.size != 3 * n_points:
        raise ValueError(f"Could not read all vector values from {vtk_file}")

    vectors = vector_values.reshape((n_points, 3))
    return points, vectors

# ============================================================
# Time extraction
# ============================================================

def infer_time_from_filename(vtk_file, index_in_sequence):
    base = os.path.basename(vtk_file)

    number_strings = re.findall(r"\d+(?:\.\d+)?", base)
    if not number_strings:
        raise ValueError(f"Could not infer any numeric token from filename: {base}")

    last_number = float(number_strings[-1])

    if use_time_from_filename:
        return last_number
    else:
        return index_in_sequence * dt_output

# ============================================================
# Main
# ============================================================

vtk_files = sorted(glob.glob(vtk_pattern))

if not vtk_files:
    raise FileNotFoundError(
        f"No VTK files found with pattern: {vtk_pattern}"
    )

rows = []

for k, vtk_file in enumerate(vtk_files):
    points, velocity = read_legacy_vtk_points_and_vectors(
        vtk_file,
        vector_name=velocity_field_name
    )

    # If needed later, add a mask here to keep only DPD-fluid particles.
    if filter_only_fluid:
        raise NotImplementedError(
            "Fluid-only filtering is not yet implemented in this generic script."
        )

    v_fluid_x = velocity[:, 0].mean()
    v_fluid_y = velocity[:, 1].mean()
    v_fluid_z = velocity[:, 2].mean()

    rows.append({
        "time": infer_time_from_filename(vtk_file, k),
        "vtk_file": os.path.basename(vtk_file),
        "n_particles": len(velocity),
        "v_fluid_x": v_fluid_x,
        "v_fluid_y": v_fluid_y,
        "v_fluid_z": v_fluid_z,
    })

df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
df.to_csv(output_csv, index=False)

print(f"Saved {output_csv} with {len(df)} rows.")
print(df.head())