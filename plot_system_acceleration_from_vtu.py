import argparse
import glob
import os
import re
from xml.etree import cElementTree as ET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import pyvista as pv
except ImportError as exc:
    raise ImportError(
        "Install PyVista first:\n"
        "  pip install pyvista"
    ) from exc


def natural_file_key(filepath):
    match = re.search(r"_(\d+)\.vtu$", os.path.basename(filepath))
    return int(match.group(1)) if match else float("inf")


def get_simulation_time_from_vtu(filepath, output_interval):
    try:
        root = ET.parse(filepath).getroot()
        for field_data in root.iter("FieldData"):
            for data_array in field_data:
                if data_array.get("Name") == "TIME" and data_array.text:
                    return float(data_array.text.strip())
    except Exception:
        pass

    match = re.search(r"_(\d+)\.vtu$", os.path.basename(filepath))
    if match:
        file_index = int(match.group(1))
        return file_index * output_interval

    return None


def find_array_name(available_names, requested_name, candidates, label):
    available_names = list(available_names)

    if requested_name is not None:
        if requested_name not in available_names:
            raise KeyError(
                f"Requested {label} array '{requested_name}' not found.\n"
                f"Available arrays: {available_names}"
            )
        return requested_name

    for candidate in candidates:
        if candidate in available_names:
            return candidate

    for candidate in candidates:
        for name in available_names:
            if name.lower() == candidate.lower():
                return name

    raise KeyError(
        f"Could not find {label} array automatically.\n"
        f"Available arrays: {available_names}"
    )


def find_velocity_array(mesh, requested_name=None):
    name = find_array_name(
        mesh.point_data.keys(),
        requested_name,
        ["VELOCITY", "Velocity", "velocity", "PARTICLE_VELOCITY"],
        "velocity",
    )
    values = np.asarray(mesh.point_data[name])
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError(f"Velocity array '{name}' is not a 3-component vector array.")
    return name


def find_radius_array(mesh, requested_name=None):
    return find_array_name(
        mesh.point_data.keys(),
        requested_name,
        ["RADIUS", "Radius", "radius", "PARTICLE_RADIUS"],
        "radius",
    )


def select_single_particle_by_radius(radii, target_radius, radius_tolerance):
    indices = np.flatnonzero(np.isclose(radii, target_radius, atol=radius_tolerance))
    if len(indices) == 0:
        raise RuntimeError(
            f"No particle found with radius {target_radius} within tolerance {radius_tolerance}."
        )
    if len(indices) > 1:
        raise RuntimeError(
            f"Expected one DEM particle with radius {target_radius}, found {len(indices)}."
        )
    return int(indices[0])


def main():
    parser = argparse.ArgumentParser(
        description="Plot system acceleration directly from VTU files."
    )

    parser.add_argument(
        "--vtkdir",
        default="Slit_Post_VTK_Files",
        help="Directory containing particle VTU files.",
    )
    parser.add_argument(
        "--pattern",
        default="Slit_Particles_*.vtu",
        help="Glob pattern for particle VTU files.",
    )
    parser.add_argument(
        "--output-interval",
        type=float,
        required=True,
        help="Physical time between VTU outputs when TIME is absent from VTU metadata.",
    )

    parser.add_argument(
        "--dem-radius",
        type=float,
        required=True,
        help="Radius of the single DEM particle.",
    )
    parser.add_argument(
        "--dpd-radius",
        type=float,
        required=True,
        help="Radius of the DPD particles.",
    )
    parser.add_argument(
        "--radius-tolerance",
        type=float,
        default=1.0e-6,
        help="Absolute radius tolerance.",
    )

    parser.add_argument(
        "--velarray",
        default=None,
        help="Exact VTU velocity-array name; auto-detected if omitted.",
    )
    parser.add_argument(
        "--radiusarray",
        default=None,
        help="Exact VTU radius-array name; auto-detected if omitted.",
    )

    parser.add_argument(
        "--force",
        type=float,
        required=True,
        help="Applied external force F0 in the forcing direction.",
    )

    parser.add_argument(
        "--dem-mass",
        type=float,
        default=None,
        help="DEM particle mass. If omitted, computed from --dem-density and --dem-radius.",
    )
    parser.add_argument(
        "--dem-density",
        type=float,
        default=None,
        help="DEM density used to compute DEM mass if --dem-mass is omitted.",
    )

    parser.add_argument(
        "--dpd-mass",
        type=float,
        default=None,
        help="DPD particle mass. If omitted, computed from --dpd-density and --dpd-radius.",
    )
    parser.add_argument(
        "--dpd-density",
        type=float,
        default=None,
        help="DPD density used to compute DPD mass if --dpd-mass is omitted.",
    )

    parser.add_argument(
        "--tstart",
        type=float,
        default=0.0,
        help="First simulation time to include.",
    )
    parser.add_argument(
        "--tend",
        type=float,
        default=np.inf,
        help="Last simulation time to include.",
    )

    parser.add_argument(
        "--smooth-window",
        type=int,
        default=9,
        help="Odd moving-average window for acceleration plot. Use 1 for no smoothing.",
    )

    parser.add_argument(
        "--csvout",
        default="system_acceleration_from_vtu.csv",
        help="Output CSV filename.",
    )
    parser.add_argument(
        "--pngout",
        default="system_acceleration_from_vtu.png",
        help="Output figure filename.",
    )

    args = parser.parse_args()

    if args.radius_tolerance <= 0.0:
        raise ValueError("--radius-tolerance must be positive.")
    if args.output_interval <= 0.0:
        raise ValueError("--output-interval must be positive.")
    if args.smooth_window < 1 or args.smooth_window % 2 == 0:
        raise ValueError("--smooth-window must be an odd integer >= 1.")

    if args.dem_mass is None:
        if args.dem_density is None:
            raise ValueError("Provide either --dem-mass or --dem-density.")
        m_dem = args.dem_density * (4.0 / 3.0) * np.pi * args.dem_radius**3
    else:
        m_dem = args.dem_mass

    if args.dpd_mass is None:
        if args.dpd_density is None:
            raise ValueError("Provide either --dpd-mass or --dpd-density.")
        m_dpd = args.dpd_density * (4.0 / 3.0) * np.pi * args.dpd_radius**3
    else:
        m_dpd = args.dpd_mass

    files = glob.glob(os.path.join(args.vtkdir, args.pattern))
    files = sorted(files, key=natural_file_key)

    if not files:
        raise FileNotFoundError(
            f"No files matched '{args.pattern}' in '{args.vtkdir}'."
        )

    first_mesh = pv.read(files[0])
    velocity_name = find_velocity_array(first_mesh, args.velarray)
    radius_name = find_radius_array(first_mesh, args.radiusarray)

    print(f"Velocity array: {velocity_name}")
    print(f"Radius array:   {radius_name}")
    print(f"Files found:    {len(files)}")

    rows = []

    for filepath in files:
        time = get_simulation_time_from_vtu(filepath, args.output_interval)
        if time is None or time < args.tstart or time > args.tend:
            continue

        mesh = pv.read(filepath)

        velocity = np.asarray(mesh.point_data[velocity_name])
        radii = np.asarray(mesh.point_data[radius_name]).reshape(-1)

        dem_index = select_single_particle_by_radius(
            radii,
            args.dem_radius,
            args.radius_tolerance,
        )

        dpd_mask = np.isclose(
            radii,
            args.dpd_radius,
            atol=args.radius_tolerance,
        )

        n_dpd = int(np.count_nonzero(dpd_mask))
        if n_dpd == 0:
            raise RuntimeError(
                f"No DPD particles with radius {args.dpd_radius} found in "
                f"{os.path.basename(filepath)}."
            )

        v_dem_x = float(velocity[dem_index, 0])
        v_dpd_x = velocity[dpd_mask, 0]

        p_dem_x = m_dem * v_dem_x
        p_dpd_x = m_dpd * np.sum(v_dpd_x)
        m_fluid = n_dpd * m_dpd
        m_total = m_dem + m_fluid

        v_system_x = (p_dem_x + p_dpd_x) / m_total

        rows.append({
            "time": time,
            "vtu_file": os.path.basename(filepath),
            "n_dpd": n_dpd,
            "v_dem_x": v_dem_x,
            "v_dpd_mean_x": float(np.mean(v_dpd_x)),
            "p_dem_x": p_dem_x,
            "p_dpd_x": p_dpd_x,
            "p_system_x": p_dem_x + p_dpd_x,
            "m_dem": m_dem,
            "m_dpd": m_dpd,
            "m_fluid": m_fluid,
            "m_total": m_total,
            "v_system_x": v_system_x,
        })

        print(
            f"t={time:.6g}, n_dpd={n_dpd}, "
            f"v_dem_x={v_dem_x:.8g}, v_sys_x={v_system_x:.8g}"
        )

    if len(rows) < 3:
        raise RuntimeError("Need at least 3 time frames to compute acceleration.")

    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)

    elapsed_time = df["time"].to_numpy(dtype=float) - df["time"].iloc[0]
    v_system = df["v_system_x"].to_numpy(dtype=float)

    a_system_measured = np.gradient(v_system, elapsed_time)
    df["elapsed_time"] = elapsed_time
    df["a_system_measured_raw"] = a_system_measured

    if args.smooth_window == 1:
        a_system_smoothed = a_system_measured.copy()
    else:
        a_system_smoothed = (
            pd.Series(a_system_measured)
            .rolling(window=args.smooth_window, center=True, min_periods=1)
            .mean()
            .to_numpy()
        )

    df["a_system_measured_smoothed"] = a_system_smoothed

    # Predicted constant whole-system acceleration from each frame mass
    df["a_system_predicted"] = args.force / df["m_total"]

    df.to_csv(args.csvout, index=False)

    mean_measured = float(np.mean(a_system_smoothed))
    mean_predicted = float(np.mean(df["a_system_predicted"]))

    print(f"\nSaved CSV: {args.csvout}")
    print(f"Mean measured acceleration  = {mean_measured:.8g}")
    print(f"Mean predicted acceleration = {mean_predicted:.8g}")

    figure, axis = plt.subplots(figsize=(10, 5.5))

    axis.plot(
        elapsed_time,
        df["a_system_measured_smoothed"],
        color="tab:purple",
        linewidth=2.2,
        label=r"Measured system acceleration $dV_{\mathrm{sys}}/dt$",
    )

    axis.plot(
        elapsed_time,
        df["a_system_measured_raw"],
        color="tab:purple",
        alpha=0.25,
        linewidth=1.0,
    )

    axis.set_xlabel("Time since first included output")
    axis.set_ylabel("Acceleration in forcing direction")
    axis.set_title("System Acceleration")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best", fontsize=10)

    text_box = f"mean measured = {mean_measured:.6g}"

    axis.text(
        0.98,
        0.04,
        text_box,
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor="gray",
            alpha=0.9
        ),
    )

    figure.tight_layout()
    figure.savefig(args.pngout, dpi=200)
    plt.show()

    print(f"Saved figure: {args.pngout}")


if __name__ == "__main__":
    main()