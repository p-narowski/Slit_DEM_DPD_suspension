import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv


def vtk_file_index(filepath):
    match = re.search(r"_(\d+)\.vtu$", os.path.basename(filepath))

    if match is None:
        return -1

    return int(match.group(1))


def find_array_name(available_names, candidates, label):
    available_names = list(available_names)

    for candidate in candidates:
        if candidate in available_names:
            return candidate

    for candidate in candidates:
        for name in available_names:
            if name.lower() == candidate.lower():
                return name

    raise KeyError(
        "Could not find {} array. Available arrays: {}"
        .format(label, available_names)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot v_x(t) and z(t) of one R=0.05 solid particle."
    )

    parser.add_argument(
        "--vtk_dir",
        default="Slit_Post_VTK_Files"
    )

    parser.add_argument(
        "--pattern",
        default="*Particles*.vtu"
    )

    parser.add_argument(
        "--target_radius",
        type=float,
        default=0.05
    )

    parser.add_argument(
        "--radius_tolerance",
        type=float,
        default=1.0e-6
    )

    parser.add_argument(
        "--output_interval",
        type=float,
        default=0.1,
        help="Time spacing between VTK files."
    )

    parser.add_argument(
        "--t_start",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--t_end",
        type=float,
        default=1.0e30
    )

    parser.add_argument(
        "--out",
        default="solid_particle_vx_and_z.png"
    )

    args = parser.parse_args()

    files = glob.glob(
        os.path.join(args.vtk_dir, args.pattern)
    )

    files = sorted(files, key=vtk_file_index)

    if not files:
        raise FileNotFoundError(
            "No VTK files found in '{}' with pattern '{}'."
            .format(args.vtk_dir, args.pattern)
        )

    first_mesh = pv.read(files[0])

    radius_name = find_array_name(
        first_mesh.point_data.keys(),
        ["radius", "RADIUS", "Radius"],
        "radius"
    )

    velocity_name = find_array_name(
        first_mesh.point_data.keys(),
        ["velocity", "VELOCITY", "Velocity"],
        "velocity"
    )

    print("Using radius array:", radius_name)
    print("Using velocity array:", velocity_name)

    times = []
    vx_values = []
    z_values = []

    target_found = False

    for filepath in files:
        file_index = vtk_file_index(filepath)

        if file_index < 0:
            continue

        simulation_time = file_index * args.output_interval

        if simulation_time < args.t_start:
            continue

        if simulation_time > args.t_end:
            break

        mesh = pv.read(filepath)

        radii = np.asarray(
            mesh.point_data[radius_name]
        ).reshape(-1)

        matching_indices = np.flatnonzero(
            np.abs(radii - args.target_radius)
            <= args.radius_tolerance
        )

        if len(matching_indices) == 0:
            if target_found:
                print(
                    "Target R={} disappeared at t={}; stopping."
                    .format(
                        args.target_radius,
                        simulation_time
                    )
                )
                break

            continue

        if len(matching_indices) > 1:
            raise RuntimeError(
                "Found {} particles with R={} +/- {} at t={}. "
                "This script requires exactly one target particle."
                .format(
                    len(matching_indices),
                    args.target_radius,
                    args.radius_tolerance,
                    simulation_time
                )
            )

        target_found = True

        target_index = int(matching_indices[0])

        point = np.asarray(mesh.points)[target_index]
        velocity = np.asarray(
            mesh.point_data[velocity_name]
        )[target_index]

        times.append(simulation_time)
        vx_values.append(velocity[0])
        z_values.append(point[2])

    if not times:
        raise RuntimeError(
            "The R={} particle was not found in the selected "
            "time interval."
            .format(args.target_radius)
        )

    times = np.asarray(times)
    vx_values = np.asarray(vx_values)
    z_values = np.asarray(z_values)

    print("Frames used:", len(times))
    print(
        "Time range: [{}, {}]"
        .format(times[0], times[-1])
    )
    print(
        "v_x range: [{:.8g}, {:.8g}]"
        .format(np.min(vx_values), np.max(vx_values))
    )
    print(
        "z range: [{:.8g}, {:.8g}]"
        .format(np.min(z_values), np.max(z_values))
    )

    figure, (axis_velocity, axis_z) = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True
    )

    axis_velocity.plot(
        times,
        vx_values,
        color="steelblue",
        linewidth=1.3
    )

    axis_velocity.set_ylabel(r"$v_x$")
    axis_velocity.set_title(
        r"Solid particle, $R = {:.6g}$: $v_x(t)$ and $z(t)$"
        .format(args.target_radius)
    )
    axis_velocity.grid(True, alpha=0.3)

    axis_z.plot(
        times,
        z_values,
        color="darkgreen",
        linewidth=1.3
    )

    axis_z.set_xlabel("Time (simulation units)")
    axis_z.set_ylabel(r"$z$")
    axis_z.grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(args.out, dpi=160)

    print("Saved:", args.out)


if __name__ == "__main__":
    main()