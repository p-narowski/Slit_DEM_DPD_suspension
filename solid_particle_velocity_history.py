"""
solid_particle_velocity_history.py
==================================

Track one solid Kratos DEM particle selected by radius, generate:

  1. v_x(t), including a linear fit over a chosen fit window.
  2. Unwrapped particle trajectory s(t), rather than wrapped x(t).

This version is designed for VTU output that contains, for example:

    ['material', 'radius', 'velocity', 'displacement', 'total_forces']

and does not contain node IDs.

For periodic x boundaries, raw x(t) jumps from x_max to x_min. The script
constructs an unwrapped trajectory:

    s(t_0) = 0
    s(t_i) = s(t_{i-1}) + delta_x_unwrapped

where delta_x_unwrapped is corrected by +/- period_x when a boundary crossing
is detected.

Example:
python solid_particle_velocity_history.py \
    --vtk_dir Slit_Post_VTK_Files \
    --target_radius 0.05 \
    --radius_tolerance 1e-6 \
    --output_interval 0.1 \
    --t_start 1.5 \
    --t_end 19.9 \
    --steady_start 10.0 \
    --fit_start 10.0 \
    --fit_end 19.9 \
    --period_x 1.0
"""

import argparse
import glob
import os
import re
from xml.etree import cElementTree as ET

import matplotlib.pyplot as plt
import numpy as np

try:
    import pyvista as pv
except ImportError as exc:
    raise ImportError("Install PyVista first: pip install pyvista") from exc


def get_sim_time_from_vtu(filepath, output_interval=0.1):
    """
    Extract physical time from VTU FieldData.

    If TIME is not stored in the VTU file, use:

        t = file_index * output_interval
    """
    try:
        root = ET.parse(filepath).getroot()

        for field_data in root.iter("FieldData"):
            for data_array in field_data:
                if data_array.get("Name") == "TIME" and data_array.text:
                    return float(data_array.text.strip())

    except Exception:
        pass

    match = re.search(r"_(\d+)\.vtu$", filepath)

    if match:
        file_index = int(match.group(1))
        return file_index * output_interval

    return None


def natural_file_key(filepath):
    """
    Sort Slit_Particles_2.vtu before Slit_Particles_10.vtu.

    Use the trailing numeric output index, independent of file-name
    lexicographic ordering.
    """
    match = re.search(r"_(\d+)\.vtu$", os.path.basename(filepath))

    if match:
        return int(match.group(1))

    return float("inf")


def find_array_name(available_names, requested_name, candidates, label):
    """Find a VTK point-data array by explicit name or common alternatives."""
    available_names = list(available_names)

    if requested_name is not None:
        if requested_name not in available_names:
            raise KeyError(
                "Requested {} array '{}' was not found.\n"
                "Available point arrays: {}".format(
                    label,
                    requested_name,
                    available_names
                )
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
        "No {} array found automatically.\n"
        "Available point arrays: {}\n"
        "Specify it explicitly with --{}_array.".format(
            label,
            available_names,
            label
        )
    )


def find_velocity_array(mesh, requested_name=None):
    """Return the name of a three-component velocity point-data array."""
    array_name = find_array_name(
        mesh.point_data.keys(),
        requested_name,
        [
            "VELOCITY",
            "Velocity",
            "velocity",
            "PARTICLE_VELOCITY",
            "v",
            "VEL"
        ],
        "vel"
    )

    values = np.asarray(mesh.point_data[array_name])

    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError(
            "Velocity array '{}' is not a three-component vector array."
            .format(array_name)
        )

    return array_name


def find_radius_array(mesh, requested_name=None):
    """Return the name of the scalar radius point-data array."""
    return find_array_name(
        mesh.point_data.keys(),
        requested_name,
        [
            "RADIUS",
            "Radius",
            "radius",
            "PARTICLE_RADIUS"
        ],
        "radius"
    )


def select_particle_by_exact_radius(
    mesh,
    radius_array,
    target_radius,
    radius_tolerance
):
    """
    Select exactly one target particle satisfying:

        abs(radius - target_radius) <= radius_tolerance.
    """
    radii = np.asarray(mesh.point_data[radius_array]).reshape(-1)

    matching_indices = np.flatnonzero(
        np.abs(radii - target_radius) <= radius_tolerance
    )

    if len(matching_indices) == 0:
        nearest_index = int(np.argmin(np.abs(radii - target_radius)))

        raise RuntimeError(
            "No particle matches target radius R = {} within tolerance {}.\n"
            "Nearest particle: index = {}, radius = {}, difference = {}."
            .format(
                target_radius,
                radius_tolerance,
                nearest_index,
                radii[nearest_index],
                abs(radii[nearest_index] - target_radius)
            )
        )

    if len(matching_indices) > 1:
        raise RuntimeError(
            "Found {} particles matching R = {} within tolerance {}.\n"
            "Matching radii: {}"
            .format(
                len(matching_indices),
                target_radius,
                radius_tolerance,
                radii[matching_indices]
            )
        )

    return int(matching_indices[0])


def read_velocity_history(
    vtk_dir,
    pattern,
    t_start,
    t_end,
    target_radius,
    radius_tolerance,
    velocity_array,
    radius_array,
    output_interval=0.1
):
    """
    Read the selected solid-particle history.

    Before the target particle is inserted, frames without target radius are
    skipped. Once it has been found, a missing target is treated as its removal
    and terminates tracking safely.
    """
    files = glob.glob(os.path.join(vtk_dir, pattern))
    files = sorted(files, key=natural_file_key)

    if not files:
        raise FileNotFoundError(
            "No VTU files matching '{}' were found in '{}'."
            .format(pattern, vtk_dir)
        )

    print("Inspecting first file:", os.path.basename(files[0]))

    first_mesh = pv.read(files[0])

    print("Point arrays:", list(first_mesh.point_data.keys()))

    velocity_name = find_velocity_array(first_mesh, velocity_array)
    radius_name = find_radius_array(first_mesh, radius_array)

    print("Using velocity array:", repr(velocity_name))
    print("Using radius array:", repr(radius_name))
    print(
        "Target particle: R = {} +/- {}"
        .format(target_radius, radius_tolerance)
    )
    print(
        "Fallback VTK output interval: {}"
        .format(output_interval)
    )

    history = []
    particle_has_appeared = False

    for filepath in files:
        simulation_time = get_sim_time_from_vtu(
            filepath,
            output_interval
        )

        if simulation_time is None:
            print(
                "[warning] Could not determine time for '{}'; skipping."
                .format(os.path.basename(filepath))
            )
            continue

        if simulation_time < t_start or simulation_time > t_end:
            continue

        mesh = pv.read(filepath)

        try:
            target_index = select_particle_by_exact_radius(
                mesh,
                radius_name,
                target_radius,
                radius_tolerance
            )

        except RuntimeError as error:
            if not particle_has_appeared:
                print(
                    "[info] Target R = {} is not present at t = {}; "
                    "skipping pre-insertion frame."
                    .format(target_radius, simulation_time)
                )
                continue

            print(
                "[info] Target R = {} is no longer present at t = {}; "
                "stopping tracking.\n{}"
                .format(target_radius, simulation_time, error)
            )
            break

        particle_has_appeared = True

        points = np.asarray(mesh.points)
        velocity = np.asarray(mesh.point_data[velocity_name])
        radii = np.asarray(mesh.point_data[radius_name]).reshape(-1)

        position = points[target_index]
        target_velocity = velocity[target_index]
        actual_radius = radii[target_index]

        history.append(
            [
                simulation_time,
                position[0],
                position[1],
                position[2],
                target_velocity[0],
                target_velocity[1],
                target_velocity[2],
                actual_radius
            ]
        )

        print(
            "Loaded t={:.8g}, R={:.12g}, x={:.8g}, v_x={:.8g}"
            .format(
                simulation_time,
                actual_radius,
                position[0],
                target_velocity[0]
            )
        )

    if not history:
        raise RuntimeError(
            "No frames containing the selected target particle were found "
            "in t=[{}, {}].".format(t_start, t_end)
        )

    history = np.asarray(history, dtype=float)

    return history[np.argsort(history[:, 0])], velocity_name, radius_name


def unwrap_periodic_trajectory(x, period_x):
    """
    Construct travelled x-distance s(t) from a periodic coordinate x(t).

    If period_x > 0:
      - raw jump dx < -period_x/2 is interpreted as forward crossing:
            dx_corrected = dx + period_x
      - raw jump dx > +period_x/2 is interpreted as backward crossing:
            dx_corrected = dx - period_x

    s[0] = 0, then s[i] accumulates corrected dx values.

    Set period_x <= 0 to use nonperiodic displacement s=x-x[0].
    """
    if len(x) == 0:
        return np.array([])

    if period_x <= 0.0:
        return x - x[0]

    dx = np.diff(x)

    dx_corrected = dx.copy()
    dx_corrected[dx < -0.5 * period_x] += period_x
    dx_corrected[dx > 0.5 * period_x] -= period_x

    s = np.zeros_like(x)
    s[1:] = np.cumsum(dx_corrected)

    return s


def time_weighted_average(time, values):
    """Compute integral(values dt) / total time using trapezoidal integration."""
    if len(time) == 1:
        return float(values[0])

    duration = time[-1] - time[0]

    if duration <= 0.0:
        return float(np.mean(values))

    return float(np.trapezoid(values, time) / duration)


def linear_fit(time, values, fit_mask):
    """
    Fit values = slope * time + intercept over the selected fit mask.

    Returns fitted values over all supplied times, slope, intercept, and R^2.
    """
    n_fit = np.count_nonzero(fit_mask)

    if n_fit < 2:
        raise RuntimeError(
            "At least two frames are required for the v_x(t) linear fit."
        )

    slope, intercept = np.polyfit(
        time[fit_mask],
        values[fit_mask],
        1
    )

    fitted_all = slope * time + intercept
    fitted_window = slope * time[fit_mask] + intercept

    residuals = values[fit_mask] - fitted_window
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum(
        (values[fit_mask] - np.mean(values[fit_mask])) ** 2
    )

    if ss_tot > 0.0:
        r_squared = 1.0 - ss_res / ss_tot
    else:
        r_squared = np.nan

    return fitted_all, slope, intercept, r_squared


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Track a solid DEM particle by radius, plot v_x(t), "
            "fit v_x(t), and calculate an unwrapped trajectory s(t)."
        )
    )

    parser.add_argument(
        "--vtk_dir",
        default="Slit_Post_VTK_Files",
        help="Directory containing Kratos *Particles*.vtu files"
    )

    parser.add_argument(
        "--pattern",
        default="*Particles*.vtu",
        help="Glob pattern for particle VTU files"
    )

    parser.add_argument(
        "--target_radius",
        type=float,
        default=0.05,
        help="Solid-particle radius to track; default: 0.05"
    )

    parser.add_argument(
        "--radius_tolerance",
        type=float,
        default=1.0e-6,
        help="Absolute tolerance for radius selection; default: 1e-6"
    )

    parser.add_argument(
        "--vel_array",
        default=None,
        help="Exact VTK velocity-array name; auto-detected if omitted"
    )

    parser.add_argument(
        "--radius_array",
        default=None,
        help="Exact VTK radius-array name; auto-detected if omitted"
    )

    parser.add_argument(
        "--output_interval",
        type=float,
        default=0.1,
        help=(
            "Physical time between VTK files when TIME is absent "
            "from VTU FieldData; default: 0.1"
        )
    )

    parser.add_argument(
        "--t_start",
        type=float,
        default=0.0,
        help="First physical simulation time included"
    )

    parser.add_argument(
        "--t_end",
        type=float,
        default=np.inf,
        help="Last physical simulation time included"
    )

    parser.add_argument(
        "--steady_start",
        type=float,
        default=None,
        help=(
            "Start time for the late-window mean; "
            "default: same as --t_start"
        )
    )

    parser.add_argument(
        "--fit_start",
        type=float,
        default=None,
        help=(
            "Start time for the v_x(t) linear fit; "
            "default: --steady_start"
        )
    )

    parser.add_argument(
        "--fit_end",
        type=float,
        default=None,
        help=(
            "End time for the v_x(t) linear fit; "
            "default: last tracked time"
        )
    )

    parser.add_argument(
        "--period_x",
        type=float,
        default=1.0,
        help=(
            "Periodic x-domain length used to unwrap x(t). "
            "Use 1.0 for x in [0,1], or <=0 for nonperiodic x."
        )
    )

    parser.add_argument(
        "--out",
        default="solid_particle_vx_and_trajectory.png",
        help="Output PNG figure name"
    )

    parser.add_argument(
        "--csv_out",
        default="solid_particle_vx_and_trajectory.csv",
        help="Output CSV file name"
    )

    args = parser.parse_args()

    if args.radius_tolerance <= 0.0:
        raise ValueError("--radius_tolerance must be positive.")

    if args.output_interval <= 0.0:
        raise ValueError("--output_interval must be positive.")

    history, velocity_name, radius_name = read_velocity_history(
        args.vtk_dir,
        args.pattern,
        args.t_start,
        args.t_end,
        args.target_radius,
        args.radius_tolerance,
        args.vel_array,
        args.radius_array,
        args.output_interval
    )

    time = history[:, 0]
    x = history[:, 1]
    y = history[:, 2]
    z = history[:, 3]
    vx = history[:, 4]
    vy = history[:, 5]
    vz = history[:, 6]
    radius = history[:, 7]

    s = unwrap_periodic_trajectory(x, args.period_x)

    full_average_vx = time_weighted_average(time, vx)

    steady_start = (
        args.t_start
        if args.steady_start is None
        else args.steady_start
    )

    late_mask = time >= steady_start

    if not np.any(late_mask):
        raise RuntimeError(
            "No samples at t >= steady_start = {}."
            .format(steady_start)
        )

    late_average_vx = time_weighted_average(
        time[late_mask],
        vx[late_mask]
    )

    late_std_vx = (
        float(np.std(vx[late_mask], ddof=1))
        if np.count_nonzero(late_mask) > 1
        else 0.0
    )

    fit_start = steady_start if args.fit_start is None else args.fit_start
    fit_end = time[-1] if args.fit_end is None else args.fit_end

    fit_mask = (time >= fit_start) & (time <= fit_end)

    fitted_vx, velocity_slope, velocity_intercept, velocity_r2 = linear_fit(
        time,
        vx,
        fit_mask
    )

    print("\n--- Solid-particle results ---")
    print("Velocity array:", velocity_name)
    print("Radius array:", radius_name)
    print("Frames used:", len(time))
    print(
        "Tracked time range: [{:.12g}, {:.12g}]"
        .format(time[0], time[-1])
    )
    print(
        "Selected-radius range: [{:.12g}, {:.12g}]"
        .format(np.min(radius), np.max(radius))
    )
    print(
        "Full-window average <v_x>: {:.12g}"
        .format(full_average_vx)
    )
    print(
        "Late-window average <v_x>: {:.12g}"
        .format(late_average_vx)
    )
    print(
        "Late-window std(v_x): {:.12g}"
        .format(late_std_vx)
    )
    print(
        "Linear-fit time interval: [{:.12g}, {:.12g}]"
        .format(time[fit_mask][0], time[fit_mask][-1])
    )
    print(
        "Linear fit: v_x(t) = ({:.12g})*t + ({:.12g})"
        .format(velocity_slope, velocity_intercept)
    )
    print(
        "Linear-fit slope dv_x/dt: {:.12g}"
        .format(velocity_slope)
    )
    print(
        "Linear-fit R^2: {:.12g}"
        .format(velocity_r2)
    )
    print(
        "Net unwrapped trajectory s(t_end): {:.12g}"
        .format(s[-1])
    )

    if args.period_x > 0.0:
        print(
            "Periodic x length used for unwrapping: {}"
            .format(args.period_x)
        )

    output_table = np.column_stack(
        (
            time,
            x,
            s,
            y,
            z,
            vx,
            vy,
            vz,
            radius,
            late_mask.astype(int),
            fit_mask.astype(int),
            fitted_vx
        )
    )

    np.savetxt(
        args.csv_out,
        output_table,
        delimiter=",",
        header=(
            "time,x_wrapped,s_unwrapped,y,z,vx,vy,vz,radius,"
            "in_late_window,in_velocity_fit_window,vx_linear_fit"
        ),
        comments=""
    )

    print("Saved CSV:", repr(args.csv_out))

    figure, (axis_velocity, axis_trajectory) = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        sharex=True
    )

    axis_velocity.plot(
        time,
        vx,
        color="steelblue",
        linewidth=1.4,
        label=r"$v_x(t)$"
    )

    axis_velocity.plot(
        time[fit_mask],
        fitted_vx[fit_mask],
        color="black",
        linestyle="--",
        linewidth=2.0,
        label=(
            r"Linear fit: $dv_x/dt$ = "
            "{:.4g}".format(velocity_slope)
        )
    )

    axis_velocity.axhline(
        full_average_vx,
        color="gray",
        linestyle=":",
        linewidth=1.5,
        label=r"Full mean = {:.6g}".format(full_average_vx)
    )

    axis_velocity.axhline(
        late_average_vx,
        color="tomato",
        linestyle="-.",
        linewidth=1.8,
        label=r"Late mean = {:.6g}".format(late_average_vx)
    )

    axis_velocity.axvspan(
        time[fit_mask][0],
        time[fit_mask][-1],
        color="tomato",
        alpha=0.10,
        label="Velocity-fit window"
    )

    axis_velocity.set_ylabel(r"$v_x$ (simulation units)")
    axis_velocity.set_title(
        r"Solid particle with $R = {:.6g}$: velocity and trajectory"
        .format(args.target_radius)
    )
    axis_velocity.grid(True, alpha=0.3)
    axis_velocity.legend(loc="upper center", ncol=2)

    result_text = (
        rf"$\langle v_x \rangle_{{\rm full}}$ = {full_average_vx:.6g}"
        "\n"
        rf"$\langle v_x \rangle_{{\rm late}}$ = {late_average_vx:.6g}"
        "\n"
        rf"$dv_x/dt$ = {velocity_slope:.6g}"
        "\n"
        rf"$R^2_{{\rm fit}}$ = {velocity_r2:.6g}"
    )

    axis_velocity.text(
        0.98,
        0.04,
        result_text,
        transform=axis_velocity.transAxes,
        horizontalalignment="right",
        verticalalignment="bottom",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="lightyellow",
            edgecolor="gray",
            alpha=0.90
        )
    )

    axis_trajectory.plot(
        time,
        s,
        color="darkgreen",
        linewidth=1.7,
        label=r"Unwrapped trajectory $s(t)$"
    )

    axis_trajectory.set_xlabel("Time (simulation units)")
    axis_trajectory.set_ylabel(r"Travelled distance $s$")
    axis_trajectory.grid(True, alpha=0.3)
    axis_trajectory.legend(loc="best")

    axis_trajectory.text(
        0.98,
        0.05,
        rf"$s(t_{{\rm end}})$ = {s[-1]:.6g}",
        transform=axis_trajectory.transAxes,
        horizontalalignment="right",
        verticalalignment="bottom",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="lightyellow",
            edgecolor="gray",
            alpha=0.90
        )
    )

    figure.tight_layout()
    figure.savefig(args.out, dpi=160)

    print("Saved figure:", repr(args.out))


if __name__ == "__main__":
    main()