#!/usr/bin/env python3
"""
Plot solid velocity, fluid velocity, and solid-fluid relative velocity.

Fluid selection:
    fluid_radius_min < radius <= fluid_radius_max

Outputs:
    <out_prefix>.csv
    <out_prefix>.png

The material array is not used.
"""

import argparse
import csv
import json
import re
import warnings
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pyvista as pv
except ImportError as exc:
    raise ImportError(
        "Install dependencies with: pip install numpy matplotlib pyvista"
    ) from exc


def find_array(mesh, requested, candidates):
    names = list(mesh.point_data.keys())

    if requested is not None:
        if requested not in names:
            raise KeyError(
                f"Array {requested!r} not found. Available arrays: {names}"
            )
        return requested

    lookup = {name.lower(): name for name in names}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    raise KeyError(
        f"No matching array for {candidates}. Available arrays: {names}"
    )


def frame_index(path):
    match = re.search(r"_(\d+)\.vtu$", path.name)

    if match is None:
        raise ValueError(
            f"Cannot extract output index from {path.name!r}. "
            "Expected a filename ending in _NUMBER.vtu."
        )

    return int(match.group(1))


def frame_time(mesh, path, output_interval):
    for name in mesh.field_data.keys():
        if name.lower() == "time":
            values = np.asarray(mesh.field_data[name]).ravel()

            if values.size == 1 and np.isfinite(values[0]):
                return float(values[0]), False

    return frame_index(path) * output_interval, True


def minimum_image(delta, periods):
    result = delta.copy()

    for axis, length in enumerate(periods):
        if length > 0.0:
            result[:, axis] -= length * np.rint(
                result[:, axis] / length
            )

    return result


def load_settings(args, parser):
    parameter_path = Path(args.parameters)
    settings = {}

    if parameter_path.exists():
        with parameter_path.open() as handle:
            settings = json.load(handle)

    interval = args.output_interval

    if interval is None:
        interval = settings.get("OutputTimeStep")

    if interval is None or not np.isfinite(interval) or interval <= 0:
        parser.error(
            "Supply --output_interval or a parameters JSON "
            "containing a positive OutputTimeStep."
        )

    if args.periods is not None:
        periods = np.asarray(args.periods, dtype=float)

    elif settings.get("PeriodicDomainOption", False):
        try:
            periods = np.array([
                settings[f"BoundingBoxMax{axis}"]
                - settings[f"BoundingBoxMin{axis}"]
                for axis in "XYZ"
            ], dtype=float)
        except KeyError as exc:
            parser.error(
                f"Missing bounding-box setting {exc}. "
                "Supply --periods Lx Ly Lz explicitly."
            )

        if np.any(periods <= 0):
            parser.error("Periodic bounding-box lengths must be positive.")

    elif parameter_path.exists():
        periods = np.zeros(3)

    else:
        parser.error(
            "No parameters JSON found. Supply --periods Lx Ly Lz; "
            "use 0 for each nonperiodic direction."
        )

    if not np.isfinite(periods).all() or np.any(periods < 0):
        parser.error("Periodic lengths must be finite and nonnegative.")

    return float(interval), periods


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--vtk_dir", default="Slit_Post_VTK_Files"
    )
    parser.add_argument(
        "--pattern", default="*Particles*.vtu"
    )
    parser.add_argument(
        "--parameters", default="ProjectParametersDEM.json"
    )

    parser.add_argument(
        "--target_radius", type=float, default=0.2
    )
    parser.add_argument(
        "--radius_tolerance", type=float, default=1e-6
    )

    parser.add_argument(
        "--fluid_radius_min",
        type=float,
        default=0.0,
        help="Exclusive lower radius bound for fluid particles."
    )
    parser.add_argument(
        "--fluid_radius_max",
        type=float,
        required=True,
        help="Inclusive upper radius bound for fluid particles."
    )

    parser.add_argument("--vel_array")
    parser.add_argument("--radius_array")
    parser.add_argument(
        "--mass_array",
        help=(
            "Optional point-data particle-mass array. "
            "If omitted, means use equal particle weights."
        )
    )

    parser.add_argument(
        "--output_interval",
        type=float,
        help="Time per output index if the VTK TIME field is absent."
    )
    parser.add_argument(
        "--periods",
        type=float,
        nargs=3,
        metavar=("LX", "LY", "LZ"),
        help="Override periodic lengths; use 0 for nonperiodic directions."
    )

    parser.add_argument(
        "--exclude_factor",
        type=float,
        default=2.0,
        help=(
            "Background fluid excludes centres at distance "
            "< exclude_factor * actual solid radius."
        )
    )

    parser.add_argument("--t_start", type=float, default=1.5)
    parser.add_argument("--t_end", type=float, default=float("inf"))

    parser.add_argument(
        "--mu",
        type=float,
        help="Optional viscosity for a stationary-fluid Stokes reference."
    )
    parser.add_argument(
        "--force",
        type=float,
        default=0.5,
        help="Signed x force used only for the optional Stokes reference."
    )
    parser.add_argument(
        "--out_prefix",
        default="solid_fluid_relative_velocity"
    )

    args = parser.parse_args()

    if not np.isfinite(args.target_radius) or args.target_radius <= 0:
        parser.error("--target_radius must be finite and positive.")

    if (
        not np.isfinite(args.radius_tolerance)
        or not 0 < args.radius_tolerance < args.target_radius
    ):
        parser.error(
            "--radius_tolerance must be positive and below target radius."
        )

    if not (
        np.isfinite(args.fluid_radius_min)
        and np.isfinite(args.fluid_radius_max)
        and 0 <= args.fluid_radius_min < args.fluid_radius_max
    ):
        parser.error(
            "Require finite 0 <= fluid_radius_min < fluid_radius_max."
        )

    if args.fluid_radius_max >= (
        args.target_radius - args.radius_tolerance
    ):
        parser.error(
            "--fluid_radius_max must be below the target-radius "
            "selection range."
        )

    if (
        not np.isfinite(args.exclude_factor)
        or args.exclude_factor < 1
    ):
        parser.error("--exclude_factor must be finite and at least 1.")

    if not np.isfinite(args.t_start) or args.t_end <= args.t_start:
        parser.error("--t_end must exceed finite --t_start.")

    if np.isnan(args.t_end):
        parser.error("--t_end cannot be NaN.")

    if args.mu is not None and (
        not np.isfinite(args.mu) or args.mu <= 0
    ):
        parser.error("--mu must be finite and positive.")

    if not np.isfinite(args.force):
        parser.error("--force must be finite.")

    output_interval, periods = load_settings(args, parser)

    paths = list(Path(args.vtk_dir).glob(args.pattern))

    if not paths:
        raise FileNotFoundError(
            f"No files matched {args.pattern!r} in {args.vtk_dir!r}."
        )

    paths.sort(key=frame_index)

    print("Fluid selection uses radius only; material labels are ignored.")
    print(
        f"Fluid radius interval: "
        f"({args.fluid_radius_min:g}, {args.fluid_radius_max:g}]"
    )
    print("Periodic lengths:", periods)
    print("Fallback output interval:", output_interval)
    print(
        "Average weighting:",
        f"mass array {args.mass_array!r}"
        if args.mass_array else "equal particle weights"
    )

    rows = []
    fallback_count = 0

    for path in paths:
        mesh = pv.read(path)

        time, fallback = frame_time(
            mesh, path, output_interval
        )

        if not args.t_start <= time <= args.t_end:
            continue

        velocity_name = find_array(
            mesh,
            args.vel_array,
            ["velocity", "particle_velocity", "vel", "v"]
        )
        radius_name = find_array(
            mesh,
            args.radius_array,
            ["radius", "particle_radius"]
        )

        points = np.asarray(mesh.points, dtype=float)
        velocity = np.asarray(
            mesh.point_data[velocity_name], dtype=float
        )
        radius = np.asarray(
            mesh.point_data[radius_name], dtype=float
        ).ravel()

        if velocity.shape != (len(points), 3):
            raise ValueError(
                f"{path.name}: velocity must have shape (n_points, 3)."
            )

        if len(radius) != len(points):
            raise ValueError(
                f"{path.name}: radius must contain one value per point."
            )

        target_indices = np.flatnonzero(
            np.isfinite(radius)
            & (
                np.abs(radius - args.target_radius)
                <= args.radius_tolerance
            )
        )

        if len(target_indices) == 0:
            if rows:
                raise RuntimeError(
                    f"Target disappeared after tracking began: {path.name}"
                )
            continue

        if len(target_indices) != 1:
            raise RuntimeError(
                f"{path.name}: found {len(target_indices)} particles "
                "matching target radius; cannot identify a unique solid."
            )

        target = int(target_indices[0])

        fluid = (
            np.isfinite(radius)
            & (radius > args.fluid_radius_min)
            & (radius <= args.fluid_radius_max)
        )
        fluid[target] = False

        if not fluid.any():
            raise RuntimeError(
                f"{path.name}: no particles inside the requested "
                "fluid-radius interval."
            )

        selected = fluid.copy()
        selected[target] = True

        if (
            not np.isfinite(points[selected]).all()
            or not np.isfinite(velocity[selected]).all()
        ):
            raise ValueError(
                f"{path.name}: nonfinite selected positions or velocities."
            )

        delta = minimum_image(points - points[target], periods)
        distance = np.linalg.norm(delta, axis=1)

        exclusion_radius = args.exclude_factor * radius[target]
        background = fluid & (distance >= exclusion_radius)

        if not background.any():
            raise RuntimeError(
                f"{path.name}: no background fluid points remain "
                f"outside centre distance {exclusion_radius:g}. "
                "Reduce --exclude_factor."
            )

        if args.mass_array is not None:
            if args.mass_array not in mesh.point_data:
                raise KeyError(
                    f"{path.name}: mass array {args.mass_array!r} missing."
                )

            weights = np.asarray(
                mesh.point_data[args.mass_array], dtype=float
            ).ravel()

            if len(weights) != len(points):
                raise ValueError(
                    "Mass array must contain one value per point."
                )

            if (
                not np.isfinite(weights[fluid]).all()
                or np.any(weights[fluid] <= 0)
            ):
                raise ValueError(
                    f"{path.name}: selected fluid masses must be "
                    "finite and positive."
                )
        else:
            weights = np.ones(len(points))

        vp = velocity[target]
        u_bulk = np.average(
            velocity[fluid], axis=0, weights=weights[fluid]
        )
        u_background = np.average(
            velocity[background],
            axis=0,
            weights=weights[background]
        )

        n_fluid = int(fluid.sum())
        n_other = len(points) - n_fluid - 1

        if not rows:
            print("\nFirst accepted frame:", path.name)
            print("Point arrays:", list(mesh.point_data.keys()))
            print("Total points:", len(points))
            print("Tracked solid radius:", radius[target])
            print("Selected fluid points:", n_fluid)
            print(
                "Selected fluid radius range:",
                float(radius[fluid].min()),
                "to",
                float(radius[fluid].max())
            )
            print("Other excluded points:", n_other)
            print("Background fluid points:", int(background.sum()))

            if n_other > 0:
                warnings.warn(
                    f"{n_other} points other than the target are outside "
                    "the fluid-radius interval. Verify whether these are "
                    "other solids or excluded DPD fluid."
                )

        row = {
            "time": time,
            "source_vtu": path.name,
            "radius": float(radius[target]),
            "n_total": len(points),
            "n_fluid": n_fluid,
            "n_background": int(background.sum()),
            "n_other_excluded": n_other,
            "fluid_radius_min_selected": float(radius[fluid].min()),
            "fluid_radius_max_selected": float(radius[fluid].max()),
            "background_exclusion_radius": float(exclusion_radius),
        }

        vectors = {
            "vp": vp,
            "u_bulk": u_bulk,
            "u_background": u_background,
            "w_bulk": vp - u_bulk,
            "w_background": vp - u_background,
        }

        for prefix, vector in vectors.items():
            for axis, value in zip("xyz", vector):
                row[f"{prefix}_{axis}"] = float(value)

        rows.append(row)
        fallback_count += int(fallback)

    if len(rows) < 2:
        raise RuntimeError(
            "Need at least two accepted frames containing the target."
        )

    rows.sort(key=lambda row: row["time"])
    time = np.array([row["time"] for row in rows])

    if np.any(np.diff(time) <= 0):
        raise RuntimeError(
            "Duplicate/non-increasing frame times. "
            "Check the filename pattern and TIME fields."
        )

    if fallback_count:
        warnings.warn(
            f"{fallback_count} accepted frames lack a usable TIME field. "
            f"Used filename index * {output_interval:g}; "
            "verify this mapping matches your output."
        )

    def column(name):
        return np.array([row[name] for row in rows], dtype=float)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    csv_path = Path(str(out_prefix) + ".csv")
    png_path = Path(str(out_prefix) + ".png")

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(
        2, 1, figsize=(11, 9), sharex=True
    )

    axes[0].plot(
        time, column("vp_x"),
        label=r"Solid $V_{p,x}$", linewidth=2
    )
    axes[0].plot(
        time, column("u_bulk_x"),
        label=r"Fluid bulk mean $\overline{U}_{f,x}$",
        linewidth=1.8
    )
    axes[0].plot(
        time, column("u_background_x"),
        label=(
            r"Fluid background $U_{\mathrm{bg},x}$"
            f" (r >= {args.exclude_factor:g} R)"
        ),
        linewidth=1.8
    )
    axes[0].set_ylabel("Laboratory-frame x velocity")
    axes[0].set_title(
        "Solid and radius-selected DPD fluid: same-frame comparison"
    )

    axes[1].plot(
        time, column("w_background_x"),
        label=r"$V_{p,x}-U_{\mathrm{bg},x}$",
        linewidth=2
    )
    axes[1].plot(
        time, column("w_bulk_x"),
        label=r"$V_{p,x}-\overline{U}_{f,x}$",
        linestyle="--",
        linewidth=1.8
    )

    if args.mu is not None:
        reference = (
            args.force
            / (6.0 * np.pi * args.mu * args.target_radius)
        )
        axes[1].axhline(
            reference,
            color="black",
            linestyle=":",
            linewidth=1.8,
            label=(
                "Stationary-fluid Stokes reference "
                f"= {reference:.6g}"
            )
        )

    axes[1].set_xlabel("Simulation time")
    axes[1].set_ylabel("Relative x velocity")

    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(loc="best")

    figure.tight_layout()
    figure.savefig(png_path, dpi=200)
    plt.close(figure)

    print(
        f"\nProcessed {len(rows)} frames, "
        f"time {time[0]:g} to {time[-1]:g}"
    )
    print(
        "Fluid count range:",
        int(column("n_fluid").min()),
        "to",
        int(column("n_fluid").max())
    )
    print("Saved CSV:", csv_path)
    print("Saved plot:", png_path)


if __name__ == "__main__":
    main()