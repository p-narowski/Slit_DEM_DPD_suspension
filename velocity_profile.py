"""
velocity_profile.py
====================
Reads Kratos DEM particle VTK output files (Slit_Particles_*.vtu),
bins particle u_x by z-coordinate, time-averages over a chosen window,
fits a parabola and plots the result.

Usage:
    python velocity_profile.py --vtk_dir Slit_Post_VTK_Files --t_start 5.0 --n_bins 40
"""

import os
import glob
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt

try:
    import pyvista as pv
    USE_PYVISTA = True
except ImportError:
    USE_PYVISTA = False

try:
    from xml.etree import cElementTree as ET
except ImportError:
    from xml.etree import ElementTree as ET


# ─────────────────────────────────────────────────────────────────────────────
def get_sim_time_from_vtu(filepath):
    """Extract simulation time from VTU FieldData, or fall back to filename index."""
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
        for fd in root.iter("FieldData"):
            for da in fd:
                if da.get("Name") == "TIME":
                    return float(da.text.strip())
    except Exception:
        pass
    m = re.search(r"_(\d+)\.vtu$", filepath)
    return int(m.group(1)) if m else None


def find_velocity_array(mesh, candidates=None):
    """
    Return the name of the velocity point-data array.
    Tries a list of candidates (case-insensitive) then falls back to any
    3-component array.  Prints available arrays on the first call.
    """
    if candidates is None:
        candidates = ["VELOCITY", "Velocity", "velocity",
                      "PARTICLE_VELOCITY", "v", "VEL"]

    available = mesh.point_data.keys()

    for name in candidates:
        if name in available:
            return name
        # case-insensitive
        for a in available:
            if a.lower() == name.lower():
                return a

    # last resort: first 3-component array
    for a in available:
        arr = mesh.point_data[a]
        if arr.ndim == 2 and arr.shape[1] == 3:
            print(f"  [warn] velocity array not found by name; using '{a}' "
                  f"(first 3-component point array)")
            return a

    raise KeyError(
        f"No velocity array found in VTU file.\n"
        f"Available point arrays: {list(available)}\n"
        f"Pass --vel_array <NAME> to specify it explicitly."
    )


def read_vtu_pyvista(filepath, vel_array=None):
    mesh = pv.read(filepath)
    pts  = np.array(mesh.points)                           # (N, 3)

    if vel_array is None:
        vel_array = find_velocity_array(mesh)

    vel = np.array(mesh.point_data[vel_array])             # (N, 3)
    return pts, vel, vel_array


def read_all_vtu(vtk_dir, pattern="*Particles*.vtu",
                 t_start=0.0, t_end=np.inf, vel_array=None):
    files = sorted(glob.glob(os.path.join(vtk_dir, pattern)))
    if not files:
        raise FileNotFoundError(
            f"No VTU files matching '{pattern}' found in '{vtk_dir}'.\n"
            f"Check --vtk_dir and --pattern."
        )

    # print arrays available in the first file so user can verify
    print(f"\n[info] Inspecting first file: {os.path.basename(files[0])}")
    mesh0 = pv.read(files[0])
    print(f"  Point arrays : {list(mesh0.point_data.keys())}")
    print(f"  Field arrays : {list(mesh0.field_data.keys())}")

    detected = vel_array or find_velocity_array(mesh0)
    print(f"  Using velocity array: '{detected}'\n")

    frames = []
    for fpath in files:
        t = get_sim_time_from_vtu(fpath)
        if t is None:
            continue
        if not (t_start <= t <= t_end):
            continue
        pts, vel, _ = read_vtu_pyvista(fpath, vel_array=detected)
        z  = pts[:, 2]
        ux = vel[:, 0]
        frames.append((t, z, ux))
        print(f"  loaded t={t:.4f}  n_particles={len(z)}")

    return frames


# ─────────────────────────────────────────────────────────────────────────────
def compute_averaged_profile(frames, n_bins=40, z_min=0.0, z_max=0.5):
    edges   = np.linspace(z_min, z_max, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    sum_ux  = np.zeros(n_bins)
    count   = np.zeros(n_bins, dtype=int)

    for _, z, ux in frames:
        idx = np.searchsorted(edges[1:], z)
        idx = np.clip(idx, 0, n_bins - 1)
        np.add.at(sum_ux, idx, ux)
        np.add.at(count,  idx, 1)

    mean_ux = np.where(count > 0, sum_ux / np.maximum(count, 1), np.nan)
    return centres, mean_ux, count


def fit_parabola(z, ux):
    mask   = ~np.isnan(ux)
    coeffs = np.polyfit(z[mask], ux[mask], 2)
    fitted = np.polyval(coeffs, z)
    return fitted, coeffs


def compute_slip_lengths(coeffs, z_max):
    """Return (Ls_bottom, Ls_top) from parabola coefficients [a, b, c]."""
    a, b, c = coeffs
    u_bottom  = c
    dudz_bottom = b
    u_top     = np.polyval(coeffs, z_max)
    dudz_top  = 2 * a * z_max + b
    Ls_bottom = u_bottom  / dudz_bottom if abs(dudz_bottom) > 1e-12 else float('inf')
    Ls_top    = u_top     / abs(dudz_top) if abs(dudz_top)  > 1e-12 else float('inf')
    return Ls_bottom, Ls_top


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Time-averaged DPD velocity profile")
    parser.add_argument("--vtk_dir",   default="vtk_output",
                        help="Directory with *Particles*.vtu files")
    parser.add_argument("--pattern",   default="*Particles*.vtu")
    parser.add_argument("--vel_array", default=None,
                        help="Exact point-data array name for velocity "
                             "(auto-detected if omitted)")
    parser.add_argument("--t_start",   type=float, default=5.0)
    parser.add_argument("--t_end",     type=float, default=1e9)
    parser.add_argument("--n_bins",    type=int,   default=40)
    parser.add_argument("--z_min",     type=float, default=0.0)
    parser.add_argument("--z_max",     type=float, default=0.5)
    parser.add_argument("--out",       default="velocity_profile.png")
    args = parser.parse_args()

    if not USE_PYVISTA:
        raise ImportError("Install pyvista:  pip install pyvista")

    print(f"Reading VTK files from '{args.vtk_dir}' "
          f"with t in [{args.t_start}, {args.t_end}] ...")

    frames = read_all_vtu(args.vtk_dir, args.pattern,
                          args.t_start, args.t_end, args.vel_array)

    if not frames:
        raise RuntimeError(
            f"No frames found in t=[{args.t_start}, {args.t_end}].\n"
            "Lower --t_start or check TIME FieldData in VTU files."
        )

    print(f"\nAveraging {len(frames)} frames  "
          f"t=[{frames[0][0]:.3f}, {frames[-1][0]:.3f}]")

    z_c, mean_ux, counts = compute_averaged_profile(
        frames, n_bins=args.n_bins, z_min=args.z_min, z_max=args.z_max)

    fitted_ux, coeffs = fit_parabola(z_c, mean_ux)
    print(f"\nParabola fit:  u_x(z) = {coeffs[0]:.4f}*z^2 "
          f"+ {coeffs[1]:.4f}*z + {coeffs[2]:.4f}")

    Ls_bottom, Ls_top = compute_slip_lengths(coeffs, args.z_max)
    print(f"\nSlip length  bottom wall (z=0):             Ls = {Ls_bottom:.5f}")
    print(f"Slip length  top wall    (z={args.z_max}):  Ls = {Ls_top:.5f}")
    print(f"Target: Ls < 0.01  (< 0.5 x particle radius)")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax.plot(mean_ux,  z_c, "o-", color="steelblue", label="Binned mean $u_x$")
    ax.plot(fitted_ux, z_c, "--", color="tomato",    label="Parabola fit")
    ax.set_xlabel("$u_x$ (sim units)")
    ax.set_ylabel("$z$ (gap direction)")
    ax.set_title(f"Time-averaged velocity profile\n"
                 f"(t\u2208[{frames[0][0]:.1f},{frames[-1][0]:.1f}], "
                 f"{len(frames)} frames, {args.n_bins} bins)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Slip length annotation box
    slip_text = (
        f"Slip lengths (from fit):\n"
        f"  bottom (z=0):      $L_s$ = {Ls_bottom:+.4f}\n"
        f"  top    (z={args.z_max}):  $L_s$ = {Ls_top:+.4f}\n"
        f"  target: $|L_s|$ < 0.01"
    )
    ax.text(
        0.97, 0.05, slip_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                  edgecolor="gray", alpha=0.85),
    )

    ax2.bar(z_c, counts, width=(args.z_max - args.z_min) / args.n_bins,
            color="steelblue", alpha=0.6)
    ax2.set_xlabel("$z$")
    ax2.set_ylabel("Particle-samples per bin")
    ax2.set_title("Sampling statistics")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved: '{args.out}'")
    plt.show()


if __name__ == "__main__":
    main()
