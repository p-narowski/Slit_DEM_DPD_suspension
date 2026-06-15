"""
velocity_profile.py
====================
Reads Kratos DEM particle VTK output files (Slit_Particles_*.vtu),
bins particle u_x by z-coordinate, time-averages over a chosen window,
fits a parabola and plots the result.

Usage:
    python velocity_profile.py                         # defaults below
    python velocity_profile.py --vtk_dir ./vtk_output --t_start 15.0 --n_bins 40
"""

import os
import glob
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt

# ── VTK / XML parser ──────────────────────────────────────────────────────────
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
    """Extract simulation time from the VTU file FieldData or filename."""
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
        for fd in root.iter("FieldData"):
            for da in fd:
                if da.get("Name") == "TIME":
                    return float(da.text.strip())
    except Exception:
        pass
    # fallback: extract index from filename and approximate
    m = re.search(r"_(\d+)\.vtu$", filepath)
    return int(m.group(1)) if m else None


def parse_vtu_manual(filepath):
    """
    Minimal pure-Python VTU reader (ASCII / inline base64 not supported here).
    Falls back to pyvista if available.
    """
    raise NotImplementedError("Use pyvista or vtk library.")


def read_vtu_pyvista(filepath):
    mesh = pv.read(filepath)
    pts = np.array(mesh.points)          # (N, 3) — x, y, z
    vel = np.array(mesh["VELOCITY"])     # (N, 3)
    return pts, vel


def read_all_vtu(vtk_dir, pattern="*Particles*.vtu", t_start=0.0, t_end=np.inf):
    """
    Returns list of (sim_time, z_coords, ux_values) tuples
    for files whose simulation time falls in [t_start, t_end].
    """
    files = sorted(glob.glob(os.path.join(vtk_dir, pattern)))
    if not files:
        raise FileNotFoundError(
            f"No VTU files matching '{pattern}' found in '{vtk_dir}'.\n"
            f"Check --vtk_dir path and pattern."
        )

    frames = []
    for fpath in files:
        t = get_sim_time_from_vtu(fpath)
        if t is None:
            continue
        if not (t_start <= t <= t_end):
            continue
        pts, vel = read_vtu_pyvista(fpath)
        z = pts[:, 2]       # gap direction
        ux = vel[:, 0]      # flow direction
        frames.append((t, z, ux))
        print(f"  loaded t={t:.4f}  n_particles={len(z)}")

    return frames


# ─────────────────────────────────────────────────────────────────────────────
def compute_averaged_profile(frames, n_bins=40, z_min=0.0, z_max=0.5):
    """
    Bins all (z, ux) data from every frame, returns bin centres and mean ux.
    """
    edges = np.linspace(z_min, z_max, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    sum_ux = np.zeros(n_bins)
    count  = np.zeros(n_bins, dtype=int)

    for _, z, ux in frames:
        idx = np.searchsorted(edges[1:], z)   # bin index for each particle
        idx = np.clip(idx, 0, n_bins - 1)
        np.add.at(sum_ux, idx, ux)
        np.add.at(count,  idx, 1)

    mask = count > 0
    mean_ux = np.where(mask, sum_ux / np.maximum(count, 1), np.nan)
    return centres, mean_ux, count


def fit_parabola(z, ux):
    """Fit u = a*z^2 + b*z + c and return fitted values + coeffs."""
    mask = ~np.isnan(ux)
    coeffs = np.polyfit(z[mask], ux[mask], 2)
    fitted = np.polyval(coeffs, z)
    return fitted, coeffs


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Time-averaged DPD velocity profile")
    parser.add_argument("--vtk_dir",  default="vtk_output",
                        help="Directory containing *Particles*.vtu files")
    parser.add_argument("--pattern",  default="*Particles*.vtu")
    parser.add_argument("--t_start",  type=float, default=15.0,
                        help="Start of averaging window (sim time)")
    parser.add_argument("--t_end",    type=float, default=1e9,
                        help="End of averaging window (sim time, default=all)")
    parser.add_argument("--n_bins",   type=int,   default=40,
                        help="Number of z-bins")
    parser.add_argument("--z_min",    type=float, default=0.0)
    parser.add_argument("--z_max",    type=float, default=0.5,
                        help="Slit height (BoundingBoxMaxZ in ProjectParametersDEM.json)")
    parser.add_argument("--out",      default="velocity_profile.png")
    args = parser.parse_args()

    if not USE_PYVISTA:
        raise ImportError(
            "pyvista is required.  Install with:  pip install pyvista"
        )

    print(f"\nReading VTK files from '{args.vtk_dir}' "
          f"with t in [{args.t_start}, {args.t_end}] ...")
    frames = read_all_vtu(args.vtk_dir, args.pattern, args.t_start, args.t_end)
    if not frames:
        raise RuntimeError(
            f"No frames found in the time window [{args.t_start}, {args.t_end}].\n"
            f"Lower --t_start or check that the VTU files contain TIME FieldData."
        )

    print(f"\nAveraging {len(frames)} frames over "
          f"t = [{frames[0][0]:.3f}, {frames[-1][0]:.3f}]")

    z_centres, mean_ux, counts = compute_averaged_profile(
        frames, n_bins=args.n_bins, z_min=args.z_min, z_max=args.z_max
    )

    fitted_ux, coeffs = fit_parabola(z_centres, mean_ux)
    print(f"\nParabola fit:  u_x(z) = {coeffs[0]:.4f}*z² + {coeffs[1]:.4f}*z + {coeffs[2]:.4f}")

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(mean_ux, z_centres, "o-", color="steelblue", label="Binned mean $u_x$")
    ax.plot(fitted_ux, z_centres, "--", color="tomato",  label="Parabola fit")
    ax.set_xlabel("$u_x$ (sim units)")
    ax.set_ylabel("$z$ (gap direction)")
    ax.set_title(f"Time-averaged velocity profile\n"
                 f"(t ∈ [{frames[0][0]:.1f}, {frames[-1][0]:.1f}], "
                 f"{len(frames)} frames, {args.n_bins} bins)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.bar(z_centres, counts, width=(args.z_max - args.z_min) / args.n_bins,
            color="steelblue", alpha=0.6)
    ax2.set_xlabel("$z$")
    ax2.set_ylabel("Particle-samples per bin")
    ax2.set_title("Sampling statistics")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"\nSaved plot to '{args.out}'")
    plt.show()


if __name__ == "__main__":
    main()
