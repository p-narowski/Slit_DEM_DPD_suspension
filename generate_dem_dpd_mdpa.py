#!/usr/bin/env python3
"""Generate a mixed DEM/DPD MDPA with one central DEM particle."""

from pathlib import Path
import argparse
import numpy as np


# Geometry
BOX_MIN = np.array([0.0, 0.0, 0.0])
BOX_MAX = np.array([2.0, 2.0, 2.0])

DEM_CENTER = 0.5 * (BOX_MIN + BOX_MAX)
DEM_RADIUS = 0.4
DPD_RADIUS = 0.02
N_DPD = 80_000

# Element names and material-property IDs.
DEM_ELEMENT_TYPE = "SphericParticle3D"
DPD_ELEMENT_TYPE = "DPDParticle"

DEM_PROPERTY_ID = 1
DPD_PROPERTY_ID = 0

# True: DPD spheres do not overlap the DEM sphere.
# False: only DPD centers are excluded from the DEM interior.
EXCLUDE_FULL_DPD_SPHERE = True

# Keep entire DPD spheres inside the box.
WALL_CLEARANCE = DPD_RADIUS

# Small clearance to avoid roundoff placing a node on the exclusion surface.
GEOMETRIC_TOLERANCE = 1.0e-10


def generate_dpd_positions(seed):
    """Uniform random sampling outside the central exclusion sphere."""
    rng = np.random.default_rng(seed)

    lower = BOX_MIN + WALL_CLEARANCE
    upper = BOX_MAX - WALL_CLEARANCE

    if np.any(upper <= lower):
        raise ValueError("Wall clearance leaves no available box volume.")

    exclusion_radius = DEM_RADIUS
    if EXCLUDE_FULL_DPD_SPHERE:
        exclusion_radius += DPD_RADIUS
    exclusion_radius += GEOMETRIC_TOLERANCE

    positions = np.empty((N_DPD, 3), dtype=float)
    filled = 0

    while filled < N_DPD:
        remaining = N_DPD - filled
        batch_size = max(4096, int(1.2 * remaining))

        candidates = rng.uniform(lower, upper, size=(batch_size, 3))
        offsets = candidates - DEM_CENTER
        distance_squared = np.einsum("ij,ij->i", offsets, offsets)

        accepted = candidates[
            distance_squared >= exclusion_radius * exclusion_radius
        ]

        take = min(len(accepted), remaining)
        positions[filled:filled + take] = accepted[:take]
        filled += take

    validate_positions(positions, exclusion_radius, lower, upper)
    return positions


def validate_positions(positions, exclusion_radius, lower, upper):
    if positions.shape != (N_DPD, 3):
        raise RuntimeError("Incorrect DPD particle count.")

    if not np.all(np.isfinite(positions)):
        raise RuntimeError("Non-finite particle coordinates.")

    if np.any(positions < lower) or np.any(positions > upper):
        raise RuntimeError("DPD particle outside permitted box region.")

    distances = np.linalg.norm(positions - DEM_CENTER, axis=1)
    if np.any(distances < exclusion_radius):
        raise RuntimeError("DPD particle inside the exclusion sphere.")


def write_submodelpart(stream, name, ids):
    stream.write(f"Begin SubModelPart {name}\n")

    stream.write("  Begin SubModelPartNodes\n")
    for particle_id in ids:
        stream.write(f"    {particle_id}\n")
    stream.write("  End SubModelPartNodes\n\n")

    stream.write("  Begin SubModelPartElements\n")
    for particle_id in ids:
        stream.write(f"    {particle_id}\n")
    stream.write("  End SubModelPartElements\n")

    stream.write("End SubModelPart\n\n")


def write_mdpa(output_path, positions):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Node IDs and element IDs are identical:
    # DEM: 1
    # DPD: 2 ... 80001
    dpd_ids = range(2, N_DPD + 2)

    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("Begin ModelPartData\n")
        stream.write("End ModelPartData\n\n")

        for property_id in sorted({DEM_PROPERTY_ID, DPD_PROPERTY_ID}):
            stream.write(f"Begin Properties {property_id}\n")
            stream.write("End Properties\n\n")

        stream.write("Begin Nodes\n")
        x, y, z = DEM_CENTER
        stream.write(f"1 {x:.16e} {y:.16e} {z:.16e}\n")

        for node_id, (x, y, z) in zip(dpd_ids, positions):
            stream.write(f"{node_id} {x:.16e} {y:.16e} {z:.16e}\n")

        stream.write("End Nodes\n\n")

        stream.write(f"Begin Elements {DEM_ELEMENT_TYPE}\n")
        stream.write(f"1 {DEM_PROPERTY_ID} 1\n")
        stream.write("End Elements\n\n")

        stream.write(f"Begin Elements {DPD_ELEMENT_TYPE}\n")
        for particle_id in dpd_ids:
            stream.write(
                f"{particle_id} {DPD_PROPERTY_ID} {particle_id}\n"
            )
        stream.write("End Elements\n\n")

        stream.write("Begin NodalData RADIUS\n")
        stream.write(f"1 0 {DEM_RADIUS:.16e}\n")
        for node_id in dpd_ids:
            stream.write(f"{node_id} 0 {DPD_RADIUS:.16e}\n")
        stream.write("End NodalData\n\n")

        write_submodelpart(
            stream, "DEMPartsParticlesPart", range(1, N_DPD + 2)
        )
        write_submodelpart(stream, "DEMParticles", range(1, 2))
        write_submodelpart(stream, "DPDParticles", dpd_ids)

    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="OneDEM_80000DPDDEM.mdpa",
        help="Output MDPA filename.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible particle positions.",
    )
    args = parser.parse_args()

    positions = generate_dpd_positions(args.seed)
    output_path = write_mdpa(args.output, positions)

    distances = np.linalg.norm(positions - DEM_CENTER, axis=1)
    minimum_distance = distances.min()
    minimum_surface_gap = minimum_distance - DEM_RADIUS - DPD_RADIUS

    print(f"Written: {output_path.resolve()}")
    print(f"DEM particles: 1")
    print(f"DPD particles: {len(positions)}")
    print(f"Total nodes/elements: {len(positions) + 1}")
    print(f"DEM center: {DEM_CENTER.tolist()}")
    print(f"DEM radius: {DEM_RADIUS}")
    print(f"DPD radius: {DPD_RADIUS}")
    print(f"Minimum DEM-to-DPD center distance: {minimum_distance:.12f}")
    print(f"Minimum DEM-to-DPD surface gap: {minimum_surface_gap:.12f}")


if __name__ == "__main__":
    main()