#!/usr/bin/env python3
"""
triple_particles.py
-------------------
Reads SlitDEM.mdpa and produces SlitDEM_x3.mdpa with 3x as many DPD particles
(1500 instead of 500).  Wall geometry is preserved unchanged.

Usage:
    python triple_particles.py [input.mdpa] [output.mdpa]
Defaults:
    input  = SlitDEM.mdpa
    output = SlitDEM_x3.mdpa
"""

import re
import sys
import random
import math

# ── CLI ──────────────────────────────────────────────────────────────────────
INPUT  = sys.argv[1] if len(sys.argv) > 1 else "SlitDEM.mdpa"
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else "SlitDEM_x3.mdpa"

# ── Parameters (auto-detected from original file where possible) ─────────────
N_NEW       = 1500          # target particle count (3 × 500)
RADIUS      = 0.02          # particle radius (same as original)
MIN_OVERLAP = 0.8           # minimum centre-to-centre distance as fraction of 2R
MAX_TRIES   = 100_000       # placement attempts before giving up

# Slit domain boundaries for particle centres (walls sit at z=0 and z=0.5)
# Keep particles away from walls by at least one radius
Lx, Ly     = 1.0, 1.0
Z_LO, Z_HI = RADIUS + 0.001, 0.5 - RADIUS - 0.001   # inside the gap

MIN_DIST = MIN_OVERLAP * 2 * RADIUS   # hard-core exclusion distance

# ── Read original file ───────────────────────────────────────────────────────
with open(INPUT, "r") as fh:
    text = fh.read()

# ── Extract wall nodes (ID >= 1000 in the original) ─────────────────────────
nodes_block = re.search(r"Begin Nodes(.*?)End Nodes", text, re.S).group(1)
wall_lines = []
for line in nodes_block.splitlines():
    s = line.strip()
    if not s or s.startswith("//") or s.startswith("#"):
        continue
    parts = s.split()
    try:
        nid = int(parts[0])
    except (ValueError, IndexError):
        continue
    if nid >= 501:          # keep only wall nodes
        wall_lines.append(f"    {nid}  {parts[1]}  {parts[2]}  {parts[3]}")

# ── Extract wall elements (everything outside DPDParticle block) ──────────────
# We will reconstruct just the wall SubModelParts and FEM boundary conditions.
# Grab everything after "End Elements" that belongs to walls.

# Detect original wall node IDs so we can rebuild SubModelPart headers correctly
wall_node_ids = []
for line in wall_lines:
    wall_node_ids.append(int(line.strip().split()[0]))

# Grab RigidFace elements
rigid_block_match = re.search(
    r"(Begin Elements RigidFace3D3N.*?End Elements)", text, re.S
)
rigid_elements = rigid_block_match.group(1) if rigid_block_match else ""

# Grab RADIUS nodaldata for walls (IDs >= 501)
wall_radius_block_match = re.search(
    r"Begin NodalData RADIUS.*?End NodalData", text, re.S
)
wall_radius_lines = []
if wall_radius_block_match:
    for line in wall_radius_block_match.group(0).splitlines():
        s = line.strip()
        parts = s.split()
        try:
            nid = int(parts[0])
            if nid >= 501:
                wall_radius_lines.append(f"    {nid}   {parts[1]}  {parts[2]}")
        except (ValueError, IndexError):
            pass

# Grab the RigidFacePart SubModelPart
rigid_submodel_match = re.search(
    r"(Begin SubModelPart DEM-FEM-Wall_SlitWalls.*?End SubModelPart)", text, re.S
)
rigid_submodel = rigid_submodel_match.group(1) if rigid_submodel_match else ""

# ── Generate 1500 particle positions without hard-core overlap ───────────────
random.seed(42)   # reproducible

positions = []
placed = 0
tries  = 0

while placed < N_NEW and tries < MAX_TRIES:
    tries += 1
    x = random.uniform(RADIUS, Lx - RADIUS)
    y = random.uniform(RADIUS, Ly - RADIUS)
    z = random.uniform(Z_LO, Z_HI)

    ok = True
    for px, py, pz in positions:
        dist = math.sqrt((x-px)**2 + (y-py)**2 + (z-pz)**2)
        if dist < MIN_DIST:
            ok = False
            break
    if ok:
        positions.append((x, y, z))
        placed += 1

if placed < N_NEW:
    print(f"WARNING: only placed {placed}/{N_NEW} particles after {MAX_TRIES} tries.")
    print("  The box may be too dense for hard-core placement with these radii.")
    print("  Consider reducing MIN_OVERLAP (currently {MIN_OVERLAP}) or particle radius.")
    N_NEW = placed

print(f"Placed {placed} particles.")

# ── Build the new .mdpa content ───────────────────────────────────────────────

# 1. Nodes block
node_lines = ["Begin Nodes"]
for i, (x, y, z) in enumerate(positions, start=1):
    node_lines.append(f"    {i:5d}  {x:.6f}  {y:.6f}  {z:.6f}")
node_lines.extend(wall_lines)
node_lines.append("End Nodes")
nodes_block_new = "\n".join(node_lines)

# 2. Elements block (DPDParticle)
elem_lines = ["Begin Elements DPDParticle // GUI group identifier: ParticlesPart"]
for i in range(1, N_NEW + 1):
    elem_lines.append(f"    {i} 0 {i}")
elem_lines.append("End Elements")
dpd_elements_new = "\n".join(elem_lines)

# 3. RADIUS NodalData
radius_lines = ["Begin NodalData RADIUS // GUI group identifier: ParticlesPart"]
for i in range(1, N_NEW + 1):
    radius_lines.append(f"    {i:5d}   0  {RADIUS}")
radius_lines.extend(wall_radius_lines)
radius_lines.append("End NodalData")
radius_block_new = "\n".join(radius_lines)

# 4. VELOCITY NodalData (zero initial velocity)
vel_lines = ["Begin NodalData VELOCITY // GUI group identifier: ParticlesPart"]
for i in range(1, N_NEW + 1):
    vel_lines.append(f"    {i:5d}   0  0.0  0.0  0.0")
vel_lines.append("End NodalData")
vel_block_new = "\n".join(vel_lines)

# 5. DEMParts SubModelPart
sub_nodes = "\n".join(f"    {i}" for i in range(1, N_NEW + 1))
sub_elems = "\n".join(f"    {i}" for i in range(1, N_NEW + 1))
particles_submodel_new = (
    "Begin SubModelPart DEMParts_ParticlesPart // Group ParticlesPart // Subtree DEMParts\n"
    "  Begin SubModelPartNodes\n"
    f"{sub_nodes}\n"
    "  End SubModelPartNodes\n\n"
    "  Begin SubModelPartElements\n"
    f"{sub_elems}\n"
    "  End SubModelPartElements\n"
    "End SubModelPart"
)

# ── Assemble full file ────────────────────────────────────────────────────────
header = (
    "Begin ModelPartData\n"
    "//  VARIABLE_NAME value\n"
    "End ModelPartData\n\n"
    "Begin Properties 0\n"
    "End Properties\n"
)

parts = [
    header,
    nodes_block_new,
    "",
    dpd_elements_new,
    "",
    rigid_elements,
    "",
    radius_block_new,
    "",
    vel_block_new,
    "",
    rigid_submodel,
    "",
    particles_submodel_new,
    "",
]

with open(OUTPUT, "w") as fh:
    fh.write("\n".join(parts))

print(f"Written: {OUTPUT}")
