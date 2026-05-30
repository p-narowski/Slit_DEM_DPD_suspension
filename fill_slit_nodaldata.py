#!/usr/bin/env python3
import re
from pathlib import Path

# Input template (your original Slit.mdpa with comments / wall geometry)
TEMPLATE = Path("Slit.mdpa")
OUTFILE  = Path("Slit_full.mdpa")

text = TEMPLATE.read_text()

# ----------------------------------------------------------------------
# Parameters for the slit system
# ----------------------------------------------------------------------
Lx, Ly, H = 1.0, 1.0, 0.5
nx, ny, nz = 10, 10, 5
dx, dy, dz = Lx/nx, Ly/ny, H/nz
n_particles = nx * ny * nz  # should be 500

radius_value = 0.02  # adjust if needed

# ----------------------------------------------------------------------
# 1. Rebuild Nodes block
# ----------------------------------------------------------------------
nodes_head = """Begin Nodes
  # Particles: IDs 1..{n}
  # Regular grid: xi = (i+0.5)*dx, yi = (j+0.5)*dy, zi = (k+0.5)*dz
  # with dx = Lx/{nx}, dy = Ly/{ny}, dz = H/{nz}
""".format(n=n_particles, nx=nx, ny=ny, nz=nz)

node_lines = [nodes_head]

node_id = 1
for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            x = (i + 0.5) * dx
            y = (j + 0.5) * dy
            z = (k + 0.5) * dz
            node_lines.append(f"  {node_id:4d} {x:.6f} {y:.6f} {z:.6f}")
            node_id += 1

# Append the wall nodes from your template (1001–1104)
# We extract them from the original Nodes block to avoid hardcoding again.
nodes_block_match = re.search(r"Begin Nodes(.*?End Nodes)", text, re.S)
if not nodes_block_match:
    raise RuntimeError("Could not find 'Begin Nodes' block in template")

orig_nodes_block = nodes_block_match.group(1)
wall_lines = []
for line in orig_nodes_block.splitlines():
    line_stripped = line.strip()
    if not line_stripped or line_stripped.startswith("#"):
        continue
    parts = line_stripped.split()
    try:
        nid = int(parts[0])
    except (ValueError, IndexError):
        continue
    if nid >= 1000:
        wall_lines.append(line)

node_lines.extend(wall_lines)
node_lines.append("End Nodes")

nodes_block_new = "\n".join(node_lines)

# Replace entire Nodes block
text = re.sub(r"Begin Nodes.*?End Nodes", nodes_block_new, text, flags=re.S)

# ----------------------------------------------------------------------
# 2. Rebuild Elements block for particles 1..500
# ----------------------------------------------------------------------
elem_lines = [
    "Begin Elements SphericParticle3D",
    "  # Automatically generated: 500 particles, each one-node element",
    "  # Element format:  id  properties_id  node_id",
]
for eid in range(1, n_particles + 1):
    elem_lines.append(f"  {eid:4d}   1  {eid}")
elem_lines.append("End Elements")

elements_block_new = "\n".join(elem_lines)
text = re.sub(r"Begin Elements SphericParticle3D.*?End Elements",
              elements_block_new, text, flags=re.S)

# ----------------------------------------------------------------------
# 3. Rebuild RADIUS and VELOCITY NodalData blocks for particles 1..500
# ----------------------------------------------------------------------
radius_lines = [
    "Begin NodalData RADIUS",
    "  # Automatically generated for particle nodes 1..%d" % n_particles,
]
for nid in range(1, n_particles + 1):
    radius_lines.append(f"  {nid:4d}   0  {radius_value:.6g}")
radius_lines.append("End NodalData")
radius_block_new = "\n".join(radius_lines)

vel_lines = [
    "Begin NodalData VELOCITY",
    "  # Automatically generated: all initial velocities zero",
]
for nid in range(1, n_particles + 1):
    vel_lines.append(f"  {nid:4d}   0  0.0  0.0  0.0")
vel_lines.append("End NodalData")
vel_block_new = "\n".join(vel_lines)

def replace_block(text, block_name, new_block):
    pattern = rf"Begin NodalData {block_name}.*?End NodalData"
    if re.search(pattern, text, re.S):
        return re.sub(pattern, new_block, text, flags=re.S)
    else:
        return text.rstrip() + "\n\n" + new_block + "\n"

text = replace_block(text, "RADIUS",   radius_block_new)
text = replace_block(text, "VELOCITY", vel_block_new)

# ----------------------------------------------------------------------
# 4. Optionally update ParticlesPart SubModelPart to include 1..500
# ----------------------------------------------------------------------
text = re.sub(r"Begin SubModelPart ParticlesPart.*?End SubModelPart",
              "Begin SubModelPart ParticlesPart\n"
              "  Begin SubModelPartNodes\n" +
              "".join(f"    {i}\n" for i in range(1, n_particles+1)) +
              "  End SubModelPartNodes\n\n"
              "  Begin SubModelPartElements\n" +
              "".join(f"    {i}\n" for i in range(1, n_particles+1)) +
              "  End SubModelPartElements\n"
              "End SubModelPart",
              text, flags=re.S)

OUTFILE.write_text(text)
print(f"Wrote fully generated slit to {OUTFILE}")