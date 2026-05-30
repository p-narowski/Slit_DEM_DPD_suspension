import random

# Settings
input_file  = "SlitDEM.mdpa"
output_file = "SlitDEM_jittered.mdpa"
spacing = 0.1
jitter  = 0.025  # 25% of spacing

random.seed(42)  # remove for different jitter each run

with open(input_file, 'r') as f:
    lines = f.readlines()

out_lines = []
in_nodes_block = False

for line in lines:
    stripped = line.strip()

    if stripped == "Begin Nodes":
        in_nodes_block = True
        out_lines.append(line)
        continue

    if stripped == "End Nodes":
        in_nodes_block = False
        out_lines.append(line)
        continue

    if in_nodes_block:
        parts = stripped.split()
        if len(parts) == 4:
            nid = parts[0]
            x = float(parts[1]) + random.uniform(-jitter, jitter)
            y = float(parts[2]) + random.uniform(-jitter, jitter)
            z = float(parts[3]) + random.uniform(-jitter, jitter)
            # clamp to stay inside domain
            x = max(0.001, min(0.999, x))
            y = max(0.001, min(0.999, y))
            z = max(0.001, min(0.499, z))
            out_lines.append(f"  {nid:>6}  {x:.6f}  {y:.6f}  {z:.6f}\n")
            continue

    out_lines.append(line)

with open(output_file, 'w') as f:
    f.writelines(out_lines)

print(f"Done — written to {output_file}")