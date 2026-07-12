"""
preprocess_mdpa.py

Rewrites SlitDEM.mdpa so that particles inside a chosen spatial region
are declared as SuspendedParticle elements instead of DPDParticle,
adds a new SubModelPart (DEMParts_SuspendedPart) listing them, and
ensures a "Begin Properties 3 / End Properties" block exists (required
by ModelPartIO regardless of MaterialsDEM.json assignment).

This must run BEFORE MainKratos.py.

Usage:
    python preprocess_mdpa.py
"""

import shutil

INPUT_FILE = "SlitDEM.mdpa"
OUTPUT_FILE = "SlitDEM.mdpa"
BACKUP_FILE = "SlitDEM_original_backup.mdpa"

SUSPENDED_REGION = {
    "x_min": 0.40, "x_max": 0.60,
    "y_min": 0.40, "y_max": 0.60,
    "z_min": 0.00, "z_max": 0.50,
}
SUSPENDED_RADIUS = 0.06
NEW_PROP_ID = 3


def parse_nodes(lines):
    nodes = {}
    in_nodes = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Begin Nodes"):
            in_nodes = True
            continue
        if stripped.startswith("End Nodes"):
            in_nodes = False
            continue
        if in_nodes and stripped:
            parts = stripped.split()
            node_id = int(parts[0])
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            nodes[node_id] = (x, y, z)
    return nodes


def parse_element_block(lines, block_name):
    start_idx = None
    end_idx = None
    element_lines = []
    in_block = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(f"Begin Elements {block_name}"):
            in_block = True
            start_idx = i
            continue
        if in_block and stripped.startswith("End Elements"):
            end_idx = i
            break
        if in_block and stripped:
            parts = stripped.split()
            elem_id = int(parts[0])
            prop_id = int(parts[1])
            node_id = int(parts[2])
            element_lines.append((line, elem_id, prop_id, node_id))
    return start_idx, end_idx, element_lines


def existing_properties_ids(lines):
    ids = set()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Begin Properties"):
            parts = stripped.split()
            if len(parts) >= 3:
                ids.add(int(parts[2]))
    return ids


def insert_properties_block(lines, new_id):
    """Insert 'Begin Properties N / End Properties' right after the last existing Properties block."""
    last_end_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("End Properties"):
            last_end_idx = i
    if last_end_idx is None:
        insertion = [f"Begin Properties {new_id}\nEnd Properties\n\n"]
        return insertion + lines
    insertion = [f"Begin Properties {new_id}\nEnd Properties\n\n"]
    return lines[:last_end_idx + 1] + ["\n"] + insertion + lines[last_end_idx + 1:]


def main():
    shutil.copyfile(INPUT_FILE, BACKUP_FILE)
    print(f"[INFO] Backup written to {BACKUP_FILE}")

    with open(INPUT_FILE, "r") as f:
        lines = f.readlines()

    existing_props = existing_properties_ids(lines)
    print(f"[INFO] Existing Properties IDs found in mdpa: {sorted(existing_props)}")

    if NEW_PROP_ID not in existing_props:
        lines = insert_properties_block(lines, NEW_PROP_ID)
        print(f"[INFO] Inserted 'Begin Properties {NEW_PROP_ID} / End Properties' block")
    else:
        print(f"[INFO] Properties {NEW_PROP_ID} block already present, skipping insertion")

    nodes = parse_nodes(lines)
    print(f"[INFO] Parsed {len(nodes)} nodes")

    start_idx, end_idx, dpd_elements = parse_element_block(lines, "DPDParticle")
    if start_idx is None:
        raise RuntimeError("Could not find 'Begin Elements DPDParticle' block")
    print(f"[INFO] Found {len(dpd_elements)} DPDParticle elements")

    keep_elements = []
    suspended_elements = []
    suspended_node_ids = []

    for raw_line, elem_id, prop_id, node_id in dpd_elements:
        x, y, z = nodes[node_id]
        inside = (
            SUSPENDED_REGION["x_min"] <= x <= SUSPENDED_REGION["x_max"] and
            SUSPENDED_REGION["y_min"] <= y <= SUSPENDED_REGION["y_max"] and
            SUSPENDED_REGION["z_min"] <= z <= SUSPENDED_REGION["z_max"]
        )
        if inside:
            suspended_elements.append((elem_id, prop_id, node_id))
            suspended_node_ids.append(node_id)
        else:
            keep_elements.append(raw_line)

    print(f"[INFO] Selected {len(suspended_elements)} particles as SuspendedParticle")

    if not suspended_elements:
        print("[WARNING] No particles matched the region. Nothing to convert. Exiting.")
        return

    suspended_block_lines = ["Begin Elements SuspendedParticle\n"]
    for elem_id, _old_prop_id, node_id in suspended_elements:
        suspended_block_lines.append(f"  {elem_id}  {NEW_PROP_ID}  {node_id}\n")
    suspended_block_lines.append("End Elements\n")

    new_dpd_block_lines = ["Begin Elements DPDParticle\n"] + keep_elements + ["End Elements\n"]

    new_lines = (
        lines[:start_idx]
        + new_dpd_block_lines
        + ["\n"]
        + suspended_block_lines
        + lines[end_idx + 1:]
    )

    submodelpart_lines = [
        "\nBegin SubModelPart DEMParts_SuspendedPart\n",
        "    Begin SubModelPartNodes\n",
    ]
    for node_id in suspended_node_ids:
        submodelpart_lines.append(f"        {node_id}\n")
    submodelpart_lines.append("    End SubModelPartNodes\n")
    submodelpart_lines.append("    Begin SubModelPartElements\n")
    for elem_id, _old_prop_id, _node_id in suspended_elements:
        submodelpart_lines.append(f"        {elem_id}\n")
    submodelpart_lines.append("    End SubModelPartElements\n")
    submodelpart_lines.append("End SubModelPart\n")

    new_lines += submodelpart_lines

    radius_start, radius_end = None, None
    for i, line in enumerate(new_lines):
        stripped = line.strip()
        if stripped.startswith("Begin NodalData RADIUS"):
            radius_start = i
        elif radius_start is not None and stripped.startswith("End NodalData"):
            radius_end = i
            break

    if radius_start is not None:
        existing_ids = set()
        updated_block = [new_lines[radius_start]]
        for line in new_lines[radius_start + 1:radius_end]:
            parts = line.strip().split()
            if not parts:
                continue
            node_id = int(parts[0])
            if node_id in suspended_node_ids:
                updated_block.append(f"  {node_id}  0  {SUSPENDED_RADIUS}\n")
                existing_ids.add(node_id)
            else:
                updated_block.append(line)
        for node_id in suspended_node_ids:
            if node_id not in existing_ids:
                updated_block.append(f"  {node_id}  0  {SUSPENDED_RADIUS}\n")
        updated_block.append(new_lines[radius_end])
        new_lines = new_lines[:radius_start] + updated_block + new_lines[radius_end + 1:]
    else:
        radius_block = ["\nBegin NodalData RADIUS\n"]
        for node_id in suspended_node_ids:
            radius_block.append(f"  {node_id}  0  {SUSPENDED_RADIUS}\n")
        radius_block.append("End NodalData\n")
        new_lines += radius_block
        print("[INFO] No existing 'Begin NodalData RADIUS' block found; appended a new one.")
        print("[WARNING] Verify default RADIUS is still assigned to non-suspended particles elsewhere.")

    with open(OUTPUT_FILE, "w") as f:
        f.writelines(new_lines)

    print(f"[INFO] Wrote updated mdpa to {OUTPUT_FILE}")
    print(f"[INFO] {len(suspended_elements)} particles converted to SuspendedParticle (Properties {NEW_PROP_ID})")


if __name__ == "__main__":
    main()