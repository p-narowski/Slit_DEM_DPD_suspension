#!/usr/bin/env python3
"""
double_particles.py
Doubles the number of DPDParticle particles in a Kratos .mdpa file.
Each existing particle is duplicated with a small random jitter so the
new particle doesn't sit exactly on top of its parent.
Usage:
    python double_particles.py SlitDEM.mdpa SlitDEM_x2.mdpa
"""

import sys
import re
import random

JITTER = 0.01   # max displacement in each axis for the duplicate particle


def make_block_pattern(block_name):
    """
    Build a regex pattern for a Kratos mdpa block.
    Words in block_name are separated by \\s+ so spaces don't need escaping.
    Example: 'Elements DPDParticle' -> r'Elements\s+DPDParticle'
    """
    escaped_words = [re.escape(w) for w in block_name.split()]
    name_pattern = r'\s+'.join(escaped_words)
    return re.compile(
        r'(Begin\s+' + name_pattern + r'[^\n]*\n)'
        r'(.*?)'
        r'(End\s+' + name_pattern + r')',
        re.DOTALL
    )


def parse_block(text, block_name):
    """Return (start, end, header_line, body, footer_line) for a named block."""
    pattern = make_block_pattern(block_name)
    m = pattern.search(text)
    if m is None:
        raise ValueError(f"Block '{block_name}' not found.")
    return m.start(), m.end(), m.group(1), m.group(2), m.group(3)


def double_particles(src_path, dst_path):
    random.seed(42)

    with open(src_path, 'r') as f:
        text = f.read()

    # ── 1. Nodes ────────────────────────────────────────────────────────────
    ns, ne, nh, node_body, nf = parse_block(text, 'Nodes')
    node_lines = [l for l in node_body.split('\n') if l.strip()]
    n_orig = len(node_lines)
    print(f"Original nodes: {n_orig}")

    node_re = re.compile(r'^\s*(\d+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)')
    new_node_lines = []
    for line in node_lines:
        m = node_re.match(line)
        if not m:
            continue
        nid = int(m.group(1))
        x = float(m.group(2))
        y = float(m.group(3))
        z = float(m.group(4))
        # apply jitter; clamp z strictly inside the slit (0, 0.5)
        new_x = (x + random.uniform(-JITTER, JITTER)) % 1.0
        new_y = (y + random.uniform(-JITTER, JITTER)) % 1.0
        new_z = z + random.uniform(-JITTER, JITTER)
        new_z = max(0.001, min(0.499, new_z))
        new_id = nid + n_orig
        new_node_lines.append(f"  {new_id:6d}  {new_x:.6f}  {new_y:.6f}  {new_z:.6f}")

    new_node_body = node_body + '\n'.join(new_node_lines) + '\n'
    new_nodes_block = nh + new_node_body + nf

    # ── 2. Elements (DPDParticle) ────────────────────────────────────────────
    es, ee, eh, elem_body, ef = parse_block(text, 'Elements DPDParticle')
    elem_lines = [l for l in elem_body.split('\n') if l.strip()]
    n_elem = len(elem_lines)
    print(f"Original elements: {n_elem}")

    elem_re = re.compile(r'^\s*(\d+)\s+(\d+)\s+(\d+)')
    new_elem_lines = []
    for line in elem_lines:
        m = elem_re.match(line)
        if not m:
            continue
        eid = int(m.group(1))
        pid = int(m.group(2))
        nid = int(m.group(3))
        new_eid = eid + n_elem
        new_nid = nid + n_orig   # points to the duplicated node
        new_elem_lines.append(f"  {new_eid:6d}  {pid}  {new_nid}")

    new_elem_body = elem_body + '\n'.join(new_elem_lines) + '\n'
    new_elems_block = eh + new_elem_body + ef

    # ── 3. NodalData RADIUS ──────────────────────────────────────────────────
    rs, re_, rh, rad_body, rf = parse_block(text, 'NodalData RADIUS')
    rad_lines = [l for l in rad_body.split('\n') if l.strip()]
    rad_re = re.compile(r'^\s*(\d+)\s+(\d+)\s+([\d.eE+\-]+)')
    new_rad_lines = []
    for line in rad_lines:
        m = rad_re.match(line)
        if not m:
            continue
        nid = int(m.group(1))
        step = int(m.group(2))
        val = m.group(3)
        new_rad_lines.append(f"  {nid + n_orig:6d}  {step}  {val}")
    new_rad_body = rad_body + '\n'.join(new_rad_lines) + '\n'
    new_rad_block = rh + new_rad_body + rf

    # ── 4. Optional scalar NodalData blocks (VELOCITY_X/Y/Z, etc.) ──────────
    def double_nodal_scalar(txt, var_name):
        try:
            vs, ve, vh, vbody, vf = parse_block(txt, f'NodalData {var_name}')
        except ValueError:
            return txt  # block absent – skip silently
        vlines = [l for l in vbody.split('\n') if l.strip()]
        vr = re.compile(r'^\s*(\d+)\s+(\d+)\s+([\d.eE+\-]+)')
        new_vlines = []
        for line in vlines:
            mv = vr.match(line)
            if not mv:
                continue
            nid = int(mv.group(1))
            step = mv.group(2)
            val = mv.group(3)
            new_vlines.append(f"  {nid + n_orig:6d}  {step}  {val}")
        new_vbody = vbody + '\n'.join(new_vlines) + '\n'
        new_block = vh + new_vbody + vf
        return txt[:vs] + new_block + txt[ve:]

    # ── 5. Assemble output (replace in reverse order to keep offsets valid) ──
    blocks_to_replace = sorted([
        (ns, ne, new_nodes_block),
        (es, ee, new_elems_block),
        (rs, re_, new_rad_block),
    ], key=lambda t: t[0], reverse=True)

    out = text
    for start, end, replacement in blocks_to_replace:
        out = out[:start] + replacement + out[end:]

    for var in ('VELOCITY_X', 'VELOCITY_Y', 'VELOCITY_Z'):
        out = double_nodal_scalar(out, var)

    with open(dst_path, 'w') as f:
        f.write(out)

    print(f"Done. Written to '{dst_path}'")
    print(f"  Nodes:    {n_orig}  ->  {n_orig * 2}")
    print(f"  Elements: {n_elem}  ->  {n_elem * 2}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python double_particles.py <input.mdpa> <output.mdpa>")
        sys.exit(1)
    double_particles(sys.argv[1], sys.argv[2])
