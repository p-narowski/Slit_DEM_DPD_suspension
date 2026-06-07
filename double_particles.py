#!/usr/bin/env python3
"""
double_particles.py
Doubles the number of DPDParticle particles in a Kratos .mdpa file.
Usage:
    python double_particles.py SlitDEM.mdpa SlitDEM_x2.mdpa
"""

import sys
import re
import random

JITTER = 0.01  # max displacement per axis for duplicate particle


def find_block(text, begin_keyword):
    """
    Find a block that starts with a line beginning with 'Begin <begin_keyword>'
    (anything may follow on that line) and ends with 'End <begin_keyword>'.
    Returns (start_idx, end_idx, header_line, body, end_tag) or raises ValueError.
    """
    # Build pattern: Begin, whitespace, then the keyword words joined by \s+
    words = begin_keyword.split()
    kw_pat = r'\s+'.join(re.escape(w) for w in words)
    pat = re.compile(
        r'([ \t]*Begin[ \t]+' + kw_pat + r'[^\n]*\n)'  # header
        r'(.*?)'                                          # body
        r'([ \t]*End[ \t]+' + kw_pat + r'[^\n]*)',       # footer
        re.DOTALL
    )
    m = pat.search(text)
    if m is None:
        raise ValueError(f"Block 'Begin {begin_keyword}' not found in file.")
    return m.start(), m.end(), m.group(1), m.group(2), m.group(3)


def double_particles(src, dst):
    random.seed(42)
    with open(src) as f:
        text = f.read()

    # ── 1. Nodes ─────────────────────────────────────────────────────────────
    ns, ne, nh, node_body, nf = find_block(text, 'Nodes')
    node_re = re.compile(r'^\s*(\d+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)', re.M)
    orig_nodes = node_re.findall(node_body)
    n = len(orig_nodes)
    print(f"Nodes found: {n}")

    new_node_lines = []
    for nid, x, y, z in orig_nodes:
        nid = int(nid)
        nx = (float(x) + random.uniform(-JITTER, JITTER)) % 1.0
        ny = (float(y) + random.uniform(-JITTER, JITTER)) % 1.0
        nz = float(z) + random.uniform(-JITTER, JITTER)
        nz = max(0.001, min(0.499, nz))
        new_node_lines.append(f"  {nid + n}  {nx:.6f}  {ny:.6f}  {nz:.6f}")

    new_nodes_block = nh + node_body + '\n'.join(new_node_lines) + '\n' + nf

    # ── 2. Elements DPDParticle ───────────────────────────────────────────────
    es, ee, eh, elem_body, ef = find_block(text, 'Elements DPDParticle')
    elem_re = re.compile(r'^\s*(\d+)\s+(\d+)\s+(\d+)', re.M)
    orig_elems = elem_re.findall(elem_body)
    ne_count = len(orig_elems)
    print(f"Elements found: {ne_count}")

    new_elem_lines = []
    for eid, pid, node in orig_elems:
        new_elem_lines.append(f"  {int(eid) + ne_count}  {pid}  {int(node) + n}")

    new_elems_block = eh + elem_body + '\n'.join(new_elem_lines) + '\n' + ef

    # ── 3. NodalData RADIUS ───────────────────────────────────────────────────
    rs, re_, rh, rad_body, rf = find_block(text, 'NodalData RADIUS')
    rad_re = re.compile(r'^\s*(\d+)\s+(\d+)\s+([\d.eE+\-]+)', re.M)
    orig_rads = rad_re.findall(rad_body)
    new_rad_lines = []
    for nid, step, val in orig_rads:
        new_rad_lines.append(f"  {int(nid) + n}  {step}  {val}")
    new_rad_block = rh + rad_body + '\n'.join(new_rad_lines) + '\n' + rf

    # ── 4. SubModelPartNodes (append new node IDs) ────────────────────────────
    def extend_id_block(txt, block_keyword):
        try:
            bs, be, bh, body, bf = find_block(txt, block_keyword)
        except ValueError:
            return txt
        existing = list(map(int, re.findall(r'\d+', body)))
        if not existing:
            return txt
        max_id = max(existing)
        new_ids = [str(i) for i in range(max_id + 1, max_id + 1 + len(existing))]
        new_body = body.rstrip('\n') + '\n' + '\n'.join(new_ids) + '\n'
        return txt[:bs] + bh + new_body + bf + txt[be:]

    # ── 5. Apply all replacements in reverse offset order ────────────────────
    replacements = sorted([
        (ns, ne, new_nodes_block),
        (es, ee, new_elems_block),
        (rs, re_, new_rad_block),
    ], key=lambda t: t[0], reverse=True)

    out = text
    for start, end, rep in replacements:
        out = out[:start] + rep + out[end:]

    # Extend SubModelPart node and element lists
    for kw in ('SubModelPartNodes', 'SubModelPartElements'):
        out = extend_id_block(out, kw)

    with open(dst, 'w') as f:
        f.write(out)

    print(f"Done -> '{dst}'")
    print(f"  Nodes:    {n} -> {n * 2}")
    print(f"  Elements: {ne_count} -> {ne_count * 2}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python double_particles.py <input.mdpa> <output.mdpa>")
        sys.exit(1)
    double_particles(sys.argv[1], sys.argv[2])
