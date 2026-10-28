#!/usr/bin/env python3
"""Redistribute a particle-only MDPA with an exact particle count."""
import argparse
import random
import re
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', nargs='?', default='SlitDEM.mdpa')
    p.add_argument('output', nargs='?', default='SlitDEM_72000.mdpa')
    p.add_argument('--count', type=int, default=72000)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if a.count <= 0:
        raise ValueError('Count must be positive.')
    if Path(a.input).resolve() == Path(a.output).resolve():
        raise ValueError('Output must differ from input.')
    text = Path(a.input).read_text()
    def pattern(kind):
        return rf'(?ms)^[ \t]*Begin {kind}\b[^\n]*\n(.*?)^[ \t]*End {kind}\b[^\n]*'
    def rows(body):
        return [s.split() for line in body.splitlines()
                if (s := line.split('//', 1)[0].strip())]
    def get(kind):
        matches = list(re.finditer(pattern(kind), text))
        if len(matches) != 1:
            raise ValueError(f'Expected one {kind} block.')
        return rows(matches[0].group(1))
    nodes, elements = get('Nodes'), get('Elements')
    nids = [int(r[0]) for r in nodes]
    eids = [int(r[0]) for r in elements]
    if not nodes or len(set(nids)) != len(nids) or len(set(eids)) != len(eids):
        raise ValueError('Empty nodes or duplicate IDs.')
    if any(len(r) != 3 for r in elements) or sorted(int(r[2]) for r in elements) != sorted(nids):
        raise ValueError('Requires one single-node element per particle.')
    if re.search(r'(?m)^Begin (Conditions|ElementalData)\b', text):
        raise ValueError('Conditions and ElementalData are not supported.')
    radii = {}
    for m in re.finditer(pattern('NodalData'), text):
        if m.group(0).splitlines()[0].split()[2] == 'RADIUS':
            radii = {int(r[0]): float(r[2]) for r in rows(m.group(1))}
    if set(radii) != set(nids):
        raise ValueError('Requires RADIUS for every particle.')
    copies = (a.count + len(nodes) - 1) // len(nodes)
    ns, es = max(nids), max(eids)
    node_map = {i: [] for i in nids}
    element_map = {i: [] for i in eids}
    rng = random.Random(a.seed)
    new_nodes, new_elements = [], []
    element_by_node = {int(r[2]): r for r in elements}
    for index in range(a.count):
        copy, offset = divmod(index, len(nodes))
        old = nids[offset]
        nid = old + copy * ns
        eid, prop, _ = element_by_node[old]
        new_eid = int(eid) + copy * es
        radius = radii[old]
        if not 0 < radius < 0.5:
            raise ValueError(f'Invalid radius for node {old}.')
        xyz = [rng.uniform(radius, length - radius) for length in (2., 2., 1.)]
        new_nodes.append(f'{nid} ' + ' '.join(f'{v:.12g}' for v in xyz))
        new_elements.append(f'{new_eid} {prop} {nid}')
        node_map[old].append(nid)
        element_map[int(eid)].append(new_eid)
    changes = []
    for kind in ('Nodes', 'Elements', 'NodalData', 'SubModelPartNodes', 'SubModelPartElements'):
        for m in re.finditer(pattern(kind), text):
            if kind == 'Nodes':
                expanded = new_nodes
            elif kind == 'Elements':
                expanded = new_elements
            else:
                mapping = element_map if kind == 'SubModelPartElements' else node_map
                expanded = []
                for row in rows(m.group(1)):
                    old = int(row[0])
                    if old not in mapping:
                        raise ValueError(f'Unknown ID {old} in {kind}.')
                    tail = ' ' + ' '.join(row[1:]) if len(row) > 1 else ''
                    expanded.extend(f'        {new}{tail}' for new in mapping[old])
            changes.append((m.start(1), m.end(1), '\n'.join(expanded) + ('\n' if expanded else '')))
    for start, end, body in sorted(changes, reverse=True):
        text = text[:start] + body + text[end:]
    target = Path(a.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    print(f'Particles: {len(nodes)} -> {a.count}; output: {target}')
    print('Box: (0,0,0) to (2,2,1). Particle-particle overlaps are not prevented.')


if __name__ == '__main__':
    main()