#!/usr/bin/env python3
"""Triple a particle-only MDPA and redistribute all particle centers."""
import argparse
import random
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', nargs='?', default='SlitDEM.mdpa')
    parser.add_argument('output', nargs='?', default='SlitDEM_tripled.mdpa')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    source, target = Path(args.input), Path(args.output)
    if source.resolve() == target.resolve():
        raise ValueError('Use a different output filename to protect the original.')
    text = source.read_text()

    def block(kind):
        pattern = rf'(?m)^Begin {kind}[^\n]*\n(.*?)^End {kind}\s*$'
        matches = list(re.finditer(pattern, text, re.S))
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one {kind} block.')
        return matches[0]

    def rows(body):
        return [line.split('//', 1)[0].split() for line in body.splitlines()
                if line.split('//', 1)[0].strip()]

    nodes_match, elements_match = block('Nodes'), block('Elements')
    nodes = rows(nodes_match.group(1))
    elements = rows(elements_match.group(1))
    node_ids = [int(row[0]) for row in nodes]
    element_ids = [int(row[0]) for row in elements]
    if len(set(node_ids)) != len(nodes) or len(set(element_ids)) != len(elements):
        raise ValueError('Duplicate IDs in input.')
    if len(elements) != len(nodes) or any(len(row) != 3 for row in elements):
        raise ValueError('This script requires one single-node element per particle.')
    if sorted(int(row[2]) for row in elements) != sorted(node_ids):
        raise ValueError('Particle elements must reference each node exactly once.')
    if re.search(r'(?m)^Begin (Conditions|ElementalData)\b', text):
        raise ValueError('Conditions/ElementalData require additional handling.')

    node_step, element_step = max(node_ids), max(element_ids)
    nodal_pattern = r'(?m)^Begin NodalData ([^\n]*)\n(.*?)^End NodalData\s*$'
    nodal_matches = list(re.finditer(nodal_pattern, text, re.S))
    radii = {}
    for match in nodal_matches:
        if match.group(1).split()[0] == 'RADIUS':
            radii = {int(row[0]): float(row[2]) for row in rows(match.group(2))}
    if set(radii) != set(node_ids):
        raise ValueError('Expected a RADIUS value for every particle.')

    lower, upper = (0.0, 0.0, 0.0), (2.0, 2.0, 1.0)
    rng = random.Random(args.seed)
    replacements = []
    new_nodes = []
    for copy in range(3):
        for row in nodes:
            old_id = int(row[0])
            radius = radii[old_id]
            if radius <= 0 or any(2 * radius >= b - a for a, b in zip(lower, upper)):
                raise ValueError(f'Invalid radius for node {old_id}.')
            xyz = [rng.uniform(a + radius, b - radius) for a, b in zip(lower, upper)]
            new_nodes.append(f'{old_id + copy * node_step} ' + ' '.join(f'{v:.12g}' for v in xyz))
    replacements.append((nodes_match.start(1), nodes_match.end(1), '\n'.join(new_nodes) + '\n'))
    new_elements = []
    for copy in range(3):
        for eid, prop, nid in elements:
            new_elements.append(f'{int(eid) + copy * element_step} {prop} {int(nid) + copy * node_step}')
    replacements.append((elements_match.start(1), elements_match.end(1), '\n'.join(new_elements) + '\n'))
    for match in nodal_matches:
        data = rows(match.group(2))
        if any(int(row[0]) not in radii for row in data):
            raise ValueError('NodalData references an unknown node.')
        expanded = [f'{int(row[0]) + copy * node_step} ' + ' '.join(row[1:])
                    for copy in range(3) for row in data]
        replacements.append((match.start(2), match.end(2), '\n'.join(expanded) + '\n'))
    for name, step, valid in [('SubModelPartNodes', node_step, set(node_ids)),
                              ('SubModelPartElements', element_step, set(element_ids))]:
        pattern = rf'(?m)^[ \t]*Begin {name}[^\n]*\n(.*?)^[ \t]*End {name}[ \t]*$'
        for match in re.finditer(pattern, text, re.S):
            ids = [int(row[0]) for row in rows(match.group(1))]
            if any(i not in valid for i in ids):
                raise ValueError(f'Unknown ID in {name}.')
            expanded = [f'        {i + copy * step}' for copy in range(3) for i in ids]
            replacements.append((match.start(1), match.end(1), '\n'.join(expanded) + ('\n' if expanded else '')))
    for start, end, replacement in sorted(replacements, reverse=True):
        text = text[:start] + replacement + text[end:]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    print(f'Particles: {len(nodes)} -> {3 * len(nodes)}')
    print(f'Output: {target}')
    print('Bounding box: (0, 0, 0) to (2, 2, 1); radii and properties preserved.')
    print('Random placement: particle-particle overlaps are NOT prevented.')


if __name__ == '__main__':
    main()