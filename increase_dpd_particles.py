#!/usr/bin/env python3
"""Increase DPD count in a pure DPD MDPA, keeping the original particles.
Python 3.9+, standard library only. No Kratos installation required.
New centers are uniform random samples, not coincident duplicates.
"""
import argparse
import json
import math
import random
from pathlib import Path

class Block:
    def __init__(self, header):
        self.header = header
        self.words = header.split('//', 1)[0].split()
        self.kind = self.words[1]
        self.items = []

    def rows(self):
        for line in self.items:
            if isinstance(line, str):
                tokens = line.split('//', 1)[0].split()
                if tokens:
                    yield tokens

def parse(text):
    root = Block('Begin ROOT')
    stack = [root]
    for line in text.splitlines():
        tokens = line.split('//', 1)[0].split()
        if tokens and tokens[0] == 'Begin':
            block = Block(line.strip())
            stack[-1].items.append(block)
            stack.append(block)
        elif tokens and tokens[0] == 'End':
            if len(stack) == 1 or tokens[1] != stack[-1].kind:
                raise ValueError('Unbalanced MDPA blocks: ' + line)
            stack.pop()
        else:
            stack[-1].items.append(line)
    if len(stack) != 1:
        raise ValueError('Unclosed MDPA block')
    return root

def walk(block):
    for item in block.items:
        if isinstance(item, Block):
            yield item
            yield from walk(item)

def write_block(block, stream):
    if block.kind != 'ROOT':
        stream.write(block.header + '\n')
    for item in block.items:
        if isinstance(item, Block):
            write_block(item, stream)
        else:
            stream.write(item + '\n')
    if block.kind != 'ROOT':
        stream.write('End ' + block.kind + '\n')

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input', type=Path)
    ap.add_argument('--factors', nargs='+', type=int, required=True,
                    help='Total-count multipliers, e.g. 2 3 4')
    ap.add_argument('--parameters', type=Path, default=Path('ProjectParametersDEM.json'))
    ap.add_argument('--bounds', nargs=6, type=float,
                    metavar=('XMIN','YMIN','ZMIN','XMAX','YMAX','ZMAX'))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output-dir', type=Path, default=Path('particle_variants'))
    ap.add_argument('--min-distance', type=float, default=0.0,
                    help='Minimum new-to-existing/new center spacing; 0 disables rejection')
    ap.add_argument('--exclude-sphere', nargs=4, type=float, action='append', default=[],
                    metavar=('X','Y','Z','R'), help='Reject NEW centers inside sphere; repeatable')
    ap.add_argument('--max-attempts', type=int, default=1000,
                    help='Maximum candidate attempts per new particle')
    args = ap.parse_args()
    if any(f < 2 for f in args.factors):
        ap.error('All factors must be integers >= 2')
    if args.min_distance < 0 or args.max_attempts < 1:
        ap.error('Invalid minimum distance or attempt limit')
    if any(s[3] <= 0 for s in args.exclude_sphere):
        ap.error('Exclusion radii must be positive')
    if args.bounds:
        lo, hi = args.bounds[:3], args.bounds[3:]
    else:
        config = json.loads(args.parameters.read_text())
        lo = [config['BoundingBoxMin' + a] for a in 'XYZ']
        hi = [config['BoundingBoxMax' + a] for a in 'XYZ']
    lengths = [b-a for a,b in zip(lo,hi)]
    if any(L <= 0 for L in lengths):
        ap.error('Invalid box bounds')
    text = args.input.read_text()
    root = parse(text)
    blocks = list(walk(root))
    allowed = {'ModelPartData','Properties','Nodes','Elements','NodalData',
               'SubModelPart','SubModelPartData','SubModelPartNodes',
               'SubModelPartElements','SubModelPartConditions'}
    unknown = {b.kind for b in blocks} - allowed
    if unknown:
        raise ValueError('Unsupported sections (refusing unsafe rewrite): ' + str(unknown))
    nb = [b for b in blocks if b.kind == 'Nodes']
    eb = [b for b in blocks if b.kind == 'Elements']
    if len(nb) != 1 or len(eb) != 1 or eb[0].words[2:] != ['DPDParticle']:
        raise ValueError('Requires one Nodes block and one Elements DPDParticle block')
    nodes = {int(r[0]): tuple(map(float,r[1:])) for r in nb[0].rows()}
    elements = list(eb[0].rows())
    if any(len(r) != 3 for r in elements) or any(len(p) != 3 for p in nodes.values()):
        raise ValueError('Requires 3D nodes and single-node elements')
    node_to_element = {int(r[2]): (int(r[0]),r[1]) for r in elements}
    if len(node_to_element) != len(elements) or set(node_to_element) != set(nodes):
        raise ValueError('Requires exactly one DPD element per node')
    if len({int(r[0]) for r in elements}) != len(elements):
        raise ValueError('Duplicate element IDs')
    if any(not all(math.isfinite(x) and a <= x <= b for x,a,b in zip(p,lo,hi))
           for p in nodes.values()):
        raise ValueError('Original nodes outside supplied box or nonfinite')
    for b in blocks:
        if b.kind == 'NodalData' and {int(r[0]) for r in b.rows()} != set(nodes):
            raise ValueError('NodalData must define every original node: ' + b.header)
    n = len(nodes)
    if not n:
        raise ValueError('Empty particle input')
    ids = list(nodes)
    max_node = max(ids)
    max_element = max(int(r[0]) for r in elements)
    for sphere in args.exclude_sphere:
        count = sum(sum((p[k]-sphere[k])**2 for k in range(3)) < sphere[3]**2
                    for p in nodes.values())
        if count:
            print(f'WARNING: {count} ORIGINAL centers inside exclusion sphere; retained.')
    # Generate one nested sequence: x2 additions are a subset of x3, etc.
    rng = random.Random(args.seed)
    grid = {}
    d = args.min_distance
    cells = [max(1, int(L/d)) for L in lengths] if d else None
    widths = [L/c for L,c in zip(lengths,cells)] if d else None
    def cell(p):
        return tuple(min(cells[k]-1, int((p[k]-lo[k])/widths[k])) for k in range(3))
    def insert(p):
        if d:
            grid.setdefault(cell(p), []).append(p)
    def separated(p):
        if not d:
            return True
        c = cell(p)
        neighbors = {tuple((c[k]+off[k]) % cells[k] for k in range(3))
                     for off in ((i,j,k) for i in (-1,0,1)
                                 for j in (-1,0,1) for k in (-1,0,1))}
        for key in neighbors:
            for q in grid.get(key, []):
                delta = [abs(p[k]-q[k]) for k in range(3)]
                if sum(min(v,L-v)**2 for v,L in zip(delta,lengths)) < d*d:
                    return False
        return True
    for p in nodes.values():
        insert(p)
    additions = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for factor in sorted(set(args.factors)):
        output = args.output_dir / f'{args.input.stem}_x{factor}.mdpa'
        if output.exists() or output.resolve() == args.input.resolve():
            raise FileExistsError(f'Refusing overwrite: {output}')
        target_added = (factor-1)*n
        while len(additions) < target_added:
            for attempt in range(args.max_attempts):
                p = tuple(rng.uniform(a,b) for a,b in zip(lo,hi))
                excluded = any(sum((p[k]-s[k])**2 for k in range(3)) < s[3]**2
                               for s in args.exclude_sphere)
                if not excluded and separated(p):
                    break
            else:
                raise RuntimeError('Insertion failed. Reduce --min-distance or inspect exclusions.')
            parent = rng.choice(ids)
            j = len(additions)+1
            additions.append((max_node+j, max_element+j, parent, p))
            insert(p)
        variant = parse(text)
        for b in walk(variant):
            if b.kind == 'Nodes':
                b.items.extend(f'{nid} ' + ' '.join(f'{v:.16g}' for v in p)
                               for nid,eid,parent,p in additions)
            elif b.kind == 'Elements':
                b.items.extend(f'{eid} {node_to_element[parent][1]} {nid}'
                               for nid,eid,parent,p in additions)
            elif b.kind == 'NodalData':
                values = {int(r[0]): r[1:] for r in b.rows()}
                b.items.extend(f'{nid} ' + ' '.join(values[parent])
                               for nid,eid,parent,p in additions)
            elif b.kind in ('SubModelPartNodes','SubModelPartElements'):
                members = {int(token) for r in b.rows() for token in r}
                is_node = b.kind == 'SubModelPartNodes'
                b.items.extend(str(nid if is_node else eid)
                               for nid,eid,parent,p in additions
                               if (parent if is_node else node_to_element[parent][0]) in members)
        with output.open('x') as stream:
            write_block(variant, stream)
        print(f'{output}: {n:,} -> {factor*n:,} particles; original centers/radii retained')
    print('No material, timestep, cutoff, or mass parameters were changed.')
    print('This is a number-density study, NOT automatically fixed-density refinement.')
    print('New particles inherit nodal data and group membership from sampled originals.')
    print('Use only for spatially homogeneous particle groups; relax each initial state.')

if __name__ == '__main__':
    main()