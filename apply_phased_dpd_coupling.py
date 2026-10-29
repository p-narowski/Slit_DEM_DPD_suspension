#!/usr/bin/env python3
"""Local patch helper for p-narowski/Kratos feature/dpd-small-particles.
Dry run: python apply_phased_dpd_coupling.py --repo /path/to/Kratos
Apply:   python apply_phased_dpd_coupling.py --repo /path/to/Kratos --apply
No network access, commits, or remote writes. Existing files get .pre_phased_dpd backups.
This patch targets ExplicitSolverStrategy::GetForce, not FastGetForce or MPI.
"""
import argparse
import difflib
from pathlib import Path

GET_FORCE = '    void ExplicitSolverStrategy::GetForce()\n    {\n        KRATOS_TRY\n\n        ModelPart& r_model_part = GetModelPart();\n        ProcessInfo& r_process_info = r_model_part.GetProcessInfo();\n        const double dt = r_process_info[DELTA_TIME];\n        const array_1d<double, 3>& gravity = r_process_info[GRAVITY];\n        const int number_of_particles =\n            static_cast<int>(mListOfSphericParticles.size());\n\n        std::vector<SuspendedParticle*> suspended_particles;\n        for (auto* p_particle : mListOfSphericParticles)\n        {\n            auto* p_suspended = dynamic_cast<SuspendedParticle*>(p_particle);\n            if (p_suspended != nullptr)\n            {\n                KRATOS_ERROR_IF(p_suspended->Is(DEMFlags::BELONGS_TO_A_CLUSTER))\n                    << "Phased DEM-DPD coupling does not support clustered suspended particles."\n                    << std::endl;\n                suspended_particles.push_back(p_suspended);\n            }\n        }\n\n        KRATOS_ERROR_IF(!suspended_particles.empty() &&\n                       r_model_part.GetCommunicator().TotalProcesses() != 1)\n            << "Phased DEM-DPD coupling currently supports serial/OpenMP only, not MPI."\n            << std::endl;\n\n        KRATOS_ERROR_IF(!suspended_particles.empty() &&\n                       r_process_info.Has(GLOBAL_DAMPING) &&\n                       r_process_info[GLOBAL_DAMPING] != 0.0)\n            << "Phased DEM-DPD coupling requires GLOBAL_DAMPING = 0. "\n            << "The coupling is merged after ordinary RHS damping."\n            << std::endl;\n\n        NodesArrayType& r_nodes =\n            r_model_part.GetCommunicator().LocalMesh().Nodes();\n\n        // Phase 1: reset accumulators before either force-assembly phase.\n        block_for_each(r_nodes, [](ModelPart::NodeType& r_node)\n        {\n            r_node.FastGetSolutionStepValue(DPD_TO_DEM_COUPLING_FORCE).clear();\n            r_node.FastGetSolutionStepValue(DEM_TO_DPD_COUPLING_FORCE).clear();\n        });\n\n        // Phase 2: ordinary forces only; no cross-node TOTAL_FORCES writes.\n#pragma omp parallel for schedule(dynamic, 100)\n        for (int i = 0; i < number_of_particles; ++i)\n        {\n            mListOfSphericParticles[i]->CalculateRightHandSide(\n                r_process_info, dt, gravity);\n        }\n        // The parallel-for region has completed before phase 3 begins.\n\n        const int number_of_suspended_particles =\n            static_cast<int>(suspended_particles.size());\n\n        // Phase 3: each mixed pair is evaluated from its suspended owner.\n#pragma omp parallel for schedule(dynamic, 1)\n        for (int i = 0; i < number_of_suspended_particles; ++i)\n        {\n            suspended_particles[i]->ComputeDemDpdCouplingForces(r_process_info);\n        }\n        // All locked accumulator writes finish before phase 4 begins.\n\n        // Phase 4: one iteration owns each final nodal force update.\n        block_for_each(r_nodes, [](ModelPart::NodeType& r_node)\n        {\n            auto& r_total_force = r_node.FastGetSolutionStepValue(TOTAL_FORCES);\n            noalias(r_total_force) +=\n                r_node.FastGetSolutionStepValue(DPD_TO_DEM_COUPLING_FORCE);\n            noalias(r_total_force) +=\n                r_node.FastGetSolutionStepValue(DEM_TO_DPD_COUPLING_FORCE);\n        });\n\n        KRATOS_CATCH("")\n    }\n'

def once(text, old, new, label):
    if text.count(old) != 1:
        raise RuntimeError(label + ': expected exactly one source anchor; no files changed')
    return text.replace(old, new, 1)

def transform(header, suspended, strategy, dpd):
    if 'ComputeDemDpdCouplingForces' in header or 'ComputeDemDpdCouplingForces' in suspended:
        raise RuntimeError('Patch appears applied already or source differs; no files changed')
    guard = 'dynamic_cast<SuspendedParticle *>(p_neighbour) != nullptr'
    if guard not in dpd and 'dynamic_cast<SuspendedParticle*>(p_neighbour) != nullptr' not in dpd:
        raise RuntimeError('DPD mixed-pair exclusion guard is missing; fix duplicate evaluation first')
    header = once(header, '    std::string Info() const override',
        '    // Called by the strategy only after ordinary RHS assembly has finished.\n'
        '    void ComputeDemDpdCouplingForces(const ProcessInfo& rProcessInfo);\n\n'
        '    std::string Info() const override', 'suspended header')
    anchor = '        const array_1d<double, 3> &r_my_position =\n            GetGeometry()[0].Coordinates();'
    split = ('        KRATOS_CATCH("")\n'
             '    }\n\n'
             '    void SuspendedParticle::ComputeDemDpdCouplingForces(\n'
             '        const ProcessInfo& rProcessInfo)\n'
             '    {\n'
             '        KRATOS_TRY\n\n' + anchor)
    suspended = once(suspended, anchor, split, 'split mixed-pair method')
    old = ('            // Apply the DPD-to-suspended-particle force. This is the actual\n'
           "            // force contribution to the suspended particle's integration step.\n"
           '            noalias(rAdditionalForce) += global_dpd_force;\n')
    suspended = once(suspended, old,
        '            // Buffer both sides. The strategy merges them into TOTAL_FORCES later.\n',
        'remove DEM direct force addition')
    old = ('            noalias(\n'
           '                r_dpd_node.FastGetSolutionStepValue(\n'
           '                    TOTAL_FORCES)) -= global_dpd_force;\n\n')
    suspended = once(suspended, old, '', 'remove DPD direct force addition')
    suspended = suspended.replace(
        '// Record the exact coupling force applied to the suspended particle.',
        '// Accumulate the suspended-side force for the later merge phase.')
    suspended = suspended.replace(
        '// Apply and record the equal-and-opposite force to the DPD neighbour.',
        '// Accumulate the equal-and-opposite force; do not touch TOTAL_FORCES here.')
    suspended = suspended.replace(
        '// so both updates must occur while holding this node\'s lock.',
        '// so this accumulator update must hold the node lock.')
    neighbour_anchor = ('            if (p_dpd_neighbour == nullptr)\n'
                        '            {\n'
                        '                continue;\n'
                        '            }\n')
    suspended = once(suspended, neighbour_anchor,
        neighbour_anchor + '\n'
        '            KRATOS_ERROR_IF(p_dpd_neighbour->Is(DEMFlags::BELONGS_TO_A_CLUSTER))\n'
        '                << "Phased DEM-DPD coupling does not support clustered DPD neighbours."\n'
        '                << std::endl;\n', 'clustered DPD guard')
    if '#include <algorithm>' not in suspended:
        suspended = once(suspended, '#include <cmath>', '#include <algorithm>\n#include <cmath>', 'algorithm include')
    include = '#include "custom_elements/suspended_particle.h"'
    if include not in strategy:
        strategy = once(strategy, '#include "custom_elements/dpd_particle.h"',
            '#include "custom_elements/dpd_particle.h"\n' + include, 'strategy include')
    start_token = '    void ExplicitSolverStrategy::GetForce()'
    end_token = '    void ExplicitSolverStrategy::FastGetForce()'
    if strategy.count(start_token) != 1 or strategy.count(end_token) != 1:
        raise RuntimeError('Strategy function anchors differ; no files changed')
    start, end = strategy.index(start_token), strategy.index(end_token)
    if start >= end:
        raise RuntimeError('Unexpected method order; no files changed')
    strategy = strategy[:start] + GET_FORCE + '\n' + strategy[end:]
    return header, suspended, strategy

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', required=True, type=Path)
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    base = args.repo / 'applications/DEMApplication'
    rel = ['custom_elements/suspended_particle.h',
           'custom_elements/suspended_particle.cpp',
           'custom_strategies/strategies/explicit_solver_strategy.cpp']
    paths = [base / r for r in rel]
    old = [p.read_text(encoding='utf-8') for p in paths]
    dpd = (base / 'custom_elements/dpd_particle.cpp').read_text(encoding='utf-8')
    new = transform(*old, dpd)
    for path, a, b in zip(paths, old, new):
        print(''.join(difflib.unified_diff(a.splitlines(True), b.splitlines(True),
              fromfile=str(path), tofile=str(path))), end='')
    if not args.apply:
        print('\nDRY RUN: source files unchanged. Review the diff, then add --apply.')
        return
    backups = [p.with_name(p.name + '.pre_phased_dpd') for p in paths]
    if any(p.exists() for p in backups):
        raise RuntimeError('Backup already exists; refusing to overwrite it')
    for path, backup in zip(paths, backups):
        backup.write_bytes(path.read_bytes())
    for path, text in zip(paths,new):
        path.write_text(text,encoding='utf-8')
    print('\nApplied three-file patch. Rebuild DEMApplication before testing.')

if __name__ == '__main__':
    main()