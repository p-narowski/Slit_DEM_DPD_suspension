#!/usr/bin/env python3
"""Audit one ORIGINAL particle VTU, comparing radius selection with all non-DEM points."""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('vtu',help='Exact original VTU opened in ParaView, at the same frame')
    p.add_argument('--dem-radius',type=float,default=.2)
    p.add_argument('--dpd-radius',type=float,default=.02)
    p.add_argument('--cutoff',type=float,default=.2)
    p.add_argument('--box',type=float,nargs=3,default=[2.,2.,1.])
    p.add_argument('--periodic',default='xyz')
    p.add_argument('--center',type=float,nargs=3,help='Optional manual DEM center for comparison')
    p.add_argument('--tol',type=float,default=1e-6)
    p.add_argument('--outdir',default='occupancy_audit')
    a=p.parse_args()
    import pyvista as pv
    m=pv.read(a.vtu)
    names=list(m.point_data.keys())
    rn=next((n for n in names if n.lower() in ('radius','particle_radius')),None)
    if rn is None: raise ValueError(f'No radius array. Available: {names}')
    r=np.asarray(m.point_data[rn]).reshape(-1)
    dem=np.isclose(r,a.dem_radius,rtol=0,atol=a.tol)
    dpd=np.isclose(r,a.dpd_radius,rtol=0,atol=a.tol)
    ix=np.flatnonzero(dem)
    if len(ix)!=1: raise ValueError(f'Expected 1 DEM radius match, found {len(ix)}. Check radius values.')
    center=np.asarray(m.points[ix[0]],float)
    box=np.asarray(a.box,float)
    periodic=np.array([c in a.periodic.lower() for c in 'xyz'])
    if min(*box,a.cutoff,a.dem_radius,a.dpd_radius,a.tol)<=0: raise ValueError('Lengths must be positive.')
    if a.periodic.lower()!='none' and any(c not in 'xyz' for c in a.periodic.lower()): raise ValueError('Invalid periodic axes.')
    def distances(c):
        dx=np.asarray(m.points,float)-c
        dx[:,periodic]-=box[periodic]*np.round(dx[:,periodic]/box[periodic])
        return np.linalg.norm(dx,axis=1)
    distance=distances(center)
    gap=distance-a.dem_radius
    non_dem=~dem
    outer=(gap>=0)&(gap<a.cutoff)
    inner=gap<0
    nearby=non_dem&(gap<a.cutoff)
    lines=[f'Input: {a.vtu}',f'Points: {m.n_points}',f'Cells: {m.n_cells}',
           f'Point arrays: {names}',f'Radius array: {rn}',
           f'DEM point index: {ix[0]}',f'DEM center from raw VTU: {center.tolist()}',
           f'DEM radius matches: {dem.sum()}',f'DPD radius matches: {dpd.sum()}',
           f'Non-DEM points excluded by DPD radius filter: {(non_dem&~dpd).sum()}',
           f'Filtered DPD exterior: {(dpd&outer).sum()}',
           f'Filtered DPD interior: {(dpd&inner).sum()}',
           f'ALL non-DEM exterior: {(non_dem&outer).sum()}',
           f'ALL non-DEM interior: {(non_dem&inner).sum()}',
           'All-non-DEM counts are an audit, not a validated fluid classification.']
    if a.center:
        manual_gap=distances(np.array(a.center))-a.dem_radius
        lines += [f'Manual center: {a.center}',
                  f'Manual-center filtered exterior: {(dpd&(manual_gap>=0)&(manual_gap<a.cutoff)).sum()}',
                  f'Manual-center all-non-DEM exterior: {(non_dem&(manual_gap>=0)&(manual_gap<a.cutoff)).sum()}']
    out=Path(a.outdir)
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()): raise FileExistsError('Choose an empty/new --outdir.')
    bins=pd.DataFrame({'radius_rounded_10dp':np.round(r.astype(float),10),
                       'matches_dpd':dpd,'matches_dem':dem})
    table=bins.groupby(['radius_rounded_10dp','matches_dpd','matches_dem'],dropna=False).size().reset_index(name='count')
    table.to_csv(out/'radius_counts.csv',index=False)
    lines+=['Radius groups (first 30):',table.head(30).to_string(index=False)]
    for name in names:
        if name.lower()=='material':
            material=np.asarray(m.point_data[name]).reshape(-1)
            pd.crosstab(np.round(r,10),material).to_csv(out/'radius_by_material.csv')
    report='\n'.join(lines)
    (out/'report.txt').write_text(report)
    print(report)
    m.point_data['audit_surface_gap']=gap
    m.point_data['audit_dpd_radius_match']=dpd.astype(np.int32)
    m.point_data['audit_dem_radius_match']=dem.astype(np.int32)
    m.point_data['audit_exterior']= (non_dem&outer).astype(np.int32)
    m.save(out/'audited_original.vtu')
    if nearby.any():
        # Preserve original particle centers, not glyph vertices.
        cloud=pv.PolyData(np.asarray(m.points)[nearby])
        for name in m.point_data.keys(): cloud.point_data[name]=np.asarray(m.point_data[name])[nearby]
        cloud.save(out/'all_non_dem_candidates.vtp')

if __name__=='__main__': main()