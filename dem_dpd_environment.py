#!/usr/bin/env python3
"""CSV-first diagnostics for one DEM sphere and radius-selected DPD particles.
Dependencies: numpy pandas scipy pyvista. No Kratos installation required.
"""
import argparse
import glob
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def array(mesh, requested, candidates, label):
    names = list(mesh.point_data.keys())
    if requested:
        if requested not in names:
            raise ValueError(f'{label} array {requested!r} missing. Available: {names}')
        return np.asarray(mesh.point_data[requested])
    for candidate in candidates:
        for name in names:
            if name.lower() == candidate.lower():
                return np.asarray(mesh.point_data[name])
    raise ValueError(f'No {label} array. Available: {names}')


def file_index(path):
    match = re.search(r'_(\d+)\.vtu$', str(path))
    if not match:
        raise ValueError(f'Cannot infer frame index from {path}; use VTU TIME metadata.')
    return int(match.group(1))


def frame_time(mesh, path, args):
    for name in mesh.field_data.keys():
        if name.upper() in ('TIME', 'TIME_VALUE', 'TIMEVALUE'):
            values = np.asarray(mesh.field_data[name]).ravel()
            if values.size == 1 and np.isfinite(values[0]):
                return float(values[0]), 'VTU metadata'
    if args.output_interval is None or args.output_interval <= 0:
        raise ValueError('No TIME metadata: supply --output-interval and verify filename numbering.')
    t = args.time_offset + (file_index(path)-args.index_origin)*args.output_interval
    return t, 'filename index (verify numbering)'


def metrics(w, normal, mask):
    if not np.any(mask):
        return dict(slip_rms=np.nan, slip_mean_speed=np.nan,
                    normal_mean=np.nan, normal_rms=np.nan,
                    slip_mean_x=np.nan, slip_mean_y=np.nan, slip_mean_z=np.nan)
    q, n = w[mask], normal[mask]
    wn = np.einsum('ij,ij->i', q, n)
    wt = q-wn[:, None]*n
    speed2 = np.einsum('ij,ij->i', wt, wt)
    mean = wt.mean(axis=0)
    return dict(slip_rms=float(np.sqrt(speed2.mean())),
                slip_mean_speed=float(np.sqrt(speed2).mean()),
                normal_mean=float(wn.mean()), normal_rms=float(np.sqrt((wn**2).mean())),
                slip_mean_x=float(mean[0]), slip_mean_y=float(mean[1]), slip_mean_z=float(mean[2]))


def analyze(points, velocities, radii, omega, args):
    solid = np.flatnonzero(np.isclose(radii, args.dem_radius, rtol=0, atol=args.radius_tolerance))
    if len(solid) == 0:
        return None
    if len(solid) != 1:
        raise ValueError(f'Expected one DEM particle, found {len(solid)}.')
    # Treat --dpd-radius as the maximum DPD radius.
    mask = (
        np.isfinite(radii)
        & (radii > 0.0)
        & (radii <= args.dpd_radius)
    )
    # Explicitly exclude the selected DEM particle.
    mask[solid[0]] = False
    if mask[solid[0]] or not np.any(mask):
        raise ValueError('Invalid/overlapping radius selections or no DPD particles.')
    k = solid[0]
    box = np.array(args.box, float)
    periodic = np.array([c in args.periodic for c in 'xyz'])
    disp = points[mask]-points[k]
    disp[:, periodic] -= box[periodic]*np.round(disp[:, periodic]/box[periodic])
    r = np.linalg.norm(disp, axis=1)
    d = r-args.dem_radius
    normal = np.zeros_like(disp)
    valid = r > 1.e-14
    normal[valid] = disp[valid]/r[valid, None]
    w = velocities[mask]-velocities[k]-np.cross(omega, args.dem_radius*normal)
    outside = (d >= 0) & (d < args.dem_cutoff)
    inside = d < 0
    candidates = d < args.dem_cutoff
    near = (d >= 0) & (d < args.near_width) & valid
    direction = np.zeros(3); direction['xyz'.index(args.axis)] = 1
    cosine = normal @ direction
    equator = near & (np.abs(cosine) <= np.sin(np.deg2rad(args.equator_degrees)))
    vbox = float(np.prod(box))
    vfluid = args.fluid_volume
    if vfluid is None:
        vfluid = vbox if args.volume_convention == 'box' else vbox-4*np.pi*args.dem_radius**3/3
    if vfluid <= 0:
        raise ValueError('Fluid volume must be positive.')
    n = int(mask.sum()); density = n/vfluid
    spacing = density**(-1/3)
    row = dict(n_dpd=n, fluid_volume=vfluid, number_density=density,
               nominal_spacing=spacing, spacing_over_dem_radius=spacing/args.dem_radius,
               spacing_over_dpd_cutoff=spacing/args.dpd_cutoff,
               overlap_eta=density*4*np.pi*args.dpd_cutoff**3/3,
               n_exterior_layer=int(outside.sum()), n_interior=int(inside.sum()),
               n_geometric_candidates=int(candidates.sum()),
               interior_fraction_all_dpd=float(inside.mean()),
               max_penetration=float(max(0., -d.min())),
               n_near=int(near.sum()), n_equator=int(equator.sum()),
               n_undefined_surface_normals=int((~valid).sum()))
    for prefix, selection in [('near_', near), ('equator_', equator)]:
        row.update({prefix+key: value for key, value in metrics(w, normal, selection).items()})
    if args.nearest_neighbors:
        if args.periodic not in ('xyz', ''):
            raise ValueError('Nearest neighbors support only fully periodic or nonperiodic boxes.')
        p = points[mask].copy()
        if n > 1:
            if args.periodic == 'xyz':
                p %= box
                tree = cKDTree(p, boxsize=box)
            else:
                tree = cKDTree(p)
            nearest = tree.query(p, k=2)[0][:, 1]
            row.update(nn_mean=float(nearest.mean()), nn_std=float(nearest.std()),
                       nn_over_dem_radius=float(nearest.mean()/args.dem_radius),
                       nn_over_dpd_cutoff=float(nearest.mean()/args.dpd_cutoff))
    profiles = []
    edges = np.linspace(0., args.dem_cutoff, args.radial_bins+1)
    for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        radial = (d >= lo) & (d < hi)
        shell_volume = 4*np.pi/3*((args.dem_radius+hi)**3-(args.dem_radius+lo)**3)
        for region, angular in [('all', np.ones(n, bool)),
                                ('equator', np.abs(cosine) <= np.sin(np.deg2rad(args.equator_degrees))),
                                ('positive_axis_cap', cosine > 0.5),
                                ('negative_axis_cap', cosine < -0.5)]:
            sel = radial & angular & valid
            entry = dict(radial_bin=j, d_low=lo, d_high=hi, region=region,
                         count=int(sel.sum()),
                         full_shell_number_density=float(radial.sum()/shell_volume) if region == 'all' else np.nan)
            entry.update(metrics(w, normal, sel))
            profiles.append(entry)
    return row, profiles


def summarize(df):
    t = df.time.to_numpy(float)
    rows = []
    for col in df.select_dtypes(include='number').columns:
        if col == 'time':
            continue
        y = df[col].to_numpy(float)
        good = np.isfinite(y)
        avg = np.nan
        std = np.nan
        # Do not bridge missing-data gaps when integrating.
        if len(t) > 1:
            dt = np.diff(t)
            use = good[:-1] & good[1:]
            duration = dt[use].sum()
            if duration > 0:
                a, b = y[:-1][use], y[1:][use]
                avg = float(np.sum(dt[use]*(a+b)/2)/duration)
                second = np.sum(dt[use]*(a*a+a*b+b*b)/3)/duration
                std = float(np.sqrt(max(0., second-avg*avg)))
        values = y[good]
        rows.append(dict(metric=col, frames=int(good.sum()),
                         frame_mean=float(values.mean()) if len(values) else np.nan,
                         frame_std=float(values.std()) if len(values) else np.nan,
                         minimum=float(values.min()) if len(values) else np.nan,
                         maximum=float(values.max()) if len(values) else np.nan,
                         time_mean_linear_interpolation=avg,
                         time_std_linear_interpolation=std))
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vtkdir', default='Slit_Post_VTK_Files')
    p.add_argument('--pattern', default='Slit_Particles_*.vtu')
    p.add_argument('--output-interval', type=float)
    p.add_argument('--index-origin', type=int, default=0)
    p.add_argument('--time-offset', type=float, default=0.)
    p.add_argument('--dem-radius', type=float, required=True)
    p.add_argument(
    '--dpd-radius',
    type=float,
    required=True,
    help='Maximum DPD radius: select particles with 0 < radius <= this value.',
    )
    p.add_argument('--dem-cutoff', type=float, required=True, help='Surface-gap cutoff, not center cutoff/search radius.')
    p.add_argument('--dpd-cutoff', type=float, required=True)
    p.add_argument('--box', type=float, nargs=3, required=True, metavar=('LX','LY','LZ'))
    p.add_argument('--periodic', default='xyz', help='Periodic axes, e.g. xyz, xy, or none.')
    p.add_argument('--fluid-volume', type=float)
    p.add_argument('--volume-convention', choices=['box', 'exclude-solid'], default='exclude-solid')
    p.add_argument('--near-width', type=float, default=None)
    p.add_argument('--equator-degrees', type=float, default=15.)
    p.add_argument('--axis', choices=['x','y','z'], default='x')
    p.add_argument('--radial-bins', type=int, default=20)
    p.add_argument('--radius-tolerance', type=float, default=1.e-6)
    p.add_argument('--velocity-array')
    p.add_argument('--radius-array')
    p.add_argument('--rotation', action='store_true', help='Require DEM angular velocity from VTU.')
    p.add_argument('--omega-array')
    p.add_argument('--nearest-neighbors', action='store_true')
    p.add_argument('--tstart', type=float, default=-np.inf)
    p.add_argument('--tend', type=float, default=np.inf)
    p.add_argument('--outdir', default='interaction_diagnostics')
    a = p.parse_args()
    a.periodic = '' if a.periodic.lower() == 'none' else a.periodic.lower()
    if any(c not in 'xyz' for c in a.periodic):
        p.error('Invalid periodic axes.')
    if a.near_width is None:
        a.near_width = a.dem_cutoff/4
    if min(a.dem_radius,a.dpd_radius,a.dem_cutoff,a.dpd_cutoff,a.radius_tolerance,a.near_width,*a.box) <= 0:
        p.error('Radii, cutoffs, near width, tolerance and box lengths must be positive.')
    if a.radial_bins < 1 or not 0 < a.equator_degrees < 90 or a.near_width > a.dem_cutoff or a.tend < a.tstart:
        p.error('Invalid bin count, angle, near width or time range.')
    # Full shell volumes require no self-image overlap and no wall clipping.
    if a.dem_radius+a.dem_cutoff >= min(a.box)/2:
        p.error('R + DEM cutoff must be below half the smallest box length for full-shell profiles.')
    if a.periodic != 'xyz':
        print('WARNING: full-shell density assumes shells are not clipped by walls/boundaries.')
    import pyvista as pv
    paths = sorted(glob.glob(str(Path(a.vtkdir)/a.pattern)), key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
    if not paths:
        raise FileNotFoundError('No matching VTU files.')
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    if any((out/f).exists() for f in ['frames.csv','radial_profiles.csv','summary.csv','run_settings.json']):
        raise FileExistsError('Output files already exist. Choose a new --outdir to avoid overwriting.')
    records, skipped, sources = [], 0, set()
    with (out/'radial_profiles.csv').open('w') as profile_file:
        header = True
        for path in paths:
            mesh = pv.read(path)
            t, source = frame_time(mesh, path, a)
            if not a.tstart <= t <= a.tend:
                continue
            sources.add(source)
            r = array(mesh, a.radius_array, ['RADIUS','PARTICLE_RADIUS'], 'radius').reshape(-1)
            v = array(mesh, a.velocity_array, ['VELOCITY','PARTICLE_VELOCITY'], 'velocity')
            points = np.asarray(mesh.points, float)
            if v.shape != points.shape or not np.isfinite(v).all() or not np.isfinite(points).all() or not np.isfinite(r).all():
                raise ValueError(f'Invalid positions, velocities or radii in {path}')
            solid = np.flatnonzero(np.isclose(r, a.dem_radius, rtol=0, atol=a.radius_tolerance))
            if not len(solid):
                skipped += 1
                continue
            omega = np.zeros(3)
            if a.rotation:
                om = array(mesh, a.omega_array, ['ANGULAR_VELOCITY'], 'angular velocity')
                omega = np.asarray(om[solid[0]], float)
                if omega.shape != (3,) or not np.isfinite(omega).all():
                    raise ValueError('Invalid angular velocity.')
            result = analyze(points,v,r,omega,a)
            row, profiles = result
            row.update(time=t, source_file=Path(path).name)
            records.append(row)
            pd.DataFrame([dict(time=t, **entry) for entry in profiles]).to_csv(profile_file,index=False,header=header)
            header = False
            print(f't={t:g}: exterior={row["n_exterior_layer"]}, interior={row["n_interior"]}, slip RMS={row["near_slip_rms"]:.6g}')
    if not records:
        raise RuntimeError('No usable frames in selected window.')
    df = pd.DataFrame(records).sort_values('time').reset_index(drop=True)
    if df.time.duplicated().any():
        raise ValueError('Duplicate timestamps: check pattern, metadata and index mapping.')
    df.to_csv(out/'frames.csv',index=False)
    summarize(df).to_csv(out/'summary.csv',index=False)
    settings = vars(a).copy()
    settings.update(skipped_without_dem=skipped, frames=len(df), time_sources=sorted(sources),
                    notes=['Geometric occupancy, not verified solver force-pair count.',
                           'Slip at finite particle-center gaps, not extrapolated surface slip length.',
                           'No particle attachment/residence tracking.',
                           'Time statistics assume linear interpolation between snapshots.',
                           'Use parameters archived with this run, not necessarily current repository values.'])
    (out/'run_settings.json').write_text(json.dumps(settings,indent=2))
    print(f'Saved diagnostics for {len(df)} frames. Skipped {skipped} pre-inlet frames.')


if __name__ == '__main__':
    main()