#!/usr/bin/env python3
"""Plot CSVs produced by dem_dpd_environment.py; no VTU rereading required."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--indir', default='interaction_diagnostics_baseline')
    p.add_argument('--outdir', help='Default: INDIR/figures')
    p.add_argument('--tstart', type=float, default=-np.inf)
    p.add_argument('--tend', type=float, default=np.inf)
    p.add_argument('--dpi', type=int, default=200)
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()
    if a.dpi < 1 or a.tend < a.tstart:
        p.error('Invalid DPI or time window.')
    src = Path(a.indir)
    out = Path(a.outdir) if a.outdir else src/'figures'
    df = pd.read_csv(src/'frames.csv').sort_values('time')
    df = df[df.time.between(a.tstart,a.tend)].copy()
    if df.empty:
        raise ValueError('No frames in selected time window.')
    out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.25})
    written = []

    def save(fig, name):
        dest=out/name
        if dest.exists() and not a.overwrite:
            plt.close(fig)
            raise FileExistsError(f'{dest} exists; use --overwrite or a new --outdir.')
        fig.tight_layout()
        fig.savefig(dest,dpi=a.dpi,bbox_inches='tight')
        plt.close(fig)
        written.append(name)

    def panel(ax, series, ylabel, title):
        found=False
        for column,label in series:
            if column in df and np.isfinite(df[column]).any():
                ax.plot(df.time,df[column],label=label,linewidth=1.5)
                found=True
        ax.set(xlabel='Simulation time',ylabel=ylabel,title=title)
        if found:
            ax.legend(fontsize=8)
        else:
            ax.text(.5,.5,'No finite data available',transform=ax.transAxes,ha='center')

    fig,ax=plt.subplots(figsize=(10,5))
    panel(ax,[('n_exterior_layer','Exterior interaction layer'),('n_interior','Inside DEM'),
              ('n_geometric_candidates','All geometric candidates')],
          'DPD particle count','Geometric occupancy (not verified force-pair count)')
    save(fig,'01_interaction_counts.png')

    fig,axs=plt.subplots(2,1,figsize=(10,7))
    panel(axs[0],[('max_penetration','Maximum center penetration')],
          'Penetration depth','Penetration into the DEM sphere')
    panel(axs[1],[('interior_fraction_all_dpd','Interior count / all DPD')],
          'Fraction','Interior population fraction')
    save(fig,'02_penetration.png')

    fig,axs=plt.subplots(2,1,figsize=(10,7))
    panel(axs[0],[('near_slip_rms','Near-surface RMS'),('equator_slip_rms','Equatorial RMS')],
          'Tangential relative speed','Slip proxy at finite particle-center gaps')
    panel(axs[1],[('near_normal_mean','Near-surface mean normal'),
                  ('near_normal_rms','Near-surface RMS normal')],
          'Normal relative velocity','Normal motion: positive is outward')
    save(fig,'03_near_surface_slip.png')

    fig,axs=plt.subplots(3,1,figsize=(10,10))
    panel(axs[0],[('number_density','DPD number density')],'Particles / volume','Global number density')
    panel(axs[1],[('overlap_eta','eta = n (4 pi / 3) rc^3')],'Dimensionless','Interaction-volume overlap measure')
    panel(axs[2],[('nominal_spacing','Nominal spacing n^(-1/3)'),('nn_mean','Measured mean nearest neighbor')],
          'Length','Nominal versus measured spacing')
    save(fig,'04_density_and_spacing.png')

    fig,axs=plt.subplots(2,1,figsize=(10,7))
    panel(axs[0],[('spacing_over_dem_radius','Nominal / DEM radius'),
                  ('nn_over_dem_radius','Nearest neighbor / DEM radius')],
          'Dimensionless','Spacing relative to DEM radius')
    panel(axs[1],[('spacing_over_dpd_cutoff','Nominal / DPD cutoff'),
                  ('nn_over_dpd_cutoff','Nearest neighbor / DPD cutoff')],
          'Dimensionless','Spacing relative to DPD-DPD cutoff')
    save(fig,'05_spacing_ratios.png')

    profile_path=src/'radial_profiles.csv'
    if profile_path.exists():
        prof=pd.read_csv(profile_path)
        prof=prof[prof.time.isin(df.time)]
        if not prof.empty:
            prof['gap_center']=(prof.d_low+prof.d_high)/2
            fig,axs=plt.subplots(1,2,figsize=(12,5))
            all_region=prof[prof.region=='all']
            g=all_region.groupby('radial_bin').agg(
                gap=('gap_center','mean'), density=('full_shell_number_density','mean'),
                std=('full_shell_number_density','std'))
            axs[0].plot(g.gap,g.density,label='Frame mean')
            spread=g['std'].fillna(0)
            axs[0].fill_between(g.gap,np.maximum(0,g.density-spread),g.density+spread,
                                alpha=.2,label='Across-frame +/- 1 SD')
            axs[0].set(xlabel='Surface gap d',ylabel='Particles / shell volume',
                       title='Radial number density')
            axs[0].legend(fontsize=8)
            for region,label in [('all','All angles'),('equator','Equator'),
                                 ('positive_axis_cap','Positive-axis cap'),
                                 ('negative_axis_cap','Negative-axis cap')]:
                q=prof[prof.region==region].copy()
                q=q[np.isfinite(q.slip_rms) & (q['count']>0)]
                if q.empty:
                    continue
                q['weighted_slip2']=q['count']*q.slip_rms**2
                h=q.groupby('radial_bin').agg(gap=('gap_center','mean'),
                         numerator=('weighted_slip2','sum'),denominator=('count','sum'))
                axs[1].plot(h.gap,np.sqrt(h.numerator/h.denominator),label=label)
            axs[1].set(xlabel='Surface gap d',ylabel='Tangential relative-speed RMS',
                       title='Pooled particle-sample RMS by region')
            if axs[1].lines:
                axs[1].legend(fontsize=8)
            save(fig,'06_radial_profiles.png')

            # Missing bins stay NaN; zero occupancy is distinct from zero slip.
            matrix=all_region.pivot(index='radial_bin',columns='time',values='slip_rms').sort_index().sort_index(axis=1)
            if np.isfinite(matrix.to_numpy()).any():
                times=matrix.columns.to_numpy(float)
                bins=matrix.index.to_numpy()
                centers=all_region.groupby('radial_bin').gap_center.mean().reindex(bins).to_numpy()
                def edges(x, fallback):
                    if len(x)==1:
                        return np.array([x[0]-fallback/2,x[0]+fallback/2])
                    return np.r_[x[0]-(x[1]-x[0])/2,(x[:-1]+x[1:])/2,x[-1]+(x[-1]-x[-2])/2]
                width=float((all_region.d_high-all_region.d_low).iloc[0])
                fig,ax=plt.subplots(figsize=(10,5))
                cmap=plt.get_cmap('viridis').copy();cmap.set_bad('lightgray')
                im=ax.pcolormesh(edges(times,1.),edges(centers,width),
                     np.ma.masked_invalid(matrix.to_numpy()),shading='flat',cmap=cmap)
                ax.grid(False)
                ax.set(xlabel='Simulation time',ylabel='Surface gap d',
                       title='Tangential RMS versus time and gap (gray = no samples)')
                fig.colorbar(im,ax=ax,label='Tangential relative-speed RMS')
                save(fig,'07_slip_time_gap_map.png')

    note = ('Plots use the selected saved frames without smoothing.\n'
            'CSV values and physical units are unchanged.\n'
            'Radial density is an equal-frame mean; shading is across-frame SD.\n'
            'Radial slip curves pool particle samples, weighting squared RMS by count.\n'
            'These are finite-gap slip proxies, not extrapolated surface slip lengths.\n'
            'Nearest-neighbor curves require --nearest-neighbors during CSV generation.\n'
            'Heatmap cells represent snapshot samples, not resolved between-frame dynamics.\n')
    (out/'plot_notes.txt').write_text(note)
    print(f'Saved {len(written)} PNG figures to {out}')
    for name in written:
        print(name)


if __name__=='__main__':
    main()