#!/usr/bin/env python3
"""Fit a constant-drive quadratic-drag model to a comparison CSV.

Default: fit relative displacement, using the measured initial relative velocity.
Requires numpy, pandas, scipy, matplotlib. Does not alter the input CSV.
The fit is empirical when the background fluid accelerates.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def quadratic(t, g, vt, v0):
    q = g * np.asarray(t) / vt
    h = np.tanh(q)
    velocity = vt * (v0 + vt * h) / (vt + v0 * h)
    # Stable log(cosh(q)); no overflow for large positive q.
    log_cosh = np.abs(q) + np.log1p(np.exp(-2 * np.abs(q))) - np.log(2.)
    displacement = vt**2 / g * (log_cosh + np.log1p((v0 / vt) * h))
    return velocity, displacement


def linear(t, g, vt, v0):
    tau = vt / g
    decay_integral = -np.expm1(-np.asarray(t) / tau)
    velocity = vt + (v0 - vt) * np.exp(-np.asarray(t) / tau)
    displacement = vt * np.asarray(t) + (v0 - vt) * tau * decay_integral
    return velocity, displacement


def fit_model(model, t, measured, v0, fit_to, speed_scale):
    duration = t[-1]
    slope_scale = max(speed_scale / duration, 1e-12)
    lower = np.array([slope_scale * 1e-6, speed_scale * 1e-3])
    upper = np.array([slope_scale * 1e4, speed_scale * 1e4])
    scale = max(np.ptp(measured), np.max(np.abs(measured)), 1e-12)
    index = 1 if fit_to == 'position' else 0
    def residual(log_parameters):
        g, vt = np.exp(log_parameters)
        return (model(t, g, vt, v0)[index] - measured) / scale
    best = None
    for factor in (0.8, 1.5, 4., 20., 200.):
        start = np.clip([slope_scale, factor * speed_scale], lower*1.01, upper/1.01)
        result = least_squares(residual, np.log(start),
                               bounds=(np.log(lower), np.log(upper)),
                               max_nfev=6000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        if result.success and (best is None or result.cost < best.cost):
            best = result
    if best is None:
        raise RuntimeError('Parameter optimization failed for all starting guesses.')
    g, vt = np.exp(best.x)
    predicted_v, predicted_s = model(t, g, vt, v0)
    predicted = predicted_s if fit_to == 'position' else predicted_v
    errors = predicted - measured
    sse = float(errors @ errors)
    total = float(np.sum((measured - measured.mean())**2))
    singular = np.linalg.svd(best.jac, compute_uv=False)
    condition = float(singular[0]/singular[-1]) if singular[-1] > 0 else float('inf')
    report = dict(g_prime=float(g), v_terminal=float(vt), characteristic_time=float(vt/g),
                  rmse=float(np.sqrt(np.mean(errors**2))),
                  r_squared=float(1-sse/total) if total > 0 else None,
                  jacobian_condition=condition,
                  near_parameter_bound=bool(np.any(np.minimum(best.x-np.log(lower),
                                                               np.log(upper)-best.x) < .05)))
    return predicted_v, predicted_s, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', default='stokes_relative_velocity_comparison.csv')
    p.add_argument('--out-prefix', default='quadratic_drag_fit')
    p.add_argument('--fit-to', choices=['position', 'velocity'], default='position')
    p.add_argument('--frame', choices=['relative', 'lab'], default='relative')
    p.add_argument('--t-start', type=float, default=None, help='Input CSV time, not elapsed time.')
    p.add_argument('--t-end', type=float, default=None)
    p.add_argument('--from-rest', action='store_true', help='Force initial velocity to zero.')
    p.add_argument('--position-column', help='Verified unwrapped solid x displacement/position.')
    a = p.parse_args()
    d = pd.read_csv(a.csv)
    d.columns = d.columns.str.strip()
    required = ['time', 'vx', 'v_fluid_x']
    missing = [c for c in required if c not in d]
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    d = d.sort_values('time').reset_index(drop=True)
    if a.t_start is not None:
        d = d.loc[d.time >= a.t_start]
    if a.t_end is not None:
        d = d.loc[d.time <= a.t_end]
    d = d.reset_index(drop=True)
    if len(d) < 6:
        raise ValueError('Need at least six frames in the selected window.')
    if not np.isfinite(d[required].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite time or velocity values.')
    t = d.time.to_numpy(dtype=float)
    t = t - t[0]
    if np.any(np.diff(t) <= 0):
        raise ValueError('CSV times must be unique and increasing.')
    vp = d.vx.to_numpy(dtype=float)
    uf = d.v_fluid_x.to_numpy(dtype=float)
    w = vp - uf
    d['v_relative_sim'] = w
    velocity = w if a.frame == 'relative' else vp
    if np.min(velocity) < 0:
        raise ValueError('This implementation expects nonnegative motion. Choose an appropriate window/direction.')
    position_column = a.position_column
    if position_column is None:
        position_column = next((c for c in ['x_sim_unwrapped', 's_unwrapped'] if c in d), None)
    if position_column:
        if position_column not in d:
            raise ValueError(f'No position column {position_column!r}.')
        solid_position = d[position_column].to_numpy(dtype=float)
        if not np.isfinite(solid_position).all():
            raise ValueError('Nonfinite position values.')
        solid_position = solid_position - solid_position[0]
        position_source = position_column
    else:
        solid_position = cumulative_trapezoid(vp, t, initial=0)
        position_source = 'integrated vx (no independent position column)'
    measured_position = solid_position.copy()
    if a.frame == 'relative':
        measured_position -= cumulative_trapezoid(uf, t, initial=0)
    v0 = 0. if a.from_rest else float(velocity[0])
    measured = measured_position if a.fit_to == 'position' else velocity
    speed_scale = max(float(np.max(np.abs(velocity))), 1e-10)
    qv, qs, qr = fit_model(quadratic, t, measured, v0, a.fit_to, speed_scale)
    lv, ls, lr = fit_model(linear, t, measured, v0, a.fit_to, speed_scale)
    qr['observed_fraction_of_terminal_speed'] = float(np.max(velocity)/qr['v_terminal'])
    notes = ['Fitted terminal speed is an extrapolation, not an observed plateau.',
             'A better fit does not establish high Reynolds number or identify added mass.',
             'For accelerating fluid, relative-frame fit parameters are effective, not exact force-law constants.']
    if qr['observed_fraction_of_terminal_speed'] < .8:
        notes.append('No near-terminal data: terminal speed may be weakly constrained; test longer and alternative fit windows.')
    if qr['near_parameter_bound'] or qr['jacobian_condition'] > 1e4:
        notes.append('WARNING: poorly conditioned or bound-limited quadratic fit; do not interpret terminal speed physically.')
    if a.from_rest and abs(velocity[0]) > .01*speed_scale:
        notes.append('WARNING: rest initial condition conflicts with the first measured velocity.')
    report = dict(input_csv=str(a.csv), fit_to=a.fit_to, frame=a.frame,
                  initial_velocity=v0, position_source=position_source, n_frames=len(d),
                  time_start=float(d.time.iloc[0]), time_end=float(d.time.iloc[-1]),
                  quadratic=qr, fitted_linear_comparison=lr, notes=notes)
    prefix = Path(a.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    d['elapsed_time_fit'] = t
    d['fit_observed_velocity'] = velocity
    d['fit_observed_displacement'] = measured_position
    d['v_quadratic_fit'] = qv
    d['s_quadratic_fit'] = qs
    d['v_linear_fit'] = lv
    d['s_linear_fit'] = ls
    d.to_csv(str(prefix)+'.csv', index=False)
    Path(str(prefix)+'_parameters.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(t, vp, color='tab:blue', lw=1.8, label='DEM laboratory velocity')
    ax.plot(t, uf, color='tab:orange', lw=1.8, label='DPD bulk velocity')
    ax.plot(t, w, color='tab:green', lw=2.2, label='Simulation relative velocity')
    if 'v_stokes_relative' in d:
        ax.plot(t, d.v_stokes_relative, 'k--', lw=2, label='Original stationary-fluid Stokes baseline')
    if 'stokes_terminal_velocity' in d:
        ax.axhline(float(d.stokes_terminal_velocity.iloc[0]), color='red', ls=':', lw=1.8,
                   label='Original Stokes terminal reference')
    ax.plot(t, qv, color='red', ls='--', lw=2.4,
            label=f"Quadratic fit ({a.frame}): g'={qr['g_prime']:.5g}, vt={qr['v_terminal']:.5g}")
    ax.set(xlabel='Time since first selected output', ylabel='Velocity in forcing direction',
           title='DEM-DPD velocity comparison with fitted quadratic-drag model')
    ax.grid(alpha=.25)
    ax.legend(fontsize=9, loc='best')
    fig.tight_layout()
    fig.savefig(str(prefix)+'.png', dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, measured_position, color='tab:green', lw=2, label=f'Simulation {a.frame} displacement')
    axes[0].plot(t, qs, 'r--', lw=2, label='Quadratic-drag fit')
    axes[0].plot(t, ls, color='gray', ls='-.', label='Two-parameter fitted linear-drag model')
    axes[0].set_ylabel('Displacement from selected initial state')
    axes[0].legend()
    axes[1].plot(t, velocity, color='tab:green', lw=2, label='Simulation velocity')
    axes[1].plot(t, qv, 'r--', lw=2, label='Quadratic fit velocity')
    axes[1].plot(t, lv, color='gray', ls='-.', label='Fitted linear model velocity')
    axes[1].set(xlabel='Time since first selected output', ylabel='Velocity')
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(str(prefix)+'_position.png', dpi=200)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()