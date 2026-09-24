#!/usr/bin/env python3

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COMPONENTS = ("x", "y", "z")


def vector_norm(data_frame, prefix):
    columns = [prefix + "_" + component for component in COMPONENTS]
    return np.sqrt(sum(data_frame[column].to_numpy() ** 2 for column in columns))


def component_direction(external_force):
    norm = np.linalg.norm(external_force)

    if norm == 0.0:
        return np.array([1.0, 0.0, 0.0])

    return external_force / norm


def projected_quantity(data_frame, prefix, direction):
    return sum(
        direction[index] * data_frame[prefix + "_" + component].to_numpy()
        for index, component in enumerate(COMPONENTS)
    )


def save_figure(file_name):
    plt.tight_layout()
    plt.savefig(file_name, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot DEM-DPD momentum-conservation diagnostics."
    )
    parser.add_argument(
        "--input",
        default="momentum_history.csv",
        help="Diagnostic CSV written by MomentumConservationTracker.",
    )
    parser.add_argument(
        "--output-dir",
        default="momentum_diagnostics",
        help="Directory for figures and text summary.",
    )
    parser.add_argument(
        "--force-tolerance",
        type=float,
        default=1.0e-10,
        help="Absolute warning tolerance for coupling residual norm.",
    )
    parser.add_argument(
        "--balance-tolerance",
        type=float,
        default=1.0e-8,
        help="Absolute warning tolerance for dP/dt - F_expected norm.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data = pd.read_csv(args.input)
    data = data.sort_values("time").reset_index(drop=True)

    if len(data) < 2:
        raise RuntimeError(
            "At least two diagnostic rows are required to evaluate dP/dt."
        )

    required_columns = [
        "time",
        "P_dem_x",
        "P_dem_y",
        "P_dem_z",
        "P_dpd_x",
        "P_dpd_y",
        "P_dpd_z",
        "P_total_x",
        "P_total_y",
        "P_total_z",
        "F_external_x",
        "F_external_y",
        "F_external_z",
        "F_wall_x",
        "F_wall_y",
        "F_wall_z",
        "F_dpd_to_dem_x",
        "F_dpd_to_dem_y",
        "F_dpd_to_dem_z",
        "F_dem_to_dpd_x",
        "F_dem_to_dpd_y",
        "F_dem_to_dpd_z",
        "F_coupling_residual_x",
        "F_coupling_residual_y",
        "F_coupling_residual_z",
        "F_coupling_residual_norm",
        "dP_total_dt_x",
        "dP_total_dt_y",
        "dP_total_dt_z",
        "F_expected_x",
        "F_expected_y",
        "F_expected_z",
        "momentum_balance_error_x",
        "momentum_balance_error_y",
        "momentum_balance_error_z",
        "momentum_balance_error_norm",
    ]

    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise RuntimeError(
            "Missing required columns in {}:\n{}".format(
                args.input,
                "\n".join(missing),
            )
        )

    time = data["time"].to_numpy()

    external_force_mean = np.array(
        [
            data["F_external_" + component].mean()
            for component in COMPONENTS
        ]
    )

    direction = component_direction(external_force_mean)

    p_dem_parallel = projected_quantity(data, "P_dem", direction)
    p_dpd_parallel = projected_quantity(data, "P_dpd", direction)
    p_total_parallel = projected_quantity(data, "P_total", direction)
    f_external_parallel = projected_quantity(data, "F_external", direction)
    f_wall_parallel = projected_quantity(data, "F_wall", direction)
    f_expected_parallel = projected_quantity(data, "F_expected", direction)
    dpdt_parallel = projected_quantity(data, "dP_total_dt", direction)

    f_dpd_to_dem_parallel = projected_quantity(
        data,
        "F_dpd_to_dem",
        direction,
    )

    f_dem_to_dpd_parallel = projected_quantity(
        data,
        "F_dem_to_dpd",
        direction,
    )

    residual_parallel = projected_quantity(
        data,
        "F_coupling_residual",
        direction,
    )

    residual_norm = data["F_coupling_residual_norm"].to_numpy()
    balance_error_norm = data["momentum_balance_error_norm"].to_numpy()

    valid_rate = np.isfinite(dpdt_parallel)
    valid_balance = np.isfinite(balance_error_norm)

    mean_expected_force = np.nanmean(f_expected_parallel)

    reference_momentum = (
        p_total_parallel[0]
        + mean_expected_force * (time - time[0])
    )

    # 1. Momentum: this figure intentionally plots momentum, not velocity.
    plt.figure(figsize=(10, 6))
    plt.plot(
        time,
        p_dem_parallel,
        label=r"$P_{\mathrm{DEM},\parallel}$",
        linewidth=1.6,
    )
    plt.plot(
        time,
        p_dpd_parallel,
        label=r"$P_{\mathrm{DPD},\parallel}$",
        linewidth=1.6,
    )
    plt.plot(
        time,
        p_total_parallel,
        label=r"$P_{\mathrm{total},\parallel}$",
        linewidth=2.2,
        color="black",
    )
    plt.plot(
        time,
        reference_momentum,
        "--",
        label=(
            r"$P_{\mathrm{total}}(0)"
            r"+ \langle F_{\mathrm{expected},\parallel}\rangle t$"
        ),
        linewidth=1.5,
    )
    plt.xlabel("Time")
    plt.ylabel("Momentum in forcing direction")
    plt.title("DEM–DPD total momentum evolution")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_figure(
        os.path.join(args.output_dir, "01_total_momentum_vs_time.png")
    )

    # 2. Both coupling-force totals, with the expected equal-and-opposite relation.
    plt.figure(figsize=(10, 6))
    plt.plot(
        time,
        f_dpd_to_dem_parallel,
        label=r"$F_{\mathrm{DPD}\rightarrow\mathrm{DEM},\parallel}$",
        linewidth=1.6,
    )
    plt.plot(
        time,
        f_dem_to_dpd_parallel,
        label=r"$F_{\mathrm{DEM}\rightarrow\mathrm{DPD},\parallel}$",
        linewidth=1.6,
    )
    plt.plot(
        time,
        residual_parallel,
        label=r"$R_{\mathrm{coupling},\parallel}$",
        linewidth=1.8,
        color="black",
    )
    plt.axhline(0.0, color="gray", linewidth=1.0)
    plt.xlabel("Time")
    plt.ylabel("Force in forcing direction")
    plt.title("DEM–DPD coupling action–reaction force balance")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_figure(
        os.path.join(args.output_dir, "02_coupling_force_balance.png")
    )

    # 3. Vector residual. Add machine epsilon only to permit log display.
    plt.figure(figsize=(10, 6))
    residual_for_log = np.maximum(residual_norm, np.finfo(float).tiny)
    plt.semilogy(
        time,
        residual_for_log,
        label=r"$\left\|\mathbf{F}_{\mathrm{DPD}\to\mathrm{DEM}}"
        r"+\mathbf{F}_{\mathrm{DEM}\to\mathrm{DPD}}\right\|$",
        linewidth=1.8,
    )
    plt.axhline(
        args.force_tolerance,
        color="red",
        linestyle="--",
        label="Force-residual tolerance",
    )
    plt.xlabel("Time")
    plt.ylabel("Coupling residual norm")
    plt.title("Per-timestep DEM–DPD coupling residual")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    save_figure(
        os.path.join(args.output_dir, "03_coupling_residual.png")
    )

    # 4. Global rate balance: dP/dt against external + wall force.
    plt.figure(figsize=(10, 6))
    plt.plot(
        time[valid_rate],
        dpdt_parallel[valid_rate],
        label=r"$dP_{\mathrm{total},\parallel}/dt$",
        linewidth=1.8,
    )
    plt.plot(
        time,
        f_external_parallel,
        label=r"$F_{\mathrm{external},\parallel}$",
        linewidth=1.4,
    )
    plt.plot(
        time,
        f_wall_parallel,
        label=r"$F_{\mathrm{wall},\parallel}$",
        linewidth=1.4,
    )
    plt.plot(
        time,
        f_expected_parallel,
        "--",
        label=r"$F_{\mathrm{expected},\parallel}"
        r"=F_{\mathrm{external},\parallel}+F_{\mathrm{wall},\parallel}$",
        linewidth=1.8,
        color="black",
    )
    plt.xlabel("Time")
    plt.ylabel("Force / momentum rate in forcing direction")
    plt.title("Global momentum-rate balance")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_figure(
        os.path.join(args.output_dir, "04_global_momentum_balance.png")
    )

    time_span = time[-1] - time[0]

    if time_span > 0.0:
        fitted_slope = np.polyfit(time, p_total_parallel, 1)[0]
    else:
        fitted_slope = float("nan")

    max_residual_norm = np.nanmax(residual_norm)
    rms_residual_norm = np.sqrt(np.nanmean(residual_norm ** 2))

    if np.any(valid_balance):
        max_balance_error = np.nanmax(balance_error_norm[valid_balance])
        rms_balance_error = np.sqrt(
            np.nanmean(balance_error_norm[valid_balance] ** 2)
        )
    else:
        max_balance_error = float("nan")
        rms_balance_error = float("nan")

    summary_path = os.path.join(
        args.output_dir,
        "momentum_diagnostic_summary.txt",
    )

    with open(summary_path, "w") as summary:
        summary.write("DEM-DPD momentum-conservation diagnostic summary\n")
        summary.write("===============================================\n\n")
        summary.write("Input CSV: {}\n".format(args.input))
        summary.write("Number of timesteps: {}\n".format(len(data)))
        summary.write(
            "Time interval: [{:.16e}, {:.16e}]\n\n".format(
                time[0],
                time[-1],
            )
        )

        summary.write("Forcing direction used for scalar projections:\n")
        summary.write(
            "  [{:.16e}, {:.16e}, {:.16e}]\n\n".format(
                direction[0],
                direction[1],
                direction[2],
            )
        )

        summary.write("Coupling action-reaction check\n")
        summary.write("------------------------------\n")
        summary.write(
            "max ||F_DPD->DEM + F_DEM->DPD|| = {:.16e}\n".format(
                max_residual_norm
            )
        )
        summary.write(
            "RMS ||F_DPD->DEM + F_DEM->DPD|| = {:.16e}\n".format(
                rms_residual_norm
            )
        )
        summary.write(
            "Force residual tolerance = {:.16e}\n".format(
                args.force_tolerance
            )
        )
        summary.write(
            "Force residual status = {}\n\n".format(
                "PASS"
                if max_residual_norm <= args.force_tolerance
                else "CHECK"
            )
        )

        summary.write("Global total-momentum check\n")
        summary.write("---------------------------\n")
        summary.write(
            "Fitted dP_total/dt in forcing direction = {:.16e}\n".format(
                fitted_slope
            )
        )
        summary.write(
            "Mean expected force in forcing direction = {:.16e}\n".format(
                mean_expected_force
            )
        )
        summary.write(
            "Slope minus expected force = {:.16e}\n\n".format(
                fitted_slope - mean_expected_force
            )
        )

        summary.write("Per-step momentum-rate balance\n")
        summary.write("------------------------------\n")
        summary.write(
            "max ||dP_total/dt - F_expected|| = {:.16e}\n".format(
                max_balance_error
            )
        )
        summary.write(
            "RMS ||dP_total/dt - F_expected|| = {:.16e}\n".format(
                rms_balance_error
            )
        )
        summary.write(
            "Momentum-rate tolerance = {:.16e}\n".format(
                args.balance_tolerance
            )
        )
        summary.write(
            "Momentum-rate status = {}\n".format(
                "PASS"
                if np.isfinite(max_balance_error)
                and max_balance_error <= args.balance_tolerance
                else "CHECK"
            )
        )

    print("Wrote diagnostic figures and summary to: {}".format(args.output_dir))
    print("Maximum coupling residual norm: {:.6e}".format(max_residual_norm))
    print(
        "Fitted momentum slope: {:.6e}; mean expected force: {:.6e}".format(
            fitted_slope,
            mean_expected_force,
        )
    )


if __name__ == "__main__":
    main()