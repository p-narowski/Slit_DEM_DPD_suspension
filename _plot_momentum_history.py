#!/usr/bin/env python3

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_history(csv_file_name):
    times = []

    total_px = []
    total_py = []
    total_pz = []
    total_pnorm = []

    dpd_px = []
    dpd_py = []
    dpd_pz = []
    dpd_pnorm = []

    suspended_px = []
    suspended_py = []
    suspended_pz = []
    suspended_pnorm = []

    residual_x = []
    residual_y = []
    residual_z = []
    residual_norm = []

    with open(csv_file_name, "r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        required_columns = [
            "time",
            "total_px",
            "total_py",
            "total_pz",
            "total_pnorm",
            "dpd_px",
            "dpd_py",
            "dpd_pz",
            "dpd_pnorm",
            "suspended_px",
            "suspended_py",
            "suspended_pz",
            "suspended_pnorm",
            "momentum_residual_x",
            "momentum_residual_y",
            "momentum_residual_z",
            "momentum_residual_norm"
        ]

        if reader.fieldnames is None:
            raise RuntimeError(
                "The CSV file has no header row: {}".format(
                    csv_file_name))

        missing_columns = [
            column for column in required_columns
            if column not in reader.fieldnames
        ]

        if missing_columns:
            raise RuntimeError(
                "CSV is missing required columns:\n  {}\n\n"
                "Available columns:\n  {}".format(
                    "\n  ".join(missing_columns),
                    "\n  ".join(reader.fieldnames)))

        for row_number, row in enumerate(reader, start=2):
            try:
                times.append(float(row["time"]))

                total_px.append(float(row["total_px"]))
                total_py.append(float(row["total_py"]))
                total_pz.append(float(row["total_pz"]))
                total_pnorm.append(float(row["total_pnorm"]))

                dpd_px.append(float(row["dpd_px"]))
                dpd_py.append(float(row["dpd_py"]))
                dpd_pz.append(float(row["dpd_pz"]))
                dpd_pnorm.append(float(row["dpd_pnorm"]))

                suspended_px.append(float(row["suspended_px"]))
                suspended_py.append(float(row["suspended_py"]))
                suspended_pz.append(float(row["suspended_pz"]))
                suspended_pnorm.append(
                    float(row["suspended_pnorm"]))

                residual_x.append(
                    float(row["momentum_residual_x"]))

                residual_y.append(
                    float(row["momentum_residual_y"]))

                residual_z.append(
                    float(row["momentum_residual_z"]))

                residual_norm.append(
                    float(row["momentum_residual_norm"]))

            except (TypeError, ValueError) as error:
                print(
                    "Skipping invalid CSV row {}: {}".format(
                        row_number,
                        error),
                    file=sys.stderr)

    if len(times) == 0:
        raise RuntimeError(
            "No valid data rows found in {}".format(
                csv_file_name))

    return {
        "time": times,

        "total_px": total_px,
        "total_py": total_py,
        "total_pz": total_pz,
        "total_pnorm": total_pnorm,

        "dpd_px": dpd_px,
        "dpd_py": dpd_py,
        "dpd_pz": dpd_pz,
        "dpd_pnorm": dpd_pnorm,

        "suspended_px": suspended_px,
        "suspended_py": suspended_py,
        "suspended_pz": suspended_pz,
        "suspended_pnorm": suspended_pnorm,

        "residual_x": residual_x,
        "residual_y": residual_y,
        "residual_z": residual_z,
        "residual_norm": residual_norm
    }


def create_plot(data, output_file_name):
    figure, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(11, 11),
        sharex=True,
        constrained_layout=True)

    time = data["time"]

    axes[0].plot(
        time,
        data["total_px"],
        label=r"Total $P_x$",
        color="tab:blue",
        linewidth=1.5)

    axes[0].plot(
        time,
        data["total_py"],
        label=r"Total $P_y$",
        color="tab:orange",
        linewidth=1.5)

    axes[0].plot(
        time,
        data["total_pz"],
        label=r"Total $P_z$",
        color="tab:green",
        linewidth=1.5)

    axes[0].plot(
        time,
        data["total_pnorm"],
        label=r"Total $\|\mathbf{P}\|$",
        color="black",
        linestyle="--",
        linewidth=1.4)

    axes[0].set_ylabel("Total momentum")
    axes[0].set_title("Total particle momentum")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(
        time,
        data["suspended_px"],
        label=r"Suspended $P_x$",
        color="tab:red",
        linewidth=1.5)

    axes[1].plot(
        time,
        data["dpd_px"],
        label=r"DPD $P_x$",
        color="tab:purple",
        linewidth=1.5)

    axes[1].plot(
        time,
        data["total_px"],
        label=r"Total $P_x$",
        color="black",
        linestyle="--",
        linewidth=1.2)

    axes[1].set_ylabel(r"$x$-momentum")
    axes[1].set_title(
        "Momentum transfer between suspended and DPD phases")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    axes[2].plot(
        time,
        data["residual_x"],
        label=r"Residual $R_x$",
        color="tab:red",
        linewidth=1.3)

    axes[2].plot(
        time,
        data["residual_y"],
        label=r"Residual $R_y$",
        color="tab:orange",
        linewidth=1.3)

    axes[2].plot(
        time,
        data["residual_z"],
        label=r"Residual $R_z$",
        color="tab:green",
        linewidth=1.3)

    axes[2].plot(
        time,
        data["residual_norm"],
        label=r"$\|\mathbf{R}\|$",
        color="black",
        linestyle="--",
        linewidth=1.3)

    axes[2].set_xlabel("Simulation time")
    axes[2].set_ylabel("Momentum residual")
    axes[2].set_title(
        r"$\mathbf{R}(t)=\mathbf{P}(t)-\mathbf{P}(0)"
        r"-\int\mathbf{F}_{\mathrm{external}}dt$")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best")

    figure.savefig(
        output_file_name,
        dpi=200)

    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot total, DPD-phase, and suspended-phase momentum "
            "from a Kratos momentum-history CSV file."))

    parser.add_argument(
        "csv_file",
        nargs="?",
        default="momentum_history.csv",
        help=(
            "Momentum CSV file. "
            "Default: momentum_history.csv"))

    parser.add_argument(
        "-o",
        "--output",
        default="total_momentum_vs_time.png",
        help=(
            "Output PNG file. "
            "Default: total_momentum_vs_time.png"))

    arguments = parser.parse_args()

    if not os.path.isfile(arguments.csv_file):
        raise FileNotFoundError(
            "CSV file not found: {}".format(
                arguments.csv_file))

    data = read_history(arguments.csv_file)

    create_plot(data, arguments.output)

    print(
        "Saved momentum plot with {} samples to:\n{}".format(
            len(data["time"]),
            os.path.abspath(arguments.output)))


if __name__ == "__main__":
    main()