#!/usr/bin/env python3

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def trapezoidal_integral(values, time):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(values, time)
    return np.trapz(values, time)


def read_momentum_history(csv_file_name):
    required_columns = [
        "time",
        "P_total_x",
        "P_total_y",
        "P_total_z",
    ]
    rows = []

    with open(csv_file_name, "r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        if reader.fieldnames is None:
            raise RuntimeError(
                "CSV file has no header row: {}".format(csv_file_name)
            )

        missing_columns = [
            column
            for column in required_columns
            if column not in reader.fieldnames
        ]

        if missing_columns:
            raise RuntimeError(
                "CSV file is missing required columns:\n"
                "  {}\n\n"
                "Available columns:\n"
                "  {}".format(
                    "\n  ".join(missing_columns),
                    "\n  ".join(reader.fieldnames),
                )
            )

        for row_number, row in enumerate(reader, start=2):
            try:
                rows.append(
                    (
                        float(row["time"]),
                        float(row["P_total_x"]),
                        float(row["P_total_y"]),
                        float(row["P_total_z"]),
                    )
                )
            except (TypeError, ValueError) as error:
                print(
                    "Skipping invalid CSV row {}: {}".format(
                        row_number,
                        error,
                    ),
                    file=sys.stderr,
                )

    if len(rows) < 2:
        raise RuntimeError(
            "At least two valid CSV rows are required for analysis."
        )

    rows.sort(key=lambda values: values[0])

    time = np.asarray([row[0] for row in rows], dtype=float)
    px = np.asarray([row[1] for row in rows], dtype=float)
    py = np.asarray([row[2] for row in rows], dtype=float)
    pz = np.asarray([row[3] for row in rows], dtype=float)

    unique_time_mask = np.concatenate(
        (np.asarray([True]), np.diff(time) > 0.0)
    )

    time = time[unique_time_mask]
    px = px[unique_time_mask]
    py = py[unique_time_mask]
    pz = pz[unique_time_mask]

    if len(time) < 2:
        raise RuntimeError(
            "At least two distinct time values are required for analysis."
        )

    return time, px, py, pz


def cumulative_trapezoidal_integral(time, values):
    integral = np.zeros_like(values, dtype=float)
    increments = 0.5 * (values[:-1] + values[1:]) * np.diff(time)
    integral[1:] = np.cumsum(increments)
    return integral


def running_time_average(time, values, cumulative_integral):
    duration = time - time[0]
    average = np.empty_like(values, dtype=float)
    average[0] = values[0]

    valid_mask = duration > 0.0
    average[valid_mask] = (
        cumulative_integral[valid_mask] / duration[valid_mask]
    )

    return average


def numerical_derivative(time, values):
    return np.gradient(values, time)


def component_analysis(time, values):
    momentum_integral = cumulative_trapezoidal_integral(time, values)
    running_average = running_time_average(
        time,
        values,
        momentum_integral,
    )
    average_derivative = numerical_derivative(time, running_average)
    derivative_integral = trapezoidal_integral(
        average_derivative,
        time,
    )

    duration = time[-1] - time[0]
    final_average = (
        momentum_integral[-1] / duration
        if duration > 0.0
        else values[-1]
    )

    return {
        "momentum": values,
        "momentum_integral": momentum_integral,
        "running_average": running_average,
        "average_derivative": average_derivative,
        "derivative_integral": derivative_integral,
        "final_average": final_average,
        "momentum_change": values[-1] - values[0],
        "final_derivative": average_derivative[-1],
    }


def create_plot(time, analyses, output_file_name):
    figure, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(18, 12),
        sharex="col",
        constrained_layout=True,
    )

    components = [
        ("x", r"P_x", "tab:blue"),
        ("y", r"P_y", "tab:orange"),
        ("z", r"P_z", "tab:green"),
    ]

    for column, (component, label, color) in enumerate(components):
        analysis = analyses[component]

        axes[0, column].plot(
            time,
            analysis["momentum"],
            color=color,
            linewidth=1.4,
        )
        axes[0, column].set_title(
            "A: Total ${}(t)$".format(label)
        )
        axes[0, column].set_ylabel("Momentum")
        axes[0, column].grid(True, alpha=0.3)

        axes[1, column].plot(
            time,
            analysis["running_average"],
            color=color,
            linewidth=1.4,
        )
        axes[1, column].set_title(
            r"B: Running average $\overline{%s}(t)$" % label
        )
        axes[1, column].set_ylabel("Average momentum")
        axes[1, column].grid(True, alpha=0.3)

        axes[2, column].plot(
            time,
            analysis["average_derivative"],
            color=color,
            linewidth=1.2,
        )
        axes[2, column].axhline(
            0.0,
            color="black",
            linestyle="--",
            linewidth=0.9,
        )
        axes[2, column].set_title(
            r"C: $d\overline{%s}/dt$" % label
        )
        axes[2, column].set_xlabel("Simulation time")
        axes[2, column].set_ylabel("Average-momentum rate")
        axes[2, column].grid(True, alpha=0.3)

    figure.suptitle(
        "Total momentum components, running averages, and rates",
        fontsize=16,
    )

    figure.savefig(output_file_name, dpi=220)
    plt.close(figure)


def write_summary(summary_file_name, time, analyses):
    duration = time[-1] - time[0]

    lines = [
        "Total momentum component analysis",
        "=================================",
        "",
        "Number of samples: {}".format(len(time)),
        "Start time: {:.16g}".format(time[0]),
        "End time: {:.16g}".format(time[-1]),
        "Simulation duration: {:.16g}".format(duration),
        "",
        "D: Integrals over simulation time",
        "",
    ]

    for component in ("x", "y", "z"):
        analysis = analyses[component]
        component_upper = component.upper()

        lines.extend(
            [
                "P_{} component".format(component_upper),
                "  Integral P_{} dt = {:.16g}".format(
                    component_upper,
                    analysis["momentum_integral"][-1],
                ),
                "  Final running average P_{} = {:.16g}".format(
                    component_upper,
                    analysis["final_average"],
                ),
                "  Integral d(average P_{})/dt dt = {:.16g}".format(
                    component_upper,
                    analysis["derivative_integral"],
                ),
                "  Change in average P_{} = {:.16g}".format(
                    component_upper,
                    analysis["running_average"][-1]
                    - analysis["running_average"][0],
                ),
                "  Direct change P_{}(end)-P_{}(start) = {:.16g}".format(
                    component_upper,
                    component_upper,
                    analysis["momentum_change"],
                ),
                "  Final d(average P_{})/dt = {:.16g}".format(
                    component_upper,
                    analysis["final_derivative"],
                ),
                "",
            ]
        )

    text = "\n".join(lines)

    with open(summary_file_name, "w", encoding="utf-8") as summary_file:
        summary_file.write(text)

    return text


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot total Px, Py, and Pz in separate panels; calculate "
            "running averages, their time derivatives, and time integrals."
        )
    )

    parser.add_argument(
        "csv_file",
        nargs="?",
        default="momentum_history.csv",
        help="Input CSV file. Default: momentum_history.csv",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="total_momentum_component_analysis.png",
        help=(
            "Output figure file. "
            "Default: total_momentum_component_analysis.png"
        ),
    )

    parser.add_argument(
        "-s",
        "--summary",
        default="total_momentum_component_analysis.txt",
        help=(
            "Output text summary file. "
            "Default: total_momentum_component_analysis.txt"
        ),
    )

    arguments = parser.parse_args()

    if not os.path.isfile(arguments.csv_file):
        raise FileNotFoundError(
            "CSV file not found: {}".format(arguments.csv_file)
        )

    time, px, py, pz = read_momentum_history(arguments.csv_file)

    analyses = {
        "x": component_analysis(time, px),
        "y": component_analysis(time, py),
        "z": component_analysis(time, pz),
    }

    create_plot(time, analyses, arguments.output)

    summary = write_summary(
        arguments.summary,
        time,
        analyses,
    )

    print(summary)
    print("\nSaved figure to:\n{}".format(
        os.path.abspath(arguments.output)
    ))
    print("\nSaved text summary to:\n{}".format(
        os.path.abspath(arguments.summary)
    ))


if __name__ == "__main__":
    main()
