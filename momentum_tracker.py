import csv
import math
import os

import KratosMultiphysics as KM
import KratosMultiphysics.DEMApplication as DEM


class MomentumTracker:
    """
    Momentum tracker for a single parent particle model part.

    Particle classification
    -----------------------
    DEM/suspended particles:
        RADIUS >= suspended_radius_threshold

    DPD/fluid particles:
        RADIUS < suspended_radius_threshold

    Current capability
    ------------------
    This tracker computes and writes:
        P_dem
        P_dpd
        P_total = P_dem + P_dpd
        dP_total/dt

    It intentionally writes NaN for DEM-DPD coupling-force fields until
    actual coupling-only force accumulators are supplied from the force-
    assembly implementation.

    Important:
    TOTAL_FORCES must not be used as a substitute for DEM-DPD coupling
    forces unless it is known to contain only that interaction. In this
    simulation, TOTAL_FORCES may also contain prescribed external force,
    wall force, gravity, DEM contact, or other terms.
    """

    _HEADER = [
        "step",
        "time",
        "dt",

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

    def __init__(
        self,
        particles_model_part,
        output_directory=".",
        suspended_radius_threshold=0.0,
        output_file_name="momentum_history.csv",
        append=False,
    ):
        """
        Parameters
        ----------
        particles_model_part
            Parent model part containing both DPD particles and suspended
            DEM particles. In the present case this is SpheresPart.

        output_directory : str
            Directory where the CSV file will be written.

        suspended_radius_threshold : float
            Nodes with RADIUS >= this value are treated as DEM/suspended
            particles. Nodes below this value are treated as DPD particles.

        output_file_name : str
            Name of output CSV file.

        append : bool
            If True, append to an existing CSV. Default False overwrites
            the old history and writes a fresh header.
        """
        self._file = None
        self._writer = None

        self.particles_model_part = particles_model_part
        self.suspended_radius_threshold = float(
            suspended_radius_threshold
        )

        self._previous_time = None
        self._previous_total_momentum = None

        if output_directory is None:
            output_directory = "."

        os.makedirs(output_directory, exist_ok=True)

        self.output_file_name = os.path.join(
            output_directory,
            output_file_name,
        )

        mode = "a" if append else "w"

        file_is_empty = (
            not os.path.exists(self.output_file_name)
            or os.path.getsize(self.output_file_name) == 0
        )

        self._file = open(
            self.output_file_name,
            mode,
            newline="",
        )

        self._writer = csv.DictWriter(
            self._file,
            fieldnames=self._HEADER,
        )

        if not append or file_is_empty:
            self._writer.writeheader()
            self._file.flush()

    @staticmethod
    def _zero():
        return [0.0, 0.0, 0.0]

    @staticmethod
    def _nan_vector():
        return [float("nan"), float("nan"), float("nan")]

    @staticmethod
    def _as_list(vector):
        return [
            float(vector[0]),
            float(vector[1]),
            float(vector[2]),
        ]

    @staticmethod
    def _add(a, b):
        return [
            a[0] + b[0],
            a[1] + b[1],
            a[2] + b[2],
        ]

    @staticmethod
    def _subtract(a, b):
        return [
            a[0] - b[0],
            a[1] - b[1],
            a[2] - b[2],
        ]

    @staticmethod
    def _scale(vector, scalar):
        return [
            scalar * vector[0],
            scalar * vector[1],
            scalar * vector[2],
        ]

    @staticmethod
    def _norm(vector):
        return math.sqrt(
            vector[0] * vector[0]
            + vector[1] * vector[1]
            + vector[2] * vector[2]
        )

    @staticmethod
    def _write_vector(row, prefix, vector):
        row[prefix + "_x"] = vector[0]
        row[prefix + "_y"] = vector[1]
        row[prefix + "_z"] = vector[2]

    def _is_dem_particle(self, node):
        return node.Is(DEM.DEMFlags.IS_SUSPENDED_PARTICLE)

    @staticmethod
    def _get_nodal_mass(node):
        """
        Obtain the mass used by the particle time integration.

        NODAL_MASS is expected for this DEM/DPD setup. The alternatives are
        retained to make failure messages clearer if a variable is stored
        non-historically instead.
        """
        if node.SolutionStepsDataHas(KM.NODAL_MASS):
            return float(
                node.GetSolutionStepValue(KM.NODAL_MASS)
            )

        if node.SolutionStepsDataHas(KM.MASS):
            return float(
                node.GetSolutionStepValue(KM.MASS)
            )

        if node.Has(KM.NODAL_MASS):
            return float(node.GetValue(KM.NODAL_MASS))

        if node.Has(KM.MASS):
            return float(node.GetValue(KM.MASS))

        raise RuntimeError(
            "Could not find NODAL_MASS or MASS on node {}. "
            "Update _get_nodal_mass() to use the mass variable used by "
            "your particle formulation.".format(node.Id)
        )

    @staticmethod
    def _get_nodal_velocity(node):
        if node.SolutionStepsDataHas(KM.VELOCITY):
            return MomentumTracker._as_list(
                node.GetSolutionStepValue(KM.VELOCITY)
            )

        if node.Has(KM.VELOCITY):
            return MomentumTracker._as_list(
                node.GetValue(KM.VELOCITY)
            )

        raise RuntimeError(
            "Could not find VELOCITY on node {}.".format(node.Id)
        )

    def _split_momentum(self):
        """
        Return P_dem and P_dpd by summing m*v over the relevant nodes.
        """
        p_dem = self._zero()
        p_dpd = self._zero()

        for node in self.particles_model_part.Nodes:
            mass = self._get_nodal_mass(node)
            velocity = self._get_nodal_velocity(node)
            particle_momentum = self._scale(velocity, mass)

            if self._is_dem_particle(node):
                p_dem = self._add(p_dem, particle_momentum)
            else:
                p_dpd = self._add(p_dpd, particle_momentum)

        return p_dem, p_dpd

    def _get_time(self):
        process_info = self.particles_model_part.ProcessInfo

        if not process_info.Has(KM.TIME):
            raise RuntimeError(
                "TIME was not found in particles_model_part.ProcessInfo."
            )

        return float(process_info[KM.TIME])

    def _get_step(self):
        process_info = self.particles_model_part.ProcessInfo

        if process_info.Has(KM.STEP):
            return int(process_info[KM.STEP])

        return -1

    def _sum_external_applied_force_on_dem(self):
        """
        Sum the prescribed external force stored on selected DEM nodes.

        This is meaningful for your current MainKratos.py because it sets
        EXTERNAL_APPLIED_FORCE on every radius-selected target particle.

        It does not include gravity unless gravity has separately been added
        into EXTERNAL_APPLIED_FORCE by your simulation setup.
        """
        total_external_force = self._zero()

        for node in self.particles_model_part.Nodes:
            if not self._is_dem_particle(node):
                continue

            if node.SolutionStepsDataHas(KM.EXTERNAL_APPLIED_FORCE):
                force = node.GetSolutionStepValue(
                    KM.EXTERNAL_APPLIED_FORCE
                )
                total_external_force = self._add(
                    total_external_force,
                    self._as_list(force),
                )

            elif node.Has(KM.EXTERNAL_APPLIED_FORCE):
                force = node.GetValue(KM.EXTERNAL_APPLIED_FORCE)
                total_external_force = self._add(
                    total_external_force,
                    self._as_list(force),
                )

        return total_external_force

    def record(
        self,
        dem_dpd_force_on_dem=None,
        dem_dpd_force_on_dpd=None,
        external_force_on_system=None,
        wall_force_on_system=None,
    ):
        """
        Write one row of momentum diagnostics.

        Parameters
        ----------
        dem_dpd_force_on_dem : length-3 sequence or None
            Sum of coupling-only forces exerted by DPD particles on DEM
            particles. Until a real coupling accumulator is connected,
            leave as None. The CSV then writes NaN.

        dem_dpd_force_on_dpd : length-3 sequence or None
            Sum of reaction coupling-only forces exerted by DEM particles on
            DPD particles. Until a real accumulator is connected, leave as
            None. The CSV then writes NaN.

        external_force_on_system : length-3 sequence or None
            Net known external force on the full DEM+DPD system. If None,
            this tracker sums EXTERNAL_APPLIED_FORCE on DEM-selected nodes.

        wall_force_on_system : length-3 sequence or None
            Net force applied by walls to DEM+DPD. Leave as None until it is
            explicitly measured; the corresponding global-balance fields
            will then be written as NaN.
        """
        time = self._get_time()
        step = self._get_step()

        p_dem, p_dpd = self._split_momentum()
        p_total = self._add(p_dem, p_dpd)

        if external_force_on_system is None:
            f_external = self._sum_external_applied_force_on_dem()
        else:
            f_external = self._as_list(external_force_on_system)

        wall_force_is_available = wall_force_on_system is not None

        if wall_force_is_available:
            f_wall = self._as_list(wall_force_on_system)
            f_expected = self._add(f_external, f_wall)
        else:
            f_wall = self._nan_vector()
            f_expected = self._nan_vector()

        coupling_forces_are_available = (
            dem_dpd_force_on_dem is not None
            and dem_dpd_force_on_dpd is not None
        )

        if coupling_forces_are_available:
            f_dpd_to_dem = self._as_list(dem_dpd_force_on_dem)
            f_dem_to_dpd = self._as_list(dem_dpd_force_on_dpd)

            f_coupling_residual = self._add(
                f_dpd_to_dem,
                f_dem_to_dpd,
            )

            f_coupling_residual_norm = self._norm(
                f_coupling_residual
            )
        else:
            f_dpd_to_dem = self._nan_vector()
            f_dem_to_dpd = self._nan_vector()
            f_coupling_residual = self._nan_vector()
            f_coupling_residual_norm = float("nan")

        if self._previous_time is None:
            dt = 0.0
            dp_total_dt = self._nan_vector()
            momentum_balance_error = self._nan_vector()
            momentum_balance_error_norm = float("nan")
        else:
            dt = time - self._previous_time

            if dt <= 0.0:
                raise RuntimeError(
                    "Non-positive diagnostic timestep: current time = {}, "
                    "previous time = {}.".format(
                        time,
                        self._previous_time,
                    )
                )

            dp_total_dt = self._scale(
                self._subtract(
                    p_total,
                    self._previous_total_momentum,
                ),
                1.0 / dt,
            )

            if wall_force_is_available:
                momentum_balance_error = self._subtract(
                    dp_total_dt,
                    f_expected,
                )

                momentum_balance_error_norm = self._norm(
                    momentum_balance_error
                )
            else:
                momentum_balance_error = self._nan_vector()
                momentum_balance_error_norm = float("nan")

        row = {
            "step": step,
            "time": time,
            "dt": dt,
            "F_coupling_residual_norm": f_coupling_residual_norm,
            "momentum_balance_error_norm": (
                momentum_balance_error_norm
            ),
        }

        self._write_vector(row, "P_dem", p_dem)
        self._write_vector(row, "P_dpd", p_dpd)
        self._write_vector(row, "P_total", p_total)

        self._write_vector(row, "F_external", f_external)
        self._write_vector(row, "F_wall", f_wall)

        self._write_vector(row, "F_dpd_to_dem", f_dpd_to_dem)
        self._write_vector(row, "F_dem_to_dpd", f_dem_to_dpd)

        self._write_vector(
            row,
            "F_coupling_residual",
            f_coupling_residual,
        )

        self._write_vector(row, "dP_total_dt", dp_total_dt)
        self._write_vector(row, "F_expected", f_expected)

        self._write_vector(
            row,
            "momentum_balance_error",
            momentum_balance_error,
        )

        self._writer.writerow(row)
        self._file.flush()

        self._previous_time = time
        self._previous_total_momentum = p_total[:]
    def _sum_coupling_variable(
        self,
        variable,
        select_dem_particles,
):
        total = self._zero()

        for node in self.particles_model_part.Nodes:
            if self._is_dem_particle(node) != select_dem_particles:
                continue

            value = node.GetSolutionStepValue(variable)

            total = self._add(
                total,
                self._as_list(value),
            )

        return total


    def _get_coupling_force_totals(self):
        f_dpd_to_dem = self._sum_coupling_variable(
            DEM.DPD_TO_DEM_COUPLING_FORCE,
            select_dem_particles=True,
        )

        f_dem_to_dpd = self._sum_coupling_variable(
            DEM.DEM_TO_DPD_COUPLING_FORCE,
            select_dem_particles=False,
        )

        return f_dpd_to_dem, f_dem_to_dpd
    
    def Execute(self):
        f_dpd_to_dem, f_dem_to_dpd = (
            self._get_coupling_force_totals()
        )

        self.record(
            dem_dpd_force_on_dem=f_dpd_to_dem,
            dem_dpd_force_on_dpd=f_dem_to_dpd,
            external_force_on_system=None,
            wall_force_on_system=None,
        )

    def close(self):
        """
        Close the CSV safely, including if __init__ failed partway through.
        """
        file_handle = getattr(self, "_file", None)

        if file_handle is not None:
            file_handle.close()
            self._file = None

    def __del__(self):
        self.close()