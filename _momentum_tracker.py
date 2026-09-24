import csv
import os

import KratosMultiphysics as KM


class MomentumTracker:
    """
    Tracks total linear momentum of free particles in a DEM model part.

    The tracker writes:
        momentum_history.csv
        total_momentum_vs_time.png

    It reports momentum of all free particle nodes in the supplied model part:

        P(t) = sum_i m_i v_i

    It also separately reports:
        - total momentum of DPD particles,
        - total momentum of large/suspended particles selected by radius,
        - total prescribed external force on selected large particles.

    The prescribed-force impulse is accumulated as:

        J_ext(t) = integral F_ext dt

    and the diagnostic residual is:

        R(t) = P(t) - P(0) - J_ext(t)

    This residual is meaningful for a closed system with no other external
    influences. In the slit simulation, walls, inlet/outlet events, gravity,
    and fixed constraints also contribute external momentum exchange.
    """

    def __init__(
        self,
        model_part,
        output_directory=".",
        suspended_radius_threshold=0.049):
        self.model_part = model_part
        self.output_directory = output_directory
        self.suspended_radius_threshold = suspended_radius_threshold

        self.csv_file_name = os.path.join(
            self.output_directory,
            "momentum_history.csv")

        self.png_file_name = os.path.join(
            self.output_directory,
            "total_momentum_vs_time.png")

        self.initial_total_momentum = None
        self.previous_time = None

        self.external_impulse = self._ZeroVector()

        self.times = []

        self.total_px_history = []
        self.total_py_history = []
        self.total_pz_history = []
        self.total_pnorm_history = []

        self.dpd_px_history = []
        self.dpd_py_history = []
        self.dpd_pz_history = []
        self.dpd_pnorm_history = []

        self.suspended_px_history = []
        self.suspended_py_history = []
        self.suspended_pz_history = []
        self.suspended_pnorm_history = []

        self.external_fx_history = []
        self.external_fy_history = []
        self.external_fz_history = []

        self.residual_x_history = []
        self.residual_y_history = []
        self.residual_z_history = []
        self.residual_norm_history = []

        self._CreateOutputFile()

    def _ZeroVector(self):
        vector = KM.Array3()
        vector[0] = 0.0
        vector[1] = 0.0
        vector[2] = 0.0
        return vector

    def _CopyVector(self, vector):
        copied_vector = self._ZeroVector()
        copied_vector[0] = vector[0]
        copied_vector[1] = vector[1]
        copied_vector[2] = vector[2]
        return copied_vector

    def _VectorNorm(self, vector):
        return (
            vector[0] * vector[0]
            + vector[1] * vector[1]
            + vector[2] * vector[2]
        ) ** 0.5

    def _AddScaledVector(self, destination, source, scale):
        destination[0] += scale * source[0]
        destination[1] += scale * source[1]
        destination[2] += scale * source[2]

    def _AddVector(self, destination, source):
        destination[0] += source[0]
        destination[1] += source[1]
        destination[2] += source[2]

    def _IsFreeParticleNode(self, node):
        return not (
            node.IsFixed(KM.VELOCITY_X)
            or node.IsFixed(KM.VELOCITY_Y)
            or node.IsFixed(KM.VELOCITY_Z)
        )

    def _GetNodeMass(self, node):
        mass = node.GetSolutionStepValue(KM.NODAL_MASS)

        if mass <= 0.0:
            raise RuntimeError(
                "Node {} has non-positive NODAL_MASS = {}. "
                "Momentum cannot be evaluated.".format(
                    node.Id,
                    mass))

        return mass

    def _IsSuspendedParticle(self, node):
        radius = node.GetSolutionStepValue(KM.RADIUS)
        return radius >= self.suspended_radius_threshold

    def _IsDPDParticle(self, node):
        return not self._IsSuspendedParticle(node)

    def _GetExternalAppliedForce(self, node):
        force = self._ZeroVector()

        try:
            nodal_force = node.GetSolutionStepValue(
                KM.EXTERNAL_APPLIED_FORCE)

            force[0] = nodal_force[0]
            force[1] = nodal_force[1]
            force[2] = nodal_force[2]

        except RuntimeError:
            pass

        return force

    def _CreateOutputFile(self):
        with open(self.csv_file_name, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)

            writer.writerow([
                "time",

                "number_of_free_particles",
                "number_of_dpd_particles",
                "number_of_suspended_particles",

                "total_mass",
                "dpd_mass",
                "suspended_mass",

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

                "external_fx",
                "external_fy",
                "external_fz",

                "external_impulse_x",
                "external_impulse_y",
                "external_impulse_z",

                "momentum_residual_x",
                "momentum_residual_y",
                "momentum_residual_z",
                "momentum_residual_norm"
            ])

    def Execute(self):
        current_time = self.model_part.ProcessInfo[KM.TIME]

        total_momentum = self._ZeroVector()
        dpd_momentum = self._ZeroVector()
        suspended_momentum = self._ZeroVector()

        total_external_force = self._ZeroVector()

        total_mass = 0.0
        dpd_mass = 0.0
        suspended_mass = 0.0

        number_of_free_particles = 0
        number_of_dpd_particles = 0
        number_of_suspended_particles = 0

        for node in self.model_part.Nodes:
            if not self._IsFreeParticleNode(node):
                continue

            mass = self._GetNodeMass(node)

            velocity = node.GetSolutionStepValue(KM.VELOCITY)

            self._AddScaledVector(
                total_momentum,
                velocity,
                mass)

            total_mass += mass
            number_of_free_particles += 1

            external_force = self._GetExternalAppliedForce(node)

            self._AddVector(
                total_external_force,
                external_force)

            if self._IsSuspendedParticle(node):
                self._AddScaledVector(
                    suspended_momentum,
                    velocity,
                    mass)

                suspended_mass += mass
                number_of_suspended_particles += 1

            else:
                self._AddScaledVector(
                    dpd_momentum,
                    velocity,
                    mass)

                dpd_mass += mass
                number_of_dpd_particles += 1

        if self.initial_total_momentum is None:
            self.initial_total_momentum = self._CopyVector(
                total_momentum)

        if self.previous_time is not None:
            delta_time = current_time - self.previous_time

            self._AddScaledVector(
                self.external_impulse,
                total_external_force,
                delta_time)

        momentum_residual = self._ZeroVector()

        momentum_residual[0] = (
            total_momentum[0]
            - self.initial_total_momentum[0]
            - self.external_impulse[0])

        momentum_residual[1] = (
            total_momentum[1]
            - self.initial_total_momentum[1]
            - self.external_impulse[1])

        momentum_residual[2] = (
            total_momentum[2]
            - self.initial_total_momentum[2]
            - self.external_impulse[2])

        self.times.append(current_time)

        self.total_px_history.append(total_momentum[0])
        self.total_py_history.append(total_momentum[1])
        self.total_pz_history.append(total_momentum[2])
        self.total_pnorm_history.append(
            self._VectorNorm(total_momentum))

        self.dpd_px_history.append(dpd_momentum[0])
        self.dpd_py_history.append(dpd_momentum[1])
        self.dpd_pz_history.append(dpd_momentum[2])
        self.dpd_pnorm_history.append(
            self._VectorNorm(dpd_momentum))

        self.suspended_px_history.append(
            suspended_momentum[0])
        self.suspended_py_history.append(
            suspended_momentum[1])
        self.suspended_pz_history.append(
            suspended_momentum[2])
        self.suspended_pnorm_history.append(
            self._VectorNorm(suspended_momentum))

        self.external_fx_history.append(
            total_external_force[0])
        self.external_fy_history.append(
            total_external_force[1])
        self.external_fz_history.append(
            total_external_force[2])

        self.residual_x_history.append(
            momentum_residual[0])
        self.residual_y_history.append(
            momentum_residual[1])
        self.residual_z_history.append(
            momentum_residual[2])
        self.residual_norm_history.append(
            self._VectorNorm(momentum_residual))

        with open(self.csv_file_name, "a", newline="") as csv_file:
            writer = csv.writer(csv_file)

            writer.writerow([
                current_time,

                number_of_free_particles,
                number_of_dpd_particles,
                number_of_suspended_particles,

                total_mass,
                dpd_mass,
                suspended_mass,

                total_momentum[0],
                total_momentum[1],
                total_momentum[2],
                self._VectorNorm(total_momentum),

                dpd_momentum[0],
                dpd_momentum[1],
                dpd_momentum[2],
                self._VectorNorm(dpd_momentum),

                suspended_momentum[0],
                suspended_momentum[1],
                suspended_momentum[2],
                self._VectorNorm(suspended_momentum),

                total_external_force[0],
                total_external_force[1],
                total_external_force[2],

                self.external_impulse[0],
                self.external_impulse[1],
                self.external_impulse[2],

                momentum_residual[0],
                momentum_residual[1],
                momentum_residual[2],
                self._VectorNorm(momentum_residual)
            ])

        self.previous_time = current_time