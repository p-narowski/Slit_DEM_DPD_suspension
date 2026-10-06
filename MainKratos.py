import sys
import time

import KratosMultiphysics
from KratosMultiphysics import Logger
from KratosMultiphysics.DEMApplication.DEM_analysis_stage import DEMAnalysisStage

from momentum_tracker import MomentumTracker


class DEMAnalysisStageWithFlush(DEMAnalysisStage):

    def __init__(self, model, project_parameters, flush_frequency=10.0):
        super().__init__(model, project_parameters)

        self.flush_frequency = flush_frequency
        self.last_flush = time.time()

        # -------------------------------------------------------------
        # Suspended-particle force settings
        # -------------------------------------------------------------

        # Select the inlet-generated suspended particle(s), while excluding
        # smaller DPD fluid particles.
        self.minimum_target_radius = 0.099

        # The DEM inlet stops at t = 1.5 s. The driving force is applied at
        # the FIRST simulation time step after that time.
        self.force_start_time = 1.5

        # Prescribed external force on each selected suspended particle.
        # Example direction: +X.
        self.driving_force_x = 0.5
        self.driving_force_y = 0.0
        self.driving_force_z = 0.0

        # Parent model part containing DPD and suspended particles.
        self.particles_model_part_name = "SpheresPart"

        # Runtime state / diagnostics.
        self.target_particle_ids = set()
        self.force_has_been_activated = False
        self.force_activation_time = None

        # -------------------------------------------------------------
        # Momentum tracking settings
        # -------------------------------------------------------------
        self.momentum_output_interval = 1.0e-2
        self.next_momentum_output_time = 0.0
        self.momentum_tracker = None

    def Initialize(self):
        super().Initialize()

        particles_model_part = self._GetParticlesModelPart()

        self.momentum_tracker = MomentumTracker(
            particles_model_part,
            output_directory=".",
            suspended_radius_threshold=self.minimum_target_radius)

        self.next_momentum_output_time = particles_model_part.ProcessInfo[
            KratosMultiphysics.TIME]

        self.momentum_tracker.Execute()

        print("---- Suspended-particle force configuration ----")
        print(
            "Selection criterion: RADIUS >= {}".format(
                self.minimum_target_radius))
        print(
            "Force start rule: first step after t = {} s".format(
                self.force_start_time))
        print(
            "External force per selected particle: [{}, {}, {}]".format(
                self.driving_force_x,
                self.driving_force_y,
                self.driving_force_z))
        print("Particle model part: {}".format(
            self.particles_model_part_name))

    def _GetParticlesModelPart(self):
        if not self.model.HasModelPart(self.particles_model_part_name):
            raise RuntimeError(
                "Particle model part '{}' was not found. "
                "Available model parts: {}".format(
                    self.particles_model_part_name,
                    list(self.model.GetModelPartNames())))

        return self.model.GetModelPart(self.particles_model_part_name)

    def _GetTargetParticles(self):
        particles_model_part = self._GetParticlesModelPart()
        target_particles = []

        for node in particles_model_part.Nodes:
            radius = node.GetSolutionStepValue(KratosMultiphysics.RADIUS)

            if radius >= self.minimum_target_radius:
                target_particles.append(node)

        return target_particles

    def _CreateDrivingForce(self):
        force = KratosMultiphysics.Array3()
        force[0] = self.driving_force_x
        force[1] = self.driving_force_y
        force[2] = self.driving_force_z
        return force

    def _CreateZeroForce(self):
        force = KratosMultiphysics.Array3()
        force[0] = 0.0
        force[1] = 0.0
        force[2] = 0.0
        return force

    def _SetExternalForceOnTargetParticles(self, force):
        target_particles = self._GetTargetParticles()
        current_ids = set()

        for node in target_particles:
            node.SetSolutionStepValue(
                KratosMultiphysics.EXTERNAL_APPLIED_FORCE,
                force)
            current_ids.add(node.Id)

        self.target_particle_ids = current_ids
        return target_particles

    def _ApplyDrivingForceToTargetParticles(self):
        driving_force = self._CreateDrivingForce()

        target_particles = self._SetExternalForceOnTargetParticles(
            driving_force)

        return len(target_particles)

    def _ClearForceFromTargetParticles(self):
        zero_force = self._CreateZeroForce()

        self._SetExternalForceOnTargetParticles(zero_force)

    def InitializeSolutionStep(self):
        super().InitializeSolutionStep()

        particles_model_part = self._GetParticlesModelPart()
        current_time = particles_model_part.ProcessInfo[
            KratosMultiphysics.TIME]
        delta_time = particles_model_part.ProcessInfo[
            KratosMultiphysics.DELTA_TIME]

        # "First time step after 1.5 s" means:
        #
        #     t > 1.5 s
        #
        # The tolerance avoids an accidental activation at a numerically
        # represented time equal to 1.5.
        should_apply_force = (
            current_time > self.force_start_time + 1.0e-15)

        if should_apply_force:
            n_target_particles = self._ApplyDrivingForceToTargetParticles()

            if not self.force_has_been_activated:
                self.force_has_been_activated = True
                self.force_activation_time = current_time

                print(
                    "---- Driving force activated ----\n"
                    "Requested start time: {}\n"
                    "Actual first active time: {}\n"
                    "Time step: {}\n"
                    "Selected suspended particles: {}".format(
                        self.force_start_time,
                        self.force_activation_time,
                        delta_time,
                        n_target_particles))
        else:
            # Explicitly leave zero force before t = 1.5 s.
            self._ClearForceFromTargetParticles()

    def FinalizeSolutionStep(self):
        super().FinalizeSolutionStep()

        current_simulation_time = self.spheres_model_part.ProcessInfo[
            KratosMultiphysics.TIME]

        if (
            self.momentum_tracker is not None
            and current_simulation_time + 1.0e-15
            >= self.next_momentum_output_time
        ):
            self.momentum_tracker.Execute()

            while (
                self.next_momentum_output_time
                <= current_simulation_time + 1.0e-15
            ):
                self.next_momentum_output_time += (
                    self.momentum_output_interval)

        if self.parallel_type == "OpenMP":
            now = time.time()

            if now - self.last_flush > self.flush_frequency:
                sys.stdout.flush()
                self.last_flush = now

    def Finalize(self):
        if self.momentum_tracker is not None:
            self.momentum_tracker.Execute()

        super().Finalize()


if __name__ == "__main__":
    Logger.GetDefaultOutput().SetSeverity(Logger.Severity.INFO)

    with open("ProjectParametersDEM.json", "r") as parameter_file:
        parameters = KratosMultiphysics.Parameters(
            parameter_file.read())

    global_model = KratosMultiphysics.Model()

    DEMAnalysisStageWithFlush(
        global_model,
        parameters).Run()