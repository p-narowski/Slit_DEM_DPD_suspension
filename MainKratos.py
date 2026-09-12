import time
import sys

import KratosMultiphysics
from KratosMultiphysics.DEMApplication.DEM_analysis_stage import DEMAnalysisStage
from KratosMultiphysics import Logger


class DEMAnalysisStageWithFlush(DEMAnalysisStage):

    def __init__(self, model, project_parameters, flush_frequency=10.0):
        super(DEMAnalysisStageWithFlush, self).__init__(
            model,
            project_parameters
        )

        self.flush_frequency = flush_frequency
        self.last_flush = time.time()

        # -------------------------------------------------------------
        # User settings
        # -------------------------------------------------------------

        # Apply the specified force to every particle with:
        # RADIUS >= minimum_target_radius
        #
        # Choose a value larger than every DPD-fluid particle radius and
        # less than or equal to the radius of all intended solid particles.
        self.minimum_target_radius = 0.049

        # Prescribed force applied to EACH selected large particle.
        # Units must be consistent with your Kratos model units.
        #
        # Example physical value for a sphere with:
        # R = 0.05 m, rho = 900 kg/m^3, g = 10 m/s^2:
        # F = rho * (4/3)*pi*R^3 * g = 4.71238898038 N.
        self.driving_force_x = 0.00000000471238898038

        # Force direction / components.
        # The example below applies the force in +X only.
        self.driving_force_y = 0.0
        self.driving_force_z = 0.0

        # Parent part containing both DPD particles and inlet-generated
        # SphericParticle3D particles.
        self.particles_model_part_name = "SpheresPart"

        # Node IDs are retained only for concise diagnostic output.
        self.target_particle_ids = set()

    def Initialize(self):
        super(DEMAnalysisStageWithFlush, self).Initialize()

        print("---- DEBUG Initialize ----")
        print(
            "Radius-selection criterion: RADIUS >= {}"
            .format(self.minimum_target_radius)
        )
        print(
            "External force per selected particle: [{}, {}, {}]"
            .format(
                self.driving_force_x,
                self.driving_force_y,
                self.driving_force_z
            )
        )

        print("Model parts:")
        for name in self.model.GetModelPartNames():
            print("  ", name)

    def _GetParticlesModelPart(self):
        if not self.model.HasModelPart(self.particles_model_part_name):
            raise RuntimeError(
                "Particle model part '{}' was not found. Available model "
                "parts are: {}".format(
                    self.particles_model_part_name,
                    list(self.model.GetModelPartNames())
                )
            )

        return self.model.GetModelPart(self.particles_model_part_name)

    def _GetTargetParticles(self):
        particles_model_part = self._GetParticlesModelPart()

        target_particles = []

        for node in particles_model_part.Nodes:
            radius = node.GetSolutionStepValue(
                KratosMultiphysics.RADIUS
            )

            if radius >= self.minimum_target_radius:
                target_particles.append(node)

        return target_particles

    def _ApplyForceToTargetParticles(self):
        target_particles = self._GetTargetParticles()

        force = KratosMultiphysics.Array3()
        force[0] = self.driving_force_x
        force[1] = self.driving_force_y
        force[2] = self.driving_force_z

        current_ids = set()

        for node in target_particles:
            node.SetSolutionStepValue(
                KratosMultiphysics.EXTERNAL_APPLIED_FORCE,
                force
            )
            current_ids.add(node.Id)

        newly_detected = current_ids - self.target_particle_ids

        for node_id in sorted(newly_detected):
            node = self._GetParticlesModelPart().GetNode(node_id)
            radius = node.GetSolutionStepValue(
                KratosMultiphysics.RADIUS
            )
            mass = node.GetSolutionStepValue(
                KratosMultiphysics.NODAL_MASS
            )

            print(
                "Selected large particle: "
                "node ID = {}, R = {}, mass = {}, force = [{}, {}, {}]"
                .format(
                    node_id,
                    radius,
                    mass,
                    self.driving_force_x,
                    self.driving_force_y,
                    self.driving_force_z
                )
            )

        self.target_particle_ids = current_ids

        return len(target_particles)

    def InitializeSolutionStep(self):
        super(DEMAnalysisStageWithFlush, self).InitializeSolutionStep()

        n_target_particles = self._ApplyForceToTargetParticles()

        if self.time % 1.0 < self.spheres_model_part.ProcessInfo[
            KratosMultiphysics.DELTA_TIME
        ]:
            print(
                "Time = {:.6g}; forced particles (R >= {}): {}"
                .format(
                    self.time,
                    self.minimum_target_radius,
                    n_target_particles
                )
            )

    def FinalizeSolutionStep(self):
        super(DEMAnalysisStageWithFlush, self).FinalizeSolutionStep()

        if self.parallel_type == "OpenMP":
            now = time.time()

            if now - self.last_flush > self.flush_frequency:
                sys.stdout.flush()
                self.last_flush = now


if __name__ == "__main__":
    Logger.GetDefaultOutput().SetSeverity(Logger.Severity.INFO)

    with open("ProjectParametersDEM.json", "r") as parameter_file:
        parameters = KratosMultiphysics.Parameters(
            parameter_file.read()
        )

    global_model = KratosMultiphysics.Model()

    DEMAnalysisStageWithFlush(global_model, parameters).Run()