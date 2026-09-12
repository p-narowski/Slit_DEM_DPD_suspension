import time
import sys

import KratosMultiphysics
from KratosMultiphysics.DEMApplication.DEM_analysis_stage import DEMAnalysisStage
from KratosMultiphysics import Logger


class DEMAnalysisStageWithFlush(DEMAnalysisStage):
    def __init__(self, model, project_parameters, flush_frequency=10.0):
        super(DEMAnalysisStageWithFlush, self).__init__(model, project_parameters)

        self.flush_frequency = flush_frequency
        self.last_flush = time.time()

        # -------------------------------------------------------------
        # User settings
        # -------------------------------------------------------------

        # Node ID of the SphericParticle3D that must move.
        # Obtain it from the "Begin Nodes" section in SlitDEM.mdpa.
        self.target_solid_particle_id = 51

        # Gravity magnitude in your consistent model units.
        self.gravity_acceleration = 10.0

        # Force direction:
        # -z direction: [0.0, 0.0, -1.0]
        # -y direction: [0.0, -1.0, 0.0]
        self.gravity_direction = KratosMultiphysics.Array3()
        self.gravity_direction[0] = 1.0
        self.gravity_direction[1] = 0.0
        self.gravity_direction[2] = 0.0

        self.target_solid_node = None
        self.target_solid_force = KratosMultiphysics.Array3()
        self.target_solid_force[0] = 1.0
        self.target_solid_force[1] = 0.0
        self.target_solid_force[2] = 0.0

    def Initialize(self):
        super(DEMAnalysisStageWithFlush, self).Initialize()

        self.target_solid_node = self._GetTargetSolidParticle()

        self.target_solid_force = self._CalculateGravityForce(self.target_solid_node)

        print("---- DEBUG Initialize ----")
        for name in self.model.GetModelPartNames():
            print("model part:", name)

        print("target solid particle node id:", self.target_solid_particle_id)

        print("target solid particle F = m*g:", self.target_solid_force)

    def _GetTargetSolidParticle(self):
        self.solid_part_name = "DEMInletPart.Inlet_SuspendedPart"

        if not self.model.HasModelPart(self.solid_part_name):
            raise RuntimeError(
                "Solid inlet model part '{}' was not found. "
                "Print self.model.GetModelPartNames() and verify its exact name."
                .format(self.solid_part_name)
            )

        solid_part = self.model.GetModelPart(self.solid_part_name)

        if solid_part.NumberOfNodes() != 1:
            raise RuntimeError(
                "Expected one solid particle in '{}', but found {} nodes."
                .format(
                    self.solid_part_name,
                    solid_part.NumberOfNodes()
                )
            )

        node = next(iter(solid_part.Nodes))

        print("Selected particle model part:", self.solid_part_name)
        print("Selected particle node ID:", node.Id)

        return node


    def _CalculateGravityForce(self, node):
        mass = node.GetSolutionStepValue(KratosMultiphysics.NODAL_MASS)

        force = KratosMultiphysics.Array3()
        force[0] = mass * self.gravity_acceleration
        force[1] = 0.0
        force[2] = 0.0
        
        radius = node.GetSolutionStepValue(KratosMultiphysics.RADIUS)

        print("Target radius:", radius)
        print("Target NODAL_MASS:", mass)
        print(
            "Density inferred from nodal mass:",
            mass / ((4.0 / 3.0) * 3.141592653589793 * radius**3)
        )

        return force

    def _ApplyGravityForceToTargetSolidParticle(self):
        self.target_solid_node.SetSolutionStepValue(
            KratosMultiphysics.EXTERNAL_APPLIED_FORCE, self.target_solid_force
        )

    def InitializeSolutionStep(self):
        super(DEMAnalysisStageWithFlush, self).InitializeSolutionStep()
        
        if self.target_solid_node is None:
            self.target_solid_node = self._GetTargetSolidParticle()
            self.target_solid_force = self._CalculateGravityForce(
                self.target_solid_node
            )
        # Apply force only to the selected SphericParticle3D.
        # No force is assigned to the DPD particles.
        self._ApplyGravityForceToTargetSolidParticle()

        print(
            "target solid particle external force:",
            self.target_solid_node.GetSolutionStepValue(
                KratosMultiphysics.EXTERNAL_APPLIED_FORCE
            ),
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
        parameters = KratosMultiphysics.Parameters(parameter_file.read())

    global_model = KratosMultiphysics.Model()

    DEMAnalysisStageWithFlush(global_model, parameters).Run()
