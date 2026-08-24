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

    def Initialize(self):
        super(DEMAnalysisStageWithFlush, self).Initialize()

        self.fluid_part_name = "SpheresPart.DEMParts_ParticlesPart"

        self.flow_acceleration = KratosMultiphysics.Array3()
        self.flow_acceleration[0] = 0.02
        self.flow_acceleration[1] = 0.0
        self.flow_acceleration[2] = 0.0

        print("---- DEBUG Initialize ----")
        for name in self.model.GetModelPartNames():
            print("model part:", name)

    def _ApplyDrivingForceToFluidParticles(self):
        if not self.model.HasModelPart(self.fluid_part_name):
            return

        fluid_part = self.model.GetModelPart(self.fluid_part_name)

        for node in fluid_part.Nodes:
            mass = node.GetSolutionStepValue(KratosMultiphysics.NODAL_MASS)

            driving_force = KratosMultiphysics.Array3()
            driving_force[0] = mass * self.flow_acceleration[0]
            driving_force[1] = mass * self.flow_acceleration[1]
            driving_force[2] = mass * self.flow_acceleration[2]

            node.SetSolutionStepValue(
                KratosMultiphysics.EXTERNAL_APPLIED_FORCE,
                driving_force
            )

    def InitializeSolutionStep(self):
        super(DEMAnalysisStageWithFlush, self).InitializeSolutionStep()

        self._ApplyDrivingForceToFluidParticles()

        if self.model.HasModelPart(self.fluid_part_name):
            fluid_part = self.model.GetModelPart(self.fluid_part_name)
            print("---- DEBUG InitializeSolutionStep ----")
            print("fluid nodes:", fluid_part.NumberOfNodes())

            if fluid_part.NumberOfNodes() > 0:
                first_node = next(iter(fluid_part.Nodes))
                print("first fluid node id:", first_node.Id)
                print(
                    "first fluid node mass:",
                    first_node.GetSolutionStepValue(KratosMultiphysics.NODAL_MASS)
                )
                print(
                    "first fluid node external force:",
                    first_node.GetSolutionStepValue(KratosMultiphysics.EXTERNAL_APPLIED_FORCE)
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