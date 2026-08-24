import time
import sys
import numpy as np

import KratosMultiphysics
from KratosMultiphysics.DEMApplication.DEM_analysis_stage import DEMAnalysisStage
from KratosMultiphysics import Logger
import KratosMultiphysics.DEMApplication as DEM


class DEMAnalysisStageWithFlush(DEMAnalysisStage):
    def __init__(self, model, project_parameters, flush_frequency=10.0):
        super(DEMAnalysisStageWithFlush, self).__init__(model, project_parameters)
        self.flush_frequency = flush_frequency
        self.last_flush = time.time()

    def Initialize(self):
        super(DEMAnalysisStageWithFlush, self).Initialize()

        print("---- DEBUG variables ----")
        for name in dir(DEM):
            if "FORCE" in name or "GRAV" in name:
                print("DEM variable:", name)

        for name in dir(KratosMultiphysics):
            if "FORCE" in name or "GRAV" in name:
                print("Kratos variable:", name)

        self.gravity = KratosMultiphysics.Array3()
        self.gravity[0] = self.project_parameters["GravityX"].GetDouble()
        self.gravity[1] = self.project_parameters["GravityY"].GetDouble()
        self.gravity[2] = self.project_parameters["GravityZ"].GetDouble()

        print("---- DEBUG Initialize ----")
        for name in self.model.GetModelPartNames():
            print("model part:", name)

        self.suspended_part_name = "Inlet_SuspendedPart"

    def _CancelGravityOnSuspendedParticles(self):
        full_part_name = "DEMInletPart.Inlet_SuspendedPart"

        if not self.model.HasModelPart(full_part_name):
            return

        suspended_part = self.model.GetModelPart(full_part_name)

        for node in suspended_part.Nodes:
            mass = node.GetSolutionStepValue(KratosMultiphysics.NODAL_MASS)

            counter_force = KratosMultiphysics.Array3()
            counter_force[0] = -mass * self.gravity[0]
            counter_force[1] = -mass * self.gravity[1]
            counter_force[2] = -mass * self.gravity[2]

            node.SetSolutionStepValue(KratosMultiphysics.EXTERNAL_APPLIED_FORCE, counter_force)
            print("suspended inlet nodes:", suspended_part.NumberOfNodes())

    def InitializeSolutionStep(self):
        super(DEMAnalysisStageWithFlush, self).InitializeSolutionStep()

        full_part_name = "DEMInletPart.Inlet_SuspendedPart"
        print("---- DEBUG InitializeSolutionStep ----")
        print("Has model part?", self.model.HasModelPart(full_part_name))

        if self.model.HasModelPart(full_part_name):
            suspended_part = self.model.GetModelPart(full_part_name)
            print("suspended inlet nodes:", suspended_part.NumberOfNodes())

            if suspended_part.NumberOfNodes() > 0:
                first_node = next(iter(suspended_part.Nodes))
                print("first node id:", first_node.Id)
                print(
                    "first node mass:",
                    first_node.GetSolutionStepValue(KratosMultiphysics.NODAL_MASS),
                )

        self._CancelGravityOnSuspendedParticles()

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
