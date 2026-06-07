import time
import sys

import KratosMultiphysics
import KratosMultiphysics.DEMApplication as KratosDEM
from KratosMultiphysics.DEMApplication.DEM_analysis_stage import DEMAnalysisStage
from KratosMultiphysics import Logger


class DEMAnalysisStageWithFlush(DEMAnalysisStage):

    def __init__(self, model, project_parameters, flush_frequency=10.0):
        super().__init__(model, project_parameters)
        self.flush_frequency = flush_frequency
        self.last_flush = time.time()

    def Initialize(self):
        super().Initialize()
        self._assign_constitutive_laws()

    def ModifyBeforeSolutionLoop(self):
        super().ModifyBeforeSolutionLoop()
        # Re-assign here so the solver re-binds law pointers AFTER our assignment
        self._assign_constitutive_laws()

    def _assign_constitutive_laws(self):
        fluid_law  = "DEM_DPD_SPH_LIKE"
        hertz_law  = "DEM_D_Hertz_viscous_Coulomb"

        law_map = {1: fluid_law, 2: hertz_law}

        for prop in self.spheres_model_part.Properties:
            law = law_map.get(prop.Id)
            if law:
                prop[KratosDEM.DEM_DISCONTINUUM_CONSTITUTIVE_LAW_NAME] = law
                print(f"[DBG] Prop {prop.Id} law → '{law}'")

        # Also cover wall model part
        for prop in self.rigid_face_model_part.Properties:
            law = law_map.get(prop.Id)
            if law:
                prop[KratosDEM.DEM_DISCONTINUUM_CONSTITUTIVE_LAW_NAME] = law

        sys.stdout.flush()

    def FinalizeSolutionStep(self):
        super().FinalizeSolutionStep()
        if self.parallel_type == "OpenMP":
            now = time.time()
            if now - self.last_flush > self.flush_frequency:
                sys.stdout.flush()
                self.last_flush = now

    def OutputSolutionStep(self):
        super().OutputSolutionStep()
        # Only print at output steps (not every time step)
        max_f = 0.0
        for node in self.model.GetModelPart("SpheresPart").Nodes:
            f = node.GetSolutionStepValue(KratosMultiphysics.TOTAL_FORCES)
            mag = (f[0]**2 + f[1]**2 + f[2]**2)**0.5
            if mag > max_f:
                max_f = mag
        #print(f"[t={self.time:.4f}] Max |TOTAL_FORCES| = {max_f:.6e}")


if __name__ == "__main__":
    Logger.GetDefaultOutput().SetSeverity(Logger.Severity.INFO)
    with open("ProjectParametersDEM.json", 'r') as parameter_file:
        parameters = KratosMultiphysics.Parameters(parameter_file.read())

    global_model = KratosMultiphysics.Model()
    DEMAnalysisStageWithFlush(global_model, parameters).Run()