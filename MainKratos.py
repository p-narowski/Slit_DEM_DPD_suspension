
import time
import sys
import numpy as np

import KratosMultiphysics
from KratosMultiphysics.DEMApplication.DEM_analysis_stage import DEMAnalysisStage
from KratosMultiphysics import Logger

class DEMAnalysisStageWithFlush(DEMAnalysisStage):

    def __init__(self, model, project_parameters, flush_frequency=10.0):
        super(DEMAnalysisStageWithFlush, self).__init__(model, project_parameters)
        self.flush_frequency = flush_frequency
        self.last_flush = time.time()

    def FinalizeSolutionStep(self):
        super(DEMAnalysisStageWithFlush, self).FinalizeSolutionStep()

        if self.parallel_type == "OpenMP":
            now = time.time()
            if now - self.last_flush > self.flush_frequency:
                sys.stdout.flush()
                self.last_flush = now
    def OutputSolutionStep(self):
        super().OutputSolutionStep()
        import numpy as np
        vx_list = []
        for node in self.model.GetModelPart("SpheresPart").Nodes:
            v = node.GetSolutionStepValue(KratosMultiphysics.VELOCITY)
            vx_list.append(v[0])
        arr = np.array(vx_list)
        print(f"[t={self.time:.4f}] Vx: mean={arr.mean():.4e}  std={arr.std():.4e}  max={arr.max():.4e}")


if __name__ == "__main__":
    Logger.GetDefaultOutput().SetSeverity(Logger.Severity.INFO)
    with open("ProjectParametersDEM.json", 'r') as parameter_file:
        parameters = KratosMultiphysics.Parameters(parameter_file.read())

    global_model = KratosMultiphysics.Model()
    DEMAnalysisStageWithFlush(global_model, parameters).Run()
