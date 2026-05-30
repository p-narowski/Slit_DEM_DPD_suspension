# diagnostic.py
import KratosMultiphysics
from KratosMultiphysics.DEMApplication.DEM_analysis_stage import DEMAnalysisStage

class DiagnosticDEM(DEMAnalysisStage):
    def SetMaterials(self):
        print("=== REGISTERED MODEL PART NAMES ===")
        for name in self.model.GetModelPartNames():
            print(" ", name)
        print("====================================")
        # don't call super — skip the crash

with open("ProjectParametersDEM.json", 'r') as f:
    params = KratosMultiphysics.Parameters(f.read())

model = KratosMultiphysics.Model()
DiagnosticDEM(model, params).Run()