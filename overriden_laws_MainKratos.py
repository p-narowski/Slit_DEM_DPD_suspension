import time
import sys

import KratosMultiphysics
import KratosMultiphysics.DEMApplication as KratosDEM
from KratosMultiphysics.DEMApplication.DEM_analysis_stage import DEMAnalysisStage
from KratosMultiphysics import Logger
from KratosMultiphysics.vtk_output_process import VtkOutputProcess


class DEMAnalysisStageWithFlush(DEMAnalysisStage):

    def __init__(self, model, project_parameters, flush_frequency=10.0):
        super().__init__(model, project_parameters)
        self.flush_frequency = flush_frequency
        self.last_flush = time.time()
        self.suspended_vtk_process = None

    def Initialize(self):
        super().Initialize()
        self._assign_constitutive_laws()
        self._initialize_suspended_vtk_output()

    def _initialize_suspended_vtk_output(self):
        """Set up a dedicated VtkOutputProcess for the DEMParts_SuspendedPart submodelpart.
        This writes separate .vtu files containing only the suspended particles.
        """
        suspended_part_name = "SpheresPart.DEMParts_SuspendedPart"

        # Check the submodelpart exists before proceeding
        spheres = self.model.GetModelPart("SpheresPart")
        if not spheres.HasSubModelPart("DEMParts_SuspendedPart"):
            Logger.PrintWarning("SuspendedVTK", "Submodelpart 'DEMParts_SuspendedPart' not found - skipping dedicated VTK output.")
            return

        vtk_params = KratosMultiphysics.Parameters("""
        {
            "model_part_name"                    : "SpheresPart.DEMParts_SuspendedPart",
            "file_format"                        : "ascii",
            "output_control_type"                : "step",
            "output_interval"                    : 1,
            "write_deformed_configuration"       : true,
            "output_sub_model_parts"             : false,
            "folder_name"                        : "VTK_Output_Suspended",
            "save_output_files_in_folder"        : true,
            "nodal_solution_step_data_variables" : ["VELOCITY", "DISPLACEMENT", "TOTAL_FORCES", "RADIUS"]
        }
        """)

        self.suspended_vtk_process = VtkOutputProcess(self.model, vtk_params)
        self.suspended_vtk_process.ExecuteInitialize()
        self.suspended_vtk_process.ExecuteBeforeSolutionLoop()
        Logger.PrintInfo("SuspendedVTK", "Dedicated VtkOutputProcess for DEMParts_SuspendedPart initialized.")

    def ModifyBeforeSolutionLoop(self):
        super().ModifyBeforeSolutionLoop()
        self._assign_constitutive_laws()

    def _assign_constitutive_laws(self):
        fluid_law = "DEM_DPD_SPH_LIKE"
        hertz_law = "DEM_D_Hertz_viscous_Coulomb"

        law_map = {1: fluid_law, 2: hertz_law, 3: hertz_law}

        for prop in self.spheres_model_part.Properties:
            law = law_map.get(prop.Id)
            if law:
                prop[KratosDEM.DEM_DISCONTINUUM_CONSTITUTIVE_LAW_NAME] = law
                print(f"[DBG] Prop {prop.Id} law -> '{law}'")

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

        # Write dedicated VTK output for suspended particles
        if self.suspended_vtk_process is not None:
            self.suspended_vtk_process.ExecuteInitializeSolutionStep()
            if self.suspended_vtk_process.IsOutputStep():
                self.suspended_vtk_process.PrintOutput()
            self.suspended_vtk_process.ExecuteFinalizeSolutionStep()

        max_f = 0.0
        for node in self.model.GetModelPart("SpheresPart").Nodes:
            f = node.GetSolutionStepValue(KratosMultiphysics.TOTAL_FORCES)
            mag = (f[0] ** 2 + f[1] ** 2 + f[2] ** 2) ** 0.5
            if mag > max_f:
                max_f = mag
        # print(f"[t={self.time:.4f}] Max |TOTAL_FORCES| = {max_f:.6e}")

    def Finalize(self):
        super().Finalize()
        if self.suspended_vtk_process is not None:
            self.suspended_vtk_process.ExecuteFinalize()


if __name__ == "__main__":
    Logger.GetDefaultOutput().SetSeverity(Logger.Severity.INFO)
    with open("ProjectParametersDEM.json", 'r') as parameter_file:
        parameters = KratosMultiphysics.Parameters(parameter_file.read())

    global_model = KratosMultiphysics.Model()
    DEMAnalysisStageWithFlush(global_model, parameters).Run()
