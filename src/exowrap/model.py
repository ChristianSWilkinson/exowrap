import os
import json
import logging
import subprocess
import tempfile
from pathlib import Path

import h5py
import pandas as pd
import numpy as np

from .namelist import build_namelist

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

M_JUPITER_KG = 1.898e27

class Simulation:
    def __init__(self, params: dict, keep_run_files: bool = False, output_dir: str = None):
            """
            Initialize the simulation with user parameters.
            
            Args:
                params (dict): Dictionary of planet parameters.
                keep_run_files (bool): Keep the temporary Fortran run directory.
                output_dir (str): If provided, saves the resulting HDF5 file to this folder.
            """
            self.params = params
            self.keep_run_files = keep_run_files
            self.output_dir = Path(output_dir) if output_dir else None
            self._load_backend_config()
        
    def _load_backend_config(self):
        config_path = Path.home() / ".exowrap" / "config.json"
        if not config_path.exists():
            raise FileNotFoundError("exowrap backend not initialized! Run `exowrap init`.")
            
        with open(config_path, "r") as f:
            config = json.load(f)
            
        self.base_path = Path(config["EXOREM_BASE_PATH"])
        self.exe_path = Path(config["EXOREM_EXE"])
        self.data_path = Path(config["EXOREM_DATA"])
        
        if not self.exe_path.exists():
            raise FileNotFoundError(f"ExoREM executable missing at {self.exe_path}.")

    def _map_params_to_namelist(self) -> dict:
        nml_updates = {
            "output_files": {
                "output_files_suffix": "exowrap_run"
            }
        }
        
        target_updates = {}
        if "mass" in self.params:
            target_updates["target_mass"] = float(self.params["mass"]) * M_JUPITER_KG
            target_updates["use_gravity"] = False
        if "g_1bar" in self.params:
            target_updates["target_equatorial_gravity"] = float(self.params["g_1bar"])
            target_updates["use_gravity"] = True
        if "T_int" in self.params:
            target_updates["target_internal_temperature"] = float(self.params["T_int"])
            
        nml_updates["target_parameters"] = target_updates
        
        if "T_irr" in self.params:
            t_irr = float(self.params["T_irr"])
            sigma = 5.670374419e-8 
            irradiation_flux = 4 * sigma * (t_irr**4) 
            nml_updates["light_source_parameters"] = {
                "add_light_source": t_irr > 1, 
                "use_irradiation": True,
                "light_source_irradiation": irradiation_flux
            }
            
        atm_updates = {}
        if "Met" in self.params:
            atm_updates["metallicity"] = 10**float(self.params["Met"])
        if "kzz" in self.params:
            atm_updates["eddy_diffusion_coefficient"] = 10**float(self.params["kzz"])
            
        nml_updates["atmosphere_parameters"] = atm_updates
        
        if "f_sed" in self.params:
            f_sed = float(self.params["f_sed"])
            nml_updates["clouds_parameters"] = {
                "sedimentation_parameter": [f_sed, f_sed] 
            }

        return nml_updates

    def _read_hdf5_results(self, h5_file: Path) -> pd.DataFrame:
        if not h5_file.exists():
            logging.error(f"HDF5 output not found at {h5_file}")
            return pd.DataFrame()
            
        data_dict = {}
        def extract_dataset(name, node):
            if isinstance(node, h5py.Dataset):
                val = node[()]
                if isinstance(val, np.ndarray) and val.size == 1:
                    val = val.item()
                data_dict[f"/{name}"] = val
                
        try:
            with h5py.File(h5_file, 'r') as f:
                f.visititems(extract_dataset)
            return pd.DataFrame([data_dict])
        except Exception as e:
            logging.error(f"Failed to read HDF5 file: {e}")
            return pd.DataFrame()

    def run(self) -> pd.DataFrame:
        logging.info("Starting ExoREM Simulation...")
        temp_manager = tempfile.TemporaryDirectory() if not self.keep_run_files else None
        run_dir_str = temp_manager.name if temp_manager else "./exowrap_debug_run"
        run_dir = Path(run_dir_str).resolve()
        
        outputs_dir = run_dir / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. Build the inputs
            nml_updates = self._map_params_to_namelist()
            nml_path = run_dir / "input.nml"
            build_namelist(nml_updates, nml_path, self.base_path, run_dir)
            
            logging.info(f"Generated namelist at {nml_path}")
            
            # 2. Execute Fortran
            bin_dir = self.exe_path.parent 
            cmd = ["./exorem.exe", str(nml_path)]
            
            logging.info(f"Running Fortran backend from {bin_dir}...")
            result = subprocess.run(
                cmd, 
                cwd=bin_dir, 
                capture_output=True, 
                text=True
            )
            
            # Save the raw terminal output directly to the Python object
            self.last_stdout = result.stdout
            self.last_stderr = result.stderr
            
            # Check 1: Did Fortran actually crash? (Segmentation fault, etc.)
            if result.returncode != 0:
                error_msg = (
                    f"\n{'='*40}\n"
                    f"❌ EXOREM FORTRAN CRASHED (Code {result.returncode})\n"
                    f"{'='*40}\n"
                    f"--- STDOUT ---\n{self.last_stdout}\n"
                    f"--- STDERR ---\n{self.last_stderr}\n"
                    f"{'='*40}\n"
                    f"Enable keep_run_files=True to debug the raw files in {run_dir}."
                )
                raise RuntimeError(error_msg)
                
            # 3. Read Outputs
            expected_output_file = outputs_dir / "exowrap_run.h5"
            
            # Check 2: Did it run but fail to converge/output data?
            if not expected_output_file.exists():
                error_msg = (
                    f"\n{'='*40}\n"
                    f"⚠️ EXOREM FAILED TO CONVERGE OR NO OUTPUT GENERATED\n"
                    f"{'='*40}\n"
                    f"The Fortran code exited normally, but no HDF5 file was created.\n"
                    f"This usually means the atmosphere model failed to converge.\n\n"
                    f"--- STDOUT ---\n{self.last_stdout}\n"
                    f"--- STDERR ---\n{self.last_stderr}\n"
                    f"{'='*40}\n"
                    f"Enable keep_run_files=True to check the raw text outputs in {run_dir}."
                )
                raise RuntimeError(error_msg)

            logging.info(f"Parsing results from {expected_output_file}...")
            results_df = self._read_hdf5_results(expected_output_file)
            
            # NEW: Save the HDF5 file permanently if the user requested it
            if self.output_dir:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                # Let's give it a name based on the planet's T_int and gravity
                filename = f"exorem_Tint{self.params.get('T_int', 'X')}_g{self.params.get('g_1bar', 'X')}.h5"
                saved_path = self.output_dir / filename
                
                import shutil
                shutil.copy(expected_output_file, saved_path)
                logging.info(f"💾 Permanently saved HDF5 to: {saved_path}")

            logging.info("Simulation complete.")
            return results_df
            
        finally:
            if temp_manager:
                temp_manager.cleanup()