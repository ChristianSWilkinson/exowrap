"""
Core Simulation Module for exowrap.

Handles the setup, execution, and data extraction for an ExoREM simulation.
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .namelist import build_namelist

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Physical Constants
M_JUPITER_KG = 1.898e27


class Simulation:
    """
    Manages the setup, execution, and data extraction for an ExoREM simulation.

    Attributes:
        params (dict): Physical parameters for the planetary atmosphere.
        keep_run_files (bool): Whether to keep the temporary Fortran run directory.
        output_dir (Path, optional): Directory to permanently save HDF5 outputs.
        last_stdout (str): Raw standard output from the last Fortran execution.
        last_stderr (str): Raw standard error from the last Fortran execution.
    """

    def __init__(
        self, 
        params: dict, 
        keep_run_files: bool = False, 
        output_dir: str = None,
        resolution: int = 50
    ):
        """
        Initialize the simulation with user parameters.

        Args:
            params (dict): Dictionary of planet parameters (mass, T_int, T_irr, etc.).
            keep_run_files (bool, optional): Keep the temporary Fortran run directory. 
                Defaults to False.
            output_dir (str, optional): If provided, saves the resulting HDF5 file 
                to this folder. Defaults to None.
        """
        self.params = params
        self.keep_run_files = keep_run_files
        self.output_dir = Path(output_dir) if output_dir else None
        self.resolution = resolution
        
        self.last_stdout = ""
        self.last_stderr = ""

        self._load_backend_config()

    def _load_backend_config(self):
        """
        Load the compiled Fortran backend paths from the CLI's config.json.

        Raises:
            FileNotFoundError: If the backend configuration or executable is missing.
        """
        config_path = Path.home() / ".exowrap" / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                "exowrap backend not initialized! Run `exowrap init`."
            )

        with open(config_path, "r") as f:
            config = json.load(f)

        self.base_path = Path(config["EXOREM_BASE_PATH"])
        self.exe_path = Path(config["EXOREM_EXE"])
        self.data_path = Path(config["EXOREM_DATA"])

        if not self.exe_path.exists():
            raise FileNotFoundError(f"ExoREM executable missing at {self.exe_path}.")

    def _map_params_to_namelist(self) -> dict:
        """
        Map simple user inputs to the nested dictionary structure 
        expected by the Fortran Namelist.

        Returns:
            dict: The parsed namelist updates.
        """
        nml_updates = {"output_files": {"output_files_suffix": "exowrap_run"}}

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
                "light_source_irradiation": irradiation_flux,
            }

        atm_updates = {}
        if "Met" in self.params:
            atm_updates["metallicity"] = 10 ** float(self.params["Met"])
        if "kzz" in self.params:
            atm_updates["eddy_diffusion_coefficient"] = 10 ** float(self.params["kzz"])

        nml_updates["atmosphere_parameters"] = atm_updates

        if "f_sed" in self.params:
            f_sed = float(self.params["f_sed"])
            nml_updates["clouds_parameters"] = {
                "sedimentation_parameter": [f_sed, f_sed]
            }

        spec_updates = {}
        if "wavenumber_min" in self.params:
            spec_updates["wavenumber_min"] = float(self.params["wavenumber_min"])
        if "wavenumber_max" in self.params:
            spec_updates["wavenumber_max"] = float(self.params["wavenumber_max"])
            
        # If the user specifies a step, use it. Otherwise, scale it intelligently!
        if "wavenumber_step" in self.params:
            spec_updates["wavenumber_step"] = float(self.params["wavenumber_step"])
        else:
            # Base step is 200 for R=50. Scale inversely with resolution.
            # R=50 -> step 200. R=500 -> step 20. R=20000 -> step 0.5.
            smart_step = 200.0 * (50.0 / self.resolution)
            spec_updates["wavenumber_step"] = smart_step

        if spec_updates:
            nml_updates["spectrum_parameters"] = spec_updates

        return nml_updates

    def _read_hdf5_results(self, h5_file: Path) -> pd.DataFrame:
        """
        Safely open the ExoREM HDF5 output and flatten it into a 1-row DataFrame.

        Args:
            h5_file (Path): Path to the generated HDF5 file.

        Returns:
            pd.DataFrame: Flattened data, or an empty DataFrame if reading fails.
        """
        if not h5_file.exists():
            logging.error(f"HDF5 output not found at {h5_file}")
            return pd.DataFrame()

        data_dict = {}

        def extract_dataset(name, node):
            """Callback for h5py to recursively extract datasets."""
            if isinstance(node, h5py.Dataset):
                val = node[()]
                if isinstance(val, np.ndarray) and val.size == 1:
                    val = val.item()
                data_dict[f"/{name}"] = val

        try:
            with h5py.File(h5_file, "r") as f:
                f.visititems(extract_dataset)
            return pd.DataFrame([data_dict])
        except Exception as e:
            logging.error(f"Failed to read HDF5 file: {e}")
            return pd.DataFrame()

    def run(self) -> pd.DataFrame:
        """
        Generate the namelist, execute the Fortran binary, and parse results.

        Returns:
            pd.DataFrame: The parsed HDF5 results.

        Raises:
            RuntimeError: If the Fortran backend crashes or fails to converge.
        """
        logging.info("Starting ExoREM Simulation...")
        temp_manager = (
            tempfile.TemporaryDirectory() if not self.keep_run_files else None
        )
        run_dir_str = temp_manager.name if temp_manager else "./exowrap_debug_run"
        run_dir = Path(run_dir_str).resolve()

        outputs_dir = run_dir / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        try:
            # NEW: Verify the requested K-tables actually exist locally
            k_table_path = self.data_path / "k_coefficients_tables" / f"R{self.resolution}"
            if not k_table_path.exists():
                raise FileNotFoundError(
                    f"K-tables for Resolution {self.resolution} not found at {k_table_path}.\n"
                    f"Please run this in your terminal first:\n"
                    f"  exowrap download-tables --res {self.resolution}"
                )

            # 1. Build the inputs (pass the resolution!)
            nml_updates = self._map_params_to_namelist()
            nml_path = run_dir / "input.nml"
            build_namelist(nml_updates, nml_path, self.base_path, run_dir, self.resolution)

            logging.info(f"Generated namelist at {nml_path}")

            # 2. Execute Fortran
            bin_dir = self.exe_path.parent
            cmd = ["./exorem.exe", str(nml_path)]

            logging.info(f"Running Fortran backend from {bin_dir}...")
            
            # --- NEW: Fix for parallel HDF5 reading ---
            # Disable strict HDF5 file locking so multiple cores can read K-tables safely
            run_env = os.environ.copy()
            run_env["HDF5_USE_FILE_LOCKING"] = "FALSE"
            
            result = subprocess.run(
                cmd,
                cwd=bin_dir,
                capture_output=True,
                text=True,
                env=run_env
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
                    f"Enable keep_run_files=True to debug raw files in {run_dir}."
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
                    f"Enable keep_run_files=True to check raw outputs in {run_dir}."
                )
                raise RuntimeError(error_msg)

            logging.info(f"Parsing results from {expected_output_file}...")
            results_df = self._read_hdf5_results(expected_output_file)

            # 4. Save the HDF5 file permanently if the user requested it
            if self.output_dir:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                
                # Dynamically name file based on the planet's T_int and gravity
                t_int = self.params.get("T_int", "X")
                g_1bar = self.params.get("g_1bar", "X")
                filename = f"exorem_Tint{t_int}_g{g_1bar}.h5"
                saved_path = self.output_dir / filename

                shutil.copy(expected_output_file, saved_path)
                logging.info(f"💾 Permanently saved HDF5 to: {saved_path}")

            logging.info("Simulation complete.")
            return results_df

        finally:
            if temp_manager:
                temp_manager.cleanup()