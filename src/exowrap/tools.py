"""
Advanced Workflow Tools for exowrap.

Contains high-level automation functions, such as resolution upgrading
and forward-model spectra generation.
"""

from pathlib import Path
from typing import Dict, Any, Union

import numpy as np
import pandas as pd

from .model import Simulation
from .output import ExoremOut


def upgrade_resolution(
    results: Union[pd.DataFrame, ExoremOut],
    base_params: Dict[str, Any],
    target_resolution: int = 500,
    output_dir: str = "../data/high_res_spectra"
) -> pd.DataFrame:
    """
    Takes a converged low-resolution model and instantly generates a 
    high-resolution spectrum by locking the P-T profile and forcing a 
    0-iteration forward radiative transfer pass.

    Args:
        results (pd.DataFrame or ExoremOut): The converged low-resolution results.
        base_params (Dict[str, Any]): The original parameter dictionary used.
        target_resolution (int, optional): The new K-table resolution. Defaults to 500.
        output_dir (str, optional): Directory to save the new run. Defaults to "./data/high_res_spectra".

    Returns:
        pd.DataFrame: The raw DataFrame containing the high-resolution output.
    """
    print(f"🚀 Upgrading simulation to R={target_resolution}...")
    
    # 1. Ensure we are working with our clean ExoremOut object
    if isinstance(results, pd.DataFrame):
        exo_data = ExoremOut(results)
    else:
        exo_data = results
        
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 2. Extract the converged arrays
    p_converged = exo_data.pressure_profile
    t_converged = exo_data.temperature_profile
    
    # 3. Save the profile to a temporary text file for Fortran to read
    # (ExoREM typically expects two columns: Pressure, Temperature)
    pt_file_path = out_path / "locked_pt_profile.dat"
    np.savetxt(
        pt_file_path, 
        np.column_stack((p_converged, t_converged)), 
        fmt="%.6e"
    )
    
    # 4. Modify the parameters for a 0-iteration forward pass
    high_res_params = base_params.copy()
    
    # NOTE: You may need to adjust these two exact dictionary keys to match 
    # whatever your specific ExoREM Fortran namelist expects!
    high_res_params["max_iter"] = 0               # Force zero convective iterations
    high_res_params["file_t_p"] = str(pt_file_path.absolute()) # Feed the locked profile
    
    print(f"💾 Locked P-T profile saved to: {pt_file_path}")
    print("⚡ Executing 0-iteration forward pass...")
    
    # 5. Initialize and run the new high-resolution Simulation
    model = Simulation(
        params=high_res_params,
        resolution=target_resolution,
        output_dir=str(out_path)
    )
    
    new_df = model.run()
    
    # Clean up the temporary P-T file to keep the directory tidy
    if pt_file_path.exists():
        pt_file_path.unlink()
        
    print("🎉 High-resolution upgrade complete!")
    return new_df