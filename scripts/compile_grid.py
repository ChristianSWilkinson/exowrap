import os
import pickle
import logging
import numpy as np
import pandas as pd
import h5py
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def compile_exoweave_grid(input_dir: str, output_prefix: str):
    """
    Scans a directory of ExoWeave .pkl outputs and compiles them into a 
    searchable CSV catalog and a highly compressed HDF5 binary data store.
    """
    in_path = Path(input_dir)
    if not in_path.exists():
        logging.error(f"Directory {input_dir} not found!")
        return

    csv_path = Path(f"{output_prefix}_catalog.csv")
    h5_path = Path(f"{output_prefix}_data.h5")
    
    pkl_files = list(in_path.glob("**/*.pkl"))
    total_files = len(pkl_files)
    
    if total_files == 0:
        logging.error("No .pkl files found to compile.")
        return

    logging.info(f"📦 Found {total_files} models. Compiling grid...")

    summary_catalog = []

    # Open the master HDF5 file in write mode
    with h5py.File(h5_path, 'w') as h5f:
        
        for idx, pkl_file in enumerate(pkl_files):
            # Generate a unique, clean ID for this model
            model_id = f"model_{idx:05d}"
            
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
            except Exception as e:
                logging.warning(f"Failed to read {pkl_file.name}: {e}")
                continue

            # Extract Parameters
            params = data.get('final_params', data.get('parameters', {}))
            status = data.get('status', 'failed')

            # 1. Build the lightweight Catalog Entry
            catalog_entry = {
                'model_id': model_id,
                'status': status,
                'target_mass_Mjup': params.get('mass', np.nan),
                'T_int_dial_K': params.get('T_int_input_dial', params.get('T_int', np.nan)),
                'T_int_true_K': params.get('T_int', np.nan),
                'T_irr_K': params.get('T_irr', np.nan),
                'metallicity': params.get('Met', np.nan),
                'core_mass_Me': params.get('core_mass_earth', np.nan),
                'f_sed': params.get('f_sed', np.nan),
                'kzz': params.get('kzz', np.nan),
                'iterations': data.get('iterations', np.nan),
                'mass_error': data.get('mass_error', np.nan),
                'original_file': pkl_file.name
            }

            # 2. Extract and Store Heavy Arrays (if converged)
            if status == 'converged':
                prof_df = data.get('stitched_profile')
                atm_df = data.get('atmosphere_raw')

                # Append final physical properties to the searchable catalog
                if prof_df is not None:
                    # Handle flexible radius column names
                    r_cols = [c for c in prof_df.columns if 'rad' in c.lower()]
                    if r_cols:
                        catalog_entry['R_total'] = prof_df[r_cols[0]].max()
                    catalog_entry['P_link_bar'] = params.get('p_link_bar', np.nan)

                # Create the HDF5 group for this specific model
                model_grp = h5f.create_group(model_id)

                # Save the continuous stitched profile
                if prof_df is not None:
                    prof_grp = model_grp.create_group('stitched_profile')
                    for col in prof_df.columns:
                        prof_grp.create_dataset(col, data=prof_df[col].values, compression="gzip")

                # Save the raw ExoREM atmosphere (intelligently parsing arrays vs scalars)
                if atm_df is not None and not atm_df.empty:
                    atm_grp = model_grp.create_group('atmosphere_raw')
                    for col in atm_df.columns:
                        val = atm_df[col].iloc[0]
                        
                        # Store strings and single numbers as lightweight HDF5 attributes
                        if isinstance(val, (str, bytes, int, float, bool, np.number)):
                            atm_grp.attrs[col] = val
                        # Store actual physical profiles as compressed datasets
                        else:
                            try:
                                atm_grp.create_dataset(col, data=np.asarray(val), compression="gzip")
                            except Exception:
                                pass # Skip un-serializable Fortran artifacts

            summary_catalog.append(catalog_entry)
            
            # Print progress every 50 files
            if (idx + 1) % 50 == 0:
                logging.info(f"Processed {idx + 1}/{total_files} files...")

    # 3. Save the Catalog as a clean CSV
    df_catalog = pd.DataFrame(summary_catalog)
    df_catalog.to_csv(csv_path, index=False)
    
    logging.info(f"✅ Grid Compilation Complete!")
    logging.info(f"📊 Catalog saved to: {csv_path}")
    logging.info(f"🗄️  Data stored in:  {h5_path}")

if __name__ == "__main__":
    # Point this to your grid output directory
    TARGET_GRID_DIR = "outputs/grid_run_v2"
    
    # Provide a prefix for the compiled files
    OUTPUT_PREFIX = "outputs/master_grid"
    
    compile_exoweave_grid(TARGET_GRID_DIR, OUTPUT_PREFIX)