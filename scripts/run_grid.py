"""
Parallel Grid Generator for exowrap.

This script generates a grid of planetary atmospheres by running ExoREM 
concurrently across multiple CPU cores. It includes smart checkpointing 
to resume interrupted runs without recalculating existing models.
"""

import hashlib
import itertools
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import exowrap

# ==========================================
# 1. Define your Grid
# ==========================================
FIXED_PARAMS = {
    "mass": 1.0,         
    "f_sed": 2.0,        
    "kzz": 8.0,          
    "T_irr": 0.0         
}

GRID_VARS = {
    "T_int": [300, 400, 500, 600],       
    "g_1bar": [10.0, 15.0, 20.0],        
    "Met": [0.0, 0.5]                    
}

RESOLUTION = 50
OUTPUT_DIR = Path("./data/grid")
MAX_WORKERS = 4

# ==========================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=OUTPUT_DIR / 'grid_run.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s'
)


def get_run_id(params: dict) -> str:
    """Generate a unique, deterministic hash for a specific parameter combination."""
    # Sort the dictionary so the order of keys doesn't change the hash
    param_str = "_".join([f"{k}-{v}" for k, v in sorted(params.items())])
    short_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
    return f"run_{short_hash}"


def run_single_model(params: dict) -> pd.DataFrame:
    """Task to run a single ExoREM model with built-in resume capability."""
    
    # 1. Check if we already completed this exact parameter combination!
    run_id = get_run_id(params)
    pkl_path = OUTPUT_DIR / f"{run_id}.pkl"
    
    if pkl_path.exists():
        logging.info(f"⏭️ Resuming: Loaded existing data for {run_id} {params}")
        return pd.read_pickle(pkl_path)
        
    # 2. If it doesn't exist, we run the model
    model = exowrap.Simulation(
        params=params,
        resolution=RESOLUTION,
        output_dir=OUTPUT_DIR,
        keep_run_files=False
    )
    
    try:
        df = model.run()
        
        # Inject the input parameters so we can filter the Master DataFrame later
        for key, val in params.items():
            df[f"input_param_{key}"] = val
            
        # 3. Save this specific run instantly so we never lose the progress!
        df.to_pickle(pkl_path)
        return df
        
    except RuntimeError as e:
        logging.error(f"Failed to converge for params: {params}\nError: {e}")
        return pd.DataFrame()


def main():
    print(f"🚀 Initializing Smart ExoREM Grid Generation...")
    print(f"📁 Outputs and checkpoints saved to: {OUTPUT_DIR}")
    
    keys, values = zip(*GRID_VARS.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    tasks = []
    for combo in combinations:
        full_params = FIXED_PARAMS.copy()
        full_params.update(combo)
        tasks.append(full_params)
        
    total_runs = len(tasks)
    print(f"🔢 Total models in grid: {total_runs}")
    print(f"⚡ Running with {MAX_WORKERS} parallel workers...\n")

    successful_results = []
    failed_count = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_params = {executor.submit(run_single_model, p): p for p in tasks}
        
        for future in tqdm(as_completed(future_to_params), total=total_runs, desc="Grid Progress"):
            try:
                df = future.result()
                if not df.empty:
                    successful_results.append(df)
                else:
                    failed_count += 1
            except Exception as exc:
                failed_count += 1
                logging.error(f"Unexpected Python exception: {exc}")

    print("\n" + "="*40)
    print("🏁 GRID GENERATION COMPLETE")
    print(f"✅ Successful runs: {len(successful_results)}")
    print(f"❌ Failed/Non-converged runs: {failed_count}")
    print("="*40)

    if successful_results:
        print("💾 Compiling master grid DataFrame...")
        master_df = pd.concat(successful_results, ignore_index=True)
        master_path = OUTPUT_DIR / "master_grid_results.pkl"
        master_df.to_pickle(master_path)
        print(f"🎉 Master grid saved to {master_path}!")


if __name__ == "__main__":
    main()