# exowrap

**A modern, Pythonic wrapper for the ExoREM atmosphere model.**

Exo-REM is a powerful 1D radiative-equilibrium model developed for the simulation of young gas giants and brown dwarfs. However, configuring Fortran namelists and handling raw HDF5 outputs can be tedious. 

`exowrap` provides a clean Python interface to define planet parameters, automatically handle background configurations, run the Fortran backend, and return the results as friendly Pandas DataFrames.

## ⚠️ Prerequisites (Read Before Installing)

Because ExoREM relies on a compiled Fortran backend and HDF5, you need a few system-level tools installed *before* initializing the backend.

**1. Basic Build Tools**
You will need `git`, `make`, and `tar` installed on your system.
* **macOS:** `brew install git make` (or install Xcode Command Line Tools)
* **Ubuntu/Debian:** `sudo apt install git make build-essential`

**2. The Compiler & HDF5 (The Conda Route - Recommended)**
If you are using Anaconda/Miniconda, the native HDF5 wrappers (`h5fc`) are incredibly strict about which compiler they use. **You must install Conda's Fortran compiler**, otherwise the build will fail:

```bash
conda install -c conda-forge fortran-compiler hdf5
```

*Non-Conda users:* Ensure you have `gfortran` and the HDF5 Fortran bindings installed via your system package manager (e.g., `brew install gcc hdf5` or `sudo apt install gfortran libhdf5-dev libhdf5-fortran-100`).

## Installation

Installing `exowrap` is a two-step process: installing the Python package, and initializing the Fortran backend.

**Step 1: Install the Python Package**
*(Note: Not yet on PyPI, placeholder for future pip install)*
```bash
git clone [https://github.com/yourusername/exowrap.git](https://github.com/yourusername/exowrap.git)
cd exowrap
pip install .
```

**Step 2: Initialize the Backend**
Run the following command in your terminal. This will automatically download the ExoREM source code, patch it for modern architectures (including Apple Silicon), compile the Fortran executable, and download the required base K-tables.

```bash
exowrap init
```
*(By default, this installs the backend to `~/.exowrap/exorem_source`. You can change this using the `--path` argument).*

## Quickstart Usage

Once installed and initialized, running an atmosphere model is as simple as defining a dictionary of parameters.

```python
import exowrap

# 1. Define your planet parameters
params = {
    "mass": 1.5,        # Jupiter masses
    "T_int": 400,       # Internal temperature (K)
    "T_irr": 1200,      # Irradiation temperature (K)
    "Met": 0.5,         # Metallicity [Fe/H]
    "f_sed": 2,         # Sedimentation efficiency
    "kzz": 8.0,         # Eddy diffusion coefficient (log10)
    "g_1bar": 15.0      # Target gravity at 1 bar (m/s^2)
}

# 2. Initialize the model 
model = exowrap.Simulation(params=params)

# 3. Run the simulation
results_df = model.run()

# 4. View the results!
print(f"Computed Effective Temperature: {results_df['T_eff'].iloc[0]:.2f} K")
```

## How it works

Behind the scenes, `exowrap`:
* Reads your `~/.exowrap/config.json` to locate the compiled executable.
* Generates a temporary directory with a dynamically built Fortran namelist (`.nml`).
* Executes the Fortran binary safely.
* Parses the output HDF5 files into a single-row Pandas DataFrame for easy analysis.
* Automatically cleans up the temporary run files.