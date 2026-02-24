# exowrap 🪐

**A modern, Pythonic wrapper for the ExoREM atmosphere model.**

ExoREM is a powerful 1D radiative-convective equilibrium model developed for the simulation of young gas giants and brown dwarfs. However, configuring Fortran namelists, managing absolute paths, and parsing complex HDF5 outputs can be tedious. 

`exowrap` provides a clean Python interface to define planet parameters, automatically handle background configurations and temporary sandboxes, execute the Fortran backend, and return the results as friendly Pandas DataFrames. It also includes a suite of publication-ready plotting tools.

---

## ✨ Features
* **Pythonic Namelist Generation:** Pass a simple Python dictionary; `exowrap` safely maps it to the Fortran namelist while preserving sequential structure.
* **Isolated Execution:** Runs Fortran in a temporary system sandbox, preventing clutter in your working directory.
* **Seamless Data Extraction:** Parses the nested ExoREM HDF5 output into a flat, single-row Pandas DataFrame for instant analysis.
* **Built-in Plotting:** Generate beautifully formatted T-P profiles, emission spectra, transmission spectra, and chemical abundance (VMR) plots with one line of code.
* **Smart Debugging:** Catch Fortran convergence failures directly in your Jupyter Notebook with formatted `STDOUT`/`STDERR` logs.

---

## ⚠️ Prerequisites

Because ExoREM relies on a compiled Fortran backend and HDF5, you must install system-level tools *before* initializing the backend.

**1. Basic Build Tools**
You will need `git`, `make`, and `tar`.
* **macOS:** `brew install git make` (or install Xcode Command Line Tools)
* **Ubuntu/Debian:** `sudo apt install git make build-essential`

**2. The Compiler & HDF5 (The Conda Route - Recommended)**
If you are using Anaconda/Miniconda, the native HDF5 wrappers (`h5fc`) are incredibly strict about which compiler they use. **You must install Conda's Fortran compiler**:
```bash
conda install -c conda-forge fortran-compiler hdf5
```

*Non-Conda users:* Ensure you have `gfortran` and the HDF5 Fortran bindings installed via your system package manager (e.g., `brew install gcc hdf5` or `sudo apt install gfortran libhdf5-dev libhdf5-fortran-100`).

---

## 🛠️ Installation

Installing `exowrap` is a two-step process:

**Step 1: Install the Python Package**
Clone the repository and install it in editable mode:
```bash
git clone git@github.com:ChristianSWilkinson/exowrap.git
cd exowrap
pip install -e .
```

**Step 2: Initialize the Backend**
Run the built-in CLI command. This downloads the ExoREM source code, patches it for modern architectures, compiles the Fortran executable, and downloads the required base K-tables.
```bash
exowrap init
```
*(By default, this installs the backend to `~/.exowrap/exorem_source`.)*

---

## 🚀 Quickstart

Running an atmosphere model is as simple as defining a dictionary of physical parameters.

```python
import exowrap
import matplotlib.pyplot as plt

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

# 2. Initialize the simulation
# Optional: Pass `output_dir="./data"` to permanently save the .h5 file
model = exowrap.Simulation(params=params)

# 3. Run the Fortran backend
results_df = model.run()

# 4. View the results!
actual_tint = results_df['/outputs/run_quality/actual_internal_temperature'].iloc[0]
print(f"Converged Internal Temperature: {actual_tint:.2f} K")
```

---

## 📊 Plotting Suite

`exowrap` comes with a built-in plotting module tailored for exoplanet science. Pass your `results_df` to any of the plotting functions to instantly visualize the physics.

### Temperature-Pressure (T-P) Profile
```python
ax = exowrap.plot_tp_profile(results_df, title="Hot Jupiter T-P Profile")
plt.show()
```

### Emission Spectrum
You can plot the total emission spectrum, and optionally include individual molecular contributions (which draw downwards from the continuum).
```python
mols = ['cia_rayleigh', 'H2O', 'CH4', 'CO']
ax = exowrap.plot_emission_spectrum(results_df, contributions=mols)
ax.set_xlim(0.5, 15) # Zoom in on JWST wavelengths
plt.show()
```

### Transmission (Transit) Spectrum
Visualize the effective radius/transit depth of the planet. Molecular contributions draw upwards as they add atmospheric opacity.
```python
ax = exowrap.plot_transmission_spectrum(results_df, contributions=['H2O', 'CH4'])
plt.show()
```

### Chemical Abundances (VMR)
Plot the Volume Mixing Ratios of various species as a function of pressure depth.
```python
ax = exowrap.plot_vmr_profile(results_df, molecules=['H2O', 'CH4', 'CO', 'NH3'])
plt.show()
```

---

## 🗄️ Accessing Raw Data

The `model.run()` command flattens the entire ExoREM HDF5 output tree into a single-row Pandas DataFrame. Every parameter, array, and convergence metric is stored using its absolute HDF5 path as the column name.

**Examples of extracting data:**
```python
# Extract scalar values
chi_squared = results_df['/outputs/run_quality/chi2_retrieval'].iloc[0]
target_mass = results_df['/model_parameters/target/mass'].iloc[0]

# Extract 1D atmospheric arrays
pressures_pa = results_df['/outputs/layers/pressure'].iloc[0]
temperatures_k = results_df['/outputs/layers/temperature'].iloc[0]

# Extract specific molecular abundances
water_vmr = results_df['/outputs/layers/volume_mixing_ratios/absorbers/H2O'].iloc[0]
```

---

## 🐛 Troubleshooting & Debugging

If the Fortran model fails to converge, or if you pass invalid parameters, `exowrap` will catch the failure and raise a `RuntimeError` containing the exact `STDOUT` and `STDERR` from the Fortran executable. 

If you want to manually inspect the Fortran inputs, outputs, or namelist files, initialize the model with `keep_run_files=True`:
```python
model = exowrap.Simulation(params=params, keep_run_files=True)
results_df = model.run()
```
This will bypass the temporary system sandbox and dump all raw Fortran execution files into a local `./exowrap_debug_run` directory for manual inspection.