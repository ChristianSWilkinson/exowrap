# exowrap 🪐

**A modern, Pythonic wrapper for the ExoREM atmosphere model.**

ExoREM is a powerful 1D radiative-convective equilibrium model developed for the simulation of young gas giants and brown dwarfs. However, configuring Fortran namelists, managing absolute paths, and parsing complex HDF5 outputs can be tedious. 

`exowrap` provides a clean Python interface to define planet parameters, automatically handle background configurations and temporary sandboxes, execute the Fortran backend, and return the results as friendly Pandas DataFrames or fully parsed, object-oriented physical data structures. 

---

## ✨ Features
* **Pythonic Namelist Generation:** Pass a simple Python dictionary; `exowrap` safely maps it to the Fortran namelist while preserving sequential structure.
* **Isolated Execution:** Runs Fortran in a temporary system sandbox, preventing clutter in your working directory.
* **Dynamic Resolution Scaling:** Seamlessly switch between K-table resolutions (R50, R500, R20000). The wrapper automatically scales your wavenumber step to match the requested resolution.
* **Object-Oriented Physics (`ExoremOut`):** Instantly wrap raw HDF5 outputs into clean, unit-aware NumPy arrays (e.g., `exo_data.temperature_profile`, `exo_data.kzz`) without heavy external dependencies.
* **Parallel Grid Generation:** Generate massive atmospheric grids across multiple CPU cores with built-in HDF5 file-lock bypassing and smart checkpointing.
* **Built-in Plotting:** Generate beautifully formatted T-P profiles, emission spectra, transmission spectra, and a comprehensive 4-panel model summary with one line of code.

---

## ⚠️ Prerequisites

Because ExoREM relies on a compiled Fortran backend and HDF5, you must install system-level tools *before* initializing the backend.

**1. Basic Build Tools**
You will need `git`, `make`, and `tar`.
* **macOS:** ```bash
  brew install git make
  ```
  *(or install Xcode Command Line Tools)*
* **Ubuntu/Debian:** ```bash
  sudo apt install git make build-essential
  ```

**2. The Compiler & HDF5 (The Conda Route - Recommended)**
If you are using Anaconda/Miniconda, the native HDF5 wrappers (`h5fc`) are incredibly strict about which compiler they use. **You must install Conda's Fortran compiler**:
```bash
conda install -c conda-forge fortran-compiler hdf5
```

---

## 🛠️ Installation

Installing `exowrap` is a multi-step process to ensure the Fortran backend is properly compiled and configured.

**Step 1: Install the Python Package**
Clone the repository and install it in editable mode:
```bash
git clone git@github.com:ChristianSWilkinson/exowrap.git
cd exowrap
pip install -e .
```

**Step 2: Initialize the Backend**
Run the built-in CLI command. This downloads the ExoREM source code, patches it for modern architectures, compiles the Fortran executable, and downloads the baseline R=50 K-tables.
```bash
exowrap init
```

**Step 3: Download High-Resolution Tables (Optional)**
If you want to run high-resolution models, you can download the corresponding K-tables (e.g., R=500 or R=20000) using the CLI:
```bash
exowrap download-tables --res 500
```

---

## 🚀 Quickstart

Running an atmosphere model is as simple as defining a dictionary of physical parameters. Once the model runs, you can wrap the results in `ExoremOut` for instantaneous physical analysis and plotting.

```python
import exowrap
from exowrap.output import ExoremOut
from exowrap.plotting import plot_model_summary
import matplotlib.pyplot as plt

# 1. Define your planet parameters
params = {
    "mass": 1.5,           # Jupiter masses
    "T_int": 400,          # Internal temperature (K)
    "T_irr": 1200,         # Irradiation temperature (K)
    "Met": 0.5,            # Metallicity [Fe/H]
    "f_sed": 2,            # Sedimentation efficiency
    "kzz": 8.0,            # Eddy diffusion coefficient (log10)
    "g_1bar": 15.0,        # Target gravity at 1 bar (m/s^2)
}

# 2. Initialize and run the simulation
model = exowrap.Simulation(params=params, resolution=500)
results_df = model.run()

# 3. Wrap the data for clean, object-oriented access!
exo_data = ExoremOut(results_df)
print(f"Converged Internal Temperature: {exo_data.t_int:.2f} K")

# 4. View the comprehensive results
fig, axes = plot_model_summary(exo_data)
plt.show()
```

---

## 🌐 Parallel Grid Generation

`exowrap` is inherently thread-safe and perfect for high-performance computing. The repository includes a ready-to-use grid generation script (`scripts/run_grid.py`) powered by Python's `ProcessPoolExecutor`. 

**Running the grid from your terminal:**
```bash
python scripts/run_grid.py
```

**Analyzing the grid later in Python:**
```python
import pandas as pd
from exowrap.output import ExoremOut

# Instantly load hundreds of models into memory
master_df = pd.read_pickle("./data/grid/master_grid_results.pkl")

# Filter the DataFrame using standard Pandas logic 
hot_planets_df = master_df[master_df['input_param_T_int'] >= 500]

# Grab the first hot planet and analyze it
first_hot_planet = ExoremOut(hot_planets_df.iloc[[0]])
```

---

## 🗄️ Accessing Physical Data (`ExoremOut`)

The `ExoremOut` class replaces the need to remember long, convoluted HDF5 string paths. Simply pass your results DataFrame into it, and access pure NumPy arrays representing the physics of the atmosphere. 

```python
exo_data = ExoremOut(results_df)

# Access scalar convergence metrics
chi_sq = exo_data.chi2_retrieval
cloud_frac = exo_data.cloud_cover

# Access 1D vertical atmospheric profiles
pressures_pa = exo_data.pressure_profile
temperatures_k = exo_data.temperature_profile
gravity = exo_data.gravity
kzz = exo_data.kzz

# Access spectral data (automatically converted from wavenumber to wavelength)
wavelengths = exo_data.wavelength
total_emission = exo_data.emission_spectral_radiosity
transit_depth = exo_data.transmission

# Access specific molecular abundances (VMR) and contributions
water_vmr = exo_data.vmr_absorbers["H2O"]
methane_emission = exo_data.emission_species["CH4"]
```

---

## 📊 Plotting Suite

The plotting suite in `exowrap.plotting` is entirely backward compatible. It accepts *both* raw Pandas DataFrames and wrapped `ExoremOut` objects. Because the plotting functions accept an optional `ax` parameter, you can easily overlay multiple models on the same canvas.

### Comparing Resolutions (Emission Spectrum)
```python
import matplotlib.pyplot as plt
from exowrap.plotting import plot_emission_spectrum

fig, ax = plt.subplots(figsize=(12, 5))

# Plot baseline (R=50)
plot_emission_spectrum(results_r50, ax=ax, color='red', lw=2)

# Overlay high-resolution model (R=500)
plot_emission_spectrum(results_r500, ax=ax, color='black', lw=1.5)

ax.set_xlim(0.5, 15) # Zoom in on JWST wavelengths
ax.legend(['R=50', 'R=500'])
plt.show()
```

### Transmission Spectrum with Contributions
Visualize the transit depth of the planet, revealing which molecules are driving the opacity.
```python
import matplotlib.pyplot as plt
from exowrap.plotting import plot_transmission_spectrum

ax = plot_transmission_spectrum(exo_data, contributions=['H2O', 'CH4'])
plt.show()
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