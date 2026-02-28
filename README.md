# exowrap 🪐

**A modern, Pythonic wrapper for the ExoREM atmosphere model.**

ExoREM is a powerful 1D radiative-convective equilibrium model developed for the simulation of young gas giants and brown dwarfs. However, configuring Fortran namelists, managing absolute paths, and parsing complex HDF5 outputs can be tedious. 

`exowrap` provides a clean Python interface to define planet parameters, automatically handle background configurations and temporary sandboxes, execute the Fortran backend, and return the results as friendly Pandas DataFrames or fully parsed, object-oriented physical data structures. 

---

## ✨ Features
* **Pythonic Namelist Generation:** Pass a simple Python dictionary; `exowrap` safely maps it to the Fortran namelist while preserving sequential structure.
* **Isolated Execution:** Runs Fortran in a temporary system sandbox, preventing clutter in your working directory.
* **Dynamic Resolution Scaling:** Seamlessly switch between K-table resolutions (R50, R500, R20000). The wrapper automatically scales your wavenumber step to match the requested resolution.
* **Advanced Workflows:** Automatically upgrade converged low-resolution models to high-resolution spectra via zero-iteration forward passes, or load existing `.h5` files without re-running Fortran.
* **Synthetic Photometry:** Search the SVO Filter Profile Service (FPS) directly from Python, automatically download transmission curves, and compute photometric integrals from your high-resolution spectra.
* **Object-Oriented Physics (`ExoremOut`):** Instantly wrap raw HDF5 outputs into clean, unit-aware NumPy arrays (e.g., `exo_data.temperature_profile`, `exo_data.flux_jy`, `exo_data.density_profile`).
* **Built-in Plotting:** Generate beautifully formatted T-P profiles, emission/transmission spectra, and a comprehensive 4-panel model summary with one line of code.

---

## ⚠️ Prerequisites

Because ExoREM relies on a compiled Fortran backend and HDF5, you must install system-level tools *before* initializing the backend.

**1. Basic Build Tools**
You will need `git`, `make`, and `tar`.

* **macOS:**
```bash
brew install git make
```
*(or install Xcode Command Line Tools)*

* **Ubuntu/Debian:**
```bash
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
If you want to run high-resolution models (R=500 or R=20000), download the corresponding K-tables:
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

# 2. Initialize and run the simulation at a fast resolution (R=50)
model = exowrap.Simulation(params=params, resolution=50)
results_df = model.run()

# 3. Wrap the data for clean, object-oriented access
exo_data = ExoremOut(results_df)
print(f"Converged Internal Temperature: {exo_data.t_int:.2f} K")

# 4. View the comprehensive results
fig, axes = plot_model_summary(exo_data)
plt.show()
```

---

## 🧰 Advanced Workflows (`tools`)

Running high-resolution models from scratch can take hours. `exowrap` provides high-level tools to optimize your workflow.

### Upgrading Resolution (The 0-Iteration Pass)
Instead of waiting for a high-resolution model to converge, you can take a converged R=50 profile, lock the P-T array, and perform a single lightning-fast forward pass at R=500 to generate the detailed spectrum.

```python
import exowrap
from exowrap.output import ExoremOut

# Assuming you already ran a fast R=50 model (results_df)
print("Upgrading to R=500 for precise spectra... ⚡")

high_res_df = exowrap.upgrade_resolution(
    results=results_df, 
    base_params=params, 
    target_resolution=500
)

exo_high_res = ExoremOut(high_res_df)
```

### Loading Existing HDF5 Files
If you've already run a model and saved the `.h5` output, you don't need to spin up the Fortran backend again. Load it directly into memory:
```python
import exowrap

# Instantly load a previous run
df_loaded = exowrap.load_exorem_h5("data/my_previous_run.h5")
exo_data = exowrap.ExoremOut(df_loaded)
```

---

## 📸 Synthetic Photometry (`photometry`)

You can directly query the SVO Filter Profile Service (FPS) to download filter transmission curves, interpolate them onto your high-resolution spectrum, and numerically integrate the flux.

### 1. Search for Filters
Don't know the exact filter ID? Search for it directly from Python:
```python
from exowrap.photometry import search_svo_filters

# Find all JWST filters
jwst_filters = search_svo_filters("JWST")
print(jwst_filters[:5]) 
# Returns: ['JWST/NIRCam.F115W', 'JWST/NIRCam.F140M', ...]
```

### 2. Compute Photometry
Pass the exact filter ID to compute the effective wavelength and integrated flux:
```python
from exowrap.photometry import compute_photometry

# Calculate synthetic photometry for the 2MASS Ks band
ks_band = compute_photometry(exo_high_res, filter_id="2MASS/2MASS.Ks")

print(f"Effective Wavelength: {ks_band['effective_wavelength_um']:.3f} μm")
print(f"Integrated Flux (Jy): {ks_band['flux_Jy']:.4e} Janskys")
print(f"Integrated Flux (W):  {ks_band['flux_W_m2_um']:.4e} W/m^2/um")
```

---

## 🗄️ Accessing Physical Data (`ExoremOut`)

The `ExoremOut` class replaces the need to remember long, convoluted HDF5 string paths. Simply pass your results DataFrame into it to access pure NumPy arrays representing the physics of the atmosphere. 

```python
exo_data = ExoremOut(results_df)

# --- 1. Thermodynamics & Atmospheric Profiles ---
pressures_pa = exo_data.pressure_profile
temperatures_k = exo_data.temperature_profile
gravity = exo_data.gravity
density = exo_data.density_profile  # Computed automatically via Ideal Gas Law!
kzz = exo_data.kzz

# --- 2. Spectral Coordinates & Flux ---
wavelengths = exo_data.wavelength

# Flux in energy/frequency space (Janskys)
flux_jy = exo_data.flux_jy

# Flux in energy/wavelength space (W m^-2 um^-1)
flux_w = exo_data.flux_flambda 

# Transit depth for transmission spectroscopy
transit_depth = exo_data.transmission

# --- 3. Molecular Abundances & Clouds ---
water_vmr = exo_data.vmr_absorbers["H2O"]
methane_emission = exo_data.emission_species["CH4"]
cloud_frac = exo_data.cloud_cover
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
plot_emission_spectrum(high_res_df, ax=ax, color='black', lw=1.5)

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
