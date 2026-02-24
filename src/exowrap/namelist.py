import f90nml
from pathlib import Path
import copy

# We store the exact template you provided as a base dictionary.
# We convert Fortran arrays to Python lists, and Fortran booleans to Python booleans.
DEFAULT_NML = {
    "output_files": {
        "spectrum_file_prefix": "spectra",
        "temperature_profile_file_prefix": "temperature_profile",
        "vmr_file_prefix": "vmr",
        "output_files_suffix": "example_R50"
    },
    "target_parameters": {
        "use_gravity": False,
        "use_flattening": True,
        "target_mass": 5e25,
        "target_equatorial_gravity": 0.0,
        "target_equatorial_radius": 15000e3,
        "target_polar_radius": 0.0,
        "target_flattening": 0.0,
        "latitude": 0.0,
        "target_internal_temperature": 500.0,
        "emission_angle": 0.0,
        "cos_average_angle": 0.666666666666666
    },
    "light_source_parameters": {
        "add_light_source": True,
        "use_irradiation": False,
        "use_light_source_spectrum": False,
        "light_source_radius": 100e6,
        "light_source_range": 2e9,
        "light_source_effective_temperature": 3450.0,
        "light_source_irradiation": 20000.0,
        "light_source_spectrum_file": "spectrum_BTSettl_3500K_logg5_met0.dat",
        "incidence_angle": 0.0
    },
    "atmosphere_parameters": {
        "use_metallicity": True,
        "use_pressure_grid": False,
        "h2_vmr": 0.6,
        "he_vmr": 0.1,
        "z_vmr": 0.3,
        "metallicity": 10.0,
        "n_levels": 81,
        "n_species": 13,
        "n_cia": 3,
        "n_clouds": 2,
        "eddy_mode": "AckermanConvective",
        "eddy_diffusion_coefficient": 1e8,
        "load_kzz_profile": False,
        "pressure_min": 1e-1,
        "pressure_max": 1e7
    },
    "species_parameters": {
        "use_atmospheric_metallicity": False,
        "use_elements_metallicity": True,
        "elements_names": ["He", "Ne", "Ar", "Kr", "Xe"],
        "elements_h_ratio": [8.395e-2, 1e-30, 1e-30, 1e-30, 1e-30],
        "elements_metallicity": [1.0, 1.0, 1.0, 1.0, 1.0],
        "species_names": ["CH4", "CO", "CO2", "FeH", "H2O", "H2S", "HCN", "K", "Na", "NH3", "PH3", "TiO", "VO"],
        "species_at_equilibrium": [False] * 13,
        "cia_names": ["H2-H2", "H2-He", "H2O-H2O"],
        "load_vmr_profiles": False,
        "vmr_profiles_file": "vmr_example_ref.dat",
        "use_chemistry": True,
        "use_rayleigh": True
    },
    "spectrum_parameters": {
        "wavenumber_min": 130.0,
        "wavenumber_max": 30130.0,
        "wavenumber_step": 200.0
    },
    "clouds_parameters": {
        "cloud_mode": "fixedRadiusTime",
        "cloud_fraction": 0.15,
        "cloud_names": ["H2O", "KCl"],
        "cloud_particle_radius": [50e-6, 5e-6],
        "sedimentation_parameter": [2.0, 2.0],
        "supersaturation_parameter": [0.003, 0.003],
        "sticking_efficiency": [1.0, 1.0],
        "cloud_particle_density": [917.0, 1980.0],
        "reference_wavenumber": [1e4, 1e4],
        "load_cloud_profiles": False
    },
    "retrieval_parameters": {
        "temperature_profile_file": "temperature_profile_example_ref.dat",
        "retrieval_level_bottom": 2,
        "retrieval_level_top": 81,
        "retrieval_flux_error_bottom": 1e-3,
        "retrieval_flux_error_top": 1e-5,
        "n_iterations": 50,
        "n_non_adiabatic_iterations": 15,
        "chemistry_iteration_interval": 2,
        "cloud_iteration_interval": 4,
        "n_burn_iterations": 99,
        "retrieval_tolerance": 0.001,
        "smoothing_bottom": 0.5,
        "smoothing_top": 0.5,
        "weight_apriori": 10.0
    },
    "options": {
        "output_transmission_spectra": True,
        "output_species_spectral_contributions": True,
        "output_cia_spectral_contribution": True,
        "output_thermal_spectral_contribution": True,
        "output_fluxes": True,
        "output_hdf5": True,
        "output_full": True
    },
    "paths": {
        "path_data": "../data/",
        "path_cia": "../data/cia/",
        "path_clouds": "../data/cloud_optical_constants/",
        "path_k_coefficients": "../data/k_coefficients_tables/R50/",
        "path_temperature_profile": "../inputs/atmospheres/temperature_profiles/",
        "path_thermochemical_tables": "../data/thermochemical_tables/",
        "path_vmr_profiles": "../inputs/atmospheres/vmr_profiles/",
        "path_light_source_spectra": "../data/stellar_spectra/",
        "path_outputs": "../outputs/exorem/"
    }
}


def build_namelist(user_updates: dict, output_file_path: Path, exorem_base_path: Path, run_dir: Path):
    """
    Opens the default example.nml from the ExoREM repo, applies user overrides, 
    dynamically sets absolute paths, and writes the .nml file out preserving order.
    """
    # 1. Read the original template from the cloned repository
    template_path = exorem_base_path / "inputs" / "example.nml"
    
    if not template_path.exists():
        raise FileNotFoundError(f"ExoREM template not found at {template_path}")
        
    # f90nml.read() perfectly preserves the strict sequential order Fortran expects!
    nml_obj = f90nml.read(str(template_path))
    
    # 2. Automatically map the absolute data and input paths
    data_str = str(exorem_base_path / "data") + "/"
    inputs_str = str(exorem_base_path / "inputs") + "/"
    outputs_str = str(run_dir / "outputs") + "/"
    
    # Update the paths block
    nml_obj["paths"]["path_data"] = data_str
    nml_obj["paths"]["path_cia"] = data_str + "cia/"
    nml_obj["paths"]["path_clouds"] = data_str + "cloud_optical_constants/"
    nml_obj["paths"]["path_k_coefficients"] = data_str + "k_coefficients_tables/R50/"
    nml_obj["paths"]["path_thermochemical_tables"] = data_str + "thermochemical_tables/"
    nml_obj["paths"]["path_light_source_spectra"] = data_str + "stellar_spectra/"
    nml_obj["paths"]["path_temperature_profile"] = inputs_str + "atmospheres/temperature_profiles/"
    nml_obj["paths"]["path_vmr_profiles"] = inputs_str + "atmospheres/vmr_profiles/"
    nml_obj["paths"]["path_outputs"] = outputs_str

    # 3. Apply the user's specific overrides
    for section, overrides in user_updates.items():
        if section not in nml_obj:
            nml_obj[section] = {}
        for key, val in overrides.items():
            nml_obj[section][key] = val

    # 4. Write to disk
    nml_obj.write(str(output_file_path), force=True)
    
    # 5. Fortran EOF safeguard
    with open(output_file_path, "a") as f:
        f.write("\n\n")
        
    return output_file_path