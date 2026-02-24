"""
Namelist Generator Module for exowrap.

Handles the dynamic generation of Fortran namelist (.nml) files by applying
user parameters to the default ExoREM template while injecting safe absolute paths.
"""

from pathlib import Path

import f90nml


def build_namelist(
    user_updates: dict,
    output_file_path: Path,
    exorem_base_path: Path,
    run_dir: Path
) -> Path:
    """
    Build and save the Fortran namelist for ExoREM.

    Opens the default `example.nml` from the ExoREM repository, applies 
    user-defined parameter overrides, dynamically sets absolute paths for 
    data and outputs, and writes the final `.nml` file while preserving 
    the strict sequential block order expected by the Fortran parser.

    Args:
        user_updates (dict): Nested dictionary of user parameters to override.
        output_file_path (Path): Destination path for the generated .nml file.
        exorem_base_path (Path): Root directory of the ExoREM installation.
        run_dir (Path): Temporary execution directory for the current run.

    Returns:
        Path: The path to the successfully generated namelist file.

    Raises:
        FileNotFoundError: If the default ExoREM template cannot be found.
    """
    # 1. Read the original template from the cloned repository
    template_path = exorem_base_path / "inputs" / "example.nml"

    if not template_path.exists():
        raise FileNotFoundError(f"ExoREM template not found at {template_path}")

    # f90nml.read() perfectly preserves the strict sequential order Fortran expects
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
    nml_obj["paths"]["path_temperature_profile"] = (
        inputs_str + "atmospheres/temperature_profiles/"
    )
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
    # Ensure the file ends with trailing blank lines so the Fortran parser cleanly exits
    with open(output_file_path, "a") as f:
        f.write("\n\n")

    return output_file_path