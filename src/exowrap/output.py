"""
Output Parsing Module for exowrap.

Provides an object-oriented wrapper around the ExoREM results DataFrame, 
providing convenient access to arrays and properties in standard units.
"""

from typing import Dict, List, Literal

import numpy as np
import pandas as pd

from .constants import C_CM_S, R_GAS


class ExoremOut:
    """
    A wrapper around the DataFrame output for ExoREM simulations.
    
    This class parses a flattened ExoREM results DataFrame and exposes its
    physical variables as easily accessible properties. All properties return
    pure Python floats or NumPy arrays in standard SI/CGS units.

    Attributes:
        df (pd.DataFrame): The raw, flattened ExoREM results DataFrame.
    """

    def __init__(self, results_df: pd.DataFrame):
        """
        Initializes the ExoremOut wrapper.

        Args:
            results_df (pd.DataFrame): The single-row DataFrame returned 
                by a completed ExoREM simulation.
        """
        self.df = results_df

    def _get(self, column_name: str) -> any:
        """
        Retrieves a single value from the results DataFrame.

        If a partial column name is provided, it searches for a matching 
        suffix across all columns.

        Args:
            column_name (str): The exact or partial name of the column to retrieve.

        Returns:
            any: The value from the DataFrame corresponding to the column.

        Raises:
            ValueError: If zero or multiple columns match the partial name.
        """
        if column_name not in self.df.columns:
            matching_cols = [c for c in self.df.columns if c.endswith(column_name)]
            if len(matching_cols) != 1:
                raise ValueError(
                    f"Expected exactly one column matching '{column_name}'"
                )
            column_name = matching_cols[0]
        return self.df[column_name].iloc[0]

    def _cols_starting_with(self, prefix: str) -> List[str]:
        """
        Finds all DataFrame columns that start with a specific prefix.

        Args:
            prefix (str): The HDF5 path prefix to search for.

        Returns:
            List[str]: A list of matching column names.
        """
        return [col for col in self.df.columns if col.startswith(prefix)]

    def _name_from_path(self, path: str) -> str:
        """
        Extracts the final token from an HDF5 path string.

        Args:
            path (str): The full HDF5 path (e.g., '/outputs/layers/pressure').

        Returns:
            str: The final token of the path (e.g., 'pressure').
        """
        return path.split("/")[-1]

    def _dict_from_prefix(self, prefix: str) -> Dict[str, np.ndarray]:
        """
        Builds a dictionary from all columns starting with a specific prefix.

        Args:
            prefix (str): The HDF5 path prefix.

        Returns:
            Dict[str, np.ndarray]: A dictionary where keys are the final path 
                tokens and values are the corresponding NumPy arrays.
        """
        out = {}
        for col in self._cols_starting_with(prefix):
            out[self._name_from_path(col)] = np.asarray(self.df[col].iloc[0])
        return out

    # ==========================================
    # Properties (Scalars)
    # ==========================================
    
    @property
    def cloud_cover(self) -> float:
        """float: The cloud coverage fraction (0.0 to 1.0)."""
        return float(self._get("/model_parameters/clouds/fraction"))

    def particle_density(
        self, specie: Literal["cloud1", "cloud2", "cloud3", "cloud4"] = "cloud1"
    ) -> float:
        """
        Retrieves the cloud particle density for a specific cloud species.

        Args:
            specie (str): The cloud species identifier. Defaults to "cloud1".

        Returns:
            float: The particle density in kg/m^3.
        """
        return float(self._get(f"/model_parameters/clouds/particle_density/{specie}"))

    @property
    def fsed(self) -> Dict[str, float]:
        """Dict[str, float]: Sedimentation parameters for each cloud species."""
        return self._dict_from_prefix("/model_parameters/clouds/sedimentation_parameter/")

    @property
    def radius_1bar(self) -> float:
        """float: The planetary radius at the 1e5 Pa (1 bar) pressure level in meters."""
        return float(self._get("/model_parameters/target/radius_1e5Pa"))

    @property
    def t_int(self) -> float:
        """float: The converged internal temperature in Kelvin."""
        return float(self._get("/outputs/run_quality/actual_internal_temperature"))

    @property
    def t_eff(self) -> float:
        """float: The effective temperature from the model light source in Kelvin."""
        return float(self._get("/model_parameters/light_source/effective_temperature"))

    @property
    def t_irr(self) -> float:
        """float: The irradiation temperature in Kelvin."""
        return float(self._get("irradiation_temperature"))

    @property
    def chi2_retrieval(self) -> float:
        """float: The reduced chi-squared metric of the retrieval convergence."""
        return float(self._get("/outputs/run_quality/chi2_retrieval"))

    # ==========================================
    # 1D Atmospheric Profiles
    # ==========================================
    
    @property
    def kzz(self) -> np.ndarray:
        """np.ndarray: Eddy diffusion coefficient profile in atmospheric layers (cm^2/s)."""
        return np.asarray(self._get("/outputs/layers/eddy_diffusion_coefficient"))

    @property
    def gravity(self) -> np.ndarray:
        """np.ndarray: Gravity profile in atmospheric layers (m/s^2)."""
        return np.asarray(self._get("/outputs/layers/gravity"))

    @property
    def mean_molar_mass(self) -> np.ndarray:
        """np.ndarray: Mean molar mass profile in atmospheric layers (kg/mol)."""
        return np.asarray(self._get("/outputs/layers/molar_mass"))

    @property
    def pressure_profile(self) -> np.ndarray:
        """np.ndarray: Pressure profile evaluated at layer centers (Pa)."""
        return np.asarray(self._get("/outputs/layers/pressure"))

    @property
    def temperature_profile(self) -> np.ndarray:
        """np.ndarray: Temperature profile evaluated at layer centers (K)."""
        return np.asarray(self._get("/outputs/layers/temperature"))
    
    @property
    def density_profile(self) -> np.ndarray:
        """
        np.ndarray: Mass density profile in atmospheric layers (kg/m^3).
        
        Calculated internally using the ideal gas law: rho = (P * M) / (R * T).
        """
        p = self.pressure_profile
        m = self.mean_molar_mass
        t = self.temperature_profile
        
        return (p * m) / (R_GAS * t)

    @property
    def pressure_levels(self) -> np.ndarray:
        """np.ndarray: Pressure profile evaluated at exact boundary levels (Pa)."""
        return np.asarray(self._get("/outputs/levels/pressure"))

    @property
    def temperature_levels(self) -> np.ndarray:
        """np.ndarray: Temperature profile evaluated at exact boundary levels (K)."""
        return np.asarray(self._get("/outputs/levels/temperature"))

    @property
    def altitude_profile(self) -> np.ndarray:
        """np.ndarray: Altitude profile evaluated at exact boundary levels (m)."""
        return np.asarray(self._get("/outputs/levels/altitude"))

    # ==========================================
    # VMR / Clouds Outputs
    # ==========================================
    
    @property
    def vmr_absorbers(self) -> Dict[str, np.ndarray]:
        """Dict[str, np.ndarray]: Volume mixing ratio profiles of active absorbing species."""
        return self._dict_from_prefix("/outputs/layers/volume_mixing_ratios/absorbers/")

    @property
    def vmr_gases(self) -> Dict[str, np.ndarray]:
        """Dict[str, np.ndarray]: Volume mixing ratio profiles of non-absorbing background gases."""
        return self._dict_from_prefix("/outputs/layers/volume_mixing_ratios/gases/")

    @property
    def vmr_elements_gas_phase(self) -> Dict[str, np.ndarray]:
        """Dict[str, np.ndarray]: Gas-phase elemental abundance proxy profiles."""
        return self._dict_from_prefix("/outputs/layers/volume_mixing_ratios/elements_gas_phase/")

    @property
    def vmr_profiles(self) -> Dict[str, np.ndarray]:
        """Dict[str, np.ndarray]: A compiled dictionary of all VMR profiles (absorbers, gases, elements)."""
        out = {}
        out.update({f"absorber:{k}": v for k, v in self.vmr_absorbers.items()})
        out.update({f"gas:{k}": v for k, v in self.vmr_gases.items()})
        out.update({f"element:{k}": v for k, v in self.vmr_elements_gas_phase.items()})
        return out

    @property
    def cloud_vmrs(self) -> Dict[str, np.ndarray]:
        """Dict[str, np.ndarray]: Volume mixing ratio profiles for cloud species."""
        return self._dict_from_prefix("/outputs/layers/clouds/volume_mixing_ratio/")

    @property
    def cloud_opacities(self) -> Dict[str, np.ndarray]:
        """Dict[str, np.ndarray]: Cloud opacity (tau) profiles at the reference wavenumber."""
        return self._dict_from_prefix("/outputs/layers/clouds/opacity/")

    # ==========================================
    # Spectral Coordinates & Flux
    # ==========================================
    
    @property
    def wavenumber(self) -> np.ndarray:
        """np.ndarray: The spectral wavenumber grid (cm^-1)."""
        return np.asarray(self._get("/outputs/spectra/wavenumber"))

    @property
    def wavelength(self) -> np.ndarray:
        """np.ndarray: The spectral wavelength grid mathematically converted from wavenumber (microns)."""
        wn = self.wavenumber
        valid = wn > 0
        wl = np.zeros_like(wn)
        wl[valid] = 10000.0 / wn[valid]
        return wl

    @property
    def emission_spectral_radiosity(self) -> np.ndarray:
        """np.ndarray: Total emission spectral radiosity (W m^-2 (cm^-1)^-1)."""
        return np.asarray(self._get("/outputs/spectra/emission/spectral_radiosity"))

    @property
    def emission_species(self) -> Dict[str, np.ndarray]:
        """Dict[str, np.ndarray]: Emission radiosity contributions broken down by individual species."""
        return self._dict_from_prefix("/outputs/spectra/emission/contributions/")

    @property
    def flux_wavenumber_density(self) -> np.ndarray:
        """np.ndarray: Spectral flux density evaluated on the wavenumber grid (W m^-2 (cm^-1)^-1)."""
        return np.asarray(self._get("/outputs/spectra/flux/spectral_flux"))

    @property
    def flux_fnu(self) -> np.ndarray:
        """np.ndarray: Flux density per frequency F_nu (W m^-2 Hz^-1)."""
        return self.flux_wavenumber_density / C_CM_S

    @property
    def flux_jy(self) -> np.ndarray:
        """
        np.ndarray: Flux density per frequency F_nu in Janskys (Jy).
        
        Calculated internally using the standard astronomical conversion:
        1 Jy = 10^-26 W m^-2 Hz^-1.
        """
        return self.flux_fnu * 1e26

    @property
    def flux_flambda(self) -> np.ndarray:
        """
        np.ndarray: Flux density per wavelength F_lambda (W m^-2 um^-1).
        
        This is calculated internally using the Jacobian conversion from the 
        wavenumber density grid.
        """
        wl = self.wavelength
        valid = wl > 0
        flam = np.zeros_like(self.flux_wavenumber_density)
        flam[valid] = self.flux_wavenumber_density[valid] * (10000.0 / (wl[valid] ** 2))
        return flam

    @property
    def transmission(self) -> np.ndarray:
        """np.ndarray: Transmission spectrum evaluated as transit depth (dimensionless fraction)."""
        return np.asarray(self._get("/outputs/spectra/transmission/transit_depth"))

    @property
    def transmission_species(self) -> Dict[str, np.ndarray]:
        """Dict[str, np.ndarray]: Transmission transit depth contributions by individual species."""
        return self._dict_from_prefix("/outputs/spectra/transmission/contributions/")

    @property
    def transmission_clear(self) -> np.ndarray:
        """np.ndarray: Transmission transit depth assuming a completely clear, cloud-free atmosphere."""
        return np.asarray(self._get("/outputs/spectra/transmission/transit_depth_clear"))

    @property
    def transmission_full_cover(self) -> np.ndarray:
        """np.ndarray: Transmission transit depth assuming 100% full cloud coverage."""
        return np.asarray(self._get("/outputs/spectra/transmission/transit_depth_full_cover"))

    def summary(self) -> dict:
        """
        Generates a high-level statistical summary of the parsed atmospheric model.

        Returns:
            dict: A dictionary containing structural metrics (like spectral grid size),
                scalar temperatures, and the number of species contributors found.
        """
        return {
            "n_columns": len(self.df.columns),
            "n_lambda": len(self.wavelength),
            "t_int": self.t_int,
            "cloud_cover": self.cloud_cover,
            "n_emission_contrib": len(self.emission_species),
            "n_transmission_contrib": len(self.transmission_species),
            "n_cloud_species": len(self.cloud_vmrs),
            "n_vmr_absorbers": len(self.vmr_absorbers),
        }