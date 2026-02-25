"""
Output Parsing Module for exowrap.

Provides an object-oriented wrapper around the ExoREM results DataFrame, 
providing convenient access to arrays and properties in standard units.
"""

from typing import Dict, List, Literal
import numpy as np
import pandas as pd

from .constants import C_CM_S


class ExoremOut:
    """
    A wrapper around the DataFrame output for ExoREM simulations.
    All properties return pure floats or NumPy arrays in standard SI/CGS units.
    """

    def __init__(self, results_df: pd.DataFrame):
        self.df = results_df

    def _get(self, column_name: str):
        if column_name not in self.df.columns:
            matching_cols = [c for c in self.df.columns if c.endswith(column_name)]
            if len(matching_cols) != 1:
                raise ValueError(f"Expected exactly one column matching '{column_name}'")
            column_name = matching_cols[0]
        return self.df[column_name].iloc[0]

    def _cols_starting_with(self, prefix: str) -> List[str]:
        return [col for col in self.df.columns if col.startswith(prefix)]

    def _name_from_path(self, path: str) -> str:
        return path.split("/")[-1]

    def _dict_from_prefix(self, prefix: str) -> Dict[str, np.ndarray]:
        out = {}
        for col in self._cols_starting_with(prefix):
            out[self._name_from_path(col)] = np.asarray(self.df[col].iloc[0])
        return out

    # ==========================================
    # Properties (Scalars)
    # ==========================================
    @property
    def cloud_cover(self) -> float:
        return float(self._get("/model_parameters/clouds/fraction"))

    def particle_density(self, specie: Literal["cloud1", "cloud2", "cloud3", "cloud4"] = "cloud1") -> float:
        return float(self._get(f"/model_parameters/clouds/particle_density/{specie}"))

    @property
    def fsed(self) -> Dict[str, float]:
        return self._dict_from_prefix("/model_parameters/clouds/sedimentation_parameter/")

    @property
    def radius_1bar(self) -> float:
        return float(self._get("/model_parameters/target/radius_1e5Pa"))

    @property
    def t_int(self) -> float:
        return float(self._get("/outputs/run_quality/actual_internal_temperature"))

    @property
    def t_eff(self) -> float:
        return float(self._get("/model_parameters/light_source/effective_temperature"))

    @property
    def t_irr(self) -> float:
        return float(self._get("irradiation_temperature"))

    @property
    def chi2_retrieval(self) -> float:
        return float(self._get("/outputs/run_quality/chi2_retrieval"))

    # ==========================================
    # 1D Atmospheric Profiles
    # ==========================================
    @property
    def kzz(self) -> np.ndarray:
        return np.asarray(self._get("/outputs/layers/eddy_diffusion_coefficient"))

    @property
    def gravity(self) -> np.ndarray:
        return np.asarray(self._get("/outputs/layers/gravity"))

    @property
    def mean_molar_mass(self) -> np.ndarray:
        return np.asarray(self._get("/outputs/layers/molar_mass"))

    @property
    def pressure_profile(self) -> np.ndarray:
        return np.asarray(self._get("/outputs/layers/pressure"))

    @property
    def temperature_profile(self) -> np.ndarray:
        return np.asarray(self._get("/outputs/layers/temperature"))

    @property
    def pressure_levels(self) -> np.ndarray:
        return np.asarray(self._get("/outputs/levels/pressure"))

    @property
    def temperature_levels(self) -> np.ndarray:
        return np.asarray(self._get("/outputs/levels/temperature"))

    @property
    def altitude_profile(self) -> np.ndarray:
        return np.asarray(self._get("/outputs/levels/altitude"))

    # ==========================================
    # VMR / Clouds Outputs
    # ==========================================
    @property
    def vmr_absorbers(self) -> Dict[str, np.ndarray]:
        return self._dict_from_prefix("/outputs/layers/volume_mixing_ratios/absorbers/")

    @property
    def vmr_gases(self) -> Dict[str, np.ndarray]:
        return self._dict_from_prefix("/outputs/layers/volume_mixing_ratios/gases/")

    @property
    def vmr_elements_gas_phase(self) -> Dict[str, np.ndarray]:
        return self._dict_from_prefix("/outputs/layers/volume_mixing_ratios/elements_gas_phase/")

    @property
    def vmr_profiles(self) -> Dict[str, np.ndarray]:
        out = {}
        out.update({f"absorber:{k}": v for k, v in self.vmr_absorbers.items()})
        out.update({f"gas:{k}": v for k, v in self.vmr_gases.items()})
        out.update({f"element:{k}": v for k, v in self.vmr_elements_gas_phase.items()})
        return out

    @property
    def cloud_vmrs(self) -> Dict[str, np.ndarray]:
        return self._dict_from_prefix("/outputs/layers/clouds/volume_mixing_ratio/")

    @property
    def cloud_opacities(self) -> Dict[str, np.ndarray]:
        return self._dict_from_prefix("/outputs/layers/clouds/opacity/")

    # ==========================================
    # Spectral Coordinates & Flux
    # ==========================================
    @property
    def wavenumber(self) -> np.ndarray:
        return np.asarray(self._get("/outputs/spectra/wavenumber"))

    @property
    def wavelength(self) -> np.ndarray:
        wn = self.wavenumber
        valid = wn > 0
        wl = np.zeros_like(wn)
        wl[valid] = 10000.0 / wn[valid]
        return wl

    @property
    def emission_spectral_radiosity(self) -> np.ndarray:
        return np.asarray(self._get("/outputs/spectra/emission/spectral_radiosity"))

    @property
    def emission_species(self) -> Dict[str, np.ndarray]:
        return self._dict_from_prefix("/outputs/spectra/emission/contributions/")

    @property
    def flux_wavenumber_density(self) -> np.ndarray:
        return np.asarray(self._get("/outputs/spectra/flux/spectral_flux"))

    @property
    def flux_fnu(self) -> np.ndarray:
        return self.flux_wavenumber_density / C_CM_S

    @property
    def flux_flambda(self) -> np.ndarray:
        wl = self.wavelength
        valid = wl > 0
        flam = np.zeros_like(self.flux_wavenumber_density)
        flam[valid] = self.flux_wavenumber_density[valid] * (10000.0 / (wl[valid] ** 2))
        return flam

    @property
    def transmission(self) -> np.ndarray:
        return np.asarray(self._get("/outputs/spectra/transmission/transit_depth"))

    @property
    def transmission_species(self) -> Dict[str, np.ndarray]:
        return self._dict_from_prefix("/outputs/spectra/transmission/contributions/")

    @property
    def transmission_clear(self) -> np.ndarray:
        return np.asarray(self._get("/outputs/spectra/transmission/transit_depth_clear"))

    @property
    def transmission_full_cover(self) -> np.ndarray:
        return np.asarray(self._get("/outputs/spectra/transmission/transit_depth_full_cover"))

    def summary(self) -> dict:
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