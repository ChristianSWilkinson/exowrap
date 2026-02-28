"""
Plotting Module for exowrap.

Provides publication-ready visualization functions for exoplanet atmospheres.
Compatible with both raw pandas DataFrames and ExoremOut objects.
"""

from typing import List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter

from .output import ExoremOut


# ==========================================
# Internal Plotting Helpers
# ==========================================

def _latex_name(name: str) -> str:
    """Escapes underscores for matplotlib LaTeX rendering."""
    return name.replace("_", r"\_")

def _setup_pressure_axis(ax: plt.Axes, p: np.ndarray) -> plt.Axes:
    """Applies standard log-scale formatting to a pressure y-axis."""
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_ylabel(r"$P\ \mathrm{(bar)}$")
    ax.grid(True, alpha=0.3)
    return ax

def _sorted_spectral_index(x: np.ndarray) -> np.ndarray:
    """Returns indices to sort the spectral axis for clean line plotting."""
    return np.argsort(x)

def _x_spectral(
    exo: ExoremOut, x_axis: Literal["wavelength", "wavenumber"] = "wavelength"
) -> Tuple[np.ndarray, str, np.ndarray]:
    """Extracts and formats the appropriate X-axis for spectral plots."""
    if x_axis == "wavelength":
        x = exo.wavelength
        xlabel = r"$\lambda\ \mathrm{(\mu m)}$"
    else:
        x = exo.wavenumber
        xlabel = r"$\tilde{\nu}\ \mathrm{(cm^{-1})}$"
    idx = _sorted_spectral_index(x)
    return x, xlabel, idx


# ==========================================
# Original Public Plotting API (Backward Compatible)
# ==========================================

def plot_tp_profile(
    results_df: Union[pd.DataFrame, ExoremOut],
    ax: Optional[plt.Axes] = None,
    title: str = "T-P Profile",
    color: str = "darkred",
    lw: float = 2.0,
) -> Optional[plt.Axes]:
    """Plots the Temperature-Pressure profile."""
    # Ensure we are working with an ExoremOut object
    if isinstance(results_df, pd.DataFrame):
        if results_df.empty:
            print("Error: DataFrame is empty. Cannot plot.")
            return None
        exo = ExoremOut(results_df)
    else:
        exo = results_df

    try:
        # Convert Pa to bar for legacy compatibility
        p_bar = exo.pressure_profile / 1e5
        t = exo.temperature_profile

        created_ax = ax is None
        if created_ax:
            fig, ax = plt.subplots(figsize=(6, 8))

        ax.plot(t, p_bar, color=color, lw=lw, label=r"$T(P)$")
        _setup_pressure_axis(ax, p_bar)
        
        # Override the label from the helper to match original
        ax.set_xlabel("Temperature (K)", fontsize=14)
        ax.set_ylabel("Pressure (bar)", fontsize=14)
        ax.set_title(title, fontsize=14)

        if created_ax:
            plt.tight_layout()

        return ax

    except Exception as e:
        print(f"Error plotting T-P Profile: {e}")
        return None


def plot_emission_spectrum(
    results_df: Union[pd.DataFrame, ExoremOut],
    ax: Optional[plt.Axes] = None,
    title: str = "Emission Spectrum",
    color: str = "black",
    lw: float = 2.0,
    contributions: Union[bool, List[str]] = False,
) -> Optional[plt.Axes]:
    """Plots the Emission Spectrum (converted to wavelength in microns)."""
    if isinstance(results_df, pd.DataFrame):
        if results_df.empty:
            print("Error: DataFrame is empty. Cannot plot.")
            return None
        exo = ExoremOut(results_df)
    else:
        exo = results_df

    try:
        x, xlabel, idx = _x_spectral(exo, x_axis="wavelength")
        
        # Convert radiosity to lambda space
        valid = x > 0
        ytot = np.zeros_like(exo.emission_spectral_radiosity)
        ytot[valid] = exo.emission_spectral_radiosity[valid] * (10000.0 / (x[valid]**2))

        created_ax = ax is None
        if created_ax:
            fig, ax = plt.subplots(figsize=(12, 5))

        # Plot CONTRIBUTIONS FIRST so they stay in the background
        if contributions:
            species_dict = exo.emission_species
            
            # Filter if a specific list was provided
            if isinstance(contributions, list):
                species_dict = {
                    k: v for k, v in species_dict.items() if k in contributions
                }

            for name, q in species_dict.items():
                y = np.zeros_like(q)
                y[valid] = q[valid] * (10000.0 / (x[valid]**2))
                ax.plot(
                    x[idx], y[idx], lw=1.2, alpha=0.7, 
                    label=rf"${_latex_name(name)}$"
                )

        # Plot the TOTAL spectrum
        ax.plot(
            x[idx], ytot[idx], color=color, lw=lw,
            label="Total Spectrum", zorder=10
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Wavelength ($\\mu m$)", fontsize=14)
        ax.set_ylabel("Spectral Radiosity", fontsize=14)
        ax.set_title(title, fontsize=14)
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.xaxis.set_major_formatter(ScalarFormatter())

        if contributions:
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10)

        if created_ax:
            plt.tight_layout()

        return ax

    except Exception as e:
        print(f"Error plotting Emission Spectrum: {e}")
        return None


def plot_transmission_spectrum(
    results_df: Union[pd.DataFrame, ExoremOut],
    ax: Optional[plt.Axes] = None,
    title: str = "Transmission Spectrum",
    color: str = "black",
    lw: float = 2.0,
    contributions: Union[bool, List[str]] = False,
) -> Optional[plt.Axes]:
    """Plots the Transmission (Transit) Spectrum in percentage."""
    if isinstance(results_df, pd.DataFrame):
        if results_df.empty:
            print("Error: DataFrame is empty. Cannot plot.")
            return None
        exo = ExoremOut(results_df)
    else:
        exo = results_df

    try:
        x, xlabel, idx = _x_spectral(exo, x_axis="wavelength")
        
        # Convert depth to percentage
        ytot_pct = exo.transmission * 100.0

        created_ax = ax is None
        if created_ax:
            fig, ax = plt.subplots(figsize=(12, 5))

        # Plot the TOTAL spectrum
        ax.plot(
            x[idx], ytot_pct[idx], color=color, lw=lw,
            label="Total Spectrum", zorder=10
        )

        # Plot CONTRIBUTIONS if requested
        if contributions:
            species_dict = exo.transmission_species
            
            if isinstance(contributions, list):
                species_dict = {
                    k: v for k, v in species_dict.items() if k in contributions
                }

            for name, q in species_dict.items():
                y_pct = q * 100.0
                ax.plot(
                    x[idx], y_pct[idx], lw=1.5, alpha=0.7, 
                    label=rf"${_latex_name(name)}$"
                )

        ax.set_xscale("log")
        ax.set_xlabel("Wavelength ($\\mu m$)", fontsize=14)
        ax.set_ylabel("Transit Depth (%)", fontsize=14)
        ax.set_title(title, fontsize=14)
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.xaxis.set_major_formatter(ScalarFormatter())

        if contributions:
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10)

        if created_ax:
            plt.tight_layout()

        return ax

    except Exception as e:
        print(f"Error plotting Transmission Spectrum: {e}")
        return None


def plot_vmr_profile(
    results_df: Union[pd.DataFrame, ExoremOut],
    molecules: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Chemical Abundances (VMR)",
) -> Optional[plt.Axes]:
    """Plots the Volume Mixing Ratio (VMR) of atmospheric molecules vs Pressure."""
    if isinstance(results_df, pd.DataFrame):
        if results_df.empty:
            print("Error: DataFrame is empty. Cannot plot.")
            return None
        exo = ExoremOut(results_df)
    else:
        exo = results_df

    try:
        p_bar = exo.pressure_profile / 1e5
        species_dict = exo.vmr_absorbers

        created_ax = ax is None
        if created_ax:
            fig, ax = plt.subplots(figsize=(6, 8))

        if molecules is None:
            molecules = ["H2O", "CH4", "CO", "CO2", "NH3", "TiO", "VO"]

        plotted_any = False
        for mol in molecules:
            if mol in species_dict:
                vmr = species_dict[mol]
                ax.semilogx(
                    np.clip(vmr, 1e-30, None), p_bar, 
                    lw=2, label=rf"${_latex_name(mol)}$"
                )
                plotted_any = True

        if not plotted_any:
            print("Warning: None of the requested molecules were found in the output.")
            return ax

        _setup_pressure_axis(ax, p_bar)
        ax.set_xlabel("Volume Mixing Ratio (VMR)", fontsize=14)
        ax.set_ylabel("Pressure (bar)", fontsize=14)
        ax.set_title(title, fontsize=14)
        ax.legend(loc="best")

        if created_ax:
            plt.tight_layout()

        return ax

    except Exception as e:
        print(f"Error plotting VMR profiles: {e}")
        return None

# ==========================================
# New Advanced Plotting Tools
# ==========================================

def plot_model_summary(results_df: Union[pd.DataFrame, ExoremOut]):
    """Generates a comprehensive 4-panel summary of the atmospheric model."""
    if isinstance(results_df, pd.DataFrame):
        exo = ExoremOut(results_df)
    else:
        exo = results_df
        
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax_text, ax_pt, ax_flux, ax_tr = axes.flatten()

    # Panel 1: Text Summary
    ax_text.axis("off")
    summary_text = (
        rf"$T_{{\mathrm{{int}}}} = {exo.t_int:.2f}\ \mathrm{{K}}$" "\n"
        rf"$f_{{\mathrm{{cloud}}}} = {exo.cloud_cover:.3f}$" "\n"
        rf"$\chi^2 = {exo.chi2_retrieval:.3g}$" "\n"
        rf"$N_{{\lambda}} = {len(exo.wavelength)}$"
    )
    ax_text.text(0.03, 0.97, summary_text, va="top", ha="left", fontsize=14)

    # Panel 2: P-T Profile
    plot_tp_profile(exo, ax=ax_pt, title="")

    # Panel 3: Flux
    x, _, idx = _x_spectral(exo, x_axis="wavelength")
    ax_flux.plot(x[idx], exo.flux_flambda[idx], lw=1.8, color="tab:blue")
    ax_flux.set_xlabel(r"$\lambda\ \mathrm{(\mu m)}$")
    ax_flux.set_yscale('log')
    ax_flux.set_xscale('log')
    ax_flux.set_ylabel(r"$F_{\lambda}\ \mathrm{(W\ m^{-2}\ \mu m^{-1})}$")
    ax_flux.set_title("Emission Flux")
    ax_flux.grid(True, alpha=0.3)

    # Panel 4: Transmission
    ax_tr.plot(x[idx], exo.transmission[idx], lw=1.8, color="tab:green")
    ax_tr.set_xlabel(r"$\lambda\ \mathrm{(\mu m)}$")
    ax_tr.set_ylabel(r"$D_{\mathrm{tr}}\ \mathrm{(-)}$")
    ax_tr.set_title("Transmission")
    ax_tr.set_xscale('log')
    ax_tr.grid(True, alpha=0.3)

    fig.suptitle("ExoREM Model Summary", y=0.995, fontsize=16, fontweight='bold')
    fig.tight_layout()
    return fig, axes