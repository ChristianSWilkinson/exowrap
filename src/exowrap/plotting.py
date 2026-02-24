import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd

def plot_tp_profile(results_df: pd.DataFrame, ax=None, title="T-P Profile", color='darkred', lw=2):
    """
    Plots the Temperature-Pressure profile from an ExoREM results DataFrame.
    """
    if results_df.empty:
        print("Error: DataFrame is empty. Cannot plot.")
        return None
        
    try:
        # Extract data (ExoREM pressure is in Pascals, convert to bar)
        pressure_pa = results_df['/outputs/layers/pressure'].iloc[0]
        pressure_bar = np.array(pressure_pa) / 1e5
        temperature_k = results_df['/outputs/layers/temperature'].iloc[0]
        
        # Create plot if no axis is provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 8))
            
        ax.plot(temperature_k, pressure_bar, color=color, lw=lw)
        
        # Atmospheric science standard formatting
        ax.set_yscale('log')
        if not ax.yaxis_inverted():
            ax.invert_yaxis() # High pressure (deep atmosphere) at the bottom
            
        ax.set_xlabel('Temperature (K)', fontsize=14)
        ax.set_ylabel('Pressure (bar)', fontsize=14)
        ax.set_title(title, fontsize=14)
        ax.grid(True, which="both", ls="--", alpha=0.5)
        
        if ax is None:
            plt.tight_layout()
            
        return ax
        
    except KeyError as e:
        print(f"Error: Missing key in HDF5 results: {e}")
        return None

def plot_emission_spectrum(results_df: pd.DataFrame, ax=None, title="Emission Spectrum", color='black', lw=2, contributions=False):
    """
    Plots the Emission Spectrum (converted to wavelength in microns).
    
    Args:
        results_df: The DataFrame containing the ExoREM HDF5 output.
        ax: Optional matplotlib axis.
        title: Plot title.
        color: Color of the total spectrum line.
        lw: Line width of the total spectrum.
        contributions (bool or list): If True, plots all available contributions. 
                                      If a list of strings (e.g., ['H2O', 'CH4', 'cia_rayleigh']), plots only those.
    """
    if results_df.empty:
        print("Error: DataFrame is empty. Cannot plot.")
        return None
        
    try:
        wavenumber = results_df['/outputs/spectra/wavenumber'].iloc[0]
        total_radiosity = results_df['/outputs/spectra/emission/spectral_radiosity'].iloc[0]
        
        # Convert wavenumber (cm^-1) to wavelength (microns)
        valid_idx = wavenumber > 0
        wavenumber = wavenumber[valid_idx]
        total_radiosity = total_radiosity[valid_idx]
        
        wavelength_um = 10000.0 / wavenumber
        
        # Sort by wavelength for clean plotting
        sort_idx = np.argsort(wavelength_um)
        wavelength_um = wavelength_um[sort_idx]
        total_radiosity = total_radiosity[sort_idx]

        created_ax = ax is None
        if created_ax:
            # Make the figure slightly wider to accommodate the external legend
            fig, ax = plt.subplots(figsize=(12, 5))
            
        # Plot CONTRIBUTIONS FIRST so they stay in the background
        if contributions:
            prefix = '/outputs/spectra/emission/contributions/'
            contrib_cols = [c for c in results_df.columns if c.startswith(prefix)]
            
            # If a list of specific molecules was provided, filter the columns
            if isinstance(contributions, list):
                contrib_cols = [c for c in contrib_cols if c.split('/')[-1] in contributions]
                
            for col in contrib_cols:
                name = col.split('/')[-1] # Extract the molecule/component name
                contrib_rad = results_df[col].iloc[0]
                
                # Apply the same wavelength filtering and sorting
                contrib_rad = contrib_rad[valid_idx]
                contrib_rad = contrib_rad[sort_idx]
                
                # Plot the contribution with a thinner line and transparency
                # These will visually "hang down" from the continuum as absorption features
                ax.plot(wavelength_um, contrib_rad, lw=1.2, alpha=0.7, label=name)

        # Plot the TOTAL spectrum on top (zorder=10 ensures it draws over everything else)
        ax.plot(wavelength_um, total_radiosity, color=color, lw=lw, label='Total Spectrum', zorder=10)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Wavelength ($\\mu m$)', fontsize=14)
        ax.set_ylabel('Spectral Radiosity', fontsize=14)
        ax.set_title(title, fontsize=14)
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        
        # Only show legend if we plotted contributions
        if contributions:
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)
        
        if created_ax:
            plt.tight_layout()
            
        return ax

    except KeyError as e:
        print(f"Error: Missing spectral keys in HDF5 results: {e}")
        return None
    
def plot_transmission_spectrum(results_df: pd.DataFrame, ax=None, title="Transmission Spectrum", color='black', lw=2, contributions=False):
    """
    Plots the Transmission (Transit) Spectrum.
    Y-axis is the Transit Depth in percentage (%).
    X-axis is Wavelength in microns.
    
    Args:
        results_df: The DataFrame containing the ExoREM HDF5 output.
        ax: Optional matplotlib axis.
        title: Plot title.
        color: Color of the total spectrum line.
        lw: Line width of the total spectrum.
        contributions (bool or list): If True, plots all available contributions. 
                                      If a list of strings (e.g., ['H2O', 'CH4']), plots only those.
    """
    if results_df.empty:
        print("Error: DataFrame is empty. Cannot plot.")
        return None
        
    try:
        wavenumber = results_df['/outputs/spectra/wavenumber'].iloc[0]
        transit_depth = results_df['/outputs/spectra/transmission/transit_depth'].iloc[0]

        # Convert wavenumber (cm^-1) to wavelength (microns)
        valid_idx = wavenumber > 0
        wavenumber = wavenumber[valid_idx]
        transit_depth = transit_depth[valid_idx]
        
        wavelength_um = 10000.0 / wavenumber
        
        # Convert transit depth to percentage
        transit_depth_pct = transit_depth * 100.0
        
        # Sort by wavelength
        sort_idx = np.argsort(wavelength_um)
        wavelength_um = wavelength_um[sort_idx]
        transit_depth_pct = transit_depth_pct[sort_idx]

        created_ax = ax is None
        if created_ax:
            # Make the figure slightly wider to accommodate the legend
            fig, ax = plt.subplots(figsize=(12, 5))
            
        # Plot the TOTAL spectrum (draw it on top with a higher zorder)
        ax.plot(wavelength_um, transit_depth_pct, color=color, lw=lw, label='Total Spectrum', zorder=10)
        
        # Plot CONTRIBUTIONS if requested
        if contributions:
            prefix = '/outputs/spectra/transmission/contributions/'
            # Find all column names that start with this prefix
            contrib_cols = [c for c in results_df.columns if c.startswith(prefix)]
            
            # If a list of specific molecules was provided, filter the columns
            if isinstance(contributions, list):
                contrib_cols = [c for c in contrib_cols if c.split('/')[-1] in contributions]
                
            for col in contrib_cols:
                name = col.split('/')[-1] # Extract the molecule name (e.g., 'H2O')
                contrib_depth = results_df[col].iloc[0]
                
                # Apply the same wavelength filtering, scaling, and sorting
                contrib_depth = contrib_depth[valid_idx]
                contrib_depth_pct = contrib_depth * 100.0
                contrib_depth_pct = contrib_depth_pct[sort_idx]
                
                # Plot the contribution with a thinner line and some transparency
                ax.plot(wavelength_um, contrib_depth_pct, lw=1.5, alpha=0.7, label=name)

        ax.set_xscale('log')
        ax.set_xlabel('Wavelength ($\\mu m$)', fontsize=14)
        ax.set_ylabel('Transit Depth (%)', fontsize=14)
        ax.set_title(title, fontsize=14)
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        
        # Only show legend if we plotted contributions
        if contributions:
            # Place the legend outside the plot area so it doesn't cover the data
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)
        
        if created_ax:
            # Adjust layout so the external legend isn't cut off
            plt.tight_layout()
            
        return ax

    except KeyError as e:
        print(f"Error: Missing spectral keys in HDF5 results: {e}")
        return None
    
def plot_vmr_profile(results_df: pd.DataFrame, molecules=None, ax=None, title="Chemical Abundances (VMR)"):
    """
    Plots the Volume Mixing Ratio (VMR) of atmospheric molecules vs Pressure.
    If molecules is None, plots a few major expected species.
    """
    if results_df.empty:
        print("Error: DataFrame is empty. Cannot plot.")
        return None
        
    try:
        pressure_pa = results_df['/outputs/layers/pressure'].iloc[0]
        pressure_bar = np.array(pressure_pa) / 1e5
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 8))
            
        # Default molecules to look for if the user doesn't specify
        if molecules is None:
            molecules = ['H2O', 'CH4', 'CO', 'CO2', 'NH3', 'TiO', 'VO']
            
        plotted_any = False
        for mol in molecules:
            key = f'/outputs/layers/volume_mixing_ratios/absorbers/{mol}'
            if key in results_df.columns:
                vmr = results_df[key].iloc[0]
                ax.plot(vmr, pressure_bar, lw=2, label=mol)
                plotted_any = True
                
        if not plotted_any:
            print("Warning: None of the requested molecules were found in the output.")
            return ax
            
        # Format the plot
        ax.set_yscale('log')
        ax.set_xscale('log')
        if not ax.yaxis_inverted():
            ax.invert_yaxis()
            
        ax.set_xlabel('Volume Mixing Ratio (VMR)', fontsize=14)
        ax.set_ylabel('Pressure (bar)', fontsize=14)
        ax.set_title(title, fontsize=14)
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(loc='best')
        
        if ax is None:
            plt.tight_layout()
            
        return ax

    except KeyError as e:
        print(f"Error: Missing keys for VMR plot: {e}")
        return None