"""
exowrap: A modern, Pythonic wrapper for the ExoREM atmosphere model.
"""

__version__ = "0.1.0"

from .model import Simulation
from .plotting import plot_tp_profile, plot_emission_spectrum, plot_transmission_spectrum, plot_vmr_profile

__all__ = [
    "Simulation", 
    "plot_tp_profile", 
    "plot_emission_spectrum", 
    "plot_transmission_spectrum",
    "plot_vmr_profile"
]
