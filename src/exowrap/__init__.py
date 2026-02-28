__version__ = "0.1.0"

from .model import Simulation
from .plotting import (
    plot_tp_profile, 
    plot_emission_spectrum, 
    plot_transmission_spectrum, 
    plot_vmr_profile,
    plot_model_summary
)
from .output import ExoremOut
from .tools import upgrade_resolution
from .photometry import compute_photometry, get_svo_filter # <-- Added

__all__ = [
    "Simulation", 
    "plot_tp_profile", 
    "plot_emission_spectrum", 
    "plot_transmission_spectrum", 
    "plot_vmr_profile",
    "plot_model_summary",
    "ExoremOut",
    "upgrade_resolution",
    "compute_photometry",
    "get_svo_filter",
    "search_svo_filters"
]