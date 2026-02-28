"""
Photometry Module for exowrap.

Provides tools to seamlessly download filter transmission curves from the 
SVO Filter Profile Service (FPS) and compute synthetic photometry from 
ExoREM atmospheric spectra.
"""

import urllib.request
import urllib.error
import re
from io import StringIO
from typing import Dict, Tuple, List

import numpy as np

from .output import ExoremOut


def _clean_url(raw_url: str) -> str:
    """Aggressively strips markdown/HTML hyperlink formatting injected by text editors."""
    # If the editor made it [url](url), split by ] and take the first part
    clean = raw_url.split("](")[0] 
    # Remove any remaining brackets or angle brackets
    for char in ["[", "]", "<", ">"]:
        clean = clean.replace(char, "")
    return clean


def search_svo_filters(facility: str) -> List[str]:
    """
    Searches the SVO FPS database for all available filters for a given facility.
    """
    # Splitting the string so the text editor doesn't auto-hyperlink it on paste!
    host = "https://" + "svo2.cab.inta-csic.es/theory/fps/fps.php?Facility="
    url = _clean_url(f"{host}{facility}")
    
    try:
        response = urllib.request.urlopen(url)
        data = response.read().decode('utf-8')
    except Exception as e:
        print(f"Failed to query SVO FPS using URL {url}: {e}")
        return []
        
    matches = re.findall(r'<TD>([A-Za-z0-9_\-\+]+/[A-Za-z0-9_\-\.\+]+)</TD>', data)
    return sorted(list(set(matches)))


def get_svo_filter(filter_id: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downloads a filter transmission curve from the SVO Filter Profile Service.
    """
    # Splitting the string again to hide it from the editor
    host = "https://" + "svo2.cab.inta-csic.es/theory/fps/getdata.php?format=ascii&id="
    url = _clean_url(f"{host}{filter_id}")
    
    try:
        response = urllib.request.urlopen(url)
        data = response.read().decode('utf-8')
    except urllib.error.URLError as e:
        raise ValueError(f"Failed to connect to SVO FPS using URL {url}: {e}")

    if not data.strip() or "<html" in data.lower() or "No filter found" in data:
        raise ValueError(f"Filter '{filter_id}' could not be found on SVO FPS.")

    try:
        parsed_data = np.genfromtxt(StringIO(data))
        if parsed_data.size == 0 or parsed_data.ndim != 2:
            raise ValueError
    except Exception:
        raise ValueError(f"Could not parse the filter data for '{filter_id}'.")

    wav_angstroms = parsed_data[:, 0]
    transmission = parsed_data[:, 1]

    # Convert Angstroms to Microns to match ExoREM
    wav_microns = wav_angstroms * 1e-4

    return wav_microns, transmission


def compute_photometry(
    exo: ExoremOut, 
    filter_id: str, 
    photon_counting: bool = True
) -> Dict[str, float]:
    """
    Computes synthetic photometry for an ExoREM spectrum using a specific filter.
    """
    filt_wav, filt_trans = get_svo_filter(filter_id)

    exo_wl = exo.wavelength
    valid = exo_wl > 0
    exo_wl = exo_wl[valid]
    exo_flux = exo.flux_flambda[valid]

    sort_idx = np.argsort(exo_wl)
    exo_wl = exo_wl[sort_idx]
    exo_flux = exo_flux[sort_idx]

    interp_trans = np.interp(exo_wl, filt_wav, filt_trans, left=0.0, right=0.0)

    if np.sum(interp_trans) == 0:
        raise ValueError(
            f"The filter '{filter_id}' ({filt_wav[0]:.2f} - {filt_wav[-1]:.2f} μm) "
            "does not overlap with the ExoREM spectral range."
        )

    eff_wav = np.trapz(interp_trans * exo_wl, exo_wl) / np.trapz(interp_trans, exo_wl)

    if photon_counting:
        numerator = np.trapz(exo_flux * interp_trans * exo_wl, exo_wl)
        denominator = np.trapz(interp_trans * exo_wl, exo_wl)
    else:
        numerator = np.trapz(exo_flux * interp_trans, exo_wl)
        denominator = np.trapz(interp_trans, exo_wl)

    phot_flux_flambda = numerator / denominator

    c_um_s = 299792458.0 * 1e6
    phot_flux_fnu = phot_flux_flambda * (eff_wav**2) / c_um_s
    phot_flux_jy = phot_flux_fnu * 1e26

    return {
        "filter_id": filter_id,
        "effective_wavelength_um": eff_wav,
        "flux_W_m2_um": phot_flux_flambda,
        "flux_Jy": phot_flux_jy
    }