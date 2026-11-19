"""
Module for downloading and saving transit light curves using Lightkurve.
"""

import os
import lightkurve as lk


def download_lightcurve(target_name: str, cache_dir: str = "data", exptime: int = 120):
    """
    Downloads and caches the light curve data for a specific exoplanet transit event.

    Parameters:
    target_name: str
        Name of the host star containing the transiting exoplanet on the MAST archive
    cache_dir: str
        Defines the location where downloaded lightcurves are stored
    exptime: int
        Exposure time (cadence) in seconds requested from the archive
    """

    os.makedirs(cache_dir, exist_ok=True)  # Creating cache directory if it doesn't exist

    clean_name = target_name.replace(" ", "_").lower()  # Clean the target name for safe file naming
    cache_path = os.path.join(cache_dir, f"{clean_name}_lightcurve.fits")

    if os.path.exists(cache_path):
        print(f"Loading cached light curve for {target_name} from {cache_path}")
        return lk.read(cache_path)

    search_result = lk.search_lightcurve(target_name, author="SPOC", exptime=exptime)

    if len(search_result) == 0:
        raise ValueError(f"No light curve data found for target: {target_name}")

    # Download, remove NaNs, and normalize the light curve data
    lc = search_result[0].download().remove_nans().normalize()

    # Save to cache
    lc.to_fits(cache_path, overwrite=True)
    print(f"Successfully downloaded and cached to {cache_path}")

    return lc
