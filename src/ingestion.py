"""Module for downloading and saving transit light curves using Lightkurve."""

import os
import lightkurve as lk



def download_lightcurve(target_name: str, cache_dir: str = "data", exptime: int = 120):
    """
    Downloads and caches the light curve data for a specific exoplanet transit event.
    
    parameters:
    target_name: str
        name of the host star containing the transiting exoplanet on the MAST archive
    chache_dir: str
        defines the location where downloaded lightcurves are stored
    exptime: int
        defines the duration in seconds of the transit requested from the archive
    """

    os.makedirs(cache_dir, exist_ok=True) # creating cache directory if it doesn't exist

    clean_name = target_name.replace(" ", "_").lower()  # clean the target name for file naming
    cache_path = os.path.join(cache_dir, f"{clean_name}_lightcurve.fits")

    if os.path.exists(cache_path):
        print(f"Loading cached light curve for {target_name} from {cache_path}")
        return lk.read(cache_path)

    search_result = lk.search_lightcurve(target_name, author="SPOC", exptime=exptime)

    if len(search_result) == 0:
        raise ValueError(f"No light curve data found for target: {target_name}")

    # download, stitch, and remove NaNs from the light curve data
    lc = search_result[0].download().remove_nans().normalize()
    
    # save to cache
    lc.to_fits(cache_path, overwrite=True)
    print(f"Successfully downloaded and cached to {cache_path}")
    
    return lc