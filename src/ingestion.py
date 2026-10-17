"""Module for fetching and saving transit light curves using Lightkurve."""

import os
import lightkurve as lk



def download_lightcurve(target_name: str, cache_dir = "ExoplanetTransit/data/"):
    """Downloads and caches the light curve data for a specific exoplanet transit event."""

    os.makedirs(cache_dir, exist_ok=True) # creating cache directory if it doesn't exist

    clean_name = target_name.replace(" ", "_").lower()  # clean the target name for file naming
    cache_path = os.path.join(cache_dir, f"{clean_name}_lightcurve.fits")

    if os.path.exists(cache_path):
        print(f"Loading cached light curve for {target_name} from {cache_path}")
        return lk.read(cache_path)

    search_result = lk.search_lightcurve(target_name, author="SPOC")

    if len(search_result) == 0:
        raise ValueError(f"No light curve data found for target: {target_name}")

    # download, stitch, and remove NaNs from the light curve data
    lc = search_result.download_all().stitch().remove_nans()
    
    # save to cache
    lc.to_fits(cache_path, overwrite=True)
    print(f"Successfully downloaded and cached to {cache_path}")
    
    return lc