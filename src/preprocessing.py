"""Module for preprocessing light curves using Gaussian process and savitzky-golay filtering techniques."""

import numpy as np
import lightkurve as lk
import celerite2 as c2
from celerite2 import terms



def filter_savitzky_golay(
    lc: lk.LightCurve,
    window_length: int = 101,
    polynomial_order: int = 2,
    sigma_upper: float = 5.0,
    sigma_lower: float = 20.0,
    ):

    """
    flatten & denoise a light curve using a Savitzky-Golay filter.        
            
    parameters:

    lc : lk.LightCurve
        input light curve.
    window_length : int
        width of the window where polynomial is fitted (odd to ensure exact center)
    sigma_upper / sigma_lower : float
        Outlier clipping thresholds (lower is set lenient to preserve transits).
    """

    cleaned_lc = lc.remove_nans().remove_outliers(sigma_upper=sigma_upper,sigma_lower=sigma_lower)
    flattened_lc = cleaned_lc.flatten(window_length=window_length, polyorder=polynomial_order)
    return flattened_lc



def filter_gaussian_process(
    lc: lk.LightCurve,
    rho: float = 2.0,
    Quality: float = 1.0 / np.sqrt(2.0),
    sigma_upper: float = 5.0,
    sigma_lower: float = 20.0,
    ):

    """
    flatten a light curve using Gaussian Process (GP) detrending via celerite2.
    GP regression often outperforms SG in accuracy due to GP incorporating probability and covariance

    parameters:

    lc : lk.LightCurve
        input light curve.
    rho : float 
        characteristic timescale (period/decay) for the SHO kernel.
    Quality : float
        Quality factor for the SHO kernel (default 1/sqrt(2) for critically damped).
    sigma_upper / sigma_lower : float
        Outlier clipping thresholds (lower is set lenient to preserve transits).
    """

    cleaned_lc = lc.remove_nans().remove_outliers(sigma_upper=sigma_upper,sigma_lower=sigma_lower)
    cleaned_lc.sort("time")
    
    x = np.array(cleaned_lc.time.value, dtype=float)
    flux = np.array(cleaned_lc.flux.value, dtype=float)

    # Normalize uncertainties; celerite2 requires positive, finite errors.
    if cleaned_lc.flux_err is not None:
        yerr = cleaned_lc.flux_err.value
        yerr = np.nan_to_num(yerr, nan=np.nanmedian(yerr))
        yerr[yerr <= 0] = np.nanmedian(yerr)
    else:
        yerr = np.full_like(flux, np.nanstd(flux))

    # Mean-center the flux for GP modeling
    median_flux = np.nanmedian(flux)
    y = flux - median_flux

    # Mask downward outliers (> 3 sigma below median) so the GP only fits
    # out-of-transit stellar variability and does not absorb transit dips
    mad_std = 1.4826 * np.nanmedian(np.abs(y)) 
    # 1.4826 is a conversion factor to change between Median Absolute Deviation (MAD)
    # and the equivilent standard deviation for the data
    
    oot_mask = y > (-3.0 * mad_std)
    if np.sum(oot_mask) < 10:
        oot_mask = np.ones_like(y, dtype=bool)

    # Estimate white-noise jitter from point-to-point scatter and add to diagonal variance
    # (In celerite2, diagonal jitter is passed via `diag` in gp.compute rather than a separate JitterTerm)
    jitter = 1.4826 * np.nanmedian(np.abs(np.diff(y[oot_mask]))) / np.sqrt(2.0)
    diag = yerr**2 + jitter**2

    # Initialize celerite2 SHO kernel and condition only on out-of-transit points
    kernel_sigma = np.nanstd(y[oot_mask])
    kernel = terms.SHOTerm(sigma=kernel_sigma, rho=rho, Q=Quality)
    gp = c2.GaussianProcess(kernel, mean=0.0)
    gp.compute(x[oot_mask], diag=diag[oot_mask])

    # Predict the stellar baseline trend across all timestamps (interpolating over transits)
    trend = gp.predict(y[oot_mask], t=x) + median_flux

    # Multiplicative detrending (flux / trend)
    flat_flux = flux / trend
    flat_flux_err = yerr / trend

    # Return a new LightCurve object preserving time and metadata
    flat_lc = lk.LightCurve(
        time=cleaned_lc.time,
        flux=flat_flux,
        flux_err=flat_flux_err,
        meta=cleaned_lc.meta,
    )
    return flat_lc