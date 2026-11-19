"""
Module for running transit searches on light curves using Box Least Squares and Transit Least Squares methods.
"""

import time
import lightkurve as lk
import numpy as np
from transitleastsquares import transitleastsquares as tls
from transitleastsquares.warmup import ensure_compiled # warms up numba compiler -> accurate runtime calc


def run_bls(
    lc: lk.LightCurve,
    min_period: float = 0.5,
    max_period: float = 20.0,
    n_periods: int = 20000,
):
    """
    Fast grid search modeling transits as a step function using Box Least Squares (BLS).
    Evaluates a period grid derived from uniformly spaced frequency points against
    15 candidate transit durations spanning 0.02 days to min(0.25 days, 80% of min_period)
    to detect periodic transits.

    Parameters:
    lc: lightkurve.LightCurve
        Target light curve to search for transits.
    min_period: float
        Minimum period to search for (in days).
    max_period: float
        Maximum period to search for (in days).
    n_periods: int
        Number of uniformly spaced frequency grid points to convert into trial periods.
    returns: dict
        Dictionary containing the best-fit BLS parameters.
    """
    t_start = time.perf_counter()

    # Uniform spacing of frequency (f = 1/P) -> dP = P^2 * df, accounting for
    # phase sliping where trial periods do not align with the true period resulting
    # in the phase folded light curve being smeared
    freq_grid = np.linspace(1.0 / max_period, 1.0 / min_period, n_periods)
    period_grid = 1.0 / freq_grid[::-1]  # ascending period order

    # 15 trial transit durations bounded below min_period
    duration_grid = np.linspace(0.02, min(0.25, min_period * 0.8), 15)

    # Run the BLS model on the uniform frequency period grid.
    bls_model = lc.to_periodogram(method="bls", period=period_grid, duration=duration_grid)

    runtime = time.perf_counter() - t_start

    # Calculate the Signal Detection Efficiency (SDE)
    power = np.asarray(bls_model.power.value, dtype=float)
    power_std = np.nanstd(power)
    sde = float((np.nanmax(power) - np.nanmean(power)) / power_std) if power_std > 0 else 0.0

    return {
        "period": float(bls_model.period_at_max_power.value),
        "t0": float(bls_model.transit_time_at_max_power.value),
        "duration": float(bls_model.duration_at_max_power.value),
        "depth": float(bls_model.depth_at_max_power.value),
        "sde": sde,
        "runtime": runtime,
        "model": bls_model,
    }


def run_tls(
    lc: lk.LightCurve,
    min_period: float = 0.5,
    max_period: float = 20.0,
    oversampling_factor: int = 5,
):
    """
    Transit search using limb-darkened transit models via Transit Least Squares.

    parameters:
    lc: lightkurve.LightCurve
        Target light curve to search for transits.
    min_period: float
        Minimum period to search for (in days).
    max_period: float
        Maximum period to search for (in days).
    oversampling_factor: int
        TLS frequency-grid oversampling factor to resolve narrow transit peaks.
    returns: dict
        Dictionary containing the best-fit parameters and the TLS model object.
    """
    # Warm up Numba JIT kernels before starting the benchmark timer so one-time
    # compilation does not distort the first run's runtime measurement.
    ensure_compiled(None, verbose=False)

    # Start timer to measure runtime of the BLS search.
    start_timer = time.perf_counter()

    # Extract arrays from LightCurve object into inputs accepted by TLS
    time_arr = np.asarray(lc.time.value, dtype=float)
    flux = np.asarray(lc.flux.value, dtype=float)

    if lc.flux_err is not None:
        flux_err = np.asarray(lc.flux_err.value, dtype=float)
        if np.all(np.isnan(flux_err)):
            flux_err = None
        else:
            med_err = np.nanmedian(flux_err)
            flux_err = np.nan_to_num(flux_err, nan=med_err)
            flux_err[flux_err <= 0] = med_err
    else:
        flux_err = None

    # Run TLS model with oversampling for finer period resolution
    model = tls(time_arr, flux, flux_err)
    results = model.power(
        period_min=min_period,
        period_max=max_period,
        oversampling_factor=oversampling_factor,
    )

    # Calculate the runtime of the TLS program
    runtime = time.perf_counter() - start_timer

    return {
        "period": float(results.period),
        "t0": float(results.T0),
        "duration": float(results.duration),
        "depth": float(1.0 - results.depth),
        "sde": float(results.SDE),
        "runtime": runtime,
        "model": results,
    }
