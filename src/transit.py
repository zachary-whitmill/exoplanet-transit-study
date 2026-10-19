"""
Module for running transit searches on light curves using Box Least Squares and Transit Least Squares methods.
"""

import time
import lightkurve as lk
import numpy as np
from transitleastsquares import transitleastsquares as tls

def run_bls(lc: lk.LightCurve, min_period: float = 0.5, max_period: float = 20.0):
    """
    Fast grid search modeling transits as a step function using Box Least Squares.
    Evaluates a 10,000-point period grid against 15 candidate transit durations
    spanning 0.02days to smallest from 6hrs / 80% of min duration to detect periodic flux dips.

    parameters:
    lc: lightkurve.LightCurve
        target light curve to search for transits.
    min_period: float
        minimum period to search for (in days).
    max_period: float
        maximum period to search for (in days).
    returns: dict
        dictionary containing the best-fit parameters and the BLS model object.
    """
    # start timer to measure runtime of the BLS search.
    t_start = time.perf_counter() 

    # define trial search space: 10,000  different periods to ensure sufficient resolution.
    period_grid = np.linspace(min_period, max_period, 10000)

    # create 15 trial transit durations to ensure sensible duration wrt period
    duration_grid = np.linspace(0.02, min(0.25, min_period * 0.8), 15)

    # run the BLS model 
    bls_model = lc.to_periodogram(method='bls', period=period_grid, duration=duration_grid)

    # measure the runtime of the BLS search.
    runtime_s = time.perf_counter() - t_start

    # compute the Signal Detection Efficiency (SDE), the significance of the detected signal.
    power = np.asarray(bls_model.power.value, dtype=float)
    power_std = np.nanstd(power)
    sde = float((np.nanmax(power) - np.nanmean(power)) / power_std) if power_std > 0 else 0.0

    return {
        "period": float(bls_model.period_at_max_power.value),
        "t0": float(bls_model.transit_time_at_max_power.value),
        "duration": float(bls_model.duration_at_max_power.value),
        "depth": float(bls_model.depth_at_max_power.value),
        "sde": sde,
        "runtime_s": runtime_s,
        "model": bls_model
    }

def run_tls(lc: lk.LightCurve, min_period: float = 0.5, max_period: float = 20.0):
    """

    """

    # start timer to measure runtime of the BLS search.
    t_start = time.perf_counter()

    # extracting data from lightkurve object into inputs tls can accept.
    time_arr = np.asarray(lc.time.value, dtype=float)
    flux = np.asarray(lc.flux.value, dtype=float)

    if lc.flux_err is not None:
        flux_err = np.asarray(lc.flux_err.value, dtype=float)
        if np.all(np.isnan(flux_err)):
            # removes all NaN entries
            flux_err = None
        else:
            med_err = np.nanmedian(flux_err)
            # replaces all invalid/NaN flux_err with med_err.
            flux_err = np.nan_to_num(flux_err, nan=med_err)
            flux_err[flux_err <= 0] = med_err
    else:
        flux_err = None

    # run TLS model.
    model = tls(time_arr, flux, flux_err)
    results = model.power(period_min=min_period, period_max=max_period)
    
    # measure the runtime of the TLS search.
    runtime_s = time.perf_counter() - t_start

    return {
        "period": float(results.period),
        "t0": float(results.T0),
        "duration": float(results.duration),
        "depth": float(1.0 - results.depth),
        "sde": float(results.SDE),
        "runtime_s": runtime_s,
        "model": results
    }