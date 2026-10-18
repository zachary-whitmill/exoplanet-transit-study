import lightkurve as lk
import numpy as np
from transitleastsquares import transitleastsquares as tls



def run_bls(lc: lk.LightCurve, min_period: float = 0.5, max_period: float = 20.0):
    """
    Fast grid search modeling transits as rectangular dips using Box Least Squares.
    """

    period_grid = np.linspace(min_period, max_period, 10000)
    duration_grid = np.linspace(0.02, min(0.25, min_period * 0.8), 15)
    bls_model = lc.to_periodogram(method='bls', period=period_grid, duration=duration_grid)
    
    return {
        "period": bls_model.period_at_max_power.value,
        "t0": bls_model.transit_time_at_max_power.value,
        "duration": bls_model.duration_at_max_power.value,
        "depth": bls_model.depth_at_max_power.value,
        "model": bls_model
    }

def run_tls(lc: lk.LightCurve):
    """
    limb-darkened model search via Transit Least Squares.
    """
    time = np.asarray(lc.time.value, dtype=float)
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

    model = tls(time, flux, flux_err)
    results = model.power()
    
    return {
        "period": results.period,
        "t0": results.T0,
        "duration": results.duration,
        "depth": 1.0 - results.depth,
        "model": results
    }
    