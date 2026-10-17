import lightkurve as lk
import numpy as np
from transitleastsquares import transitleastsquares as tls



def run_bls(lc: lk.LightCurve, min_period: float = 0.5, max_period: float = 20.0):
    """
    Fast grid search modeling transits as rectangular dips using Box Least Squares.
    """

    period_grid = np.linspace(min_period, max_period, 10000)
    bls_model = lc.to_periodogram(method='bls', period=period_grid, frequency_factor=500)
    
    return {
        "period": bls_model.period_at_max_power.value,
        "t0": bls_model.transit_time_at_max_power.value,
        "duration": bls_model.duration_at_max_power.value,
        "depth": bls_model.depth_at_max_power.value,
        "model": bls_model
    }

def run_tls(lc: lk.LightCurve):
    """
    Physical limb-darkened model search via Transit Least Squares.
    """
    
    model = tls(lc.time.value, lc.flux.value, lc.flux_err.value)
    results = model.power()
    
    return {
        "period": results.period,
        "t0": results.T0,
        "duration": results.duration,
        "depth": 1.0 - results.depth,
        "model": results
    }