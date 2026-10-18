"""needs work"""

# tests/test_preprocessing.py
import numpy as np
import lightkurve as lk
from src.preprocessing import detrend_savgol

def test_savgol_detrending():
    """
    generates a synthetic noisy light curve
    """
    time = np.linspace(0, 10, 1000)
    flux = np.ones(1000) + np.random.normal(0, 0.01, 1000)
    flux_err = np.full(1000, 0.01)
    
    lc = lk.LightCurve(time=time, flux=flux, flux_err=flux_err)
    
    # run the preprocessing function
    flat_lc = detrend_savgol(lc, window_length=15, polynomial_order=2)
    
    # assert the outputs meet strict software expectations
    assert isinstance(flat_lc, lk.LightCurve), "Output must be a LightCurve object"
    assert len(flat_lc.flux) == len(lc.flux), "Data points were lost during detrending"
    assert np.isclose(np.mean(flat_lc.flux.value), 1.0, atol=0.01), "Detrending failed to normalize flux to 1.0"