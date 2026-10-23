"""Unit tests for light curve preprocessing (Savitzky-Golay and Gaussian Process detrending)."""

# tests/test_preprocessing.py
import numpy as np
import lightkurve as lk
from src.preprocessing import filter_savitzky_golay, filter_gaussian_process

def test_savgol_detrending():
    """
    generates a synthetic noisy light curve
    """
    rng = np.random.default_rng(42)
    time = np.linspace(0, 10, 1000)
    flux = np.ones(1000) + rng.normal(0, 0.01, 1000)
    flux_err = np.full(1000, 0.01)
    
    lc = lk.LightCurve(time=time, flux=flux, flux_err=flux_err)
    
    # run the preprocessing function
    flat_lc = filter_savitzky_golay(lc, window_length=15, polynomial_order=2)
    
    # assert the outputs meet strict software expectations
    assert isinstance(flat_lc, lk.LightCurve), "Output must be a LightCurve object"
    assert len(flat_lc.flux) == len(lc.flux), "Data points were lost during detrending"
    assert np.isclose(np.mean(flat_lc.flux.value), 1.0, atol=0.01), "Detrending failed to normalize flux to 1.0"


def test_gp_detrending_preserves_transit():
    """
    verifies Gaussian Process detrending flattens a sinusoidal stellar trend while preserving a synthetic transit dip
    """
    rng = np.random.default_rng(42)
    time = np.linspace(0, 10, 1000)
    # Slow stellar variability (0.2% amplitude) + white noise (0.1%)
    stellar_trend = 1.0 + 0.002 * np.sin(2.0 * np.pi * time / 5.0)
    flux = stellar_trend + rng.normal(0, 0.001, 1000)
    # Inject a 1.2% transit dip around t = 5.0 days
    in_transit = np.abs(time - 5.0) < 0.05
    flux[in_transit] -= 0.012
    flux_err = np.full(1000, 0.001)

    lc = lk.LightCurve(time=time, flux=flux, flux_err=flux_err)
    flat_lc = filter_gaussian_process(lc, rho=2.0)

    assert isinstance(flat_lc, lk.LightCurve), "Output must be a LightCurve object"
    assert len(flat_lc.flux) == len(lc.flux), "Data points were lost during GP detrending"
    assert np.isclose(np.median(flat_lc.flux.value[~in_transit]), 1.0, atol=0.002), "Out-of-transit baseline not normalized to 1.0"
    recovered_depth = 1.0 - np.median(flat_lc.flux.value[in_transit])
    assert np.isclose(recovered_depth, 0.012, atol=0.003), f"Expected ~0.012 depth, got {recovered_depth:.4f}"
