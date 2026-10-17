import numpy as np
import lightkurve as lk

def calculate_snr(lc: lk.LightCurve, period: float, t0: float, duration: float) -> float:
    """
    Calculates the empirical transit Signal-to-Noise Ratio (SNR).
    SNR = (depth / sigma_out) * sqrt(N_in)
    """
    time = lc.time.value
    flux = lc.flux.value

    # Generate a Boolean mask for in-transit data points
    phases = (time - t0) % period
    half_dur = duration / 2.0
    in_transit_mask = (phases < half_dur) | (phases > (period - half_dur))

    flux_in = flux[in_transit_mask]
    flux_out = flux[~in_transit_mask]

    if len(flux_in) == 0 or len(flux_out) == 0:
        return 0.0

    # Calculate transit depth (delta) and out-of-transit scatter (sigma)
    depth = np.median(flux_out) - np.median(flux_in)
    sigma_out = np.std(flux_out)
    n_in = len(flux_in)

    if sigma_out == 0:
        return 0.0

    snr = (depth / sigma_out) * np.sqrt(n_in)
    return snr

def evaluate_algorithms(lc: lk.LightCurve, bls_results: dict, tls_results: dict, true_period: float = None) -> dict:
    """
    Compares recovered periods, transit depths, and SNR profiles between BLS and TLS algorithms.
    """
    bls_snr = calculate_snr(lc, bls_results["period"], bls_results["t0"], bls_results["duration"])
    tls_snr = calculate_snr(lc, tls_results["period"], tls_results["t0"], tls_results["duration"])
    
    metrics = {
        "bls_period": bls_results["period"],
        "tls_period": tls_results["period"],
        "bls_snr": bls_snr,
        "tls_snr": tls_snr,
        "snr_improvement_pct": ((tls_snr - bls_snr) / bls_snr * 100) if bls_snr > 0 else 0.0
    }
    
    if true_period:
        metrics["bls_period_error_pct"] = abs(bls_results["period"] - true_period) / true_period * 100
        metrics["tls_period_error_pct"] = abs(tls_results["period"] - true_period) / true_period * 100
        
    return metrics