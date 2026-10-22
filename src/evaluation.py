"""Module for evaluating detrending and transit search algorithm performance."""

import numpy as np
import lightkurve as lk


def calculate_snr(lc: lk.LightCurve, period: float, t0: float, duration: float):
    """
    Calculates the empirical transit Signal-to-Noise Ratio (SNR).
    SNR = (depth / sigma_outside) * sqrt(N_in)

    parameters:
    lc: lk.lightcurve
        light curve of the object we want to process.
    period: float
        calculated period of the exoplanets orbit.
    t0: float
        the transit center 'epoch', middle of the transit.
    duration: float
        the total transit duration.

    returns:
    
    """

    # strip metadata with .value and store data in an array type float
    time = np.asarray(lc.time.value, dtype=float)
    flux = np.asarray(lc.flux.value, dtype=float)

    valid = np.isfinite(time) & np.isfinite(flux)
    time = time[valid] 
    flux = flux[valid]
    # check for reasonable file data
    if len(flux) == 0 or period <= 0 or duration <= 0:
        return 0.0


    # phases from range 0 to period
    phases = (time - t0) % period 
    half_dur = duration / 2.0
    # mask identifying points inside of transit window
    in_transit_mask = (phases < half_dur) | (phases > (period - half_dur))
    flux_inside = flux[in_transit_mask]
    flux_outside = flux[~in_transit_mask]
    # check for if we observe the star inside AND outside of the transit period
    if len(flux_inside) == 0 or len(flux_outside) == 0:
        return 0.0


    # Calculate transit depth (delta) and out-of-transit scatter (MAD scaled to sigma)
    med_outside = np.nanmedian(flux_outside)
    depth = med_outside - np.nanmedian(flux_inside)
    sigma_outside = 1.4826 * np.nanmedian(np.abs(flux_outside - med_outside))
    n_in = len(flux_inside)
    if sigma_outside == 0:
        return 0.0

    snr = (depth / sigma_outside) * np.sqrt(n_in)
    return float(snr)


def check_period_harmonic(calculated_period: float, known_period: float, tolerance_pct: float = 1.0):
    """
    Checks whether a recovered period matches the true period or a common harmonic/alias.

    parameters:
    calculated_period: float
        period detected by BLS & TLS algorithms (in days).
    known_period: float
        published literature or simulation value (in days).
    tolerance_pct: float
        allowed discrepancy (in percent).

    returns: str
        returns the harmonic of the periodagram or mismatch if no harmonic is fitted
    """

    # check for reasonable file data
    if known_period is None or known_period <= 0 or calculated_period <= 0:
        return "unknown"

    harmonics = [
        (1.0, "exact"),
        (0.5, "1/2 harmonic"),
        (2.0, "2x harmonic"),
        (1.0 / 3.0, "1/3 harmonic"),
        (3.0, "3x harmonic"),
    ]
    for factor, label in harmonics:
        target = known_period * factor
        # relative error.
        err_pct = abs(calculated_period - target) / target * 100.0
        if err_pct <= tolerance_pct:
            return label

    return "mismatch"


def evaluate_single_run(
    lc: lk.LightCurve,
    search_results: dict,
    known_period: float = None,
    known_depth: float = None,
    detrend_runtime: float = None,
    ):
    """
    Evaluates a single (detrender, transit_search) run on period accuracy, depth, SNR, SDE, and runtime.
    
    parameters:
    lc: LightCurve
        lightkurve object that has been processed & detrended.
    search_results: dict
        output transit parameters dictionary.
    known_period: float
        published literature or simulation value (in days).
    known_depth: float
        published fractional transit depth (R_planet / R_star)^2.
    detrend_runtime: float
        time taken in seconds by detrending (GP or SG) algorithms.
    """
    
    
    period = float(search_results["period"])
    t0 = float(search_results["t0"])
    duration = float(search_results["duration"])
    depth = float(search_results["depth"])
    sde = float(search_results.get("sde", 0.0))
    search_runtime = float(search_results.get("runtime", 0.0))

    snr = calculate_snr(lc, period, t0, duration)

    run_metrics = {
        "period": period,
        "t0": t0,
        "duration": duration,
        "depth": depth,
        "snr": snr,
        "sde": sde,
        "search_runtime": search_runtime,
    }

    if detrend_runtime is not None:
        run_metrics["detrend_runtime"] = float(detrend_runtime)
        run_metrics["total_runtime"] = float(detrend_runtime) + search_runtime

    if known_period is not None and known_period > 0:
        run_metrics["period_error_days"] = abs(period - known_period)
        run_metrics["period_error_pct"] = abs(period - known_period) / known_period * 100.0
        run_metrics["harmonic_match"] = check_period_harmonic(period, known_period)

    if known_depth is not None and known_depth > 0:
        # signed percentage error: negative indicates transit depth attenuation
        run_metrics["depth_error_pct"] = (depth - known_depth) / known_depth * 100.0

    return run_metrics


def evaluate_algorithms(
    lc: lk.LightCurve,
    bls_results: dict,
    tls_results: dict,
    known_period: float = None,
    known_depth: float = None,
    ):
    """
    Compares recovered periods, transit depths, SDE, runtime, and SNR profiles between BLS and TLS algorithms
    on a single detrended light curve.

    parameters:
    bls_results: dict
        dictionary of transit parameters calculated by BLS.
    tls_results: dict
        dictionary of transit parameters calculated by TLS.
    """

    bls_eval = evaluate_single_run(lc, bls_results, known_period=known_period, known_depth=known_depth)
    tls_eval = evaluate_single_run(lc, tls_results, known_period=known_period, known_depth=known_depth)

    bls_snr = bls_eval["snr"]
    tls_snr = tls_eval["snr"]
    bls_sde = bls_eval["sde"]
    tls_sde = tls_eval["sde"]

    metrics = {
        "bls_period": bls_eval["period"],
        "tls_period": tls_eval["period"],
        "bls_depth": bls_eval["depth"],
        "tls_depth": tls_eval["depth"],
        "bls_snr": bls_snr,
        "tls_snr": tls_snr,
        "snr_improvement_pct": ((tls_snr - bls_snr) / bls_snr * 100.0) if bls_snr > 0 else 0.0,
        "bls_sde": bls_sde,
        "tls_sde": tls_sde,
        "sde_improvement_pct": ((tls_sde - bls_sde) / bls_sde * 100.0) if bls_sde > 0 else 0.0,
        "bls_runtime_s": bls_eval["search_runtime"],
        "tls_runtime_s": tls_eval["search_runtime"],
    }

    if known_period is not None and known_period > 0:
        metrics["bls_period_error_pct"] = bls_eval["period_error_pct"]
        metrics["tls_period_error_pct"] = tls_eval["period_error_pct"]
        metrics["bls_harmonic_match"] = bls_eval["harmonic_match"]
        metrics["tls_harmonic_match"] = tls_eval["harmonic_match"]

    if known_depth is not None and known_depth > 0:
        metrics["bls_depth_error_pct"] = bls_eval["depth_error_pct"]
        metrics["tls_depth_error_pct"] = tls_eval["depth_error_pct"]

    return metrics


def evaluate_benchmark_matrix(
    savgol_lc: lk.LightCurve,
    gp_lc: lk.LightCurve,
    savgol_bls: dict,
    savgol_tls: dict,
    gp_bls: dict,
    gp_tls: dict,
    known_period: float = None,
    known_depth: float = None,
    detrend_times: dict = None,
    ):
    """
    Evaluates the full 2x2 benchmark matrix (Savitzky-Golay vs. Gaussian Process x BLS vs. TLS).
    """
    sg_time = detrend_times.get("savgol_s") if detrend_times else None
    gp_time = detrend_times.get("gp_s") if detrend_times else None

    runs = {
        "SG + BLS": evaluate_single_run(savgol_lc, savgol_bls, known_period, known_depth, sg_time),
        "SG + TLS": evaluate_single_run(savgol_lc, savgol_tls, known_period, known_depth, sg_time),
        "GP + BLS": evaluate_single_run(gp_lc, gp_bls, known_period, known_depth, gp_time),
        "GP + TLS": evaluate_single_run(gp_lc, gp_tls, known_period, known_depth, gp_time),
    }

    sg_bls_snr = runs["SG + BLS"]["snr"]
    sg_tls_snr = runs["SG + TLS"]["snr"]
    gp_bls_snr = runs["GP + BLS"]["snr"]
    gp_tls_snr = runs["GP + TLS"]["snr"]

    sg_bls_depth = runs["SG + BLS"]["depth"]
    gp_bls_depth = runs["GP + BLS"]["depth"]
    sg_tls_depth = runs["SG + TLS"]["depth"]
    gp_tls_depth = runs["GP + TLS"]["depth"]

    comparisons = {
        # Search algorithm sensitivity comparisons (TLS vs BLS)
        "tls_vs_bls_snr_gain_on_sg_pct": ((sg_tls_snr - sg_bls_snr) / sg_bls_snr * 100.0) if sg_bls_snr > 0 else 0.0,
        "tls_vs_bls_snr_gain_on_gp_pct": ((gp_tls_snr - gp_bls_snr) / gp_bls_snr * 100.0) if gp_bls_snr > 0 else 0.0,
        # Detrending comparisons (GP vs SG)
        "gp_vs_sg_snr_gain_with_bls_pct": ((gp_bls_snr - sg_bls_snr) / sg_bls_snr * 100.0) if sg_bls_snr > 0 else 0.0,
        "gp_vs_sg_snr_gain_with_tls_pct": ((gp_tls_snr - sg_tls_snr) / sg_tls_snr * 100.0) if sg_tls_snr > 0 else 0.0,
        # Transit depth preservation ratio (>1 means GP preserved a deeper transit than SG)
        "gp_to_sg_depth_ratio_bls": (gp_bls_depth / sg_bls_depth) if sg_bls_depth > 0 else 0.0,
        "gp_to_sg_depth_ratio_tls": (gp_tls_depth / sg_tls_depth) if sg_tls_depth > 0 else 0.0,
    }

    return {
        "runs": runs,
        "comparisons": comparisons,
    }


def format_benchmark_table(benchmark: dict):
    """
    Formats the 2x2 benchmark results into a human-readable ASCII summary table.
    """
    runs = benchmark["runs"]
    comp = benchmark["comparisons"]

    header = (
        f"{'Pipeline':<10} | {'Period (d)':>10} | {'P Err (%)':>9} | {'Match':<12} | "
        f"{'Depth (ppm)':>11} | {'SNR':>7} | {'SDE':>7} | {'Search (s)':>10}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]

    for name, m in runs.items():
        p_err = f"{m['period_error_pct']:.3f}%" if "period_error_pct" in m else "N/A"
        match = m.get("harmonic_match", "N/A")
        depth_ppm = m["depth"] * 1e6
        lines.append(
            f"{name:<10} | {m['period']:>10.5f} | {p_err:>9} | {match:<12} | "
            f"{depth_ppm:>11.0f} | {m['snr']:>7.2f} | {m['sde']:>7.2f} | {m['search_runtime']:>10.2f}"
        )

    lines.append(sep)
    lines.append("Cross-Comparison Summary:")
    lines.append(f"  TLS vs BLS SNR gain (on SG): {comp['tls_vs_bls_snr_gain_on_sg_pct']:+.2f}%")
    lines.append(f"  TLS vs BLS SNR gain (on GP): {comp['tls_vs_bls_snr_gain_on_gp_pct']:+.2f}%")
    lines.append(f"  GP vs SG SNR gain (with BLS): {comp['gp_vs_sg_snr_gain_with_bls_pct']:+.2f}%")
    lines.append(f"  GP vs SG SNR gain (with TLS): {comp['gp_vs_sg_snr_gain_with_tls_pct']:+.2f}%")
    lines.append(f"  GP / SG Transit Depth Ratio (BLS): {comp['gp_to_sg_depth_ratio_bls']:.3f}x")
    lines.append(f"  GP / SG Transit Depth Ratio (TLS): {comp['gp_to_sg_depth_ratio_tls']:.3f}x")
    lines.append(sep)

    return "\n".join(lines)
