"""
Module for evaluating detrending and transit search algorithm performance.

2x2 comparison grid between SG, GP, BLS, TLS.
"""

from astropy import constants as const
from lightkurve.lightcurve import LightCurve
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

# Replaced solar/Jupiter radius ratio with astropy.constants.
R_SUN_TO_R_JUP = float((const.R_sun / const.R_jup).value)

HARMONICS = [
    (1.0, "exact"),
    (0.5, "1/2 harmonic"),
    (2.0, "2x harmonic"),
    (1.0 / 3.0, "1/3 harmonic"),
    (3.0, "3x harmonic"),
]


# Helper function calculating shared percentage-gain.
def _pct_gain(a: float, b: float) -> float:
    return ((a - b) / b * 100.0) if b > 0 else 0.0


def calculate_snr(lc: LightCurve, period: float, t0: float, duration: float) -> float:
    """
    Calculates the empirical transit Signal-to-Noise Ratio (SNR).
    SNR = (depth / sigma_outside) * sqrt(N_in)

    Parameters:
    lc: LightCurve
        Light curve of the object we want to process.
    period: float
        Calculated period of the exoplanet's orbit.
    t0: float
        The transit center 'epoch', middle of the transit.
    duration: float
        Total transit duration.
    """
    # Replaced finite filtering, phase wrapping, and 1.4826*MAD
    # with built-in lightkurve.remove_nans(), create_transit_mask(), and scipy median_abs_deviation().
    lc = lc.remove_nans()
    if len(lc) == 0 or period <= 0 or duration <= 0:
        return 0.0

    in_transit_mask = lc.create_transit_mask(period=period, transit_time=t0, duration=duration)
    flux_inside = np.asarray(lc.flux.value[in_transit_mask], dtype=float)
    flux_outside = np.asarray(lc.flux.value[~in_transit_mask], dtype=float)
    if len(flux_inside) == 0 or len(flux_outside) == 0:
        return 0.0

    depth = np.nanmedian(flux_outside) - np.nanmedian(flux_inside)
    sigma_outside = float(median_abs_deviation(flux_outside, scale="normal", nan_policy="omit"))
    if not np.isfinite(sigma_outside) or sigma_outside == 0:
        return 0.0

    snr = (depth / sigma_outside) * np.sqrt(len(flux_inside))
    return float(snr)


def check_period_harmonic(calculated_period: float, known_period: float, tolerance_pct: float = 1.0) -> str:
    """
    Checks whether a recovered period matches the true period or a common harmonic/alias.
    """
    # Swapped harmonic loop for a single generator lookup.
    if known_period is None or known_period <= 0 or calculated_period <= 0:
        return "unknown"
    return next(
        (lable for f, lable in HARMONICS if abs(calculated_period / (known_period * f) - 1.0) * 100.0 <= tolerance_pct),
        "mismatch",
    )


def evaluate_single_run(
    lc: LightCurve,
    search_results: dict,
    known_period: float = None,
    known_depth: float = None,
    stellar_radius_rsun: float = None,
) -> dict:
    """
    Evaluates a single (detrender, transit_search) run on period accuracy, depth, SNR, SDE,
    runtime, and optionally inferred planetary radius / bulk density bias.
    """
    # Removed unused detrend_runtime/total_runtime plumbing and simplified radius/density
    # bias math so R_star cancels out of rp_error_pct and density_bias_factor no longer requires R_star.
    period = float(search_results["period"])
    t0 = float(search_results["t0"])
    duration = float(search_results["duration"])
    depth = float(search_results["depth"])
    sde = float(search_results.get("sde", 0.0))
    search_runtime = float(search_results.get("runtime", 0.0))

    run_metrics = {
        "period": period,
        "t0": t0,
        "duration": duration,
        "depth": depth,
        "snr": calculate_snr(lc, period, t0, duration),
        "sde": sde,
        "search_runtime": search_runtime,
    }

    if known_period is not None and known_period > 0:
        run_metrics["period_error_days"] = abs(period - known_period)
        run_metrics["period_error_pct"] = abs(period - known_period) / known_period * 100.0
        run_metrics["harmonic_match"] = check_period_harmonic(period, known_period)

    if known_depth is not None and known_depth > 0:
        # Signed percentage error: negative indicates transit depth attenuation
        run_metrics["depth_error_pct"] = (depth - known_depth) / known_depth * 100.0
        if depth > 0:
            # Rp ~ R_star * sqrt(depth) => relative Rp error = sqrt(depth / known_depth) - 1
            # Bulk density rho_p ~ depth^(-3/2) => bias factor = (known_depth / depth)^(3/2)
            run_metrics["rp_error_pct"] = float((np.sqrt(depth / known_depth) - 1.0) * 100.0)
            run_metrics["density_bias_factor"] = float((known_depth / depth) ** 1.5)

    if stellar_radius_rsun is not None and stellar_radius_rsun > 0 and depth > 0:
        run_metrics["rp_rjup"] = float(stellar_radius_rsun * R_SUN_TO_R_JUP * np.sqrt(depth))

    return run_metrics


# Removed dead function evaluate_algorithms, superseded by evaluate_benchmark_matrix.


def evaluate_benchmark_matrix(
    detrended_lcs: dict[str, LightCurve],
    search_results: dict[str, dict],
    known_period: float = None,
    known_depth: float = None,
    stellar_radius_rsun: float = None,
) -> dict:
    """
    Evaluates the full 2x2 benchmark matrix (Savitzky-Golay vs. Gaussian Process x BLS vs. TLS).
    """
    # Replaced 6 positional light-curve/result arguments with dict inputs and a comprehension.
    runs = {
        label: evaluate_single_run(
            detrended_lcs[label.split(" + ")[0]], res, known_period, known_depth, stellar_radius_rsun
        )
        for label, res in search_results.items()
    }

    sg_bls, sg_tls = runs["SG + BLS"], runs["SG + TLS"]
    gp_bls, gp_tls = runs["GP + BLS"], runs["GP + TLS"]

    comparisons = {
        "tls_vs_bls_snr_gain_on_sg_pct": _pct_gain(sg_tls["snr"], sg_bls["snr"]),
        "tls_vs_bls_snr_gain_on_gp_pct": _pct_gain(gp_tls["snr"], gp_bls["snr"]),
        "gp_vs_sg_snr_gain_with_bls_pct": _pct_gain(gp_bls["snr"], sg_bls["snr"]),
        "gp_vs_sg_snr_gain_with_tls_pct": _pct_gain(gp_tls["snr"], sg_tls["snr"]),
        "gp_to_sg_depth_ratio_bls": (gp_bls["depth"] / sg_bls["depth"]) if sg_bls["depth"] > 0 else 0.0,
        "gp_to_sg_depth_ratio_tls": (gp_tls["depth"] / sg_tls["depth"]) if sg_tls["depth"] > 0 else 0.0,
    }
    return {"runs": runs, "comparisons": comparisons}


def format_benchmark_table(benchmark: dict) -> str:
    """
    Formats the 2x2 benchmark results into a human-readable ASCII summary table.
    """
    # CHANGED: Replaced 66 lines of duplicated if/else f-string table alignment with pandas.DataFrame.to_string(),
    # folding Density Bias directly into the main table.
    runs = benchmark["runs"]
    comp = benchmark["comparisons"]

    df = pd.DataFrame({
        name: {
            "Period (d)": f"{m['period']:.5f}",
            "P Err (%)": f"{m['period_error_pct']:.4f}%" if "period_error_pct" in m else "N/A",
            "Match": m.get("harmonic_match", "N/A"),
            "Depth (ppm)": f"{m['depth'] * 1e6:.0f}",
            **({"Rp (RJup)": f"{m['rp_rjup']:.3f}"} if "rp_rjup" in m else {}),
            **(
                {
                    "Rp Err (%)": f"{m['rp_error_pct']:+.2f}%",
                    "Density Bias": f"{m['density_bias_factor']:.2f}x",
                }
                if "rp_error_pct" in m
                else {}
            ),
            "SNR": f"{m['snr']:.2f}",
            "SDE": f"{m['sde']:.2f}",
            "Search (s)": f"{m['search_runtime']:.2f}",
        }
        for name, m in runs.items()
    }).T.fillna("N/A")

    table_str = df.to_string()
    sep = "-" * max(len(line) for line in table_str.splitlines())
    return "\n".join([
        sep,
        table_str,
        sep,
        "Cross-Comparison Summary:",
        f"  TLS vs BLS SNR gain (SG / GP): {comp['tls_vs_bls_snr_gain_on_sg_pct']:+.2f}% / {comp['tls_vs_bls_snr_gain_on_gp_pct']:+.2f}%",
        f"  GP vs SG SNR gain (BLS / TLS): {comp['gp_vs_sg_snr_gain_with_bls_pct']:+.2f}% / {comp['gp_vs_sg_snr_gain_with_tls_pct']:+.2f}%",
        f"  GP / SG Transit Depth Ratio (BLS / TLS): {comp['gp_to_sg_depth_ratio_bls']:.3f}x / {comp['gp_to_sg_depth_ratio_tls']:.3f}x",
        sep,
    ])
