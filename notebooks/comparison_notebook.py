"""End-to-end comparison script for SG vs. GP detrending and BLS vs. TLS transit searches."""

import time
import matplotlib.pyplot as plt
from src.ingestion import download_lightcurve
from src.preprocessing import filter_savitzky_golay, filter_gaussian_process
from src.transit import run_bls, run_tls
from src.evaluation import evaluate_benchmark_matrix, format_benchmark_table

# Known catalog parameters for WASP-5b (for accuracy benchmarking)
TARGET_NAME = "WASP-5"
TRUE_PERIOD_DAYS = 1.6284314
TRUE_DEPTH_FRAC = 0.0121  # ~1.21% (12,100 ppm)
MIN_PERIOD = 0.5
MAX_PERIOD = 20.0

# 1. Test data ingestion
print(f"Fetching data for {TARGET_NAME}...")
raw_lc = download_lightcurve(TARGET_NAME)

# 2. Test preprocessing (Savitzky-Golay vs. Gaussian Process)
print("Detrending with Savitzky-Golay...")
t0 = time.perf_counter()
savgol_lc = filter_savitzky_golay(raw_lc, window_length=101)
sg_time = time.perf_counter() - t0

print("Detrending with Gaussian Process (celerite2)...")
t0 = time.perf_counter()
gp_lc = filter_gaussian_process(raw_lc)
gp_time = time.perf_counter() - t0
print(f"Detrending completed (SG: {sg_time:.3f}s, GP: {gp_time:.3f}s)")

# 3. Visually confirm detrending success
ax = raw_lc.plot(label="Raw Flux", alpha=0.3)
savgol_lc.plot(ax=ax, label="Savitzky-Golay", linewidth=1.5)
gp_lc.plot(ax=ax, label="Gaussian Process", linewidth=1.5)
plt.show()

# 4. Run transit searches across all 4 combinations (2x2 matrix)
print("Running BLS & TLS on Savitzky-Golay light curve...")
savgol_bls = run_bls(savgol_lc, min_period=MIN_PERIOD, max_period=MAX_PERIOD)
savgol_tls = run_tls(savgol_lc, min_period=MIN_PERIOD, max_period=MAX_PERIOD)

print("Running BLS & TLS on Gaussian Process light curve...")
gp_bls = run_bls(gp_lc, min_period=MIN_PERIOD, max_period=MAX_PERIOD)
gp_tls = run_tls(gp_lc, min_period=MIN_PERIOD, max_period=MAX_PERIOD)

# 5. Evaluate and print the 2x2 benchmark summary
print("Evaluating benchmark metrics...")
benchmark = evaluate_benchmark_matrix(
    savgol_lc=savgol_lc,
    gp_lc=gp_lc,
    savgol_bls=savgol_bls,
    savgol_tls=savgol_tls,
    gp_bls=gp_bls,
    gp_tls=gp_tls,
    known_period=TRUE_PERIOD_DAYS,
    known_depth=TRUE_DEPTH_FRAC,
    detrend_times={"savgol_s": sg_time, "gp_s": gp_time},
)

print("\n" + format_benchmark_table(benchmark))
