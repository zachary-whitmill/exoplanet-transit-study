import matplotlib.pyplot as plt
from src.ingestion import fetch_lightcurve
from src.preprocessing import detrend_savgol, detrend_gp
from src.search import run_bls
from src.evaluation import calculate_snr

# 1. Test Ingestion
raw_lc = fetch_lightcurve("WASP-5")

# 2. Test Preprocessing
savgol_lc = detrend_savgol(raw_lc, window_length=101)
gp_lc = detrend_gp(raw_lc)

# Visually confirm detrending success
ax = raw_lc.plot(label="Raw Flux", alpha=0.3)
savgol_lc.plot(ax=ax, label="Savitzky-Golay", linewidth=1.5)
gp_lc.plot(ax=ax, label="Gaussian Process", linewidth=1.5)
plt.show()

# 3. Test Search & Evaluation
bls_results = run_bls(savgol_lc)
snr = calculate_snr(savgol_lc, bls_results["period"], bls_results["t0"], bls_results["duration"])

print(f"Recovered Period: {bls_results['period']:.4f} days")
print(f"Transit SNR: {snr:.2f}")