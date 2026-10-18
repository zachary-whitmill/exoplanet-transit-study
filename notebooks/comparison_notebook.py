""""""

import matplotlib.pyplot as plt
from src.ingestion import download_lightcurve
from src.preprocessing import filter_savitzky_golay, filter_gaussian_process
from src.transit import run_bls
from src.evaluation import calculate_snr

# test data ingestion
raw_lc = download_lightcurve("WASP-5")

# test preprocessing
savgol_lc = filter_savitzky_golay(raw_lc, window_length=101)
gp_lc = filter_gaussian_process(raw_lc)

# visually confirm detrending success
ax = raw_lc.plot(label="Raw Flux", alpha=0.3)
savgol_lc.plot(ax=ax, label="Savitzky-Golay", linewidth=1.5)
gp_lc.plot(ax=ax, label="Gaussian Process", linewidth=1.5)
plt.show()

# test search & evaluation
bls_results = run_bls(savgol_lc)
snr = calculate_snr(savgol_lc, bls_results["period"], bls_results["t0"], bls_results["duration"])

print(f"Recovered Period: {bls_results['period']:.4f} days")
print(f"Transit SNR: {snr:.2f}")