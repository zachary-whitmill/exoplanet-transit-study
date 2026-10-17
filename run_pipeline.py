# run_pipeline.py
from src.ingestion import download_lightcurve
from src.preprocessing import filter_savitzky_golay, filter_gaussian_process
from src.search import run_bls, run_tls
from src.evaluation import evaluate_algorithms

# 1. Test Ingestion
print("Fetching data...")
raw_lc = download_lightcurve("WASP-5")

# 2. Test Preprocessing
print("Detrending...")
savgol_lc = filter_savitzky_golay(raw_lc)
gp_lc = filter_gaussian_process(raw_lc) 

# 3. Test Search (using the fast Savitzky-Golay baseline for the test)
print("Running search algorithms...")
bls_results = run_bls(savgol_lc)
tls_results = run_tls(savgol_lc)

# 4. Test Evaluation
print("Evaluating metrics...")
metrics = evaluate_algorithms(savgol_lc, bls_results, tls_results)

# Print the final benchmark output
for key, value in metrics.items():
    print(f"{key}: {value}")