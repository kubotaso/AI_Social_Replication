"""Save attempt 6 results to file"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generated_analysis_retry_6 import run_analysis, score_against_ground_truth

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
wvs_path = os.path.join(base, "data", "WVS_Time_Series_1981-2022_csv_v5_0.csv")
evs1990_path = os.path.join(base, "data", "ZA4460_v3-0-0.dta")
evs_long_path = "/Users/kubotaso/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/OldFiles/Replication_Claude_IB/data/ZA4804_v3-1-0.dta/ZA4804_v3-1-0.dta"

# Capture output
import io
old_stdout = sys.stdout
sys.stdout = buffer = io.StringIO()

result_text, results = run_analysis(wvs_path, evs1990_path, evs_long_path)
print(result_text)
score = score_against_ground_truth(results)

output = buffer.getvalue()
sys.stdout = old_stdout

# Save results
outdir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(outdir, 'generated_results_attempt_6.txt'), 'w') as f:
    f.write(output)

print(f"Results saved. Score: {score:.1f}")
