"""Try loading 1992 timeseries from anesr and extracting house vote + PID."""
import subprocess
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
import warnings
warnings.filterwarnings('ignore')
import os

# Extract 1992 data via R
r_code = '''
library(anesr)
data(timeseries_1992)
# Get house vote and PID variables
# V923634 = 7-pt PID (0-6)
# V925623 = House vote choice (0=NA, 1=Dem, 2=Rep)
# Also try V926101 or other vars

# Just save all column names and first few rows
write.csv(names(timeseries_1992), "/tmp/anes_1992_colnames.csv", row.names=FALSE)

# Save a subset with relevant vars - search for house vote and PID
pid_cols <- grep("V923634|V925623|V925609|V900320|V920301|V926", names(timeseries_1992), value=TRUE)
cat("Found columns:", pid_cols, "\\n")

# Save relevant subset
if (length(pid_cols) > 0) {
    subset <- timeseries_1992[, pid_cols, drop=FALSE]
    write.csv(subset, "/tmp/anes_1992_subset.csv", row.names=FALSE)
}

# Also check total N
cat("Total N:", nrow(timeseries_1992), "\\n")
cat("Total cols:", ncol(timeseries_1992), "\\n")
'''

result = subprocess.run(['/opt/homebrew/bin/Rscript', '-e', r_code],
                       capture_output=True, text=True, timeout=120)
print("STDOUT:", result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[:500])

# Check if colnames file exists
if os.path.exists('/tmp/anes_1992_colnames.csv'):
    cols = pd.read_csv('/tmp/anes_1992_colnames.csv')
    print(f"\nTotal columns: {len(cols)}")
    # Search for house vote related variables
    house_vars = [c for c in cols['x'].values if any(s in str(c).upper() for s in ['HOUSE', 'CONG', '925', '926'])]
    print(f"House/Congress-related: {house_vars[:20]}")
    pid_vars = [c for c in cols['x'].values if any(s in str(c).upper() for s in ['PARTY', 'PID', '9236', '9003'])]
    print(f"PID-related: {pid_vars[:20]}")

if os.path.exists('/tmp/anes_1992_subset.csv'):
    subset = pd.read_csv('/tmp/anes_1992_subset.csv')
    print(f"\nSubset shape: {subset.shape}")
    print(subset.head(10))
    for c in subset.columns:
        print(f"\n{c}:")
        print(subset[c].value_counts().sort_index().head(15))
