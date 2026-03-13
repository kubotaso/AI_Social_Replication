import subprocess
import pandas as pd
import numpy as np
from statsmodels.discrete.discrete_model import Probit
import statsmodels.api as sm

# The 1972-1976 ANES panel study had respondents interviewed in 1972, 1974, and 1976.
# The anesr package has timeseries_1974 and timeseries_1976 separately.
# The CDF panel_1976.csv has 925 respondents (those with 1972-era CDF IDs appearing in 1976).
# Bartels had N=682 for House vote in 1976.
# Our CDF-based N=552.
# Maybe the individual 1974/1976 studies have more panel respondents?

# First, check timeseries_1976 for panel respondent information
r_code = '''
library(anesr)

# Load 1976 timeseries
data("timeseries_1976")
cat("1976 timeseries N:", nrow(timeseries_1976), "\\n")

# Find PID variable - should be V763389 or similar
pid_vars <- grep("V76[0-9]{4}|V763[0-9]{3}", names(timeseries_1976), value=TRUE)
cat("Number of V76 variables:", length(pid_vars), "\\n")

# Search for party identification variable
# In 1976 ANES: V763389 = party identification (7-point)
for (v in c("V763389", "V763374", "V763388")) {
  if (v %in% names(timeseries_1976)) {
    cat(v, "values:", paste(sort(unique(timeseries_1976[[v]])), collapse=", "), "\\n")
  }
}

# Search for House vote variable
# V764002 or similar
for (v in c("V764002", "V764001", "V763966", "V763967")) {
  if (v %in% names(timeseries_1976)) {
    cat(v, "values:", paste(sort(unique(timeseries_1976[[v]])), collapse=", "), "\\n")
  }
}

# Actually, let me search more broadly
house_candidates <- grep("house|cong|rep|vot", names(timeseries_1976), value=TRUE, ignore.case=TRUE)
cat("House-related vars:", paste(head(house_candidates, 20), collapse=", "), "\\n")

# Search for case ID and panel indicator
id_candidates <- grep("V7600|V7601|V760001|V760101", names(timeseries_1976), value=TRUE)
cat("ID vars:", paste(head(id_candidates, 20), collapse=", "), "\\n")

# Now load 1974 timeseries for lagged PID
data("timeseries_1974")
cat("\\n1974 timeseries N:", nrow(timeseries_1974), "\\n")

# Check first few columns
cat("1974 first 10 vars:", paste(names(timeseries_1974)[1:10], collapse=", "), "\\n")
cat("1976 first 10 vars:", paste(names(timeseries_1976)[1:10], collapse=", "), "\\n")

# In 1974 ANES: party ID might be V742513
for (v in c("V742513", "V742514", "V742515")) {
  if (v %in% names(timeseries_1974)) {
    cat("1974", v, "values:", paste(sort(unique(timeseries_1974[[v]])), collapse=", "), "\\n")
  }
}

# Check 1974 case IDs
id_1974 <- grep("V7400|V7401|V740001|V740101", names(timeseries_1974), value=TRUE)
cat("1974 ID vars:", paste(head(id_1974, 20), collapse=", "), "\\n")

# The key question: can we match 1974 and 1976 respondents?
# In the panel study, the SAME respondents were interviewed in both years.
# The case IDs should be shared between the two datasets.

# Check V760001 and V740001
if ("V760001" %in% names(timeseries_1976)) {
  cat("1976 V760001 range:", min(timeseries_1976$V760001), "-", max(timeseries_1976$V760001), "\\n")
  cat("1976 V760001 N unique:", length(unique(timeseries_1976$V760001)), "\\n")
}
if ("V740001" %in% names(timeseries_1974)) {
  cat("1974 V740001 range:", min(timeseries_1974$V740001), "-", max(timeseries_1974$V740001), "\\n")
  cat("1974 V740001 N unique:", length(unique(timeseries_1974$V740001)), "\\n")
}

# Check if case IDs overlap between 1974 and 1976
if ("V760001" %in% names(timeseries_1976) && "V740001" %in% names(timeseries_1974)) {
  overlap <- intersect(timeseries_1976$V760001, timeseries_1974$V740001)
  cat("Overlapping case IDs:", length(overlap), "\\n")
}
'''

result = subprocess.run(
    ['Rscript', '-e', r_code],
    capture_output=True, text=True, timeout=120,
    cwd='/Users/administrator/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/Replication_Claude_Bartels_v2'
)
print("STDOUT:", result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr[-500:])
