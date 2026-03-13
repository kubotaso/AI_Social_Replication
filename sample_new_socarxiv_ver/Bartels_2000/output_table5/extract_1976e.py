import subprocess

r_code = '''
library(anesr)
data("timeseries_1976")
data("timeseries_1974")

# V763381 looks like it might be the House vote:
# 0=598 (didn't vote/NA), 1=681 (cand1), 5=961 (cand2)
# That's 1642 valid votes

# Let me check if V763385 is the House vote for post-election
# V763385: 0=719, 1=824, 5=687
# V763386: 0=719, 1=434, 5=1068

# These might be like the 1992 study: 1=candidate1, 5=candidate2
# We need the party of each candidate

# In the CDF codebook, VCF0707 maps from these original variables
# Let me check the CDF documentation or the timeseries_cum_doc

# V763951: 0=339, 1=790, 2=1119 - looks like it could be
# presidential vote (0=didnt vote, 1=Carter/Dem, 2=Ford/Rep)
# With 1909 valid pres votes and 2248 total, that's 85% voting rate

# The House vote would have more non-voters
# V763381: 0=598, 1=681, 5=961 → 1642 House voters

# V763871: 0=470, 1=365, 2=876 - this has value 2!
# Could be: 0=NA/no, 1=Dem, 2=Rep for House vote
# But 365 Dem vs 876 Rep is very skewed

# Let me cross-tab V763389 (PID) with potential vote variables
# to see which one behaves like House vote

# Expected pattern: Strong Dems → Dem vote, Strong Reps → Rep vote

cat("=== Cross-tab PID (V763389) with V763381 ===\\n")
print(table(timeseries_1976$V763389, timeseries_1976$V763381))

cat("\\n=== Cross-tab PID with V763385 ===\\n")
print(table(timeseries_1976$V763389, timeseries_1976$V763385))

cat("\\n=== Cross-tab PID with V763871 ===\\n")
print(table(timeseries_1976$V763389, timeseries_1976$V763871))

cat("\\n=== Cross-tab PID with V763951 ===\\n")
print(table(timeseries_1976$V763389, timeseries_1976$V763951))

# Also check V763870 which has a promising distribution
cat("\\n=== Cross-tab PID with V763870 ===\\n")
print(table(timeseries_1976$V763389, timeseries_1976$V763870))

# V763870: 0=470, 1=680, 5=1070 - this has 1750 valid
# Could be House vote intention or actual vote

# Search for 1974 PID
cat("\\n\\n=== 1974 PID search ===\\n")
# V763398 from the 1976 study has same coding as V763389 (PID)
# Could V763398 be the "1974 PID recalled in 1976"?
# Or the actual 1974 PID from the panel

# Let me check V742xxx for 7-point PID
# V742513 had weird values. Let me look more carefully.
cat("V742513:", paste(sort(unique(timeseries_1974$V742513)), collapse=", "), "\\n")
cat("V742514:", paste(sort(unique(timeseries_1974$V742514)), collapse=", "), "\\n")

# Look for vars with typical PID distribution (7 categories, 100+ each)
cat("\\nSearching for 1974 PID:\\n")
for (i in 500:600) {
  v <- paste0("V74", sprintf("%04d", 2000+i))
  if (v %in% names(timeseries_1974)) {
    vals <- sort(unique(timeseries_1974[[v]][!is.na(timeseries_1974[[v]])]))
    if (length(vals) >= 7 && length(vals) <= 15 && 1 %in% vals && 7 %in% vals) {
      cat(v, ": ", paste(paste0(vals, "=", as.vector(table(timeseries_1974[[v]]))), collapse=", "), "\\n")
    }
  }
}

# Try a broader search
for (v in names(timeseries_1974)) {
  vals <- sort(unique(timeseries_1974[[v]][!is.na(timeseries_1974[[v]])]))
  if (length(vals) >= 7 && length(vals) <= 15 && all(1:7 %in% vals)) {
    n_vals <- table(timeseries_1974[[v]])
    # PID: categories 1-7 should each have 50+
    if (all(n_vals[as.character(1:7)] > 30)) {
      cat(v, ": ", paste(paste0(names(n_vals), "=", n_vals), collapse=", "), "\\n")
    }
  }
}
'''

result = subprocess.run(
    ['Rscript', '-e', r_code],
    capture_output=True, text=True, timeout=180,
    cwd='/Users/administrator/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/Replication_Claude_Bartels_v2'
)
print(result.stdout[:6000])
if result.returncode != 0:
    print("STDERR:", result.stderr[-500:])
