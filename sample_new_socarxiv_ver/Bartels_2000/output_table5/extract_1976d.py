import subprocess

r_code = '''
library(anesr)
data("timeseries_1976")
data("timeseries_1974")

# Look for House vote variable in 1976 post-election section
# Post-election variables are typically in the V763800+ range
# House vote should have values 1 (Dem) and 2 (Rep)

# Search specifically in the V763800-V763999 range
cat("Searching V763800-V763999 for vote vars:\\n")
for (i in 3800:3999) {
  v <- paste0("V76", i)
  if (v %in% names(timeseries_1976)) {
    vals <- sort(unique(timeseries_1976[[v]][!is.na(timeseries_1976[[v]])]))
    if (length(vals) <= 8) {
      n_vals <- table(timeseries_1976[[v]])
      if (any(names(n_vals) %in% c("1", "2")) && sum(n_vals[names(n_vals) %in% c("1","2")]) > 500) {
        cat(v, ": ", paste(paste0(names(n_vals), "=", n_vals), collapse=", "), "\\n")
      }
    }
  }
}

# Also check V763400-V763500 range (which might contain the vote question)
cat("\\nSearching V763380-V763400:\\n")
for (i in 3380:3410) {
  v <- paste0("V76", i)
  if (v %in% names(timeseries_1976)) {
    vals <- sort(unique(timeseries_1976[[v]][!is.na(timeseries_1976[[v]])]))
    if (length(vals) <= 10) {
      cat(v, ": ", paste(paste0(vals, "=", as.vector(table(timeseries_1976[[v]]))), collapse=", "), "\\n")
    }
  }
}

# V763389 is PID. Vote variables are often nearby.
# Check V763380-V763410
cat("\\nChecking vars near PID (V763380-V763420):\\n")
for (i in 3380:3420) {
  v <- paste0("V76", i)
  if (v %in% names(timeseries_1976)) {
    vals <- sort(unique(timeseries_1976[[v]][!is.na(timeseries_1976[[v]])]))
    if (length(vals) <= 15) {
      n_vals <- table(timeseries_1976[[v]])
      cat(v, ": ", paste(paste0(names(n_vals), "=", n_vals), collapse=", "), "\\n")
    }
  }
}

# Also check V763700-V763800 for post-election vote section
cat("\\nSearching V763700-V763810 for vote:\\n")
for (i in 3700:3810) {
  v <- paste0("V76", i)
  if (v %in% names(timeseries_1976)) {
    vals <- sort(unique(timeseries_1976[[v]][!is.na(timeseries_1976[[v]])]))
    if (length(vals) <= 10 && length(vals) >= 2) {
      n_vals <- table(timeseries_1976[[v]])
      # Look for vars with between 500-1500 valid responses
      total_valid <- sum(n_vals[!names(n_vals) %in% c("0", "8", "9", "98", "99")])
      if (total_valid > 500 && total_valid < 2000) {
        cat(v, ": ", paste(paste0(names(n_vals), "=", n_vals), collapse=", "), "\\n")
      }
    }
  }
}

# The 1974 PID variable - search for V742xxx with values 1-7
# V742513 had values 0, 4, 5, 12, 30, 36, 45, 52, 99 - not PID
cat("\\n\\nSearching for 1974 PID (7-point scale, values 1-7):\\n")
for (v in names(timeseries_1974)) {
  vals <- sort(unique(timeseries_1974[[v]][!is.na(timeseries_1974[[v]])]))
  if (length(vals) >= 7 && length(vals) <= 12) {
    # Check if vals contain 1-7 (or 0-6)
    if (all(1:7 %in% vals)) {
      n_vals <- table(timeseries_1974[[v]])
      # PID should have 100+ in most categories
      if (all(n_vals[as.character(1:7)] > 30)) {
        cat(v, ": ", paste(paste0(names(n_vals), "=", n_vals), collapse=", "), "\\n")
      }
    }
  }
}
'''

result = subprocess.run(
    ['Rscript', '-e', r_code],
    capture_output=True, text=True, timeout=180,
    cwd='/Users/administrator/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/Replication_Claude_Bartels_v2'
)
print(result.stdout[:5000])
if result.returncode != 0:
    print("STDERR:", result.stderr[-500:])
