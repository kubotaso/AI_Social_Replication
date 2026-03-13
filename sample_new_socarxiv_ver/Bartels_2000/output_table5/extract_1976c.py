import subprocess

r_code = '''
library(anesr)
data("timeseries_1976")
data("timeseries_1974")

# V763002 is the respondent case ID for 1976
# V742002 is the respondent case ID for 1974
cat("V763002 range:", min(timeseries_1976$V763002), "-", max(timeseries_1976$V763002), ", unique:", length(unique(timeseries_1976$V763002)), "\\n")
cat("V742002 range:", min(timeseries_1974$V742002), "-", max(timeseries_1974$V742002), ", unique:", length(unique(timeseries_1974$V742002)), "\\n")

# Check overlap
overlap <- intersect(timeseries_1976$V763002, timeseries_1974$V742002)
cat("Overlapping case IDs:", length(overlap), "\\n")

# Panel respondents in 1976 are those who were also in 1974 (and 1972)
# They should share the same case ID

# Now find the House vote variable
# In 1976 ANES, the House vote is in the post-election section
# V763826-V763840 seem to be thermometer ratings (0-100 scale)
# The vote itself might be coded differently

# Let me search for variables with values like {1, 5} or {1, 2, 7, 8, 9}
# that look like vote variables
cat("\\nSearching for House vote variable in 1976...\\n")
for (v in names(timeseries_1976)) {
  vals <- sort(unique(timeseries_1976[[v]][!is.na(timeseries_1976[[v]])]))
  # Look for vars with small number of unique values that include 1 and 5
  if (length(vals) <= 8 && 1 %in% vals && 5 %in% vals) {
    n1 <- sum(timeseries_1976[[v]] == 1, na.rm=TRUE)
    n5 <- sum(timeseries_1976[[v]] == 5, na.rm=TRUE)
    if (n1 > 100 && n5 > 100) {
      cat(v, ": vals=", paste(vals, collapse=","), ", 1=", n1, ", 5=", n5, "\\n")
    }
  }
}

# Also search for vars with values 1,2 only (possible Dem/Rep coding)
cat("\\nVars with values containing both 1 and 2 (200+ each):\\n")
count <- 0
for (v in names(timeseries_1976)) {
  vals <- sort(unique(timeseries_1976[[v]][!is.na(timeseries_1976[[v]])]))
  if (length(vals) <= 10 && 1 %in% vals && 2 %in% vals) {
    n1 <- sum(timeseries_1976[[v]] == 1, na.rm=TRUE)
    n2 <- sum(timeseries_1976[[v]] == 2, na.rm=TRUE)
    if (n1 > 200 && n2 > 200) {
      cat(v, ": vals=", paste(vals, collapse=","), ", 1=", n1, ", 2=", n2, "\\n")
      count <- count + 1
      if (count > 20) break
    }
  }
}

# Search for PID in 1974
# V763389 is 1976 PID (values 1-10, 98, 99)
# In 1974, PID is likely V742197 or similar
# V763389: 1=Strong Dem, 2=Weak Dem, 3=Ind-Dem, 4=Ind, 5=Ind-Rep, 6=Weak Rep, 7=Strong Rep
cat("\\nV763389 (1976 PID) distribution:\\n")
print(table(timeseries_1976$V763389))

# Find 1974 PID
cat("\\nSearching for 1974 PID variable (values 1-7 or 0-6)...\\n")
for (v in names(timeseries_1974)) {
  vals <- sort(unique(timeseries_1974[[v]][!is.na(timeseries_1974[[v]])]))
  if (length(vals) >= 7 && length(vals) <= 12 && 1 %in% vals && 7 %in% vals) {
    cat(v, ": vals=", paste(vals, collapse=","), "\\n")
  }
}
'''

result = subprocess.run(
    ['Rscript', '-e', r_code],
    capture_output=True, text=True, timeout=120,
    cwd='/Users/administrator/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/Replication_Claude_Bartels_v2'
)
print(result.stdout[:5000])
if result.returncode != 0:
    print("STDERR:", result.stderr[-500:])
