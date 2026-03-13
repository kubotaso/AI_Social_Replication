import subprocess

r_code = '''
library(anesr)
data("timeseries_1976")
data("timeseries_1974")

# Search for case ID - usually first few variables
cat("1976 first 20 vars:\\n")
for (v in names(timeseries_1976)[1:20]) {
  vals <- sort(unique(timeseries_1976[[v]]))
  if (length(vals) > 20) {
    cat(v, ": N_unique=", length(vals), ", range=", min(vals), "-", max(vals), "\\n")
  } else {
    cat(v, ":", paste(vals, collapse=", "), "\\n")
  }
}

cat("\\n1974 first 20 vars:\\n")
for (v in names(timeseries_1974)[1:20]) {
  vals <- sort(unique(timeseries_1974[[v]]))
  if (length(vals) > 20) {
    cat(v, ": N_unique=", length(vals), ", range=", min(vals), "-", max(vals), "\\n")
  } else {
    cat(v, ":", paste(vals, collapse=", "), "\\n")
  }
}

# Search for House vote variable in 1976
# The 1976 study uses variable numbers starting with V76
# House vote in post-election: maybe V764...
# Let me search by looking at variables with values 1 and 2 (Dem/Rep)
# near the end of the questionnaire (post-election)

# Try V763837, V763838, V763966, V763967 (common post-election vote vars)
for (v in c("V763837", "V763838", "V763826", "V763827", "V763966",
            "V763967", "V763968", "V763969")) {
  if (v %in% names(timeseries_1976)) {
    cat("1976", v, ":", paste(sort(unique(timeseries_1976[[v]])), collapse=", "), "\\n")
  }
}

# Actually, for ANES 1976: the House vote post-election is typically around V763837 or V763827
# Let me check V763826-V763840
for (i in 3826:3840) {
  v <- paste0("V76", i)
  if (v %in% names(timeseries_1976)) {
    vals <- sort(unique(timeseries_1976[[v]]))
    if (length(vals) <= 10) {
      cat(v, ":", paste(vals, collapse=", "), "\\n")
    }
  }
}

# ICPSR variable name for 1976 House vote post-election
# Typically the vote is around variable 200+ in post-election section
# Let me search for vars with exactly values {1, 2} or {0, 1, 2}
cat("\\nVars with only values 1,2 (potential vote variables):\\n")
count <- 0
for (v in names(timeseries_1976)) {
  vals <- sort(unique(timeseries_1976[[v]][!is.na(timeseries_1976[[v]])]))
  if (identical(vals, c(1L, 2L)) || identical(vals, c(0L, 1L, 2L))) {
    n1 <- sum(timeseries_1976[[v]] == 1, na.rm=TRUE)
    n2 <- sum(timeseries_1976[[v]] == 2, na.rm=TRUE)
    if (n1 > 200 && n2 > 200 && abs(n1-n2) < n1) {  # roughly balanced
      cat(v, ": 1=", n1, ", 2=", n2, "\\n")
      count <- count + 1
      if (count > 30) break
    }
  }
}

# Also check the case ID - VDSETNO might be it
cat("\\nVDSETNO 1976:", length(unique(timeseries_1976$VDSETNO)), "unique values\\n")
cat("VDSETNO 1974:", length(unique(timeseries_1974$VDSETNO)), "unique values\\n")
cat("VDSETNO overlap:", length(intersect(timeseries_1976$VDSETNO, timeseries_1974$VDSETNO)), "\\n")

# V763001 might be case ID
cat("\\nV763001 range:", min(timeseries_1976$V763001), "-", max(timeseries_1976$V763001), "\\n")
cat("V763001 unique:", length(unique(timeseries_1976$V763001)), "\\n")
cat("V742001 range:", min(timeseries_1974$V742001), "-", max(timeseries_1974$V742001), "\\n")
cat("V742001 unique:", length(unique(timeseries_1974$V742001)), "\\n")
overlap <- length(intersect(timeseries_1976$V763001, timeseries_1974$V742001))
cat("V763001/V742001 overlap:", overlap, "\\n")
'''

result = subprocess.run(
    ['Rscript', '-e', r_code],
    capture_output=True, text=True, timeout=120,
    cwd='/Users/administrator/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/Replication_Claude_Bartels_v2'
)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr[-500:])
