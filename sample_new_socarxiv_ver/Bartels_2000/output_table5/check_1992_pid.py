import subprocess

r_code = '''
library(anesr)
data("timeseries_1990")
data("timeseries_1992")

cat("=== 1990 ===\\n")
cat("N:", nrow(timeseries_1990), "\\n")

# Find PID variable in 1990
for (v in names(timeseries_1990)) {
  vals <- sort(unique(timeseries_1990[[v]][!is.na(timeseries_1990[[v]])]))
  if (length(vals) >= 7 && length(vals) <= 12 && all(1:7 %in% vals)) {
    n_vals <- table(timeseries_1990[[v]])
    if (all(n_vals[as.character(1:7)] > 30)) {
      cat(v, ": ", paste(paste0(names(n_vals), "=", n_vals), collapse=", "), "\\n")
    }
  }
}

# Find case ID variable in 1990
cat("\\n1990 vars with ID-like names:\\n")
for (v in names(timeseries_1990)) {
  if (grepl("CASEID|case|V90000|V900001|V900002|V900003|V900004", v, ignore.case=TRUE)) {
    cat(v, ": range ", min(timeseries_1990[[v]], na.rm=TRUE), "to", max(timeseries_1990[[v]], na.rm=TRUE), "\\n")
  }
}

cat("\\n=== 1992 ===\\n")
cat("N:", nrow(timeseries_1992), "\\n")

# Find case ID in 1992
for (v in names(timeseries_1992)) {
  if (grepl("CASEID|case|V92000|V920001|V920002|V920003|V920004", v, ignore.case=TRUE)) {
    cat(v, ": range ", min(timeseries_1992[[v]], na.rm=TRUE), "to", max(timeseries_1992[[v]], na.rm=TRUE), "\\n")
  }
}

# Find PID in 1992
cat("\\n1992 PID vars:\\n")
for (v in names(timeseries_1992)) {
  vals <- sort(unique(timeseries_1992[[v]][!is.na(timeseries_1992[[v]])]))
  if (length(vals) >= 7 && length(vals) <= 12 && all(1:7 %in% vals)) {
    n_vals <- table(timeseries_1992[[v]])
    if (all(n_vals[as.character(1:7)] > 30)) {
      cat(v, ": ", paste(paste0(names(n_vals), "=", n_vals), collapse=", "), "\\n")
    }
  }
}

# Find House vote in 1992 - looking for V925xxx
cat("\\n1992 House vote candidates:\\n")
for (v in names(timeseries_1992)) {
  if (grepl("V9257|V9258|V9253|V9254|V9255|V9256|V9259|V926", v)) {
    vals <- sort(unique(timeseries_1992[[v]][!is.na(timeseries_1992[[v]])]))
    if (length(vals) <= 10 && length(vals) >= 2) {
      n_vals <- table(timeseries_1992[[v]])
      total <- sum(n_vals)
      cat(v, ": ", paste(paste0(names(n_vals), "=", n_vals), collapse=", "), " (total=", total, ")\\n")
    }
  }
}

# Check case IDs: first few vars in each dataset
cat("\\nFirst 5 vars in 1990:", paste(head(names(timeseries_1990), 5), collapse=", "), "\\n")
cat("First 5 vars in 1992:", paste(head(names(timeseries_1992), 5), collapse=", "), "\\n")

# Check V900001 through V900010
cat("\\n1990 V9000xx:\\n")
for (i in 1:20) {
  v <- paste0("V", sprintf("%06d", 900000+i))
  if (v %in% names(timeseries_1990)) {
    cat(v, ": range", min(timeseries_1990[[v]], na.rm=TRUE), "to", max(timeseries_1990[[v]], na.rm=TRUE), "\\n")
  }
}

# Check V923xxx for IDs
cat("\\n1992 V923xxx:\\n")
for (i in 1:20) {
  v <- paste0("V", sprintf("%06d", 923000+i))
  if (v %in% names(timeseries_1992)) {
    cat(v, ": range", min(timeseries_1992[[v]], na.rm=TRUE), "to", max(timeseries_1992[[v]], na.rm=TRUE), "\\n")
  }
}
'''

result = subprocess.run(
    ['Rscript', '-e', r_code],
    capture_output=True, text=True, timeout=300,
    cwd='/Users/administrator/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/Replication_Claude_Bartels_v2'
)
print(result.stdout[:8000])
if result.returncode != 0:
    print("STDERR:", result.stderr[-500:])
