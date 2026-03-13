import subprocess

r_code = '''
library(anesr)

# Try to get the 1956-58-60 panel study
cat("Available datasets:\\n")
d <- data(package="anesr")
items <- d$results[, "Item"]
panel_items <- items[grep("panel|1956|1958|1960", items, ignore.case=TRUE)]
cat(paste(panel_items, collapse="\\n"), "\\n")

# Try loading panel_1956_1960
tryCatch({
  data("panel_1956_1960")
  cat("\\nLoaded panel_1956_1960\\n")
  cat("Dimensions:", nrow(panel_1956_1960), "x", ncol(panel_1956_1960), "\\n")
  cat("Column names (first 50):", paste(head(names(panel_1956_1960), 50), collapse=", "), "\\n")

  # Look for party ID in 1958 wave (V58xxxx)
  pid_vars <- names(panel_1956_1960)[grep("V58", names(panel_1956_1960))]
  cat("\\nV58 vars:", paste(head(pid_vars, 30), collapse=", "), "\\n")

  # Look for party ID in 1960 wave (V60xxxx)
  pid_vars60 <- names(panel_1956_1960)[grep("V60", names(panel_1956_1960))]
  cat("V60 vars:", paste(head(pid_vars60, 30), collapse=", "), "\\n")

  # Search for PID variable (7-point, values 1-7)
  cat("\\nSearching for PID (values 1-7) in panel data:\\n")
  for (v in names(panel_1956_1960)) {
    vals <- sort(unique(panel_1956_1960[[v]][!is.na(panel_1956_1960[[v]])]))
    if (length(vals) >= 7 && length(vals) <= 12 && all(1:7 %in% vals)) {
      n_vals <- table(panel_1956_1960[[v]])
      if (all(n_vals[as.character(1:7)] > 20)) {
        cat(v, ": ", paste(paste0(names(n_vals), "=", n_vals), collapse=", "), "\\n")
      }
    }
  }

  # Look for House vote variable
  cat("\\nSearching for House vote (values 1,2 or 0,1):\\n")
  for (v in names(panel_1956_1960)) {
    vals <- sort(unique(panel_1956_1960[[v]][!is.na(panel_1956_1960[[v]])]))
    if (length(vals) >= 2 && length(vals) <= 6) {
      n_vals <- table(panel_1956_1960[[v]])
      total <- sum(n_vals)
      # House vote: 2 main values with 300-800 each
      if (length(vals) <= 4 && max(n_vals) < 900 && min(n_vals[1:min(2,length(n_vals))]) > 200) {
        if (grepl("V60", v)) {
          cat(v, ": ", paste(paste0(names(n_vals), "=", n_vals), collapse=", "), "\\n")
        }
      }
    }
  }
}, error = function(e) {
  cat("Error loading panel_1956_1960:", e$message, "\\n")
})

# Try timeseries_1960
tryCatch({
  data("timeseries_1960")
  cat("\\nLoaded timeseries_1960\\n")
  cat("Dimensions:", nrow(timeseries_1960), "x", ncol(timeseries_1960), "\\n")

  # Search for House vote
  cat("Searching for House vote in timeseries_1960:\\n")
  for (v in names(timeseries_1960)) {
    vals <- sort(unique(timeseries_1960[[v]][!is.na(timeseries_1960[[v]])]))
    if (length(vals) >= 2 && length(vals) <= 6) {
      n_vals <- table(timeseries_1960[[v]])
      # Look for variables with 2 main categories summing to 700-1100
      main_cats <- sort(n_vals, decreasing=TRUE)[1:min(2, length(n_vals))]
      if (all(main_cats > 200) && sum(main_cats) > 600 && sum(main_cats) < 1100) {
        if (grepl("V60", v)) {
          cat(v, ": ", paste(paste0(names(n_vals), "=", n_vals), collapse=", "), "\\n")
        }
      }
    }
  }

  # PID in timeseries_1960
  cat("\\nPID search in timeseries_1960:\\n")
  for (v in names(timeseries_1960)) {
    vals <- sort(unique(timeseries_1960[[v]][!is.na(timeseries_1960[[v]])]))
    if (length(vals) >= 7 && length(vals) <= 12 && all(1:7 %in% vals)) {
      n_vals <- table(timeseries_1960[[v]])
      if (all(n_vals[as.character(1:7)] > 20)) {
        cat(v, ": ", paste(paste0(names(n_vals), "=", n_vals), collapse=", "), "\\n")
      }
    }
  }
}, error = function(e) {
  cat("Error:", e$message, "\\n")
})
'''

result = subprocess.run(
    ['Rscript', '-e', r_code],
    capture_output=True, text=True, timeout=300,
    cwd='/Users/administrator/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/Replication_Claude_Bartels_v2'
)
print(result.stdout[:8000])
if result.returncode != 0:
    print("STDERR:", result.stderr[-1000:])
