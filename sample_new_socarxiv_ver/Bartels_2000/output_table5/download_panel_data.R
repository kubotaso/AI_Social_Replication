# Try to download ANES panel study data using anesr package
# We need the original panel studies for proper N values

# Install anesr if not already installed
if (!requireNamespace("anesr", quietly = TRUE)) {
  install.packages("remotes", repos = "https://cloud.r-project.org")
  remotes::install_github("jamesmartherus/anesr")
}

library(anesr)

# List available datasets
cat("Available datasets:\n")
tryCatch({
  datasets <- data(package = "anesr")
  print(datasets$results[, "Item"])
}, error = function(e) {
  cat("Error listing datasets:", e$message, "\n")
})

# Try to load 1956-1960 panel
cat("\n\nTrying to load 1956-1960 panel data...\n")
tryCatch({
  data("timeseries_1958", package = "anesr")
  cat("1958 timeseries loaded, N=", nrow(timeseries_1958), "\n")
  cat("Column names (first 50):", paste(names(timeseries_1958)[1:min(50, ncol(timeseries_1958))], collapse=", "), "\n")
}, error = function(e) {
  cat("Error:", e$message, "\n")
})

# Try panel_1956_1960
tryCatch({
  data("panel_1956_1960", package = "anesr")
  cat("1956-1960 panel loaded, N=", nrow(panel_1956_1960), "\n")
}, error = function(e) {
  cat("panel_1956_1960 Error:", e$message, "\n")
})

# Try 1972-1976 panel
cat("\nTrying to load 1972-1976 panel data...\n")
tryCatch({
  data("timeseries_1974", package = "anesr")
  cat("1974 timeseries loaded, N=", nrow(timeseries_1974), "\n")
}, error = function(e) {
  cat("Error:", e$message, "\n")
})

tryCatch({
  data("panel_1972_1976", package = "anesr")
  cat("1972-1976 panel loaded, N=", nrow(panel_1972_1976), "\n")
}, error = function(e) {
  cat("panel_1972_1976 Error:", e$message, "\n")
})

# Try 1990-1992 panel
cat("\nTrying to load 1990-1992 panel data...\n")
tryCatch({
  data("timeseries_1990", package = "anesr")
  cat("1990 timeseries loaded, N=", nrow(timeseries_1990), "\n")
}, error = function(e) {
  cat("Error:", e$message, "\n")
})

tryCatch({
  data("panel_1990_1992", package = "anesr")
  cat("1990-1992 panel loaded, N=", nrow(panel_1990_1992), "\n")
}, error = function(e) {
  cat("panel_1990_1992 Error:", e$message, "\n")
})

# Try the ANES 1956-60 panel specifically
cat("\n\nTrying timeseries_1960...\n")
tryCatch({
  data("timeseries_1960", package = "anesr")
  cat("1960 timeseries loaded, N=", nrow(timeseries_1960), "\n")
  cat("Columns:", ncol(timeseries_1960), "\n")
}, error = function(e) {
  cat("Error:", e$message, "\n")
})
