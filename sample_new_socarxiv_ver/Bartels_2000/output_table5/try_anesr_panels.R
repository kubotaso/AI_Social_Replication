# Try to access original ANES panel data via anesr package
library(anesr)

# Check what datasets are available
cat("Available ANES datasets:\n")
datasets <- data(package = "anesr")$results[, "Item"]
print(datasets)

# Try to load the 1956-1960 panel study
tryCatch({
    data(anes_timeseries_cdf)
    cat("\nCDF loaded, dimensions:", dim(anes_timeseries_cdf), "\n")
}, error = function(e) {
    cat("Error loading CDF:", e$message, "\n")
})

# Check for panel-specific datasets
panel_datasets <- grep("panel|1956|1972|1990", datasets, value = TRUE, ignore.case = TRUE)
cat("\nPanel-related datasets:", panel_datasets, "\n")
