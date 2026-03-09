# Download ANES Cumulative Data File
options(repos = c(CRAN = "https://cloud.r-project.org"))

# Try anesr package first
tryCatch({
  if (!"anesr" %in% installed.packages()[,"Package"]) {
    cat("Installing anesr...\n")
    install.packages("anesr",
                     repos = c("https://jamesmartherus.r-universe.dev",
                               "https://cloud.r-project.org"),
                     quiet = TRUE)
  }
  library(anesr)
  cat("anesr loaded successfully\n")
  cat("Available datasets:\n")
  print(data(package="anesr")$results[,"Item"])

  # Load the cumulative data file
  data(timeseries_cum)
  cat("\nDimensions:", dim(timeseries_cum), "\n")
  cat("Column names (first 50):\n")
  print(head(names(timeseries_cum), 50))

  # Save to CSV
  write.csv(timeseries_cum, "anes_cumulative.csv", row.names = FALSE)
  cat("\nSaved to anes_cumulative.csv\n")

}, error = function(e) {
  cat("anesr approach failed:", e$message, "\n")
  cat("Trying alternative approach...\n")

  # Try dataverse package
  tryCatch({
    if (!"dataverse" %in% installed.packages()[,"Package"]) {
      install.packages("dataverse", quiet = TRUE)
    }
    library(dataverse)
    cat("Trying Harvard Dataverse...\n")
  }, error = function(e2) {
    cat("dataverse also failed:", e2$message, "\n")
  })
})
