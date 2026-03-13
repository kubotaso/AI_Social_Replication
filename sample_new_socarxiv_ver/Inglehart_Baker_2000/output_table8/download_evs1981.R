# Try to download EVS 1981 data using gesisdata package
library(gesisdata)

# Check if credentials are set up
cat("Checking GESIS credentials...\n")
tryCatch({
  # Try to download EVS Longitudinal Data File (ZA4804)
  # or EVS Wave 1 (ZA4438)

  # First try the longitudinal file
  cat("Attempting to download ZA4804 (EVS Longitudinal 1981-2008)...\n")
  gesis_download(file_id = "ZA4804",
                 path = file.path(dirname(getwd()), "data"),
                 filetype = "stata")
  cat("SUCCESS: Downloaded ZA4804\n")
}, error = function(e) {
  cat("Failed ZA4804:", e$message, "\n")

  tryCatch({
    cat("Attempting to download ZA4438 (EVS Wave 1)...\n")
    gesis_download(file_id = "ZA4438",
                   path = file.path(dirname(getwd()), "data"),
                   filetype = "stata")
    cat("SUCCESS: Downloaded ZA4438\n")
  }, error = function(e2) {
    cat("Failed ZA4438:", e2$message, "\n")
    cat("\nGESIS authentication required. Please:\n")
    cat("1. Create a free account at https://login.gesis.org/\n")
    cat("2. Set credentials with: gesisdata::gesis_remotes()\n")
    cat("3. Or manually download from https://search.gesis.org/research_data/ZA4804\n")
  })
})

# List data files to see what we have
cat("\nData directory contents:\n")
print(list.files(file.path(dirname(getwd()), "data")))
