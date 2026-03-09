# Extract individual year data for 1960 and 1996 to check if different PID coding
library(anesr)

# 1960 individual study
data(timeseries_1960)  # Error expected if not available
# Check available datasets
available <- data(package="anesr")$results[,"Item"]
cat("Available datasets:\n")
cat(available[grepl("1960|1996|1952|1956", available)], sep="\n")

# Try to check if timeseries_1960 exists
if ("timeseries_1960" %in% available) {
  cat("\ntimeseries_1960 is available\n")
} else {
  cat("\ntimeseries_1960 NOT available\n")
}

# Check 1996
data(timeseries_1996)
t96 <- timeseries_1996
cat("\n=== 1996 individual dataset ===\n")
cat("N:", nrow(t96), "\n")

# Find party ID variable in 1996 - likely V961104
pid_vars_96 <- grep("V9611[0-9]{2}", names(t96), value=TRUE)
cat("PID-like vars (V9611xx):", head(pid_vars_96, 10), "\n")

# V961104 should be party ID summary
if ("V961104" %in% names(t96)) {
  cat("\nV961104 distribution:\n")
  print(table(t96$V961104, useNA="always"))
}

# Find pres vote - V961005 or similar
vote_vars_96 <- grep("V96100[0-9]", names(t96), value=TRUE)
cat("\nVote vars (V96100x):", vote_vars_96, "\n")
if ("V961005" %in% names(t96)) {
  cat("V961005:\n")
  print(table(t96$V961005, useNA="always"))
}

# Also check the CDF for comparison
data(timeseries_cum)
cdf <- timeseries_cum
cdf_96 <- cdf[cdf$VCF0004 == 1996, ]
cat("\n=== CDF 1996 ===\n")
cat("N:", nrow(cdf_96), "\n")
cat("VCF0301 distribution:\n")
print(table(cdf_96$VCF0301, useNA="always"))
cat("VCF0704a distribution:\n")
print(table(cdf_96$VCF0704a, useNA="always"))

# Check if there are more valid responses in individual vs CDF
cat("\nCDF voters with valid PID:", sum(cdf_96$VCF0704a %in% c(1,2) & cdf_96$VCF0301 %in% 1:7), "\n")
