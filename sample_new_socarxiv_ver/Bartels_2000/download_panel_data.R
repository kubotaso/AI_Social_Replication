# Download panel data for Tables 4 and 5
# Need: 1956-58-60 panel, 1972-74-76 panel, 1990-91-92 panel
library(anesr)

# For 1976 panel: need 1974 party ID for 1976 respondents
# timeseries_1974 and timeseries_1976 are available
data(timeseries_1974)
data(timeseries_1976)

cat("=== 1974 study ===\n")
cat("N:", nrow(timeseries_1974), "\n")
# Find party ID variable - V742204 per codebook
cat("V742204 (party ID 1974):\n")
if ("V742204" %in% names(timeseries_1974)) {
  print(table(timeseries_1974$V742204, useNA="always"))
}
# Find case ID to link across waves
pid_vars_74 <- grep("V7420[0-9][0-9]", names(timeseries_1974), value=TRUE)
cat("Potential ID vars:", head(pid_vars_74, 5), "\n")

# Check for a case ID or respondent ID
id_vars_74 <- grep("V7400[0-9][0-9]", names(timeseries_1974), value=TRUE)
cat("ID-like vars:", head(id_vars_74, 10), "\n")

cat("\n=== 1976 study ===\n")
cat("N:", nrow(timeseries_1976), "\n")
if ("V763174" %in% names(timeseries_1976)) {
  cat("V763174 (party ID 1976):\n")
  print(table(timeseries_1976$V763174, useNA="always"))
}

# Check for panel indicator in 1976
# The 1972-74-76 panel respondents should have data from both waves
panel_vars_76 <- grep("V76300[0-9]", names(timeseries_1976), value=TRUE)
cat("First vars in 1976:", head(names(timeseries_1976), 20), "\n")

# For the 1990-92 panel
data(timeseries_1990)
data(timeseries_1992)
cat("\n=== 1990 study ===\n")
cat("N:", nrow(timeseries_1990), "\n")
cat("First vars:", head(names(timeseries_1990), 10), "\n")

cat("\n=== 1992 study ===\n")
cat("N:", nrow(timeseries_1992), "\n")
cat("First vars:", head(names(timeseries_1992), 10), "\n")

# Check for panel case ID in 1992
panel_id_92 <- grep("V92[0-9]{4}", names(timeseries_1992), value=TRUE)
cat("Vars starting V92:", head(panel_id_92, 10), "\n")

# In the CDF, VCF0016 might indicate panel membership
data(timeseries_cum)
cdf <- timeseries_cum

# Check VCF0016 for panel indicators
cat("\n=== CDF Panel indicators ===\n")
cat("VCF0016 by year (1972-1976, 1990-1992):\n")
for (yr in c(1972, 1974, 1976, 1990, 1992)) {
  d <- cdf[cdf$VCF0004 == yr, ]
  cat(yr, ": N=", nrow(d), " VCF0016 dist: ")
  print(table(d$VCF0016, useNA="ifany"))
}
