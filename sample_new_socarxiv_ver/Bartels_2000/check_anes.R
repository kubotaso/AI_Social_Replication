# Check ANES cumulative data for key variables
library(anesr)

# Load cumulative data
data(timeseries_cum)
df <- timeseries_cum

# Check year range
cat("Year variable VCF0004:\n")
print(table(df$VCF0004))

# Key variables:
# VCF0301 - Party ID 7-point scale
# VCF0303 - Party ID summary (3-category)
# VCF0704 - Presidential vote (2-party)
# VCF0706 - Congressional vote
# VCF0702 - Voter registration / turnout
# VCF0105a/b - Race
# VCF0113 - Region (South/Non-South)

cat("\nParty ID 7-point (VCF0301):\n")
print(table(df$VCF0301, useNA="always"))

cat("\nParty ID summary (VCF0303):\n")
print(table(df$VCF0303, useNA="always"))

cat("\nPresidential vote (VCF0704):\n")
print(table(df$VCF0704, useNA="always"))

cat("\nPresidential vote 2-party (VCF0704a):\n")
if ("VCF0704a" %in% names(df)) print(table(df$VCF0704a, useNA="always"))

cat("\nCongressional vote (VCF0706):\n")
print(table(df$VCF0706, useNA="always"))

cat("\nTurnout (VCF0702):\n")
print(table(df$VCF0702, useNA="always"))

cat("\nRace (VCF0105a):\n")
print(table(df$VCF0105a, useNA="always"))

cat("\nRegion (VCF0113):\n")
print(table(df$VCF0113, useNA="always"))

# Check for incumbency variable
cat("\nIncumbency variables:\n")
inc_vars <- grep("VCF09", names(df), value=TRUE)
print(inc_vars)

# Check for VCF0905 (incumbent party in congressional district)
if ("VCF0905" %in% names(df)) {
  cat("\nVCF0905 (incumbency):\n")
  print(table(df$VCF0905, useNA="always"))
}

# Also check panel variables
panel_vars <- grep("VCF0012|VCF0013|VCF0014|VCF0015|VCF0016|VCF0017|VCF0018|VCF0019", names(df), value=TRUE)
cat("\nPanel/study type variables:\n")
print(panel_vars)
