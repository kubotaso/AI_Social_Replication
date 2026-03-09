# Investigate N and coefficient discrepancies for 1960 and 1996
library(anesr)
data(timeseries_cum)
df <- timeseries_cum

# Check 1960 data more carefully
cat("=== 1960 Analysis ===\n")
d60 <- df[df$VCF0004 == 1960, ]
cat("Total 1960 obs:", nrow(d60), "\n")
cat("VCF0704a distribution in 1960:\n")
print(table(d60$VCF0704a, useNA="always"))
cat("VCF0301 distribution in 1960:\n")
print(table(d60$VCF0301, useNA="always"))

# How many have both valid VCF0704a and VCF0301?
valid60 <- d60[d60$VCF0704a %in% c(1,2) & d60$VCF0301 %in% 1:7, ]
cat("Valid for probit (both vote and PID):", nrow(valid60), "\n")

# Check if there's a VCF0704 (3-category) vs VCF0704a issue
cat("VCF0704 (3-cat) in 1960:\n")
print(table(d60$VCF0704, useNA="always"))

# Check 1996 data
cat("\n=== 1996 Analysis ===\n")
d96 <- df[df$VCF0004 == 1996, ]
cat("Total 1996 obs:", nrow(d96), "\n")
cat("VCF0704a distribution in 1996:\n")
print(table(d96$VCF0704a, useNA="always"))
cat("VCF0301 distribution in 1996:\n")
print(table(d96$VCF0301, useNA="always"))

valid96 <- d96[d96$VCF0704a %in% c(1,2) & d96$VCF0301 %in% 1:7, ]
cat("Valid for probit:", nrow(valid96), "\n")

# Compare with individual 1996 study
data(timeseries_1996)
t96 <- timeseries_1996
cat("\n=== Individual 1996 study ===\n")
cat("Obs:", nrow(t96), "\n")
cat("Column names containing '960':\n")
print(grep("960", names(t96), value=TRUE)[1:20])

# Check if individual year has different party ID coding
cat("\nV960420 (party ID) in 1996 individual:\n")
if ("V960420" %in% names(t96)) print(table(t96$V960420, useNA="always"))

cat("\nV961082 (pres vote) in 1996 individual:\n")
if ("V961082" %in% names(t96)) print(table(t96$V961082, useNA="always"))

# Check 1952: why are we 26 short?
cat("\n=== 1952 Analysis ===\n")
d52 <- df[df$VCF0004 == 1952, ]
cat("Total 1952 obs:", nrow(d52), "\n")
cat("VCF0704a distribution:\n")
print(table(d52$VCF0704a, useNA="always"))
cat("VCF0301 distribution:\n")
print(table(d52$VCF0301, useNA="always"))

valid52 <- d52[d52$VCF0704a %in% c(1,2) & d52$VCF0301 %in% 1:7, ]
cat("Valid for probit:", nrow(valid52), "\n")

# Check crosstab of VCF0704a and VCF0301 for 1952 - any that have vote but no PID?
cat("Crosstab for 1952 - VCF0704a by VCF0301 NA status:\n")
d52$pid_valid <- d52$VCF0301 %in% 1:7
d52$vote_valid <- d52$VCF0704a %in% c(1,2)
print(table(d52$vote_valid, d52$pid_valid, dnn=c("vote_valid", "pid_valid")))
