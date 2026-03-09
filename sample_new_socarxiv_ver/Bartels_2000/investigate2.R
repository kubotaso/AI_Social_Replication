# Check VCF0301 = 0 vs NA patterns
library(anesr)
data(timeseries_cum)
df <- timeseries_cum

years <- c(1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996)

cat("Year | Total_voters | PID_valid | PID_0 | PID_NA | Paper_N\n")
paper_n <- c(1181, 1266, 885, 1111, 911, 1587, 1322, 877, 1376, 1195, 1357, 1034)
for (i in seq_along(years)) {
  yr <- years[i]
  d <- df[df$VCF0004 == yr, ]
  voters <- d[d$VCF0704a %in% c(1,2), ]
  pid_valid <- sum(voters$VCF0301 %in% 1:7, na.rm=TRUE)
  pid_0 <- sum(voters$VCF0301 == 0, na.rm=TRUE)
  pid_na <- sum(is.na(voters$VCF0301))
  cat(sprintf("%d | %d | %d | %d | %d | %d\n", yr, nrow(voters), pid_valid, pid_0, pid_na, paper_n[i]))
}

# Try including VCF0301=0 with pure independents (VCF0301=4)
cat("\n\nIf we include VCF0301=0 as valid (mapped to pure independent/code 4):\n")
for (i in seq_along(years)) {
  yr <- years[i]
  d <- df[df$VCF0004 == yr, ]
  voters <- d[d$VCF0704a %in% c(1,2), ]
  pid_valid_incl0 <- sum(voters$VCF0301 %in% 0:7, na.rm=TRUE)
  cat(sprintf("%d: N=%d (paper: %d, diff: %d)\n", yr, pid_valid_incl0, paper_n[i],
              pid_valid_incl0 - paper_n[i]))
}
