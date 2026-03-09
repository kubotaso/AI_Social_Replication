# Try using individual year datasets for 1960 and 1996
library(anesr)

# 1996 individual study
data(timeseries_1996)
t96 <- timeseries_1996

cat("=== 1996 Individual Study ===\n")
cat("N:", nrow(t96), "\n")

# Need to find party ID and presidential vote variables
# V960420: Party ID (from CDF source vars documentation)
# V961082: Presidential vote
cat("\nV960420 (Party ID):\n")
print(table(t96$V960420, useNA="always"))

# According to 1996 codebook, V960420 codes:
# 0 = Strong Democrat
# 1 = Not very strong Democrat
# 2 = Independent, closer to Democratic Party
# 3 = Independent (neither)
# 4 = Independent, closer to Republican Party
# 5 = Not very strong Republican
# 6 = Strong Republican
# 7 = NA/don't know on initial question
# 8 = Apolitical (on initial)
# 9 = NA/DK strength or direction

cat("\nV961082 (Presidential vote):\n")
print(table(t96$V961082, useNA="always"))
# 1 = Clinton (Democrat)
# 2 = Dole (Republican)
# 3 = Perot
# 7 = Other
# 0 = not applicable/didn't vote
# 8 = DK
# 9 = NA/refused

# Recode to match CDF 7-point scale
# V960420: 0->Strong Dem(=1 in CDF), 1->Weak Dem(=2), 2->Ind-Dem(=3),
#           3->Ind-Ind(=4), 4->Ind-Rep(=5), 5->Weak Rep(=6), 6->Strong Rep(=7)
# So CDF VCF0301 = V960420 + 1 for valid codes 0-6

t96$pid7 <- ifelse(t96$V960420 %in% 0:6, t96$V960420 + 1, NA)
t96$vote_rep <- ifelse(t96$V961082 == 2, 1, ifelse(t96$V961082 == 1, 0, NA))

valid96 <- t96[!is.na(t96$pid7) & !is.na(t96$vote_rep), ]
cat("\nValid N for 1996 probit:", nrow(valid96), "\n")

# Compare with CDF
data(timeseries_cum)
cdf96 <- timeseries_cum[timeseries_cum$VCF0004 == 1996, ]
cdf_valid <- cdf96[cdf96$VCF0704a %in% c(1,2) & cdf96$VCF0301 %in% 1:7, ]
cat("CDF valid N for 1996 probit:", nrow(cdf_valid), "\n")

# Check detailed PID distribution in individual vs CDF
cat("\nPID distribution comparison (among valid probit cases):\n")
cat("Individual study (recoded):\n")
print(table(valid96$pid7))
cat("CDF:\n")
print(table(cdf_valid$VCF0301))

# Also check 1960 - get individual study if available
cat("\n\n=== 1960 Check ===\n")
# 1960 is not in the available individual year datasets in anesr
# But let me check what's in the CDF for 1960 more carefully
d60 <- timeseries_cum[timeseries_cum$VCF0004 == 1960, ]
valid60 <- d60[d60$VCF0704a %in% c(1,2) & d60$VCF0301 %in% 1:7, ]
cat("CDF valid N for 1960:", nrow(valid60), "\n")
cat("PID distribution:\n")
print(table(valid60$VCF0301))
cat("Vote distribution:\n")
print(table(valid60$VCF0704a))

# Check crosstab of PID by vote for 1960
cat("\nCrosstab PID x Vote for 1960:\n")
print(table(valid60$VCF0301, valid60$VCF0704a, dnn=c("PID", "Vote")))
