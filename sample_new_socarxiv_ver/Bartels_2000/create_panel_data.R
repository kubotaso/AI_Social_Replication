# Create panel datasets for Tables 4 and 5
library(anesr)
data(timeseries_cum)
cdf <- timeseries_cum

# === 1958-1960 Panel ===
# Link 1958 and 1960 respondents via VCF0006
cdf_58 <- cdf[cdf$VCF0004 == 1958, c("VCF0006", "VCF0006a", "VCF0301")]
cdf_60 <- cdf[cdf$VCF0004 == 1960, ]

# Rename 1958 party ID
names(cdf_58)[names(cdf_58) == "VCF0301"] <- "VCF0301_lagged"

# Merge
panel_60 <- merge(cdf_60, cdf_58[, c("VCF0006", "VCF0301_lagged")], by="VCF0006", all.x=FALSE)
cat("1958-1960 panel: matched", nrow(panel_60), "respondents\n")
cat("With valid lagged PID:", sum(panel_60$VCF0301_lagged %in% 1:7), "\n")
cat("With valid current PID:", sum(panel_60$VCF0301 %in% 1:7), "\n")
cat("With valid pres vote:", sum(panel_60$VCF0704a %in% c(1,2)), "\n")
panel_60_valid <- panel_60[panel_60$VCF0704a %in% c(1,2) & panel_60$VCF0301 %in% 1:7 & panel_60$VCF0301_lagged %in% 1:7, ]
cat("Valid for full analysis:", nrow(panel_60_valid), "(paper: 1057)\n")

# === 1974-1976 Panel ===
cdf_74 <- cdf[cdf$VCF0004 == 1974 & cdf$VCF0016 == 1, c("VCF0006", "VCF0301")]
cdf_76 <- cdf[cdf$VCF0004 == 1976 & cdf$VCF0016 == 1, ]

names(cdf_74)[names(cdf_74) == "VCF0301"] <- "VCF0301_lagged"

panel_76 <- merge(cdf_76, cdf_74[, c("VCF0006", "VCF0301_lagged")], by="VCF0006")
cat("\n1974-1976 panel: matched", nrow(panel_76), "respondents\n")
cat("With valid pres vote:", sum(panel_76$VCF0704a %in% c(1,2)), "\n")
panel_76_valid <- panel_76[panel_76$VCF0704a %in% c(1,2) & panel_76$VCF0301 %in% 1:7 & panel_76$VCF0301_lagged %in% 1:7, ]
cat("Valid for full analysis:", nrow(panel_76_valid), "(paper: 799)\n")

# Also check without VCF0016 filter for 1976
cdf_76_all <- cdf[cdf$VCF0004 == 1976, ]
panel_76b <- merge(cdf_76_all, cdf_74[, c("VCF0006", "VCF0301_lagged")], by="VCF0006")
cat("Without VCF0016 filter on 1976:", nrow(panel_76b), "\n")
panel_76b_valid <- panel_76b[panel_76b$VCF0704a %in% c(1,2) & panel_76b$VCF0301 %in% 1:7 & panel_76b$VCF0301_lagged %in% 1:7, ]
cat("Valid:", nrow(panel_76b_valid), "\n")

# === 1990-1992 Panel ===
# Use the 1992 individual dataset which has V900320 (1990 party ID)
data(timeseries_1992)
t92 <- timeseries_1992

# Identify panel members: V900320 not in {9} (9 = NA/not panel)
panel_92 <- t92[t92$V900320 %in% 0:8, ]
cat("\n1990-1992 panel:", nrow(panel_92), "respondents with 1990 data\n")

# Recode 1992 party ID (V923634): 0=Strong Dem, 1=Not very strong Dem, ...6=Strong Rep
# Map to CDF 7-point scale: V923634 + 1 for codes 0-6
panel_92$pid_current <- ifelse(panel_92$V923634 %in% 0:6, panel_92$V923634 + 1, NA)
panel_92$pid_lagged <- ifelse(panel_92$V900320 %in% 0:6, panel_92$V900320 + 1, NA)

# 1992 pres vote (V925609): 1=Clinton(Dem), 2=Bush(Rep), 3=Perot
panel_92$vote_pres <- ifelse(panel_92$V925609 %in% c(1,2), panel_92$V925609, NA)

# 1992 House vote - need to find variable
# V925701 might be the House vote
cat("V925701:", "V925701" %in% names(t92), "\n")
house_vote_vars <- grep("V9257[0-9]{2}", names(t92), value=TRUE)
cat("House vote vars:", head(house_vote_vars, 5), "\n")
if ("V925701" %in% names(t92)) {
  cat("V925701 distribution:\n")
  print(table(t92$V925701, useNA="always"))
}

# Actually look for the congressional vote variable more carefully
# In the CDF, the source for VCF0707 in 1992 would be something like V925xxx
# Let me check the CDF documentation
data(timeseries_cum_doc)
doc <- timeseries_cum_doc
idx <- which(doc$id == "vcf0707")
if (length(idx) > 0) {
  cat("\nVCF0707 text (first 2000 chars):\n")
  cat(substr(doc$text[idx], 1, 2000), "\n")
}

panel_92_valid <- panel_92[!is.na(panel_92$pid_current) & !is.na(panel_92$pid_lagged) & !is.na(panel_92$vote_pres), ]
cat("\nValid for presidential analysis:", nrow(panel_92_valid), "(paper: 729)\n")

# Save panel datasets
# For CDF-based panels (1960, 1976), save the merged data
write.csv(panel_60, "panel_1960.csv", row.names=FALSE)
write.csv(panel_76b, "panel_1976.csv", row.names=FALSE)

# For 1992, save the individual study data with recoded variables
panel_92_out <- data.frame(
  pid_current = panel_92$pid_current,
  pid_lagged = panel_92$pid_lagged,
  vote_pres = panel_92$vote_pres,
  V923634 = panel_92$V923634,
  V900320 = panel_92$V900320,
  V925609 = panel_92$V925609
)
# Also try to find House vote
if ("V925701" %in% names(panel_92)) {
  panel_92_out$house_vote_raw <- panel_92$V925701
}
write.csv(panel_92_out, "panel_1992.csv", row.names=FALSE)
cat("\nPanel datasets saved.\n")
