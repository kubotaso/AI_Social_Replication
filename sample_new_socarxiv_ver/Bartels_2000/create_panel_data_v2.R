# Create improved panel datasets for Tables 4 and 5
# Uses VCF0006a for better cross-year linkage
library(anesr)
data(timeseries_cum)
cdf <- timeseries_cum

# === 1958-1960 Panel ===
# Use VCF0006a for linkage (939 overlap vs 857 with VCF0006)
cdf_58 <- cdf[cdf$VCF0004 == 1958, c("VCF0006a", "VCF0301")]
cdf_60 <- cdf[cdf$VCF0004 == 1960, ]

names(cdf_58)[names(cdf_58) == "VCF0301"] <- "VCF0301_lagged"

panel_60 <- merge(cdf_60, cdf_58[, c("VCF0006a", "VCF0301_lagged")], by="VCF0006a", all.x=FALSE)
cat("1958-1960 panel: matched", nrow(panel_60), "respondents\n")
cat("With valid lagged PID:", sum(panel_60$VCF0301_lagged %in% 1:7), "\n")
cat("With valid current PID:", sum(panel_60$VCF0301 %in% 1:7), "\n")
cat("With valid pres vote:", sum(panel_60$VCF0704a %in% c(1,2)), "\n")
panel_60_valid <- panel_60[panel_60$VCF0704a %in% c(1,2) & panel_60$VCF0301 %in% 1:7 & panel_60$VCF0301_lagged %in% 1:7, ]
cat("Valid for full analysis:", nrow(panel_60_valid), "(paper: 1057)\n")

# House vote
cat("With valid house vote:", sum(panel_60$VCF0707 %in% c(1,2)), "\n")
panel_60_house <- panel_60[panel_60$VCF0707 %in% c(1,2) & panel_60$VCF0301 %in% 1:7 & panel_60$VCF0301_lagged %in% 1:7, ]
cat("Valid for house analysis:", nrow(panel_60_house), "(paper: 680)\n")

write.csv(panel_60, "panel_1960.csv", row.names=FALSE)

# === 1974-1976 Panel ===
# Use VCF0006a, include all 1976 respondents (not just VCF0016==1)
cdf_74_panel <- cdf[cdf$VCF0004 == 1974 & cdf$VCF0016 == 1, c("VCF0006a", "VCF0301")]
names(cdf_74_panel)[names(cdf_74_panel) == "VCF0301"] <- "VCF0301_lagged"

cdf_76_all <- cdf[cdf$VCF0004 == 1976, ]
panel_76 <- merge(cdf_76_all, cdf_74_panel[, c("VCF0006a", "VCF0301_lagged")], by="VCF0006a")
cat("\n1974-1976 panel: matched", nrow(panel_76), "respondents\n")
panel_76_valid <- panel_76[panel_76$VCF0704a %in% c(1,2) & panel_76$VCF0301 %in% 1:7 & panel_76$VCF0301_lagged %in% 1:7, ]
cat("Valid for pres analysis:", nrow(panel_76_valid), "(paper: 799)\n")
panel_76_house <- panel_76[panel_76$VCF0707 %in% c(1,2) & panel_76$VCF0301 %in% 1:7 & panel_76$VCF0301_lagged %in% 1:7, ]
cat("Valid for house analysis:", nrow(panel_76_house), "(paper: 740)\n")

write.csv(panel_76, "panel_1976.csv", row.names=FALSE)

# === 1990-1992 Panel ===
# Use 1992 individual dataset with V900320 (1990 party ID already merged)
data(timeseries_1992)
t92 <- timeseries_1992

panel_92 <- t92[t92$V900320 %in% 0:6, ]
cat("\n1990-1992 panel:", nrow(panel_92), "respondents with valid 1990 PID\n")

panel_92$pid_current <- ifelse(panel_92$V923634 %in% 0:6, panel_92$V923634 + 1, NA)
panel_92$pid_lagged <- ifelse(panel_92$V900320 %in% 0:6, panel_92$V900320 + 1, NA)
panel_92$vote_pres <- ifelse(panel_92$V925609 %in% c(1,2), panel_92$V925609, NA)

# House vote: V925701 - need to check coding
# 0=did not vote for House, 1=voted Dem, 5=voted Rep based on distribution
cat("V925701 distribution (House vote):\n")
print(table(panel_92$V925701, useNA="always"))

# Map V925701: 1=Dem, 5=Rep -> 1=Dem, 2=Rep
panel_92$vote_house <- ifelse(panel_92$V925701 == 1, 1, ifelse(panel_92$V925701 == 5, 2, NA))

panel_92_pres_valid <- panel_92[!is.na(panel_92$pid_current) & !is.na(panel_92$pid_lagged) & !is.na(panel_92$vote_pres), ]
cat("Valid for pres analysis:", nrow(panel_92_pres_valid), "(paper: 729)\n")

panel_92_house_valid <- panel_92[!is.na(panel_92$pid_current) & !is.na(panel_92$pid_lagged) & !is.na(panel_92$vote_house), ]
cat("Valid for house analysis:", nrow(panel_92_house_valid), "(paper: 701)\n")

# Save with all needed variables
panel_92_out <- data.frame(
  pid_current = panel_92$pid_current,
  pid_lagged = panel_92$pid_lagged,
  vote_pres = panel_92$vote_pres,
  vote_house = panel_92$vote_house,
  V923634 = panel_92$V923634,
  V900320 = panel_92$V900320,
  V925609 = panel_92$V925609,
  V925701 = panel_92$V925701
)
write.csv(panel_92_out, "panel_1992.csv", row.names=FALSE)

cat("\nPanel datasets saved.\n")
cat("\nSummary of N gaps:\n")
cat("1960 pres: have", nrow(panel_60_valid), "vs paper 1057 (", round(nrow(panel_60_valid)/1057*100,1), "%)\n")
cat("1976 pres: have", nrow(panel_76_valid), "vs paper 799 (", round(nrow(panel_76_valid)/799*100,1), "%)\n")
cat("1992 pres: have", nrow(panel_92_pres_valid), "vs paper 729 (", round(nrow(panel_92_pres_valid)/729*100,1), "%)\n")
