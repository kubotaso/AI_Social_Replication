# Create final panel datasets for Tables 4 and 5
# Best available approach using CDF linkage
library(anesr)
data(timeseries_cum)
cdf <- timeseries_cum

# === 1958-1960 Panel ===
# VCF0006a linkage between 1958 and 1960
cdf_58 <- cdf[cdf$VCF0004 == 1958, c("VCF0006a", "VCF0301")]
cdf_60 <- cdf[cdf$VCF0004 == 1960, ]
names(cdf_58)[names(cdf_58) == "VCF0301"] <- "VCF0301_lagged"
panel_60 <- merge(cdf_60, cdf_58[, c("VCF0006a", "VCF0301_lagged")], by="VCF0006a", all.x=FALSE)
cat("1960 panel: N =", nrow(panel_60), "(paper uses original panel study with more respondents)\n")
write.csv(panel_60, "panel_1960.csv", row.names=FALSE)

# === 1974-1976 Panel ===
# Use ALL 1974 respondents (not just VCF0016==1) via VCF0006a
cdf_74 <- cdf[cdf$VCF0004 == 1974, c("VCF0006a", "VCF0301")]
names(cdf_74)[names(cdf_74) == "VCF0301"] <- "VCF0301_lagged"
cdf_76 <- cdf[cdf$VCF0004 == 1976, ]
panel_76 <- merge(cdf_76, cdf_74[, c("VCF0006a", "VCF0301_lagged")], by="VCF0006a")
cat("1976 panel: N =", nrow(panel_76), "(paper uses original panel study with more respondents)\n")
write.csv(panel_76, "panel_1976.csv", row.names=FALSE)

# === 1990-1992 Panel ===
# 1992 individual dataset already has 1990 party ID (V900320)
data(timeseries_1992)
t92 <- timeseries_1992
panel_92 <- t92[t92$V900320 %in% 0:6, ]
panel_92$pid_current <- ifelse(panel_92$V923634 %in% 0:6, panel_92$V923634 + 1, NA)
panel_92$pid_lagged <- panel_92$V900320 + 1
panel_92$vote_pres <- ifelse(panel_92$V925609 %in% c(1,2), panel_92$V925609, NA)
panel_92$vote_house <- ifelse(panel_92$V925701 == 1, 1, ifelse(panel_92$V925701 == 5, 2, NA))

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
cat("1992 panel: N =", nrow(panel_92), "\n")

cat("\nDone.\n")
