# Check for respondent IDs to link panel waves
library(anesr)

# 1974 study
data(timeseries_1974)
t74 <- timeseries_1974
cat("=== 1974 first 5 vars ===\n")
cat(head(names(t74), 20), sep="\n")

# Check V742001 (likely case ID)
cat("\nV742001 range:", range(t74$V742001, na.rm=TRUE), "\n")
cat("V742001 unique:", length(unique(t74$V742001)), "of", nrow(t74), "\n")

# 1976 study
data(timeseries_1976)
t76 <- timeseries_1976
cat("\n=== 1976 first 5 vars ===\n")
cat(head(names(t76), 20), sep="\n")

# Check V763001 (likely case ID)
cat("\nV763001 range:", range(t76$V763001, na.rm=TRUE), "\n")
cat("V763001 unique:", length(unique(t76$V763001)), "of", nrow(t76), "\n")

# Check if IDs overlap between 1974 and 1976
common_ids <- intersect(t74$V742001, t76$V763001)
cat("\nCommon IDs between 1974 and 1976:", length(common_ids), "\n")

# Maybe the IDs are in different format. Check V763002 or other vars
cat("V763002 range:", range(t76$V763002, na.rm=TRUE), "\n")
cat("V742002 range:", range(t74$V742002, na.rm=TRUE), "\n")

# Check VDSETNO
cat("\n1974 VDSETNO:", head(unique(t74$VDSETNO)), "\n")
cat("1976 VDSETNO:", head(unique(t76$VDSETNO)), "\n")

# In the CDF, check if there's a unique respondent ID
data(timeseries_cum)
cdf <- timeseries_cum
# Check VCF0006 (case ID)
cat("\n=== CDF Case IDs ===\n")
# For 1974 panel respondents
cdf_74 <- cdf[cdf$VCF0004 == 1974 & cdf$VCF0016 == 1, ]
cdf_76 <- cdf[cdf$VCF0004 == 1976 & cdf$VCF0016 == 1, ]
cat("1974 panel N:", nrow(cdf_74), "\n")
cat("1976 panel N:", nrow(cdf_76), "\n")
cat("1974 VCF0006 sample:", head(cdf_74$VCF0006), "\n")
cat("1976 VCF0006 sample:", head(cdf_76$VCF0006), "\n")

# Check if VCF0006 overlaps between 1974 and 1976 panel members
common_cdf <- intersect(cdf_74$VCF0006, cdf_76$VCF0006)
cat("Common VCF0006 between 1974 and 1976 panel:", length(common_cdf), "\n")

# Also check VCF0006a
cat("1974 VCF0006a sample:", head(cdf_74$VCF0006a), "\n")
cat("1976 VCF0006a sample:", head(cdf_76$VCF0006a), "\n")
common_6a <- intersect(cdf_74$VCF0006a, cdf_76$VCF0006a)
cat("Common VCF0006a:", length(common_6a), "\n")
