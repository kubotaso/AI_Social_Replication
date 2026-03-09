# Check if original source variables for VCF0301 are in the CDF
library(anesr)
data(timeseries_cum)
df <- timeseries_cum

# Source vars for VCF0301 from codebook:
# 1952: V520237
# 1960: V600091
# 1996: V960420

# Check if these exist in the CDF
cat("V520237 in CDF:", "V520237" %in% names(df), "\n")
cat("V600091 in CDF:", "V600091" %in% names(df), "\n")
cat("V960420 in CDF:", "V960420" %in% names(df), "\n")

# They won't be in the CDF - source vars are from original study files
# But let me check what individual year datasets are available
available_datasets <- data(package="anesr")$results[,"Item"]
cat("\nAvailable datasets:\n")
print(sort(available_datasets))

# Check if timeseries_1952, timeseries_1960 exist
cat("\n1952 available:", "timeseries_1952" %in% available_datasets, "\n")
cat("1960 available:", "timeseries_1960" %in% available_datasets, "\n")
