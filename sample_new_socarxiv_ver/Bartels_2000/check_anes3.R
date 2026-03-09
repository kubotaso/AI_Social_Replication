# Check congressional vote and incumbency variables
library(anesr)
data(timeseries_cum)
data(timeseries_cum_doc)
df <- timeseries_cum
doc <- timeseries_cum_doc

# Find House vote variables
house_vars <- grep("vcf070[7-9]|vcf071", doc$id, value=TRUE)
cat("House vote variables:\n")
print(house_vars)

# Check VCF0707 (House vote)
for (v in c("vcf0707", "vcf0708", "vcf0709", "vcf0710")) {
  idx <- which(doc$id == v)
  if (length(idx) > 0) {
    cat("\n===", toupper(v), "===\n")
    cat("Description:", doc$description[idx], "\n")
    cat("Text:", substr(doc$text[idx], 1, 500), "\n")
  }
}

# Check VCF0900 (type of race)
for (v in c("vcf0900", "vcf0901a", "vcf0901b", "vcf0902", "vcf0904")) {
  idx <- which(doc$id == v)
  if (length(idx) > 0) {
    cat("\n===", toupper(v), "===\n")
    cat("Description:", doc$description[idx], "\n")
    cat("Text:", substr(doc$text[idx], 1, 500), "\n")
  }
}

# Check VCF0707 data
if ("VCF0707" %in% names(df)) {
  cat("\n\nVCF0707 by year:\n")
  print(table(df$VCF0707, df$VCF0004, useNA="ifany"))
}

# Check VCF0904 data
if ("VCF0904" %in% names(df)) {
  cat("\n\nVCF0904 by year:\n")
  print(table(df$VCF0904, df$VCF0004, useNA="ifany"))
}

# Check VCF0902 data (type of house race)
if ("VCF0902" %in% names(df)) {
  cat("\n\nVCF0902 (type of House race):\n")
  print(table(df$VCF0902, useNA="always"))
}

# Check panel study variables more carefully
# VCF0015a and VCF0015b seem to be about panel/cross-section
for (v in c("vcf0015a", "vcf0015b")) {
  idx <- which(doc$id == v)
  if (length(idx) > 0) {
    cat("\n===", toupper(v), "===\n")
    cat("Description:", doc$description[idx], "\n")
    cat("Text:", substr(doc$text[idx], 1, 800), "\n")
  }
}
