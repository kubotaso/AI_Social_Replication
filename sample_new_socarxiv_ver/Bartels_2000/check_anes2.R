# Get documentation and check panel/incumbency variables
library(anesr)
data(timeseries_cum)
data(timeseries_cum_doc)
df <- timeseries_cum

# Get documentation for key variables
key_vars <- c("VCF0004", "VCF0301", "VCF0303", "VCF0704", "VCF0704a",
              "VCF0706", "VCF0702", "VCF0105a", "VCF0105b", "VCF0113",
              "VCF0905", "VCF0012", "VCF0013", "VCF0014",
              "VCF0015a", "VCF0015b", "VCF0016", "VCF0017",
              "VCF0018a", "VCF0018b", "VCF0019",
              "VCF0900a", "VCF0900b", "VCF0900c")

doc <- timeseries_cum_doc
cat("Documentation columns:\n")
print(names(doc))
cat("\nDocumentation rows:", nrow(doc), "\n")

# Get doc for key variables
for (v in key_vars) {
  idx <- which(doc$vcf_name == v)
  if (length(idx) > 0) {
    cat("\n===", v, "===\n")
    cat("Label:", doc$label[idx], "\n")
    if ("notes" %in% names(doc) && !is.na(doc$notes[idx]))
      cat("Notes:", substr(doc$notes[idx], 1, 200), "\n")
  }
}

# Check study type / panel indicators
cat("\n\n=== PANEL INDICATORS ===\n")
cat("VCF0012 (study):\n")
print(table(df$VCF0012, df$VCF0004, useNA="ifany")[,c("1956","1958","1960","1972","1974","1976","1990","1992")])

cat("\nVCF0013:\n")
print(table(df$VCF0013, useNA="always"))

cat("\nVCF0014:\n")
print(table(df$VCF0014, useNA="always"))

cat("\nVCF0016:\n")
print(table(df$VCF0016, useNA="always"))

cat("\nVCF0017:\n")
print(table(df$VCF0017, useNA="always"))

# Check VCF0900a for incumbency coding
cat("\n=== INCUMBENCY ===\n")
cat("VCF0900a:\n")
if ("VCF0900a" %in% names(df)) print(table(df$VCF0900a, useNA="always"))
cat("VCF0900b:\n")
if ("VCF0900b" %in% names(df)) print(table(df$VCF0900b, useNA="always"))
cat("VCF0900c:\n")
if ("VCF0900c" %in% names(df)) print(table(df$VCF0900c, useNA="always"))

# Check VCF0905 by year
cat("\nVCF0905 by year:\n")
print(table(df$VCF0905, df$VCF0004, useNA="ifany"))
