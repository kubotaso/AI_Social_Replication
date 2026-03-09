# Extract codebook for key variables
library(anesr)
data(timeseries_cum_doc)
doc <- timeseries_cum_doc

# Key variables (lowercase)
key_vars <- c("vcf0004", "vcf0301", "vcf0303", "vcf0704", "vcf0704a",
              "vcf0706", "vcf0702", "vcf0105a", "vcf0105b", "vcf0113",
              "vcf0905", "vcf0012", "vcf0017", "vcf0018a", "vcf0018b",
              "vcf0900a")

sink("anes_codebook.txt")
for (v in key_vars) {
  idx <- which(doc$id == v)
  if (length(idx) == 0) { cat("NOT FOUND:", v, "\n\n"); next }
  cat("================================================================\n")
  cat("Variable:", toupper(v), "\n")
  cat("Description:", doc$description[idx], "\n\n")
  if (!is.na(doc$text[idx]) && nchar(doc$text[idx]) > 0) {
    cat("Text:\n", substr(doc$text[idx], 1, 3000), "\n\n")
  }
  if (!is.na(doc$marginals[idx]) && nchar(doc$marginals[idx]) > 0) {
    cat("Marginals:\n", substr(doc$marginals[idx], 1, 2000), "\n\n")
  }
}
sink()
cat("Codebook saved to anes_codebook.txt\n")
