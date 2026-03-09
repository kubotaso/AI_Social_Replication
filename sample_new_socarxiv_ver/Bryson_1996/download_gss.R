# Download GSS 1993 data using gssr package
if (!require("gssr", quietly = TRUE)) {
  install.packages("gssr", repos = c("https://kjhealy.r-universe.dev", "https://cloud.r-project.org"))
}
library(gssr)

# Get the cumulative GSS data and filter to 1993
data(gss_all)
gss93 <- gss_all[gss_all$year == 1993, ]
cat("Rows in 1993 GSS:", nrow(gss93), "\n")
cat("Columns:", ncol(gss93), "\n")

# Save full 1993 data
write.csv(gss93, "gss1993.csv", row.names = FALSE)
cat("Saved gss1993.csv\n")

# Check for music variables
music_vars <- grep("^music", names(gss93), value = TRUE, ignore.case = TRUE)
cat("Music variables found:", paste(music_vars, collapse = ", "), "\n")

# Check for key variables
key_vars <- c("educ", "income", "rincome", "prestg80", "sex", "age", "race", "hispanic", "region", "relig", "polviews")
for (v in key_vars) {
  if (v %in% names(gss93)) {
    cat(v, "- present, non-NA:", sum(!is.na(gss93[[v]])), "\n")
  } else {
    cat(v, "- NOT FOUND\n")
  }
}
