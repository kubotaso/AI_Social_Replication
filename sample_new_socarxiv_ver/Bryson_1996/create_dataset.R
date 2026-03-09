library(gssr)
data(gss_all)
gss93 <- gss_all[gss_all$year == 1993, ]

# Extract all needed variables
vars_to_keep <- c(
  # Music genres (18)
  "latin", "jazz", "blues", "musicals", "oldies", "classicl",
  "reggae", "bigband", "newage", "opera", "blugrass", "folk",
  "moodeasy", "conrock", "rap", "hvymetal", "country", "gospel",
  # Demographics
  "educ", "age", "sex", "race", "ethnic",
  # Income
  "realinc", "hompop", "income91",
  # Occupation
  "prestg80",
  # Region
  "region",
  # Religion
  "relig", "denom", "fund", "attend",
  # Political tolerance items
  "spkath", "colath", "libath",
  "spkrac", "colrac", "librac",
  "spkcom", "colcom", "libcom",
  "spkmil", "colmil", "libmil",
  "spkhomo", "colhomo", "libhomo",
  # Racism items
  "racmost", "busing", "racdif1", "racdif2", "racdif3", "racdif4",
  "racfew", "rachaf", "racmar", "racopen", "racseg",
  # Other potentially useful
  "polviews", "partyid"
)

# Convert haven labelled to numeric
out <- data.frame(id = 1:nrow(gss93))
for (v in vars_to_keep) {
  if (v %in% names(gss93)) {
    vals <- as.numeric(gss93[[v]])
    out[[v]] <- vals
  }
}

write.csv(out, "gss1993_clean.csv", row.names = FALSE)
cat("Saved gss1993_clean.csv with", nrow(out), "rows and", ncol(out), "columns\n")

# Verify key distributions
cat("\nMusic genre means (1-5 scale):\n")
music_vars <- c("latin", "jazz", "blues", "musicals", "oldies", "classicl",
                "reggae", "bigband", "newage", "opera", "blugrass", "folk",
                "moodeasy", "conrock", "rap", "hvymetal", "country", "gospel")
for (v in music_vars) {
  m <- round(mean(out[[v]], na.rm=TRUE), 2)
  cat(v, ":", m, "\n")
}

cat("\nTable 3 check - Latin/Salsa distribution:\n")
print(table(out$latin, useNA="ifany"))
