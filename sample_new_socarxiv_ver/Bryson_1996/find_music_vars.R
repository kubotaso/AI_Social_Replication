library(gssr)
data(gss_all)
gss93 <- gss_all[gss_all$year == 1993, ]

# Search for variables that might be music genre ratings
# Look for variables with values 1-5 that could be Likert scales about music
# The GSS variable names for music genres are likely: bigband, blugrass, country, etc.
possible_music <- c("bigband", "blugrass", "country", "folk", "gospel", "jazz", "latin",
                     "blues", "musicals", "newage", "opera", "pop", "rap", "reggae",
                     "classic", "classicl", "heavy", "hvymetal", "metal",
                     "easylist", "easylstn", "oldies", "showtune", "rnb",
                     "classical", "chamber", "swing", "rock", "contemp",
                     "salsa", "hiphop", "blugrss", "bluesrnb")

cat("Checking possible music variable names:\n")
for (v in possible_music) {
  if (v %in% names(gss93)) {
    vals <- table(gss93[[v]], useNA = "ifany")
    cat(v, "- FOUND:", paste(names(vals), vals, sep=":", collapse=", "), "\n")
  }
}

# Also search broadly
all_names <- names(gss93)
# Look for anything with non-NA values in 1993 that has 1-5 type distribution
# Let's grep for likely patterns
music_like <- grep("(jazz|blues|rap|gospel|folk|opera|country|metal|reggae|latin|swing|band|class|easy|old|show|blue|new.?age|pop|rock|salsa|chamber)", 
                    all_names, value = TRUE, ignore.case = TRUE)
cat("\nGrep matches:\n")
for (v in music_like) {
  n_valid <- sum(!is.na(gss93[[v]]))
  if (n_valid > 100) {
    cat(v, "- valid:", n_valid, "\n")
  }
}
