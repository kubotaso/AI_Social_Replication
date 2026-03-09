library(gssr)
library(psych)
data(gss_all)
gss93 <- gss_all[gss_all$year == 1993, ]

# Political intolerance (corrected coding)
tol_df <- data.frame(
  spkath = ifelse(gss93$spkath == 2, 1, ifelse(gss93$spkath == 1, 0, NA)),
  colath = ifelse(gss93$colath == 5, 1, ifelse(gss93$colath == 4, 0, NA)),
  libath = ifelse(gss93$libath == 1, 1, ifelse(gss93$libath == 2, 0, NA)),
  spkrac = ifelse(gss93$spkrac == 2, 1, ifelse(gss93$spkrac == 1, 0, NA)),
  colrac = ifelse(gss93$colrac == 5, 1, ifelse(gss93$colrac == 4, 0, NA)),
  librac = ifelse(gss93$librac == 1, 1, ifelse(gss93$librac == 2, 0, NA)),
  spkcom = ifelse(gss93$spkcom == 2, 1, ifelse(gss93$spkcom == 1, 0, NA)),
  colcom = ifelse(gss93$colcom == 5, 1, ifelse(gss93$colcom == 4, 0, NA)),
  libcom = ifelse(gss93$libcom == 1, 1, ifelse(gss93$libcom == 2, 0, NA)),
  spkmil = ifelse(gss93$spkmil == 2, 1, ifelse(gss93$spkmil == 1, 0, NA)),
  colmil = ifelse(gss93$colmil == 5, 1, ifelse(gss93$colmil == 4, 0, NA)),
  libmil = ifelse(gss93$libmil == 1, 1, ifelse(gss93$libmil == 2, 0, NA)),
  spkhomo = ifelse(gss93$spkhomo == 2, 1, ifelse(gss93$spkhomo == 1, 0, NA)),
  colhomo = ifelse(gss93$colhomo == 5, 1, ifelse(gss93$colhomo == 4, 0, NA)),
  libhomo = ifelse(gss93$libhomo == 1, 1, ifelse(gss93$libhomo == 2, 0, NA))
)
tol_cc <- tol_df[complete.cases(tol_df), ]
tol_sum <- rowSums(tol_cc)
cat("POLITICAL INTOLERANCE:\n")
cat("N:", nrow(tol_cc), "Mean:", round(mean(tol_sum), 2), "SD:", round(sd(tol_sum), 2), "\n")
cat("Alpha:", round(alpha(tol_cc)$total$raw_alpha, 4), "\n")
cat("Paper: N~1000, Mean=5.24, SD=4.72, Alpha=.9163\n\n")

# Racism scale: RACMOST, BUSING, RACDIF1, RACDIF2, RACDIF3
rac_df <- data.frame(
  racmost = ifelse(gss93$racmost == 1, 1, ifelse(gss93$racmost == 2, 0, NA)),
  busing = ifelse(gss93$busing == 2, 1, ifelse(gss93$busing == 1, 0, NA)),
  racdif1 = ifelse(gss93$racdif1 == 2, 1, ifelse(gss93$racdif1 == 1, 0, NA)),
  racdif2 = ifelse(gss93$racdif2 == 2, 1, ifelse(gss93$racdif2 == 1, 0, NA)),
  racdif3 = ifelse(gss93$racdif3 == 1, 1, ifelse(gss93$racdif3 == 2, 0, NA))
)
rac_cc <- rac_df[complete.cases(rac_df), ]
rac_sum <- rowSums(rac_cc)
cat("RACISM SCALE (5 items: RACMOST, BUSING, RACDIF1, RACDIF2, RACDIF3):\n")
cat("N:", nrow(rac_cc), "Mean:", round(mean(rac_sum), 2), "SD:", round(sd(rac_sum), 2), "\n")
cat("Alpha:", round(alpha(rac_cc)$total$raw_alpha, 4), "\n")
cat("Paper: mean=2.65, SD=1.56, Alpha=.54, Range=0-5\n\n")

# Try alternative racism items - maybe RACHAF instead of one of them
# Or maybe RACDIF4 instead of RACDIF2
# The paper says factor analysis removed one item from the original set
# The remaining 5 are described explicitly

# Check what happens with RACDIF4 (inborn ability) instead of RACDIF2
rac_df2 <- data.frame(
  racmost = ifelse(gss93$racmost == 1, 1, ifelse(gss93$racmost == 2, 0, NA)),
  busing = ifelse(gss93$busing == 2, 1, ifelse(gss93$busing == 1, 0, NA)),
  racdif1 = ifelse(gss93$racdif1 == 2, 1, ifelse(gss93$racdif1 == 1, 0, NA)),
  racdif3 = ifelse(gss93$racdif3 == 1, 1, ifelse(gss93$racdif3 == 2, 0, NA)),
  racdif4 = ifelse(gss93$racdif4 == 1, 1, ifelse(gss93$racdif4 == 2, 0, NA))
)
rac_cc2 <- rac_df2[complete.cases(rac_df2), ]
rac_sum2 <- rowSums(rac_cc2)
cat("ALT RACISM (RACMOST, BUSING, RACDIF1, RACDIF3, RACDIF4):\n")
cat("N:", nrow(rac_cc2), "Mean:", round(mean(rac_sum2), 2), "SD:", round(sd(rac_sum2), 2), "\n")
cat("Alpha:", round(alpha(rac_cc2)$total$raw_alpha, 4), "\n\n")

# Check with RACFEW instead of RACMOST (wording "a few" vs "most")
# Paper says "more than half" which is closest to RACMOST
# But let me try both to see which gives better match

# Musical exclusiveness scale stats
music_vars <- c("latin", "jazz", "blues", "musicals", "oldies", "classicl",
                "reggae", "bigband", "newage", "opera", "blugrass", "folk",
                "moodeasy", "conrock", "rap", "hvymetal", "country", "gospel")

music_df <- gss93[, music_vars]
# Count dislike (4) and dislike very much (5) responses
exclus <- rowSums(music_df >= 4, na.rm = FALSE)
cat("MUSICAL EXCLUSIVENESS (count 4+5 across 18 genres, listwise):\n")
cat("N valid:", sum(!is.na(exclus)), "\n")
cat("Mean:", round(mean(exclus, na.rm=TRUE), 2), "\n")
cat("SD:", round(sd(exclus, na.rm=TRUE), 2), "\n")
cat("Range:", range(exclus, na.rm=TRUE), "\n")
cat("Paper: N=912, mean=5.78, SD=3.76, range=0-18\n\n")

# Try with allowing some NA (treat "don't know" as not disliking)
# The paper says don't know treated as missing and respondents eliminated
# Let me check how many have all 18 valid
all_valid <- complete.cases(music_df)
cat("Respondents with all 18 music ratings (1-5):", sum(all_valid), "\n")

# But paper says "don't know responses are treated as missing and those respondents are eliminated"
# "leaving 912 valid cases"
# Let me check: how many have responses in 1-5 range for all 18?
music_15 <- music_df
for (v in music_vars) {
  music_15[[v]] <- ifelse(music_15[[v]] >= 1 & music_15[[v]] <= 5, music_15[[v]], NA)
}
all_valid2 <- complete.cases(music_15)
cat("Respondents with all 18 in 1-5 range:", sum(all_valid2), "\n")

exclus2 <- rowSums(music_15[all_valid2, ] >= 4)
cat("Exclusiveness (all valid, count 4+5):\n")
cat("N:", length(exclus2), "Mean:", round(mean(exclus2), 2), "SD:", round(sd(exclus2), 2), "\n")
