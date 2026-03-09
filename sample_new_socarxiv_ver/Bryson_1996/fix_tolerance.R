library(gssr)
data(gss_all)
gss93 <- gss_all[gss_all$year == 1993, ]

# Re-check coding of col* variables
# COLATH: "Should such a person be allowed to teach in a college or university?"
# 4 = YES (allowed to teach) -> TOLERANT
# 5 = NO (not allowed to teach) -> INTOLERANT
# So intolerant = 5, not 4!

tol_df <- data.frame(id = 1:nrow(gss93))
# spk*: 1=allowed, 2=not allowed -> intolerant if 2
tol_df$spkath <- ifelse(gss93$spkath == 2, 1, ifelse(gss93$spkath == 1, 0, NA))
tol_df$spkrac <- ifelse(gss93$spkrac == 2, 1, ifelse(gss93$spkrac == 1, 0, NA))
tol_df$spkcom <- ifelse(gss93$spkcom == 2, 1, ifelse(gss93$spkcom == 1, 0, NA))
tol_df$spkmil <- ifelse(gss93$spkmil == 2, 1, ifelse(gss93$spkmil == 1, 0, NA))
tol_df$spkhomo <- ifelse(gss93$spkhomo == 2, 1, ifelse(gss93$spkhomo == 1, 0, NA))

# col*: 4=yes (allowed), 5=no (not allowed) -> intolerant if 5
tol_df$colath <- ifelse(gss93$colath == 5, 1, ifelse(gss93$colath == 4, 0, NA))
tol_df$colrac <- ifelse(gss93$colrac == 5, 1, ifelse(gss93$colrac == 4, 0, NA))
tol_df$colcom <- ifelse(gss93$colcom == 5, 1, ifelse(gss93$colcom == 4, 0, NA))
tol_df$colmil <- ifelse(gss93$colmil == 5, 1, ifelse(gss93$colmil == 4, 0, NA))
tol_df$colhomo <- ifelse(gss93$colhomo == 5, 1, ifelse(gss93$colhomo == 4, 0, NA))

# lib*: 1=remove, 2=not remove -> intolerant if 1
tol_df$libath <- ifelse(gss93$libath == 1, 1, ifelse(gss93$libath == 2, 0, NA))
tol_df$librac <- ifelse(gss93$librac == 1, 1, ifelse(gss93$librac == 2, 0, NA))
tol_df$libcom <- ifelse(gss93$libcom == 1, 1, ifelse(gss93$libcom == 2, 0, NA))
tol_df$libmil <- ifelse(gss93$libmil == 1, 1, ifelse(gss93$libmil == 2, 0, NA))
tol_df$libhomo <- ifelse(gss93$libhomo == 1, 1, ifelse(gss93$libhomo == 2, 0, NA))

item_cols <- names(tol_df)[-1]
tol_df$intol_sum <- rowSums(tol_df[, item_cols], na.rm=FALSE)
tol_valid <- tol_df$intol_sum[!is.na(tol_df$intol_sum)]
cat("Political intolerance scale (corrected):\n")
cat("N:", length(tol_valid), "\n")
cat("Mean:", round(mean(tol_valid), 2), "\n")
cat("SD:", round(sd(tol_valid), 2), "\n")
cat("Range:", range(tol_valid), "\n")
cat("Alpha:", round(psych::alpha(tol_df[complete.cases(tol_df[,item_cols]), item_cols])$total$raw_alpha, 4), "\n")

# Now check racism scale
# 5 items: RACMOST, BUSING, RACDIF1, RACDIF2, RACDIF3
# Racist direction coding:
# RACMOST: 1=object (racist), 2=not object (not racist)
# BUSING: 1=favor (not racist), 2=oppose (racist)
# RACDIF1: 1=yes discrimination (not racist), 2=no (racist)
# RACDIF2: 1=yes education lack (not racist), 2=no (racist)
# RACDIF3: 1=yes motivation lack (racist), 2=no (not racist)

rac_df <- data.frame(
  racmost = ifelse(gss93$racmost == 1, 1, ifelse(gss93$racmost == 2, 0, NA)),
  busing = ifelse(gss93$busing == 2, 1, ifelse(gss93$busing == 1, 0, NA)),
  racdif1 = ifelse(gss93$racdif1 == 2, 1, ifelse(gss93$racdif1 == 1, 0, NA)),
  racdif2 = ifelse(gss93$racdif2 == 2, 1, ifelse(gss93$racdif2 == 1, 0, NA)),
  racdif3 = ifelse(gss93$racdif3 == 1, 1, ifelse(gss93$racdif3 == 2, 0, NA))
)

rac_df$racism <- rowSums(rac_df, na.rm=FALSE)
rac_valid <- rac_df$racism[!is.na(rac_df$racism)]
cat("\nRacism scale (RACMOST, BUSING, RACDIF1, RACDIF2, RACDIF3):\n")
cat("N:", length(rac_valid), "\n")
cat("Mean:", round(mean(rac_valid), 2), "\n")
cat("SD:", round(sd(rac_valid), 2), "\n")
cat("Range:", range(rac_valid), "\n")
if (length(rac_valid) > 10) {
  cat("Alpha:", round(psych::alpha(rac_df[complete.cases(rac_df), ])$total$raw_alpha, 4), "\n")
}

# Paper says: mean 2.65, SD 1.56, alpha .54, range 0-5
