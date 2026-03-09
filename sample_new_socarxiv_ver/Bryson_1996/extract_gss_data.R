library(gssr)
data(gss_all)
gss93 <- gss_all[gss_all$year == 1993, ]

# Check political tolerance / civil liberties variables
# Stouffer-type questions about 5 groups: atheist, racist, communist, militarist, homosexual
tol_vars <- c("spkath", "colath", "libath",    # anti-religious
              "spkrac", "colrac", "librac",    # racist
              "spkcom", "colcom", "libcom",    # communist
              "spkmil", "colmil", "libmil",    # militarist
              "spkhomo", "colhomo", "libhomo") # homosexual
cat("Political tolerance vars:\n")
for (v in tol_vars) {
  if (v %in% names(gss93)) {
    n_valid <- sum(!is.na(gss93[[v]]))
    cat(v, "- valid:", n_valid, ", values:", paste(sort(unique(gss93[[v]][!is.na(gss93[[v]])])), collapse=","), "\n")
  } else {
    cat(v, "- NOT FOUND\n")
  }
}

# Check racism variables
racism_vars <- c("racseg", "racpush", "racopen", "racschol", "racdin", 
                  "busing", "racmar", "rachaf", "racdif1", "racdif2", 
                  "racdif3", "racdif4", "racfew", "racmost", "raclive",
                  "workblks", "wrkwayup", "closeblk", "affrmact",
                  "racavoid", "schblks", "racchurh")
cat("\nRacism vars:\n")
for (v in racism_vars) {
  if (v %in% names(gss93)) {
    n_valid <- sum(!is.na(gss93[[v]]))
    cat(v, "- valid:", n_valid, "\n")
  }
}

# Check income and household vars
income_vars <- c("income", "income91", "rincome", "realinc", "realrinc", 
                  "coninc", "conrinc", "hompop", "family16", "incom16",
                  "income98", "income86", "income72")
cat("\nIncome vars:\n")
for (v in income_vars) {
  if (v %in% names(gss93)) {
    n_valid <- sum(!is.na(gss93[[v]]))
    if (n_valid > 0) cat(v, "- valid:", n_valid, "\n")
  }
}

# Check Hispanic/ethnicity
eth_vars <- c("hispanic", "ethnic", "race", "racecen1", "racecen2", "racecen3", "hisp")
cat("\nEthnicity vars:\n")
for (v in eth_vars) {
  if (v %in% names(gss93)) {
    n_valid <- sum(!is.na(gss93[[v]]))
    vals <- table(gss93[[v]], useNA = "ifany")
    cat(v, "- valid:", n_valid, ", vals:", paste(names(vals), vals, sep=":", collapse=", "), "\n")
  }
}

# Check religion
relig_vars <- c("relig", "denom", "fund", "attend")
cat("\nReligion vars:\n")
for (v in relig_vars) {
  if (v %in% names(gss93)) {
    n_valid <- sum(!is.na(gss93[[v]]))
    vals <- table(gss93[[v]], useNA = "ifany")
    cat(v, "- valid:", n_valid, ", vals:", paste(names(vals), vals, sep=":", collapse=", "), "\n")
  }
}

# Check region
cat("\nRegion:\n")
cat(paste(names(table(gss93$region)), table(gss93$region), sep=":", collapse=", "), "\n")

# Check sex
cat("\nSex:\n")
cat(paste(names(table(gss93$sex)), table(gss93$sex), sep=":", collapse=", "), "\n")

# Stouffer tolerance - also check allowed/not allowed coding
cat("\nspkath values:\n")
print(table(gss93$spkath, useNA="ifany"))

cat("\ncolath values:\n")
print(table(gss93$colath, useNA="ifany"))

cat("\nlibath values:\n")
print(table(gss93$libath, useNA="ifany"))
