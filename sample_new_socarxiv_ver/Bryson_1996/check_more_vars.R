library(gssr)
data(gss_all)
gss93 <- gss_all[gss_all$year == 1993, ]

# Check for RACHALF and other rac* vars
rac_vars <- grep("^rac", names(gss93), value=TRUE, ignore.case=TRUE)
cat("All rac* vars:\n")
for (v in rac_vars) {
  n <- sum(!is.na(gss93[[v]]))
  if (n > 0) cat(v, "valid:", n, "\n")
}

# Check RACHALF specifically
if ("rachalf" %in% names(gss93)) {
  cat("\nrachalf:\n")
  print(table(gss93$rachalf, useNA="ifany"))
}

# Check region variable more carefully - is this the 4-region or 9-region?
# Standard GSS has 9 regions. Let me check srcbelt or other location vars
cat("\nsrcbelt (src of sample):\n")
if ("srcbelt" %in% names(gss93)) print(table(gss93$srcbelt, useNA="ifany"))

# South in 9-region GSS = regions 5,6,7 (South Atlantic, East South Central, West South Central)
# But with 4 regions: 1=NE, 2=MW, 3=S, 4=W
# 559 in region 3 is about 34.8% which is roughly right for South

# Check denom for Conservative Protestant classification
cat("\ndenom values:\n")
print(table(gss93$denom, useNA="ifany"))

# Check fund (fundamentalism)
cat("\nfund values:\n") 
print(table(gss93$fund, useNA="ifany"))

# For conservative Protestant: denom variable codes
# GSS DENOM: 1=Baptist, 2=Methodist, 3=Lutheran, 4=Presbyterian, 5=Episcopal
# Also need OTHER denomination codes
# Check other variable
if ("other" %in% names(gss93)) {
  cat("\nother (denomination detail):\n")
  oth <- table(gss93$other, useNA="ifany")
  cat("Unique values:", length(oth), "\n")
}

# Check fund variable: 1=fundamentalist, 2=moderate, 3=liberal
# Conservative Protestant might be those classified as fundamentalist by FUND

# Check ATTEND for control
cat("\nattend values:\n")
print(table(gss93$attend, useNA="ifany"))

# Check for variables related to the political tolerance scale
# Paper says: 15 dichotomous questions, 5 groups x 3 questions
# Scale range 0-15, mean 5.24, sd 4.72, alpha .9163
# Need to verify this

# Calculate tolerance scale
tol_vars <- c("spkath", "colath", "libath",
              "spkrac", "colrac", "librac",
              "spkcom", "colcom", "libcom",
              "spkmil", "colmil", "libmil",
              "spkhomo", "colhomo", "libhomo")

# Code intolerant responses
# spk*: 1=allowed, 2=not allowed -> intolerant if 2
# col*: 4=should fire (yes), 5=should not fire (no) -> intolerant if 4
# lib*: 1=remove, 2=not remove -> intolerant if 1

tol_df <- data.frame(id = 1:nrow(gss93))
tol_df$spkath_intol <- ifelse(gss93$spkath == 2, 1, ifelse(gss93$spkath == 1, 0, NA))
tol_df$colath_intol <- ifelse(gss93$colath == 4, 1, ifelse(gss93$colath == 5, 0, NA))
tol_df$libath_intol <- ifelse(gss93$libath == 1, 1, ifelse(gss93$libath == 2, 0, NA))
tol_df$spkrac_intol <- ifelse(gss93$spkrac == 2, 1, ifelse(gss93$spkrac == 1, 0, NA))
tol_df$colrac_intol <- ifelse(gss93$colrac == 4, 1, ifelse(gss93$colrac == 5, 0, NA))
tol_df$librac_intol <- ifelse(gss93$librac == 1, 1, ifelse(gss93$librac == 2, 0, NA))
tol_df$spkcom_intol <- ifelse(gss93$spkcom == 2, 1, ifelse(gss93$spkcom == 1, 0, NA))
tol_df$colcom_intol <- ifelse(gss93$colcom == 4, 1, ifelse(gss93$colcom == 5, 0, NA))
tol_df$libcom_intol <- ifelse(gss93$libcom == 1, 1, ifelse(gss93$libcom == 2, 0, NA))
tol_df$spkmil_intol <- ifelse(gss93$spkmil == 2, 1, ifelse(gss93$spkmil == 1, 0, NA))
tol_df$colmil_intol <- ifelse(gss93$colmil == 4, 1, ifelse(gss93$colmil == 5, 0, NA))
tol_df$libmil_intol <- ifelse(gss93$libmil == 1, 1, ifelse(gss93$libmil == 2, 0, NA))
tol_df$spkhomo_intol <- ifelse(gss93$spkhomo == 2, 1, ifelse(gss93$spkhomo == 1, 0, NA))
tol_df$colhomo_intol <- ifelse(gss93$colhomo == 4, 1, ifelse(gss93$colhomo == 5, 0, NA))
tol_df$libhomo_intol <- ifelse(gss93$libhomo == 1, 1, ifelse(gss93$libhomo == 2, 0, NA))

intol_cols <- grep("_intol$", names(tol_df), value=TRUE)
tol_df$intol_sum <- rowSums(tol_df[, intol_cols], na.rm=FALSE)
tol_valid <- tol_df$intol_sum[!is.na(tol_df$intol_sum)]
cat("\nPolitical intolerance scale (complete cases):\n")
cat("N:", length(tol_valid), "\n")
cat("Mean:", mean(tol_valid), "\n")
cat("SD:", sd(tol_valid), "\n")
cat("Range:", range(tol_valid), "\n")

# Paper says: mean 5.24, sd 4.72, N is about 2/3 of sample
# Our N should be around 1000-1100
