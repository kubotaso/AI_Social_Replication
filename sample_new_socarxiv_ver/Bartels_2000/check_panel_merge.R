# Check if panel datasets have merged data from prior waves
library(anesr)

# 1992 dataset - does it include 1990 party ID?
data(timeseries_1992)
t92 <- timeseries_1992

cat("=== 1992 dataset ===\n")
cat("Total vars:", ncol(t92), "\n")

# Check for 1990 party ID variable V900320
cat("V900320 (1990 party ID) in 1992 dataset:", "V900320" %in% names(t92), "\n")
if ("V900320" %in% names(t92)) {
  cat("V900320 distribution:\n")
  print(table(t92$V900320, useNA="always"))
}

# Check for 1992 party ID variable V923634
cat("V923634 (1992 party ID) in 1992 dataset:", "V923634" %in% names(t92), "\n")
if ("V923634" %in% names(t92)) {
  cat("V923634 distribution:\n")
  print(table(t92$V923634, useNA="always"))
}

# Check for 1992 pres vote V925609
cat("V925609 (1992 pres vote) in 1992 dataset:", "V925609" %in% names(t92), "\n")
if ("V925609" %in% names(t92)) {
  cat("V925609 distribution:\n")
  print(table(t92$V925609, useNA="always"))
}

# Check for 1992 House vote
house_vars_92 <- grep("V925", names(t92), value=TRUE)
cat("\nVars with V925 prefix:", head(house_vars_92, 20), "\n")

# 1976 dataset - does it include 1974 party ID?
data(timeseries_1976)
t76 <- timeseries_1976

cat("\n=== 1976 dataset ===\n")
cat("Total vars:", ncol(t76), "\n")

# Check for 1974 party ID (V742204)
cat("V742204 (1974 party ID) in 1976 dataset:", "V742204" %in% names(t76), "\n")

# Check for any V74xxxx variables in 1976 dataset
v74_vars <- grep("^V74", names(t76), value=TRUE)
cat("1974 vars in 1976 dataset:", length(v74_vars), "\n")
cat("First 20:", head(v74_vars, 20), "\n")

# Check 1976 party ID V763174
cat("\nV763174 (1976 party ID):\n")
print(table(t76$V763174, useNA="always"))

# Check 1976 pres vote V763665
cat("V763665 (1976 pres vote):", "V763665" %in% names(t76), "\n")
if ("V763665" %in% names(t76)) {
  print(table(t76$V763665, useNA="always"))
}

# Check for 1976 House vote
cat("\nV763655:", "V763655" %in% names(t76), "\n")
house_vars_76 <- grep("V7636[0-9]{2}", names(t76), value=TRUE)
cat("V7636xx vars:", house_vars_76, "\n")
