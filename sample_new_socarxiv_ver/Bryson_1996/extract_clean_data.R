library(gssr)
data(gss_all)
gss93 <- gss_all[gss_all$year == 1993, ]

# Music genre variables
music_vars <- c("latin", "jazz", "blues", "musicals", "oldies", "classicl",
                "reggae", "bigband", "newage", "opera", "blugrass", "folk",
                "moodeasy", "conrock", "rap", "hvymetal", "country", "gospel")
music_labels <- c("Latin_Salsa", "Jazz", "Blues_RnB", "Show_Tunes", "Oldies", 
                   "Classical_Chamber", "Reggae", "Swing_BigBand", "NewAge_Space",
                   "Opera", "Bluegrass", "Folk", "Easy_Listening", "Pop_ContemRock",
                   "Rap", "Heavy_Metal", "Country_Western", "Gospel")

# Political tolerance: spk/col/lib for ath, rac, com, mil, homo
# spk: 1=allowed, 2=not allowed -> intolerant if 2
# col: 4=yes (should fire), 5=no (should not fire) -> intolerant if 4 (fire=intolerant)
# lib: 1=remove, 2=not remove -> intolerant if 1
tol_vars <- c("spkath", "colath", "libath",
              "spkrac", "colrac", "librac",
              "spkcom", "colcom", "libcom",
              "spkmil", "colmil", "libmil",
              "spkhomo", "colhomo", "libhomo")

# Racism scale items
# Paper says: racseg (RACSCHOL mapping?), busing, racdif1, racdif3, racdif4
# The 5 items from the paper:
# 1. Object to sending children to school >50% Black = RACSCHOL (not available) or RACSEG?
# Let me check what we have:
# racseg: segregation question
# busing: busing question
# racdif1: discrimination explanation
# racdif2: education chance  
# racdif3: lack of will power
# racdif4: inborn ability

# The paper describes 5 items. Let me map them:
# (1) "Would you have any objection to sending your children to a school where more than half of the children are Black?" 
#     This is likely RACSCHOL but it's not available. Could be part of RACSEG.
# Actually looking more carefully at GSS: 
# RACSEG = "Do you think white students and black students should go to the same schools?"
# That doesn't match. Let me check other vars.

# The paper's 5 racism items:
# 1. Object to children in school >50% Black = likely not available as separate var
# 2. Favor/oppose busing = BUSING  
# 3. Differences due to discrimination = RACDIF1
# 4. Don't have chance for education = RACDIF2
# 5. Don't have motivation/will power = RACDIF3

# Actually re-reading the paper more carefully:
# The questions are:
# (1) "Would you have any objection to sending your children to a school where more than half the children are Black?"
# (2) "In general, do you favor or oppose the busing of Black and White school children from one school district to another?"
# (3) "On average Blacks have worse jobs, income, and housing than White people. Do you think these differences are mainly due to discrimination?"
# (4) "...because most Blacks don't have the chance for education..."
# (5) "...because most Blacks just don't have the motivation or will power..."

# GSS vars: RACSCHOL was not asked in 1993 but might be different name
# Let me check what's available: racseg, busing, racdif1-4

# Actually RACSCHOL was not a standard variable name. The question about sending 
# children to school with >50% Black children might be captured differently.
# Let me check: do we have "schblks" or similar?
cat("Checking school-related racism vars:\n")
school_vars <- grep("(sch|school|blk|black)", names(gss93), value=TRUE, ignore.case=TRUE)
for (v in school_vars) {
  n <- sum(!is.na(gss93[[v]]))
  if (n > 0) cat(v, "valid:", n, "\n")
}

# Let me also check racopen (open housing)
cat("\nracopen values:\n")
print(table(gss93$racopen, useNA="ifany"))

cat("\nracseg values:\n") 
print(table(gss93$racseg, useNA="ifany"))

cat("\nbusing values:\n")
print(table(gss93$busing, useNA="ifany"))

cat("\nracdif1 values:\n")
print(table(gss93$racdif1, useNA="ifany"))

cat("\nracdif2 values:\n")
print(table(gss93$racdif2, useNA="ifany"))

cat("\nracdif3 values:\n")
print(table(gss93$racdif3, useNA="ifany"))

cat("\nracdif4 values:\n")
print(table(gss93$racdif4, useNA="ifany"))

cat("\nracchurh values:\n")
print(table(gss93$racchurh, useNA="ifany"))

cat("\nracmar values:\n")
print(table(gss93$racmar, useNA="ifany"))

cat("\nracfew values:\n")
print(table(gss93$racfew, useNA="ifany"))

cat("\nracmost values:\n")
print(table(gss93$racmost, useNA="ifany"))

cat("\nrachaf values:\n")
print(table(gss93$rachaf, useNA="ifany"))

# Check income91 variable for household income
cat("\nincome91 values:\n")
print(table(gss93$income91, useNA="ifany"))

cat("\nrealinc (family income in constant dollars):\n")
summary(gss93$realinc)

cat("\nhompop (persons in household):\n")
print(table(gss93$hompop, useNA="ifany"))

# Check ethnic variable for Hispanic identification
cat("\nethnic values:\n")
print(table(gss93$ethnic, useNA="ifany"))

# Region codes
cat("\nregion values:\n")
print(table(gss93$region, useNA="ifany"))
# GSS regions: 1=New England, 2=Middle Atlantic, 3=East North Central, 4=West North Central,
# 5=South Atlantic, 6=East South Central, 7=West South Central, 8=Mountain, 9=Pacific
# Wait, we only see 1-4. Let me check more carefully.
cat("\nActual region range:\n")
cat(range(gss93$region, na.rm=TRUE), "\n")

# Check if region might be recoded
cat("region unique values:", sort(unique(gss93$region)), "\n")
