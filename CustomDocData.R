# GSS sample code for R
# Downloads GSS 1993 data using gssr package and extracts selected variables

# # Install 'gssr' from 'ropensci' universe
# install.packages('gssr', repos = c('https://kjhealy.r-universe.dev', 'https://cloud.r-project.org'))

# # Also recommended: install 'gssrdoc' as well
# install.packages('gssrdoc', repos = c('https://kjhealy.r-universe.dev', 'https://cloud.r-project.org'))

rm(list = ls(all = TRUE))
library(gssr)
library(gssrdoc)
library(tidyverse)

#### make documentation of selected variables only ###################

vars <- c(
        # Individual Variables (converted to lowercase)
        "year","id","hompop","educ", "realinc", "prestg80", "sei", "sex", "age", "race", "ethnic", "relig", "denom","jew",
        "relig16", "denom16","jew16","region", "polviews", "ballot",
        "incom16","income","rincome",
        # Political Intolerance Variables
        "spkath","colath","libath","spkrac","colrac","librac",
        "spkcom","colcom","libcom","spkmil","colmil","libmil",
        "spkhomo","colhomo","libhomo",
        # Racial Attitudes Variables
        "rachaf", "busing", "racdif1", "racdif3", "racdif4",
        # Music/Culture Preference Variables (actual names from GSS 1993)
        "bigband", "blugrass", "country", "blues", "musicals", "classicl", "folk", "gospel", "jazz", "latin", "moodeasy", "newage", "opera", "rap", "reggae", "conrock", "oldies", "hvymetal"
    )

gss_select_doc <- gss_doc %>%
    filter(variable %in% vars) %>%
    print(n = Inf)

# Write human-readable text file
sink("gss_selected_variables_doc.txt")
for (i in 1:nrow(gss_select_doc)) {
    cat("================================================================================\n")
    cat("Variable:", toupper(gss_select_doc$variable[i]), "\n")
    cat("--------------------------------------------------------------------------------\n")
    cat("Description:", gss_select_doc$description[i], "\n\n")
    cat("Question:", gss_select_doc$question[i], "\n\n")
    cat("Value Labels:\n", gss_select_doc$value_labels[i], "\n\n")
}
sink()


#### Download GSS data and extract 1993 with selected variables ###################

# Load GSS 1993 data directly (faster than loading all years)
gss93_full <- gss_get_yr(1993)

# Select available variables
gss93 <- gss93_full %>%
    select(any_of(vars))

# Save the selected data
write_csv(gss93, "gss93_selected.csv")
cat("\nSaved gss93_selected.csv with", nrow(gss93), "rows and", ncol(gss93), "columns\n")
