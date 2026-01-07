# GSS sample code for R

# # Install 'gssr' from 'ropensci' universe
# install.packages('gssr', repos = c('https://kjhealy.r-universe.dev', 'https://cloud.r-project.org'))

# # Also recommended: install 'gssrdoc' as well
# install.packages('gssrdoc', repos = c('https://kjhealy.r-universe.dev', 'https://cloud.r-project.org'))

rm(list = ls(all = TRUE))

install.packages(c(
  # === Table 2 の再現に直接必要なもの ===
  "tidyverse",   # dplyr, readr, tidyr, ggplot2, stringr など一式
  "broom",       # 回帰出力の整形
  "haven",       # GSSデータのSPSS/Stata版にも対応
  "janitor",     # データのクリーンアップ補助
  "car",         # 回帰診断など
  "psych",       # 標準化やスコア計算に便利
  "data.table",  # 高速処理
  "readxl",      # Excel読み込み
  "writexl",     # Excel出力
  "lmtest",      # 回帰テスト
  "sandwich",    # ロバスト分散推定
  "Hmisc",       # summaryツール
  "stargazer",   # 表のLaTeX/HTML出力
  "gtsummary",   # 表出力用 tidy table
  "gt",          # 綺麗な表出力
  "kableExtra",   # knitr表の整形
  "gssr",
  "gssrdoc"
))

library(gssr)
library(gssrdoc)
library(tidyverse)

#### make documentation of selected variables only ###################

write_csv(gss_doc, file = "gss_all_variables_doc.csv")

vars <- c(
        # Individual Variables (converted to lowercase)
        "year","id","hompop","educ", "realinc", "prestg80", "sei", "sex", "age", "race", "ethnic", "relig", "denom","jew",
        "relig16", "denom16","jew16","region",
        "incom16","income","rincome",
        # Political Intolerance Variables 
        "spkath","colath","libath","spkrac","colrac","librac",
        "spkcom","colcom","libcom","spkmil","colmil","libmil",
        "spkhomo","colhomo","libhomo",
        # Music/Culture Preference Variables (actual names from GSS 1993)
        "bigband", "blugrass", "country", "blues", "musicals", "classicl", "folk", "gospel", "jazz", "latin", "moodeasy", "newage", "opera", "rap", "reggae", "conrock", "oldies", "hvymetal"
    )

gss_select_doc <- gss_doc %>%
    filter(variable %in% vars) %>%
    print(n = Inf) %>%
    write_csv("gss_selected_variables_doc.csv")


#### make gss data of selected variables only ###################

# Read the CSV with column type suppression to avoid warnings
gss93 <- read_csv("gss93.csv", show_col_types = FALSE) %>%
    select(all_of(vars))

# Save the selected data
write_csv(gss93, "gss93_selected.csv")



# ### Individual Variables:
# EDUC
# EDUC89
# REALINC
# NUMADULT
# PRESTG80
# PRESTIGE
# SEI
# SEX
# AGE
# RACE
# ETHNIC
# ETHNICITY
# RELIG
# DENOM
# REGION

# ### Political Intolerance Variables:
# SPKATH
# SPKRAC
# SPKCOM
# SPKMIL
# SPKHOMO
# COLATH
# COLRAC

# ### Music Preference Variables:
# MUSCLASS
# MUSOPERA
# MUSJAZZ
# MUSGOSPE
# MUSCOUN
# MUSRAP
# MUSBLUES
# MUSREGGA
# MUSLATIN
# MUSROCK
# MUSFOLK
# MUSBLUEG
# MUSPOP
# MUSNEWAG
# MUSSHOW
# MUSDANCE
# MUSALTE
# MUSHEAVY
