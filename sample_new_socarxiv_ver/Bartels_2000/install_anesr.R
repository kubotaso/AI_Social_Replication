if (!require("anesr", quietly=TRUE)) {
  install.packages("remotes", repos="https://cloud.r-project.org")
  remotes::install_github("jamesmartherus/anesr")
}
library(anesr)
cat("anesr loaded\n")
print(data(package="anesr")$results[,c("Item","Title")])
