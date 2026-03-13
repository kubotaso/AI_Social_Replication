library(dataverse)
r <- dataverse_search("European Values Study 1981", server="dataverse.harvard.edu")
cat("Found", nrow(r), "results\n")
if(nrow(r) > 0) {
  for(i in 1:min(10, nrow(r))) {
    cat(r$name[i], "-", r$global_id[i], "\n")
  }
}
