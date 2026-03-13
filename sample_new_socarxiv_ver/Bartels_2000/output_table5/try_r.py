import subprocess
import sys

# Try to use R to install and load anesr package
r_code = '''
if (!requireNamespace("anesr", quietly=TRUE)) {
  install.packages("remotes", repos="https://cloud.r-project.org", quiet=TRUE)
  remotes::install_github("jamesmartherus/anesr", quiet=TRUE)
}
cat("anesr available:", requireNamespace("anesr", quietly=TRUE), "\\n")

if (requireNamespace("anesr", quietly=TRUE)) {
  library(anesr)
  # List available datasets
  datasets <- data(package = "anesr")
  cat("Available datasets:\\n")
  for (item in datasets$results[, "Item"]) {
    cat("  ", item, "\\n")
  }
}
'''

result = subprocess.run(
    ['Rscript', '-e', r_code],
    capture_output=True, text=True, timeout=300
)
print("STDOUT:", result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)
print("Return code:", result.returncode)
