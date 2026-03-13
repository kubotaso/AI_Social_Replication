"""Try to access ANES panel data via subprocess calling R."""
import subprocess
import sys

cmd = '''
library(anesr)
datasets <- data(package = "anesr")$results[, "Item"]
cat("Available datasets:\\n")
for (d in datasets) cat(d, "\\n")
'''

result = subprocess.run(['/opt/homebrew/bin/Rscript', '-e', cmd],
                       capture_output=True, text=True, timeout=120)
print("STDOUT:", result.stdout[:2000])
print("STDERR:", result.stderr[:500])
