import subprocess

r_code = '''
library(anesr)
d <- data(package="anesr")
items <- d$results[, "Item"]
cat("All datasets in anesr:\\n")
for (item in items) {
  cat(item, "\\n")
}
'''

result = subprocess.run(
    ['Rscript', '-e', r_code],
    capture_output=True, text=True, timeout=180,
    cwd='/Users/administrator/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/Replication_Claude_Bartels_v2'
)
print(result.stdout[:5000])
if result.returncode != 0:
    print("STDERR:", result.stderr[-500:])
