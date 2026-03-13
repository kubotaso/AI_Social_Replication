"""Try to download EVS data using R via Python subprocess"""
import subprocess
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

r_script = '''
library(gesisdata)
cat("Checking GESIS credentials...\\n")
tryCatch({
  cat("Attempting to download ZA4804 (EVS Longitudinal 1981-2008)...\\n")
  gesis_download(file_id = "ZA4804",
                 path = "%s",
                 filetype = "stata")
  cat("SUCCESS: Downloaded ZA4804\\n")
}, error = function(e) {
  cat("Failed ZA4804:", e$message, "\\n")
  tryCatch({
    cat("Attempting to download ZA4438 (EVS Wave 1)...\\n")
    gesis_download(file_id = "ZA4438",
                   path = "%s",
                   filetype = "stata")
    cat("SUCCESS: Downloaded ZA4438\\n")
  }, error = function(e2) {
    cat("Failed ZA4438:", e2$message, "\\n")
    cat("\\nNeed GESIS credentials for download.\\n")
  })
})
cat("\\nData files:\\n")
print(list.files("%s"))
''' % (os.path.join(base, 'data'), os.path.join(base, 'data'), os.path.join(base, 'data'))

result = subprocess.run(['R', '--no-save', '-e', r_script],
                       capture_output=True, text=True, timeout=120,
                       cwd=base)
# Print relevant output
for line in result.stdout.split('\n'):
    if any(x in line.lower() for x in ['success', 'failed', 'error', 'download', 'attempt', 'check', 'need', 'data files', 'za4', 'evs', 'wvs', 'csv', 'dta', '"']):
        print(line)

if result.returncode != 0:
    print(f"R exit code: {result.returncode}")
    # Print last few lines of stderr
    stderr_lines = [l for l in result.stderr.split('\n') if l.strip() and not any(x in l for x in ['Copyright', 'Platform', '自由', '配布', '貢献', '引用', 'demo', 'help', 'quit', 'license', '共同', '出版', 'ブラウザ', '終了', 'R は', 'R version', '一定', '詳しく', 'また', '>'])]
    for line in stderr_lines[-10:]:
        print(f"  ERR: {line}")
