"""Try to access EVS 1981 data via R packages"""
import subprocess
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Try R approach
r_script = '''
# Check installed packages
pkgs <- installed.packages()[,"Package"]
evs_pkgs <- pkgs[grepl("evs|EVS|european.value", pkgs, ignore.case=TRUE)]
cat("EVS packages:", paste(evs_pkgs, collapse=", "), "\\n")

gesis_pkgs <- pkgs[grepl("gesis|GESIS", pkgs, ignore.case=TRUE)]
cat("GESIS packages:", paste(gesis_pkgs, collapse=", "), "\\n")

wvs_pkgs <- pkgs[grepl("wvs|WVS|world.value", pkgs, ignore.case=TRUE)]
cat("WVS packages:", paste(wvs_pkgs, collapse=", "), "\\n")

# Try to check if WVS integrated dataset exists somewhere
cat("\\nChecking for integrated WVS-EVS datasets...\\n")

# Look for any .dta, .sav, or .csv files that might contain EVS 1981
data_dir <- file.path(dirname(getwd()), "data")
files <- list.files(data_dir, full.names=TRUE)
cat("Data files:", paste(basename(files), collapse=", "), "\\n")
'''

result = subprocess.run(['R', '--no-save', '-e', r_script],
                       capture_output=True, text=True, timeout=30,
                       cwd=os.path.join(base, 'output_table8'))
print("STDOUT:", result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)
if result.stderr:
    # Filter out R startup messages
    stderr_lines = [l for l in result.stderr.split('\n') if not any(x in l for x in ['Copyright', 'Platform', '自由な', '配布条件', '貢献者', '引用', 'demo', 'help', 'quit', 'license', '共同', '出版物', 'ブラウザ', '終了', 'R は', 'R version', '一定の', '詳しく', 'また'])]
    filtered = '\n'.join(stderr_lines).strip()
    if filtered:
        print("STDERR:", filtered[:500])
