"""Check gesisdata R package API"""
import subprocess, os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

r_script = '''
library(gesisdata)
# List all exported functions
cat("Exported functions:\\n")
print(ls("package:gesisdata"))
cat("\\n")

# Try to get help on key functions
for (fn in ls("package:gesisdata")) {
  cat("Function:", fn, "\\n")
  tryCatch({
    args <- formals(fn)
    cat("  Args:", paste(names(args), collapse=", "), "\\n")
  }, error = function(e) cat("  Not a function\\n"))
}
'''

result = subprocess.run(['R', '--no-save', '-e', r_script],
                       capture_output=True, text=True, timeout=30, cwd=base)
for line in result.stdout.split('\n'):
    if line.strip() and not line.startswith('>') and not line.startswith('+'):
        print(line)
