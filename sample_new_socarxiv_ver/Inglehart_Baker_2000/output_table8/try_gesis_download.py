"""Try downloading EVS data via gesisdata R package (which is already installed)"""
import subprocess
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base, 'data')

# Check if GESIS credentials are configured
r_script = f'''
library(gesisdata)

# Check for stored credentials
cat("Checking for GESIS credentials...\\n")
tryCatch({{
  # The gesisdata package stores credentials in environment variables or .Renviron
  email <- Sys.getenv("GESIS_EMAIL")
  password <- Sys.getenv("GESIS_PASSWORD")
  cat("GESIS_EMAIL set:", nchar(email) > 0, "\\n")
  cat("GESIS_PASSWORD set:", nchar(password) > 0, "\\n")

  # Also check .Renviron
  renviron <- file.path(Sys.getenv("HOME"), ".Renviron")
  if (file.exists(renviron)) {{
    cat(".Renviron exists, checking for GESIS vars...\\n")
    env_content <- readLines(renviron)
    gesis_lines <- grep("GESIS", env_content, value=TRUE)
    cat("GESIS lines found:", length(gesis_lines), "\\n")
  }} else {{
    cat(".Renviron does not exist\\n")
  }}

  # Try to download if credentials exist
  if (nchar(email) > 0 && nchar(password) > 0) {{
    cat("\\nAttempting download with stored credentials...\\n")

    # EVS Longitudinal Data File 1981-2008
    gesis_download(file_id = "ZA4804",
                   email = email,
                   password = password,
                   download_dir = "{data_dir}")
    cat("SUCCESS: Downloaded ZA4804\\n")
  }} else {{
    cat("\\nNo GESIS credentials found.\\n")
    cat("To set up: Add GESIS_EMAIL and GESIS_PASSWORD to ~/.Renviron\\n")
  }}
}}, error = function(e) {{
  cat("Error:", e$message, "\\n")
}})

# List data files
cat("\\nData files:\\n")
print(list.files("{data_dir}"))
'''

result = subprocess.run(['R', '--no-save', '-e', r_script],
                       capture_output=True, text=True, timeout=120, cwd=base)

# Print relevant lines
for line in result.stdout.split('\n'):
    stripped = line.strip()
    if stripped and not stripped.startswith('>') and not stripped.startswith('+'):
        print(line)
