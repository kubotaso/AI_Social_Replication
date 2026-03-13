import subprocess
import os

# Strategy: Use anesr to load individual timeseries studies
# and extract panel-relevant data

# For 1976 panel: we need 1974 PID (lagged) and 1976 vote/PID (current)
# The 1972-76 panel study had the SAME respondents interviewed in 72, 74, 76
# timeseries_1976 should contain these panel respondents along with fresh sample
# timeseries_1974 should contain the 1974 wave respondents

# For 1960 panel: we need 1958 PID (lagged) and 1960 vote/PID (current)
# timeseries_1958 is NOT available in anesr
# But the CDF already has the 1958 data via panel_1960.csv

# For 1992 panel: timeseries_1990 has the 1990 wave, timeseries_1992 has 1992

# Let's extract the 1992 data first since that's our closest match

r_code = '''
library(anesr)

# Load 1992 timeseries
data("timeseries_1992")
cat("1992 timeseries N:", nrow(timeseries_1992), "\\n")
cat("1992 timeseries vars:", ncol(timeseries_1992), "\\n")

# Find PID variable
pid_vars <- grep("V923634", names(timeseries_1992), value=TRUE)
cat("PID vars (V923634):", paste(pid_vars, collapse=", "), "\\n")

# Find House vote variable
house_vars <- grep("V925701", names(timeseries_1992), value=TRUE)
cat("House vote vars (V925701):", paste(house_vars, collapse=", "), "\\n")

# Find the panel indicator or case ID
# V920001 is typically the case ID in ANES studies
id_vars <- grep("V920001|V920000|V925908|V925901|V925902", names(timeseries_1992), value=TRUE)
cat("ID vars:", paste(id_vars, collapse=", "), "\\n")

# Find any variable with "panel" in the name
panel_vars <- grep("[Pp]anel", names(timeseries_1992), value=TRUE)
cat("Panel vars:", paste(panel_vars, collapse=", "), "\\n")

# V923634 should be the 7-point PID
# V925701 should be the House vote
# V926004 might indicate panel membership

# Let's also check for V925608/V925609 (party of House candidate)
party_vars <- grep("V92560[89]|V92561[01]", names(timeseries_1992), value=TRUE)
cat("Party vars:", paste(party_vars, collapse=", "), "\\n")

# V926004 - check if it exists and indicates panel
check_vars <- grep("V926", names(timeseries_1992), value=TRUE)
cat("V926 vars:", paste(head(check_vars, 20), collapse=", "), "\\n")

# Actually, the key issue is: does timeseries_1992 have the
# party of the House candidate voted for?
# V925608 = party of House candidate 1
# V925610 = party of House candidate 2
# If both exist, we can determine the party of the voted-for candidate

for (v in c("V925608", "V925609", "V925610", "V925611", "V925701")) {
  if (v %in% names(timeseries_1992)) {
    cat(v, ":", paste(sort(unique(timeseries_1992[[v]])), collapse=", "), "\\n")
  } else {
    cat(v, ": NOT FOUND\\n")
  }
}

# Save relevant variables to CSV
vars_to_save <- c("V923634", "V925701", "V925608", "V925609", "V925610", "V925611")
available <- vars_to_save[vars_to_save %in% names(timeseries_1992)]
cat("\\nSaving variables:", paste(available, collapse=", "), "\\n")

# Also get the case ID
id_candidates <- grep("V920001|V920000|V920101", names(timeseries_1992), value=TRUE)
if (length(id_candidates) > 0) {
  available <- c(id_candidates, available)
}

df <- timeseries_1992[, available, drop=FALSE]
write.csv(df, "output_table5/anes_1992_house_vars.csv", row.names=FALSE)
cat("Saved to output_table5/anes_1992_house_vars.csv\\n")
cat("N rows:", nrow(df), "\\n")
'''

result = subprocess.run(
    ['Rscript', '-e', r_code],
    capture_output=True, text=True, timeout=120,
    cwd='/Users/administrator/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/Replication_Claude_Bartels_v2'
)
print("STDOUT:", result.stdout)
if result.stderr:
    stderr_lines = result.stderr.strip().split('\n')
    # Only show non-warning lines or last few
    important = [l for l in stderr_lines if not l.startswith('Warning') and not 'NAs introduced' in l]
    if important:
        print("STDERR (important):", '\n'.join(important[-10:]))
print("Return code:", result.returncode)
