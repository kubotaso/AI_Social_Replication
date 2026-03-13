import subprocess

r_code = '''
library(anesr)
data("timeseries_1992")

# V925701 is ballot position of House vote candidate (1=first, 5=second)
# V925609 seems to be party (1=Dem, 2=Rep based on earlier analysis)
# But we found V925609 gave wrong results

# The CDF variable VCF0707 maps from the original study
# Let's find what VCF0707 maps to in the 1992 study

# Key variables to check:
# V925701: ballot position voted for (0=NA, 1=cand1, 5=cand2)
# V925428: House vote validated? 0=506, 1=909, 2=803, 3=267
# V925430: 0=1465, 1=1020 - voted in House race
# V925431: 0=1465, 1=455, 5=535 - candidate voted for

# V925431 looks promising! Let me cross-tab with PID
cat("=== V925431 x PID (V923634) ===\\n")
print(table(timeseries_1992$V923634, timeseries_1992$V925431, useNA="ifany"))

# Also check V925419
cat("\\n=== V925419 x PID ===\\n")
print(table(timeseries_1992$V923634, timeseries_1992$V925419, useNA="ifany"))

# V925401 looks like presidential vote: 0=450, 1=598, 5=915, 7=438
# V925413 might be senate: 0=476, 1=388, 5=938, 7=589

# Let me check V925407: 0=888, 1=302, 5=1226
cat("\\n=== V925407 x PID ===\\n")
print(table(timeseries_1992$V923634, timeseries_1992$V925407, useNA="ifany"))

# Check the CDF to verify mapping
data("timeseries_cum")
cdf92 <- timeseries_cum[timeseries_cum$VCF0004 == 1992, ]
cat("\\nCDF 1992 VCF0707 dist:\\n")
print(table(cdf92$VCF0707, useNA="ifany"))

# CDF 1992 case IDs
cat("\\nCDF 1992 VCF0006a range:", min(cdf92$VCF0006a, na.rm=TRUE), "to", max(cdf92$VCF0006a, na.rm=TRUE), "\\n")

# Does V923634 (PID) match VCF0301?
cat("\\nV923634 dist (PID 1992):\\n")
print(table(timeseries_1992$V923634, useNA="ifany"))

# V900320 is the case ID? Let's check
cat("\\nV923001 range:", min(timeseries_1992$V923001, na.rm=TRUE), "to", max(timeseries_1992$V923001, na.rm=TRUE), "\\n")

# Check if we can match timeseries_1992 to timeseries_1990 via case ID
data("timeseries_1990")
cat("\\nV900004 (case ID?) in 1990:", min(timeseries_1990$V900004, na.rm=TRUE), "to", max(timeseries_1990$V900004, na.rm=TRUE), "\\n")

# How many 1992 respondents appear in 1990?
ids_92 <- timeseries_1992$V923001
ids_90 <- timeseries_1990$V900004
overlap <- intersect(ids_92, ids_90)
cat("Overlap between 1992 V923001 and 1990 V900004:", length(overlap), "\\n")

# V900320 in panel_1992.csv - check what it is
cat("\\nV900320 in 1990 study range:", min(timeseries_1990$V900320, na.rm=TRUE), "to", max(timeseries_1990$V900320, na.rm=TRUE), "\\n")
cat("V900320 dist:\\n")
print(table(timeseries_1990$V900320, useNA="ifany"))
'''

result = subprocess.run(
    ['Rscript', '-e', r_code],
    capture_output=True, text=True, timeout=300,
    cwd='/Users/administrator/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/Replication_Claude_Bartels_v2'
)
print(result.stdout[:8000])
if result.returncode != 0:
    print("STDERR:", result.stderr[-1000:])
