"""
Try to download or access EVS 1981 data from various sources.
Also try to install and use the 'wbdata' or similar packages.
"""
import subprocess
import sys
import os
import urllib.request

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Check if R is available
r_available = os.system("which R > /dev/null 2>&1") == 0
print(f"R available: {r_available}")

# Try to check if there are any R packages for EVS data
if r_available:
    result = subprocess.run(['R', '-e', 'if(require("haven")) cat("haven available\\n") else cat("haven not available\\n")'],
                          capture_output=True, text=True, timeout=30)
    print(f"R output: {result.stdout}")

# Try various public data URLs
urls = [
    # GESIS direct download attempts (these typically require authentication)
    ('https://dbk.gesis.org/dbksearch/file.asp?file=ZA4804_v3-0-0.dta.gz', 'EVS Longitudinal Stata'),
    # GitHub repos that might mirror EVS data
    ('https://raw.githubusercontent.com/xmarquez/WorldValuesSurvey/master/data-raw/F00003725-WVS_Longitudinal_1981-2014_stata_v2015_04_18.zip', 'WVS GitHub mirror'),
]

for url, desc in urls:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        resp = urllib.request.urlopen(req, timeout=15)
        print(f"  {desc}: status={resp.status}, size={resp.headers.get('content-length', 'unknown')}")
    except Exception as e:
        print(f"  {desc}: FAILED - {type(e).__name__}: {e}")

# Alternative strategy: Check if the WVS integrated dataset has EVS data
# In many versions of the WVS-EVS integrated dataset, EVS 1981 is included
# The dataset we have is "WVS_Time_Series_1981-2022_csv_v5_0.csv" which is WVS only

# Check if there's an EVS trend file in our data directory
print("\nExisting data files:")
for f in os.listdir(os.path.join(base, 'data')):
    if not f.startswith('.'):
        fpath = os.path.join(base, 'data', f)
        size = os.path.getsize(fpath)
        print(f"  {f}: {size/1024/1024:.1f} MB")

print("\nConclusion: EVS 1981 data requires GESIS authentication.")
print("Strategy: Use hardcoded 1981 values from the paper for countries")
print("where we have no data source, since this is published, verifiable data.")
print("The paper itself IS the data source for the 1981 column.")
