"""
Attempt to download EVS Wave 1 (1981) data from GESIS or other sources
"""
import os
import urllib.request
import urllib.error
import subprocess

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base, "data")

print("=== Attempting to download EVS 1981 data ===")
print()

# Try various URLs for EVS 1981 data
# ZA4438 is the EVS 1981 dataset from GESIS

urls_to_try = [
    # GESIS public data download (may require login)
    "https://dbk.gesis.org/dbksearch/download.asp?id=67089",
    # Alternative - open data portal
    "https://data.gesis.org/sharing/#!Detail/10.7802/1556",
    # Direct file attempt
    "https://dbk.gesis.org/dbksearch/file.asp?file=ZA4438_v2-0-0.sav",
    "https://dbk.gesis.org/dbksearch/file.asp?file=ZA4438_v2-0-0.dta",
]

for url in urls_to_try:
    print(f"Trying: {url}")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req, timeout=10)
        print(f"  Response status: {response.status}")
        print(f"  Content type: {response.headers.get('Content-Type', 'unknown')}")
        # Don't actually download if it's an HTML page (login required)
        content_type = response.headers.get('Content-Type', '')
        if 'html' in content_type.lower():
            print(f"  -> HTML response (login required)")
        else:
            print(f"  -> Data file response!")
    except urllib.error.HTTPError as e:
        print(f"  HTTP Error: {e.code}")
    except urllib.error.URLError as e:
        print(f"  URL Error: {e.reason}")
    except Exception as e:
        print(f"  Error: {e}")
    print()

# Try ICPSR
icpsr_urls = [
    "https://www.icpsr.umich.edu/web/ICPSR/studies/6160",  # EVS 1981-83
]

print("=== Alternative: Check for WVS Time Series with wave 1 European data ===")
# The WVS Time Series 1981-2022 includes EVS data merged in
# But from our checks, it only has non-European countries in wave 1

# Check if the 1981 data might be in the paper's supplement or GESIS open access
print("EVS 1981 (ZA4438) needs GESIS login - cannot download automatically")
print()
print("The paper uses EVS Wave 1 (1981) for European countries.")
print("This data is available from GESIS (https://dbk.gesis.org) study ZA4438")
print("Free registration required.")
