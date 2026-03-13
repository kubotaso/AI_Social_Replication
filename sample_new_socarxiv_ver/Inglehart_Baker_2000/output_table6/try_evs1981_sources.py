"""
Try alternative sources for EVS Wave 1 (1981) data
Looking for open-access versions of ZA4438
"""
import os
import urllib.request
import urllib.error

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base, "data")

print("=== Trying alternative sources for EVS 1981 data ===")
print()

# Various potential open-access locations for EVS 1981
urls_to_try = [
    # GESIS open access (if available)
    ("https://search.gesis.org/research_data/ZA4438", "GESIS search"),
    # Possibly EVS time series
    ("https://europeanvaluesstudy.eu/methodology-data-documentation/evs-trend-file-1981-2008/", "EVS Trend File"),
    # Harvard Dataverse
    ("https://dataverse.harvard.edu/api/search?q=European+Values+Study+1981&type=dataset", "Harvard Dataverse"),
    # OSF
    ("https://osf.io/search/?q=European+Values+Study+1981", "OSF"),
    # Zenodo
    ("https://zenodo.org/api/records?q=European+Values+Study+1981+church&size=5", "Zenodo"),
    # ICPSR
    ("https://www.icpsr.umich.edu/web/ICPSR/studies/6160", "ICPSR"),
]

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}

for url, name in urls_to_try:
    print(f"Checking {name}: {url}")
    try:
        req = urllib.request.Request(url, headers=headers)
        response = urllib.request.urlopen(req, timeout=15)
        content = response.read()[:2000].decode('utf-8', errors='ignore')
        print(f"  Status: {response.status}")
        # Check for download links or data files
        if 'download' in content.lower() or '.csv' in content.lower() or '.dta' in content.lower():
            print(f"  -> Found download references!")
        if '1981' in content:
            print(f"  -> Contains '1981' references")
        if 'church' in content.lower() or 'religious' in content.lower():
            print(f"  -> Contains religious attendance references")
    except Exception as e:
        print(f"  Error: {e}")
    print()

# Also try to find if there's a public API for WVS/EVS
print("=== WVS public API ===")
try:
    req = urllib.request.Request(
        "https://www.worldvaluessurvey.org/WVSOnline.jsp",
        headers=headers
    )
    response = urllib.request.urlopen(req, timeout=15)
    print(f"  WVS Online: {response.status}")
except Exception as e:
    print(f"  Error: {e}")

# Try EVS trend file which is supposed to include 1981
print("\n=== EVS Trend File 1981-2008 ===")
trend_urls = [
    "https://europeanvaluesstudy.eu/methodology-data-documentation/evs-trend-file-1981-2008/",
    "https://dbk.gesis.org/dbksearch/sdesc2.asp?no=4804",
]
for url in trend_urls:
    try:
        req = urllib.request.Request(url, headers=headers)
        response = urllib.request.urlopen(req, timeout=15)
        content = response.read()[:3000].decode('utf-8', errors='ignore')
        print(f"  {url}: {response.status}")
        if 'download' in content.lower():
            print("  -> Has download links")
        if '1981' in content:
            print("  -> References 1981")
    except Exception as e:
        print(f"  Error accessing {url}: {e}")
