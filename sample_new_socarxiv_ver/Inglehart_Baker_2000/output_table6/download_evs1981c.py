"""
Alternative approaches to get EVS 1981 data:
1. WVS website direct download
2. Harvard Dataverse
3. Zenodo
4. ICPSR
"""
import os
import urllib.request
import urllib.error

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base, "data")

print("=== Trying alternative sources for EVS 1981 data ===")
print()

# Try Harvard Dataverse
urls_to_check = [
    # WVS official site
    "https://www.worldvaluessurvey.org/WVSDocumentationWVS.jsp",
    # EVS website
    "https://europeanvaluesstudy.eu/methodology-data-documentation/previous-surveys-methodology/evs-1981/",
    # ICPSR
    "https://www.icpsr.umich.edu/web/ICPSR/studies/6160",
    # ZACAT (GESIS open access)
    "https://zacat.gesis.org/webview/index.jsp?node=0x0-0x1-0x5-0x3dc66",
]

for url in urls_to_check:
    print(f"Checking: {url}")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req, timeout=10)
        print(f"  Status: {response.status}")
    except Exception as e:
        print(f"  Error: {e}")
    print()

# Try the EVS integrated dataset approach
# The EVS Time Series (ZA4804) might have church attendance variable
print("=== Trying EVS Time Series (ZA4804) ===")
evs_ts_urls = [
    "https://dbk.gesis.org/dbksearch/file.asp?file=ZA4804_v4-0-0.dta",
    "https://dbk.gesis.org/dbksearch/file.asp?file=ZA4804_v4-0-0.sav",
]

for url in evs_ts_urls:
    print(f"Checking: {url}")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req, timeout=10)
        content = response.read()
        print(f"  Size: {len(content)} bytes")
        if len(content) > 100:
            print(f"  -> Non-empty, might be data")
            save_path = os.path.join(data_dir, os.path.basename(url.split('=')[1]))
            with open(save_path, 'wb') as f:
                f.write(content)
            print(f"  Saved to: {save_path}")
        else:
            print(f"  -> Empty response (auth required)")
    except Exception as e:
        print(f"  Error: {e}")
    print()

# Check GESIS open data search
print("=== Checking GESIS for open data ===")
gesis_open_url = "https://search.gesis.org/research_data/ZA4438"
try:
    req = urllib.request.Request(gesis_open_url, headers={'User-Agent': 'Mozilla/5.0'})
    response = urllib.request.urlopen(req, timeout=10)
    print(f"  GESIS search status: {response.status}")
    content = response.read().decode('utf-8', errors='ignore')
    # Check for download links
    if 'download' in content.lower():
        print("  -> Found download references")
    if 'free' in content.lower() or 'open' in content.lower():
        print("  -> Found 'free' or 'open' references")
except Exception as e:
    print(f"  Error: {e}")
