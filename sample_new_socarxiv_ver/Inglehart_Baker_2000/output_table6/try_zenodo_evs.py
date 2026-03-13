"""
Try Zenodo for EVS 1981 data - detailed search
"""
import os
import urllib.request
import json

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base, "data")

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}

# Search Zenodo for EVS 1981
print("=== Zenodo API Search ===")
search_queries = [
    "European+Values+Study+1981+ZA4438",
    "EVS+1981+church+attendance",
    "Inglehart+Baker+2000+replication",
    "World+Values+Survey+1981+Europe",
]

for q in search_queries:
    url = f"https://zenodo.org/api/records?q={q}&size=5"
    print(f"\nSearching: {q}")
    try:
        req = urllib.request.Request(url, headers=headers)
        response = urllib.request.urlopen(req, timeout=15)
        data = json.loads(response.read())
        hits = data.get('hits', {}).get('hits', [])
        print(f"  Found {len(hits)} results:")
        for h in hits[:3]:
            title = h.get('metadata', {}).get('title', 'No title')
            doi = h.get('doi', 'No DOI')
            print(f"  - {title[:60]} | DOI: {doi}")
            # Check for downloadable files
            files = h.get('files', [])
            for f in files[:2]:
                print(f"    File: {f.get('filename', '')} ({f.get('filesize', 0)/1024/1024:.1f} MB)")
    except Exception as e:
        print(f"  Error: {e}")

# Try Harvard Dataverse API
print("\n\n=== Harvard Dataverse Search ===")
url = "https://dataverse.harvard.edu/api/search?q=European+Values+Study+1981&type=dataset&per_page=5"
try:
    req = urllib.request.Request(url, headers=headers)
    response = urllib.request.urlopen(req, timeout=15)
    data = json.loads(response.read())
    items = data.get('data', {}).get('items', [])
    print(f"Found {len(items)} datasets:")
    for item in items[:5]:
        print(f"  - {item.get('name', 'No name')[:60]}")
        print(f"    URL: {item.get('url', '')}")
except Exception as e:
    print(f"  Error: {e}")

# Try OSF search
print("\n\n=== ICPSR STUDY 6160 CHECK ===")
icpsr_url = "https://www.icpsr.umich.edu/web/ICPSR/studies/6160/datadocumentation"
try:
    req = urllib.request.Request(icpsr_url, headers=headers)
    response = urllib.request.urlopen(req, timeout=15)
    content = response.read()[:5000].decode('utf-8', errors='ignore')
    print(f"  Status: {response.status}")
    if 'download' in content.lower():
        print("  -> Has download options")
    if 'free' in content.lower() or 'guest' in content.lower():
        print("  -> May be freely available")
    # Extract any relevant text
    import re
    download_links = re.findall(r'href="[^"]*download[^"]*"', content)
    print(f"  Download links found: {download_links[:3]}")
except Exception as e:
    print(f"  Error: {e}")
