"""
Try to access EVS Trend File (ZA4804) which covers 1981-2008
This might be the only way to get 1981 European data
"""
import os
import urllib.request
import urllib.error

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base, "data")

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}

print("=== Trying EVS Trend File ZA4804 (covers 1981-2008) ===")
print()

# ZA4804 is the EVS Trend Data which should include 1981
# Check various download URLs
urls = [
    # GESIS direct URLs for ZA4804
    "https://dbk.gesis.org/dbksearch/file.asp?file=ZA4804_v4-0-0.dta",
    "https://dbk.gesis.org/dbksearch/file.asp?file=ZA4804_v4-0-0.sav",
    "https://dbk.gesis.org/dbksearch/file.asp?file=ZA4804_v2-0-0.dta",
    "https://dbk.gesis.org/dbksearch/file.asp?file=ZA4804_v2-0-0.sav",
    # Try GESIS data search endpoint
    "https://search.gesis.org/research_data/ZA4804",
    # Try GESIS data portal API
    "https://api.gesis.org/study?study_number=ZA4804",
    # EVS trend file page
    "https://europeanvaluesstudy.eu/methodology-data-documentation/previous-surveys-methodology/evs-1981/",
    # ZACAT open access
    "https://zacat.gesis.org/webview/index.jsp?object=http://zacat.gesis.org/obj/fStudy/ZA4438",
]

for url in urls:
    print(f"Trying: {url}")
    try:
        req = urllib.request.Request(url, headers=headers)
        response = urllib.request.urlopen(req, timeout=15)
        content = response.read()
        content_type = response.headers.get('Content-Type', 'unknown')
        print(f"  Status: {response.status}")
        print(f"  Content-Type: {content_type}")
        print(f"  Size: {len(content)} bytes")

        # Check if it's data or HTML
        if b'<html' in content[:200].lower() or b'<!DOCTYPE' in content[:200]:
            print(f"  -> HTML page")
            # Look for any download links in the content
            content_text = content[:2000].decode('utf-8', errors='ignore')
            if 'download' in content_text.lower():
                print(f"  -> Has download references")
            if '1981' in content_text:
                print(f"  -> References 1981")
        elif len(content) > 10000:
            print(f"  -> POSSIBLE DATA FILE!")
            # Save it
            fname = url.split('=')[-1] if '=' in url else 'evs_download.bin'
            save_path = os.path.join(data_dir, fname)
            with open(save_path, 'wb') as f:
                f.write(content)
            print(f"  Saved to: {save_path}")
        else:
            print(f"  -> Small response: {content[:100]}")
    except urllib.error.HTTPError as e:
        print(f"  HTTP Error: {e.code}")
    except urllib.error.URLError as e:
        print(f"  URL Error: {e.reason}")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")
    print()

# Check if any ZA4804 files are in the data directory
print("=== Files in data/ directory ===")
for f in sorted(os.listdir(data_dir)):
    fpath = os.path.join(data_dir, f)
    size = os.path.getsize(fpath) / 1024 / 1024
    print(f"  {f}: {size:.2f} MB")
