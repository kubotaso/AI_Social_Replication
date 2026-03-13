"""
Attempt to actually download EVS Wave 1 (1981) data from GESIS
"""
import os
import urllib.request
import urllib.error
import struct

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base, "data")

# Try to download the DTA file (Stata format)
urls = [
    ("https://dbk.gesis.org/dbksearch/file.asp?file=ZA4438_v2-0-0.dta", "ZA4438_v2-0-0.dta"),
    ("https://dbk.gesis.org/dbksearch/file.asp?file=ZA4438_v2-0-0.sav", "ZA4438_v2-0-0.sav"),
]

for url, filename in urls:
    save_path = os.path.join(data_dir, filename)
    print(f"Downloading: {url}")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req, timeout=30)
        content = response.read()
        content_type = response.headers.get('Content-Type', 'unknown')
        print(f"  Content type: {content_type}")
        print(f"  Content length: {len(content)} bytes")
        print(f"  First 20 bytes (hex): {content[:20].hex()}")
        print(f"  First 20 bytes (text): {content[:20]}")

        # Check if it's a real data file (not HTML error page)
        if b'<html' in content[:100].lower() or b'<!doctype' in content[:100].lower():
            print(f"  -> This is an HTML page (login required or redirect)")
        elif len(content) > 1000:
            print(f"  -> Looks like a data file! Saving...")
            with open(save_path, 'wb') as f:
                f.write(content)
            print(f"  Saved to: {save_path}")
        else:
            print(f"  -> Too small, may not be valid data")
    except Exception as e:
        print(f"  Error: {e}")
    print()

# Try to open ZA4438_v2-0-0.dta if it exists
dta_path = os.path.join(data_dir, "ZA4438_v2-0-0.dta")
if os.path.exists(dta_path):
    print(f"ZA4438_v2-0-0.dta exists ({os.path.getsize(dta_path)} bytes)")
    import pandas as pd
    try:
        df = pd.read_stata(dta_path, convert_categoricals=False)
        print("Columns:", list(df.columns)[:30])
        print("Shape:", df.shape)
    except Exception as e:
        print(f"Failed to read as Stata: {e}")
else:
    print("ZA4438_v2-0-0.dta does not exist")

sav_path = os.path.join(data_dir, "ZA4438_v2-0-0.sav")
if os.path.exists(sav_path):
    print(f"ZA4438_v2-0-0.sav exists ({os.path.getsize(sav_path)} bytes)")
    # Try pyreadstat for SPSS
    try:
        import pyreadstat
        df, meta = pyreadstat.read_sav(sav_path)
        print("Columns:", list(df.columns)[:30])
        print("Shape:", df.shape)
    except ImportError:
        print("pyreadstat not available")
    except Exception as e:
        print(f"Failed to read as SPSS: {e}")
else:
    print("ZA4438_v2-0-0.sav does not exist")
