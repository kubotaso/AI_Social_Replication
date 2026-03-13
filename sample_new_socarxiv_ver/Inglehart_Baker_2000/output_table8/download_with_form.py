"""Download WVS/EVS Trend data by submitting the license form"""
import urllib.request
import urllib.parse
import http.cookiejar
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base, 'data')

# Set up cookie handling for session management
cj = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
}

# Step 1: Visit the main page first to get session cookies
try:
    req = urllib.request.Request('https://www.worldvaluessurvey.org/WVSEVStrend.jsp', headers=headers)
    resp = opener.open(req, timeout=30)
    print(f"Step 1 (main page): status={resp.status}")
    print(f"Cookies: {[c.name for c in cj]}")
except Exception as e:
    print(f"Step 1 error: {e}")

# Step 2: Visit the license page
try:
    req = urllib.request.Request('https://www.worldvaluessurvey.org/AJDownloadLicense.jsp?docid=11410', headers=headers)
    resp = opener.open(req, timeout=30)
    content = resp.read()
    print(f"Step 2 (license page): status={resp.status}, size={len(content)}")
    print(f"Cookies: {[c.name for c in cj]}")
except Exception as e:
    print(f"Step 2 error: {e}")

# Step 3: Submit the form with required fields
# docid=11410 is the Stata file
form_data = {
    'ESSION': '',
    'ESSION2': '',
    'docid': '11410',
    'ESSION5': 'Academic research',
    'ESSION3': 'Replication of Inglehart Baker 2000',
    'ESSION4': 'Academic research - replication study',
    'checkTerms': 'on',
}

try:
    data = urllib.parse.urlencode(form_data).encode('utf-8')
    req = urllib.request.Request('https://www.worldvaluessurvey.org/AJDownload.jsp',
                               data=data,
                               headers={
                                   **headers,
                                   'Content-Type': 'application/x-www-form-urlencoded',
                                   'Referer': 'https://www.worldvaluessurvey.org/AJDownloadLicense.jsp?docid=11410',
                               })
    resp = opener.open(req, timeout=120)
    content = resp.read()
    ct = resp.headers.get('Content-Type', '')
    cd = resp.headers.get('Content-Disposition', '')
    print(f"Step 3 (download): status={resp.status}, type={ct}, disp={cd}, size={len(content)}")

    if len(content) > 1000:
        # Check if it's a zip/binary file
        if content[:2] == b'PK' or content[:4] == b'\\x1f\\x8b' or 'application' in ct.lower():
            fname = 'WVS_EVS_Trend_Stata.zip'
            if cd and 'filename' in cd:
                fname = cd.split('filename=')[1].strip('"').strip("'").strip()
            fpath = os.path.join(data_dir, fname)
            with open(fpath, 'wb') as f:
                f.write(content)
            print(f"  SAVED binary: {fpath} ({len(content)} bytes)")
        else:
            # Save anyway and check
            fpath = os.path.join(data_dir, 'wvs_trend_response.dat')
            with open(fpath, 'wb') as f:
                f.write(content)
            print(f"  Saved response to {fpath}")
            print(f"  First 200 bytes: {content[:200]}")
    else:
        print(f"  Content: {content}")
except Exception as e:
    print(f"Step 3 error: {e}")

# Try alternative: direct download with all form fields in query string
try:
    params = urllib.parse.urlencode({
        'docid': '11410',
        'ESSION5': 'Academic research',
    })
    url = f'https://www.worldvaluessurvey.org/AJDownload.jsp?{params}'
    req = urllib.request.Request(url, headers={
        **headers,
        'Referer': 'https://www.worldvaluessurvey.org/AJDownloadLicense.jsp?docid=11410',
    })
    resp = opener.open(req, timeout=120)
    content = resp.read()
    ct = resp.headers.get('Content-Type', '')
    cd = resp.headers.get('Content-Disposition', '')
    print(f"\nAlternative GET: status={resp.status}, type={ct}, disp={cd}, size={len(content)}")
    if len(content) > 100:
        if content[:2] == b'PK':
            print("  ZIP file detected!")
            fpath = os.path.join(data_dir, 'WVS_EVS_Trend.zip')
            with open(fpath, 'wb') as f:
                f.write(content)
            print(f"  SAVED: {fpath}")
        else:
            print(f"  First 100 bytes: {content[:100]}")
except Exception as e:
    print(f"Alternative GET error: {e}")
