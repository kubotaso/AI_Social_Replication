"""Try to download the WVS/EVS Trend dataset using POST requests"""
import urllib.request
import urllib.parse
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base, 'data')

# The WVS website uses form submissions. Let's try to mimic the form POST.
# Based on the page source, the download function posts to AJDownload.jsp

# Try Stata format (doc_id 11410)
for doc_id, desc in [('11410', 'Stata'), ('11411', 'R_rds'), ('11409', 'SPSS')]:
    url = 'https://www.worldvaluessurvey.org/AJDownload.jsp'

    # Try GET with query parameter
    try:
        full_url = f'{url}?docid={doc_id}'
        req = urllib.request.Request(full_url, headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.worldvaluessurvey.org/WVSEVStrend.jsp',
            'Connection': 'keep-alive',
        })
        resp = urllib.request.urlopen(req, timeout=30)
        content = resp.read()
        ct = resp.headers.get('Content-Type', '')
        cd = resp.headers.get('Content-Disposition', '')
        print(f"GET docid={doc_id} ({desc}): status={resp.status}, type={ct}, disp={cd}, size={len(content)}")
        if len(content) > 10:
            print(f"  Content preview: {content[:100]}")
    except Exception as e:
        print(f"GET docid={doc_id} ({desc}): {e}")

    # Try POST with form data
    try:
        data = urllib.parse.urlencode({'docid': doc_id}).encode()
        req = urllib.request.Request(url, data=data, headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Content-Type': 'application/x-www-form-urlencoded',
            'Referer': 'https://www.worldvaluessurvey.org/WVSEVStrend.jsp',
        })
        resp = urllib.request.urlopen(req, timeout=30)
        content = resp.read()
        ct = resp.headers.get('Content-Type', '')
        cd = resp.headers.get('Content-Disposition', '')
        print(f"POST docid={doc_id} ({desc}): status={resp.status}, type={ct}, disp={cd}, size={len(content)}")
        if len(content) > 10 and 'application' in ct.lower():
            fname = f'wvs_evs_trend_{doc_id}.zip'
            if cd and 'filename' in cd:
                fname = cd.split('filename=')[1].strip('"').strip("'")
            fpath = os.path.join(data_dir, fname)
            with open(fpath, 'wb') as f:
                f.write(content)
            print(f"  SAVED: {fpath}")
        elif len(content) > 10:
            print(f"  Content preview: {content[:200]}")
    except Exception as e:
        print(f"POST docid={doc_id} ({desc}): {e}")

# Try the license download endpoint
for doc_id, desc in [('11410', 'Stata'), ('11411', 'R_rds')]:
    try:
        url = f'https://www.worldvaluessurvey.org/AJDownloadLicense.jsp?docid={doc_id}'
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0',
            'Referer': 'https://www.worldvaluessurvey.org/WVSEVStrend.jsp',
        })
        resp = urllib.request.urlopen(req, timeout=30)
        content = resp.read()
        ct = resp.headers.get('Content-Type', '')
        cd = resp.headers.get('Content-Disposition', '')
        print(f"\nLicense endpoint docid={doc_id}: status={resp.status}, type={ct}, disp={cd}, size={len(content)}")
        if 'application' in ct.lower() or (cd and 'filename' in cd):
            fname = f'wvs_evs_trend_license_{doc_id}'
            if cd and 'filename' in cd:
                fname = cd.split('filename=')[1].strip('"').strip("'")
            fpath = os.path.join(data_dir, fname)
            with open(fpath, 'wb') as f:
                f.write(content)
            print(f"  SAVED: {fpath}")
        else:
            print(f"  Content preview: {content[:200]}")
    except Exception as e:
        print(f"License endpoint docid={doc_id}: {e}")
