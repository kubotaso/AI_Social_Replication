"""
Download the WVS/EVS Trend 1981-2022 dataset which includes EVS 1981 wave data.
Try various download approaches.
"""
import urllib.request
import os
import json

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base, 'data')

# The WVS website uses JavaScript download handlers. Let's try to construct the download URL.
# The download function is SetDocDownload('11409') which likely maps to:
# https://www.worldvaluessurvey.org/AJDownload.jsp?docid=11409

# Try various document IDs for the Trend file
doc_ids = {
    '11409': 'WVS_EVS_Trend (first listed)',
    '11410': 'WVS_EVS_Trend (next)',
    '11411': 'WVS_EVS_Trend (next)',
    '11412': 'WVS_EVS_Trend (next)',
}

for doc_id, desc in doc_ids.items():
    url = f'https://www.worldvaluessurvey.org/AJDownload.jsp?docid={doc_id}'
    print(f"Trying {url} ({desc})...")
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Referer': 'https://www.worldvaluessurvey.org/WVSEVStrend.jsp',
        })
        resp = urllib.request.urlopen(req, timeout=30)
        content_type = resp.headers.get('Content-Type', 'unknown')
        content_disp = resp.headers.get('Content-Disposition', 'unknown')
        content_len = resp.headers.get('Content-Length', 'unknown')
        print(f"  Status: {resp.status}, Type: {content_type}, Disp: {content_disp}, Size: {content_len}")

        # If it's a file download, save it
        if 'application' in content_type.lower() or 'octet' in content_type.lower() or content_disp != 'unknown':
            # Extract filename from Content-Disposition
            fname = f'wvs_evs_trend_{doc_id}'
            if 'filename' in content_disp:
                fname = content_disp.split('filename=')[1].strip('"').strip("'")
            filepath = os.path.join(data_dir, fname)
            data = resp.read()
            with open(filepath, 'wb') as f:
                f.write(data)
            print(f"  SAVED: {filepath} ({len(data)} bytes)")
        else:
            # Read first 500 chars to see what it is
            data = resp.read(500)
            print(f"  Content preview: {data[:200]}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")

# Also try the direct known patterns
patterns = [
    'https://www.worldvaluessurvey.org/WVSDocumentationWVL.jsp',
]
