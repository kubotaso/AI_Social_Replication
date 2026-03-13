import urllib.request, re
url = 'https://search.gesis.org/research_data/ZA7503'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
response = urllib.request.urlopen(req, timeout=15)
content = response.read().decode('utf-8', errors='ignore')
# Find all links
links = re.findall(r'href="(https?://[^"]*)"', content)
for l in links[:30]:
    print(l)
print("\n---")
# Look for download-related text
for match in re.finditer(r'download|\.dta|\.sav|\.csv|\.zip|data.*file', content, re.I):
    start = max(0, match.start()-100)
    end = min(len(content), match.end()+100)
    print(content[start:end])
    print("---")
