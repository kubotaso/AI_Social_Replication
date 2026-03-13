"""Search for EVS data files on the machine"""
import os
import glob

# Search in the AI_WVS directory tree
base_search = '/Users/kubotaso/Library/CloudStorage/Dropbox/lib/AI_WVS'

patterns = ['**/EVS*', '**/*evs*', '**/ZA4804*', '**/ZA4438*', '**/european_value*',
            '**/*WVS_EVS*', '**/*wvs_evs*', '**/*Trend*1981*', '**/*trend*1981*',
            '**/*longitudinal*', '**/*Longitudinal*']

found_files = set()
for pattern in patterns:
    for f in glob.glob(os.path.join(base_search, pattern), recursive=True):
        if not os.path.isdir(f) and not f.endswith('.py') and not '__pycache__' in f:
            found_files.add(f)

if found_files:
    print("Found EVS/Trend related files:")
    for f in sorted(found_files):
        size = os.path.getsize(f)
        print(f"  {f} ({size/1024/1024:.1f} MB)")
else:
    print("No EVS/Trend files found in AI_WVS directory.")

# Also check home directory for any downloaded EVS files
home = os.path.expanduser('~')
for search_dir in [os.path.join(home, 'Downloads'), os.path.join(home, 'Documents'),
                   os.path.join(home, 'Desktop')]:
    if os.path.exists(search_dir):
        for pattern in ['*EVS*', '*ZA4804*', '*ZA4438*', '*evs*longitudinal*']:
            for f in glob.glob(os.path.join(search_dir, pattern)):
                if not os.path.isdir(f):
                    size = os.path.getsize(f)
                    print(f"  {f} ({size/1024/1024:.1f} MB)")
