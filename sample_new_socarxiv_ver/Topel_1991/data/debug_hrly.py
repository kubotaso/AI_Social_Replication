import pandas as pd
import numpy as np

BASE = "/Users/administrator/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/Replication_Claude_Topel_v3/psid_raw"

# Check hourly earnings raw values for each year
years_hrly = [
    (1968, 608, 612, "V337 HEAD HOURLY EARN"),
    (1969, 725, 729, "V871 HEADS HRLY EARN"),
    (1970, 848, 852, "V1567 HEADS AVG HRLY ERN"),
    (1971, 866, 870, "V2279 HEADS AVG HRLY ERN"),
    (1972, 899, 903, "V2906 HEADS AVG HRLY ERN"),
    (1973, 520, 524, "V3275 HDS AVG HRLY ERN"),
    (1974, 571, 575, "V3695 HDS AVG HRLY EARN"),
    (1975, 684, 688, "V4174 HDS AVG HRLY EARN"),
    (1976, 1400, 1404, "V5050 HD AVG HOURLY EARN"),
    # 1977: no explicit hrly var - skip
    (1978, 920, 924, "V6178 HEAD 77 AVG HRLY EARNING"),
    (1979, 991, 995, "V6771 HEAD 78 AVG HRLY EARNING"),
    (1980, 1058, 1062, "V7417 HEAD 79 AVG HRLY EARNING"),
    (1981, 1175, 1179, "V8069 HEAD 80 AVG HRLY EARNING"),
    (1982, 1032, 1036, "V8693 HEAD 81 AVG HRLY EARNING"),
    (1983, 1215, 1219, "V9379 HEAD 82 AVG HRLY EARNING"),
]

for year, start, end, label in years_hrly:
    fname = f'{BASE}/fam{year}/FAM{year}.txt'
    df = pd.read_fwf(fname, colspecs=[(start-1, end)], names=['hrly'], header=None)
    pos = df[df['hrly'] > 0]['hrly']
    if len(pos) > 0:
        print(f"{year} {label}:")
        print(f"  Positive values: n={len(pos)}, mean={pos.mean():.2f}, median={pos.median():.2f}, min={pos.min()}, max={pos.max()}")
        print(f"  Looks like {'CENTS' if pos.mean() > 100 else 'DOLLARS'}")
    else:
        print(f"{year} {label}: No positive values!")
    print()

# Also check: are 1968 values in dollars with implied decimal?
# The 1968 V337 has range 0-99.99, which suggests dollars with 2 decimal places
# Let's check the format - does it read as a float?
df68 = pd.read_fwf(f'{BASE}/fam1968/FAM1968.txt', colspecs=[(607, 612)], names=['hrly'], header=None)
print("1968 raw first 20:", df68['hrly'].head(20).tolist())
print()

# Check 1970
df70 = pd.read_fwf(f'{BASE}/fam1970/FAM1970.txt', colspecs=[(847, 852)], names=['hrly'], header=None)
print("1970 raw first 20:", df70['hrly'].head(20).tolist())
print()

# Check 1978
df78 = pd.read_fwf(f'{BASE}/fam1978/FAM1978.txt', colspecs=[(919, 924)], names=['hrly'], header=None)
print("1978 raw first 20:", df78['hrly'].head(20).tolist())
