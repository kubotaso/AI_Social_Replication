import pandas as pd
import numpy as np

BASE = "/Users/administrator/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/Replication_Claude_Topel_v3/psid_raw"

# Check 1968 family file ID
fam68 = pd.read_fwf(f'{BASE}/fam1968/FAM1968.txt', colspecs=[(1,6)], names=['id'], header=None)
print("1968 family IDs (first 10):", fam68['id'].head(10).tolist())
print("1968 family IDs range:", fam68['id'].min(), "-", fam68['id'].max())
print("1968 family IDs / 10 (first 10):", (fam68['id']//10).head(10).tolist())
print("1968 family IDs mod 10 (value counts):", (fam68['id']%10).value_counts().to_dict())
print()

# Check individual file ER30001
ind = pd.read_fwf(f'{BASE}/ind2023er/IND2023ER.txt', colspecs=[(1,5)], names=['id68'], header=None)
print("Individual ER30001 (first 10):", ind['id68'].head(10).tolist())
print("Individual ER30001 range:", ind['id68'].min(), "-", ind['id68'].max())
print("Unique individual ER30001:", ind['id68'].nunique())
print()

# Check if we should read 1968 family file with 4-digit ID (cols 2-5 instead of 2-6)
fam68_4 = pd.read_fwf(f'{BASE}/fam1968/FAM1968.txt', colspecs=[(1,5)], names=['id4'], header=None)
print("1968 family IDs 4-digit (first 10):", fam68_4['id4'].head(10).tolist())
print("1968 family IDs 4-digit range:", fam68_4['id4'].min(), "-", fam68_4['id4'].max())
print()

# Also check hourly earnings for a few years
for year, varname, start, end in [
    (1968, 'V337', 608, 612),
    (1969, 'V871', 725, 729),
    (1977, 'wages_V5283', 192, 196),
    (1977, 'hours_V5232', 60, 63),
]:
    fname = f'fam{year}/FAM{year}.txt'
    df = pd.read_fwf(f'{BASE}/{fname}', colspecs=[(start-1, end)], names=[varname], header=None)
    print(f"Year {year} {varname}: mean={df[varname].mean():.1f}, median={df[varname].median():.0f}, min={df[varname].min()}, max={df[varname].max()}")
    print(f"  Distribution: {df[varname].describe().to_dict()}")
    print()

# Check what the raw hourly earnings look like when divided by 100
df_hr68 = pd.read_fwf(f'{BASE}/fam1968/FAM1968.txt', colspecs=[(607, 612)], names=['hrly'], header=None)
hrly = df_hr68['hrly'] / 100.0
print(f"1968 hourly earnings / 100: mean={hrly.mean():.2f}, values>0 count={sum(hrly>0)}")
print(f"  Values between 1-200: {sum((hrly >= 1) & (hrly <= 200))}")
print(f"  Values between 0.01-10: {sum((hrly >= 0.01) & (hrly <= 10))}")
