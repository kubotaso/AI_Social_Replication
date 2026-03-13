import pandas as pd
import numpy as np

cdf = pd.read_csv('anes_cumulative.csv', low_memory=False)

# For 1960: panel respondents would have been interviewed in 1958 and 1960
# Check CDF for 1960 respondents
cdf60 = cdf[cdf['VCF0004']==1960].copy()
print("=== 1960 in CDF ===")
print("Total:", len(cdf60))
print("VCF0006a range:", cdf60['VCF0006a'].min(), "-", cdf60['VCF0006a'].max())

# Check for 1958-era IDs in 1960 data
panel60 = cdf60[cdf60['VCF0006a'] < 19600000]
fresh60 = cdf60[cdf60['VCF0006a'] >= 19600000]
print("Panel (ID < 19600000):", len(panel60))
print("Fresh (ID >= 19600000):", len(fresh60))

if len(panel60) > 0:
    print("Panel VCF0006a range:", panel60['VCF0006a'].min(), "-", panel60['VCF0006a'].max())

# Check 1958 data
cdf58 = cdf[cdf['VCF0004']==1958].copy()
print("\n=== 1958 in CDF ===")
print("Total:", len(cdf58))
print("VCF0006a range:", cdf58['VCF0006a'].min(), "-", cdf58['VCF0006a'].max())

# Check for matching IDs
if len(panel60) > 0:
    panel_ids = set(panel60['VCF0006a'].values)
    matched = cdf58[cdf58['VCF0006a'].isin(panel_ids)]
    print("Matched in 1958:", len(matched))

# For 1976: panel respondents from 1974
cdf76 = cdf[cdf['VCF0004']==1976].copy()
print("\n=== 1976 in CDF ===")
print("Total:", len(cdf76))
print("VCF0006a range:", cdf76['VCF0006a'].min(), "-", cdf76['VCF0006a'].max())

panel76 = cdf76[cdf76['VCF0006a'] < 19760000]
fresh76 = cdf76[cdf76['VCF0006a'] >= 19760000]
print("Panel (ID < 19760000):", len(panel76))
print("Fresh (ID >= 19760000):", len(fresh76))

if len(panel76) > 0:
    print("Panel VCF0006a range:", panel76['VCF0006a'].min(), "-", panel76['VCF0006a'].max())

cdf74 = cdf[cdf['VCF0004']==1974].copy()
print("\n=== 1974 in CDF ===")
print("Total:", len(cdf74))
print("VCF0006a range:", cdf74['VCF0006a'].min(), "-", cdf74['VCF0006a'].max())

if len(panel76) > 0:
    panel_ids76 = set(panel76['VCF0006a'].values)
    matched74 = cdf74[cdf74['VCF0006a'].isin(panel_ids76)]
    print("Matched in 1974:", len(matched74))

# Now check the panel CSVs
p60 = pd.read_csv('panel_1960.csv')
p76 = pd.read_csv('panel_1976.csv')

print("\n=== Panel CSVs ===")
print("panel_1960.csv:", len(p60), "rows")
print("  VCF0006a range:", p60['VCF0006a'].min(), "-", p60['VCF0006a'].max())
print("  VCF0707 valid:", p60['VCF0707'].isin([1.0,2.0]).sum())
print("  VCF0301 valid:", p60['VCF0301'].isin([1,2,3,4,5,6,7]).sum())
print("  VCF0301_lagged valid:", p60['VCF0301_lagged'].isin([1,2,3,4,5,6,7]).sum())

print("\npanel_1976.csv:", len(p76), "rows")
print("  VCF0006a range:", p76['VCF0006a'].min(), "-", p76['VCF0006a'].max())
print("  VCF0707 valid:", p76['VCF0707'].isin([1.0,2.0]).sum())
print("  VCF0301 valid:", p76['VCF0301'].isin([1,2,3,4,5,6,7]).sum())
print("  VCF0301_lagged valid:", p76['VCF0301_lagged'].isin([1,2,3,4,5,6,7]).sum())

# Can we use CDF to get MORE panel respondents?
# The panel CSVs were created from the CDF, so the CDF should have all of them
# But maybe the CDF approach can find MORE respondents

# Try CDF-based approach for 1960
print("\n=== CDF approach for 1960 ===")
if len(panel60) > 0 and len(matched) > 0:
    merged60 = panel60.merge(cdf58[['VCF0006a','VCF0301']], on='VCF0006a', suffixes=('','_lag'))
    valid60 = merged60[
        merged60['VCF0707'].isin([1.0,2.0]) &
        merged60['VCF0301'].isin([1,2,3,4,5,6,7]) &
        merged60['VCF0301_lag'].isin([1,2,3,4,5,6,7])
    ]
    print("Valid for analysis:", len(valid60))
    print("vs panel CSV approach: 634")
    print("vs paper target: 911")

# Try CDF-based approach for 1976
print("\n=== CDF approach for 1976 ===")
if len(panel76) > 0 and len(matched74) > 0:
    merged76 = panel76.merge(cdf74[['VCF0006a','VCF0301']], on='VCF0006a', suffixes=('','_lag'))
    valid76 = merged76[
        merged76['VCF0707'].isin([1.0,2.0]) &
        merged76['VCF0301'].isin([1,2,3,4,5,6,7]) &
        merged76['VCF0301_lag'].isin([1,2,3,4,5,6,7])
    ]
    print("Valid for analysis:", len(valid76))
    print("vs panel CSV approach: 552")
    print("vs paper target: 682")
