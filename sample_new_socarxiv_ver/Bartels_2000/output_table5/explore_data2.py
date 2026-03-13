import pandas as pd
import numpy as np

cdf = pd.read_csv('anes_cumulative.csv', low_memory=False)
cdf92 = cdf[cdf['VCF0004']==1992].copy()

# Check: VCF0006a in CDF for 1992 - are there any starting with 1990?
print("VCF0006a range for 1992:", cdf92['VCF0006a'].min(), "-", cdf92['VCF0006a'].max())
panel_ids = cdf92[cdf92['VCF0006a'] < 19920000]
print("Panel respondents (VCF0006a < 19920000):", len(panel_ids))
fresh_ids = cdf92[cdf92['VCF0006a'] >= 19920000]
print("Fresh respondents (VCF0006a >= 19920000):", len(fresh_ids))

if len(panel_ids) > 0:
    print("Panel VCF0006a range:", panel_ids['VCF0006a'].min(), "-", panel_ids['VCF0006a'].max())
    # These panel respondents should have VCF0707
    print("Panel VCF0707:", panel_ids['VCF0707'].value_counts().sort_index())
    print("Panel VCF0301:", panel_ids['VCF0301'].value_counts().sort_index())

# Actually, looking at VCF0006a = 19900001-19921126
# The panel respondents are those with VCF0006a starting with 1990
# They were first recruited in the 1990 study

# Check if there's a 1990 entry for these people too
print("\n=== Checking for 1990 respondents in CDF ===")
cdf90 = cdf[cdf['VCF0004']==1990]
print("1990 CDF respondents:", len(cdf90))
print("VCF0006a range:", cdf90['VCF0006a'].min(), "-", cdf90['VCF0006a'].max())

# The panel respondents in 1992 have VCF0006a in the 1990xxxx range
# AND they appear in the 1992 study year
# So we can find them by VCF0004==1992 and VCF0006a starting with 1990

# Actually wait - I need to check: are the 1990-1992 panel respondents
# listed under VCF0004=1992 with their 1990 ID?
# Or are they listed under VCF0004=1990?

# Let me check the cross-reference
if len(panel_ids) > 0:
    # Do these panel IDs also appear under 1990?
    panel_id_list = set(panel_ids['VCF0006a'].values)
    matched_90 = cdf90[cdf90['VCF0006a'].isin(panel_id_list)]
    print("\nPanel IDs found in 1990 CDF:", len(matched_90))
    print("Panel IDs NOT found in 1990 CDF:", len(panel_id_list) - len(matched_90))

# For 1992 panel respondents, we can use:
# - VCF0301 from VCF0004=1992 as current PID
# - VCF0301 from VCF0004=1990 (same VCF0006a) as lagged PID
# - VCF0707 from VCF0004=1992 as House vote

# But wait - the panel_1992.csv already has pid_lagged from the 1990 wave
# We just need VCF0707 for these panel respondents

# Let me check if the number of 1990-panel respondents in 1992 matches
print("\n=== Summary ===")
print("Panel_1992.csv has", 1336, "respondents")
print("CDF 1992 with 1990-era IDs:", len(panel_ids) if len(panel_ids) > 0 else 0)

# Check PID distribution of panel respondents in CDF
if len(panel_ids) > 0:
    print("\nPanel respondents in CDF 1992:")
    print("VCF0301:", panel_ids['VCF0301'].value_counts().sort_index())
    print("VCF0707:", panel_ids['VCF0707'].value_counts().sort_index())
    print("VCF0707 vs VCF0301 crosstab:")
    print(pd.crosstab(panel_ids['VCF0301'], panel_ids['VCF0707']))
