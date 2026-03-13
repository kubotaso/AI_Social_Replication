import subprocess
import pandas as pd
import numpy as np
from statsmodels.discrete.discrete_model import Probit
import statsmodels.api as sm

# Rather than trying to identify variables in the individual studies,
# let's use the CDF which has standardized variable names.
# The issue is that the CDF has fewer panel respondents than Bartels had.
#
# CDF approach already gives:
# 1960: 634 (target 911)
# 1976: 552 (target 682)
# 1992: 759 (target 760)
#
# The key insight from the individual studies is that V763002/V742002
# overlap with 897 case IDs for the 1972-76 panel. This is more than
# the 925 in our panel_1976.csv. But matching requires knowing which
# variable is the House vote in the individual study.
#
# Alternative approach: check if the CDF itself has more respondents
# that we're missing. Perhaps there are respondents with valid data
# that we're filtering out unnecessarily.

# Let's check: for the 1976 panel, are there respondents with
# VCF0707=0 (didn't vote/3rd party) who might actually have valid
# House votes in the original study?

cdf = pd.read_csv('anes_cumulative.csv', low_memory=False)
cdf76 = cdf[cdf['VCF0004']==1976].copy()
cdf76_panel = cdf76[cdf76['VCF0006a'] < 19760000].copy()
cdf74 = cdf[cdf['VCF0004']==1974].copy()

print("1976 CDF panel respondents:", len(cdf76_panel))
print("VCF0707 distribution:")
print(cdf76_panel['VCF0707'].value_counts(dropna=False).sort_index())

# Check VCF0706 (House vote alternative)
print("\nVCF0706 distribution:")
print(cdf76_panel['VCF0706'].value_counts(dropna=False).sort_index())

# Check if VCF0706 provides additional House voters
# VCF0706: 1=Dem, 2=Rep, 3=other, 4=wouldn't vote, 7=DK
valid_707 = cdf76_panel['VCF0707'].isin([1.0, 2.0])
valid_706 = cdf76_panel['VCF0706'].isin([1.0, 2.0])
print(f"\nValid VCF0707: {valid_707.sum()}")
print(f"Valid VCF0706: {valid_706.sum()}")
print(f"Both valid: {(valid_707 & valid_706).sum()}")
print(f"VCF0707 only: {(valid_707 & ~valid_706).sum()}")
print(f"VCF0706 only: {(~valid_707 & valid_706).sum()}")

# Same for 1960
cdf60 = cdf[cdf['VCF0004']==1960].copy()
cdf60_panel = cdf60[cdf60['VCF0006a'] < 19600000].copy()
print(f"\n1960 CDF panel respondents: {len(cdf60_panel)}")
print("VCF0707:", cdf60_panel['VCF0707'].value_counts(dropna=False).sort_index().to_dict())
print("VCF0706:", cdf60_panel['VCF0706'].value_counts(dropna=False).sort_index().to_dict())

valid_707_60 = cdf60_panel['VCF0707'].isin([1.0, 2.0])
valid_706_60 = cdf60_panel['VCF0706'].isin([1.0, 2.0])
print(f"Valid VCF0707: {valid_707_60.sum()}")
print(f"Valid VCF0706: {valid_706_60.sum()}")
print(f"VCF0706 only: {(~valid_707_60 & valid_706_60).sum()}")

# What if we try using VCF0706 (pre-election vote intent) for those
# missing VCF0707? This might not be what Bartels did, but it could
# give us more data.

# Actually, let me check the 1976 panel CSV to understand
df76 = pd.read_csv('panel_1976.csv')
print(f"\npanel_1976.csv VCF0707:", df76['VCF0707'].value_counts(dropna=False).sort_index().to_dict())
if 'VCF0706' in df76.columns:
    print(f"panel_1976.csv VCF0706:", df76['VCF0706'].value_counts(dropna=False).sort_index().to_dict())
