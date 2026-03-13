import pandas as pd
import numpy as np
from statsmodels.discrete.discrete_model import Probit
import statsmodels.api as sm

cdf = pd.read_csv('anes_cumulative.csv', low_memory=False)
p92 = pd.read_csv('panel_1992.csv')

# CDF panel respondents in 1992
cdf92 = cdf[cdf['VCF0004']==1992].copy()
cdf92_panel = cdf92[cdf92['VCF0006a'] < 19920000].copy()
cdf90 = cdf[cdf['VCF0004']==1990].copy()

# Merge
merged = cdf92_panel.merge(cdf90[['VCF0006a','VCF0301']], on='VCF0006a', suffixes=('','_lag'))

# Check: how many have valid House vote but different PID filtering
print("Total merged:", len(merged))
print("VCF0707 valid (1 or 2):", merged['VCF0707'].isin([1.0,2.0]).sum())
print("VCF0301 valid (1-7):", merged['VCF0301'].isin([1,2,3,4,5,6,7]).sum())
print("VCF0301_lag valid (1-7):", merged['VCF0301_lag'].isin([1,2,3,4,5,6,7]).sum())

# Both PIDs valid
both_pid = merged['VCF0301'].isin([1,2,3,4,5,6,7]) & merged['VCF0301_lag'].isin([1,2,3,4,5,6,7])
print("Both PIDs valid:", both_pid.sum())

# All three valid
all_valid = both_pid & merged['VCF0707'].isin([1.0,2.0])
print("All three valid:", all_valid.sum())

# Check: what about NaN values vs specific codes?
print("\nVCF0301 distribution (all):")
print(merged['VCF0301'].value_counts(dropna=False).sort_index())
print("\nVCF0301_lag distribution (all):")
print(merged['VCF0301_lag'].value_counts(dropna=False).sort_index())
print("\nVCF0707 distribution (all):")
print(merged['VCF0707'].value_counts(dropna=False).sort_index())

# Alternative approach: use panel_1992.csv pid_lagged matched to CDF VCF0707
# The panel_1992.csv has 1336 respondents
# Match to CDF by VCF0006a

# But panel_1992 doesn't have VCF0006a. Can we find another match key?
# Check if pid_current in panel matches VCF0301 in CDF
print("\n=== Trying to match panel_1992 to CDF ===")
print("Panel_1992 pid_current distribution:")
print(p92['pid_current'].value_counts(dropna=False).sort_index())
print("CDF panel VCF0301 distribution:")
print(cdf92_panel['VCF0301'].value_counts(dropna=False).sort_index())

# The distributions are very similar. But we can't match by PID alone.
# We need some other identifier.

# Actually V923634 is probably the ANES case ID variable from the 1992 study
# Wait, V923634 has values 0-9 (same as PID coding). So it IS the raw PID.

# Let me check the 1990 ANES study. V900320 should be the 1990 raw PID.
# In the 1990 ANES, PID is coded differently: 0=Strong Dem, ..., 6=Strong Rep
# So pid_lagged = V900320 + 1 maps 0->1, 1->2, ..., 6->7

# For matching panel to CDF: we might try matching on multiple variables
# Or just accept that the CDF approach gives N=759

# Check: what if we DON'T require lagged PID for the "current" model?
# No - the paper says "panel respondents only" so we need both waves

# Try: relax filtering slightly - maybe include VCF0301=0 (ANES missing code)
# if it doesn't actually affect the analysis
print("\nVCF0301=0.0 count in CDF panel:", (merged['VCF0301']==0).sum())
print("VCF0301_lag=0.0 count in CDF panel:", (merged['VCF0301_lag']==0).sum())
print("VCF0301 NaN count:", merged['VCF0301'].isna().sum())
print("VCF0301_lag NaN count:", merged['VCF0301_lag'].isna().sum())

# Run probit with N=760 goal: can we get exactly 760?
# Currently 759. Check what's off by 1.
df_test = merged[all_valid].copy()
print(f"\nCurrently valid N={len(df_test)}")

# Check if there's a respondent with VCF0707 valid but one PID is 0 or 8 or 9
# who should be included
edge = merged[merged['VCF0707'].isin([1.0,2.0]) & ~all_valid]
print(f"Respondents with valid vote but excluded: {len(edge)}")
if len(edge) > 0:
    print("Reasons for exclusion:")
    print("  VCF0301 invalid:", (~edge['VCF0301'].isin([1,2,3,4,5,6,7])).sum())
    print("  VCF0301_lag invalid:", (~edge['VCF0301_lag'].isin([1,2,3,4,5,6,7])).sum())
    print("  Invalid VCF0301 values:", edge.loc[~edge['VCF0301'].isin([1,2,3,4,5,6,7]), 'VCF0301'].value_counts())
    print("  Invalid VCF0301_lag values:", edge.loc[~edge['VCF0301_lag'].isin([1,2,3,4,5,6,7]), 'VCF0301_lag'].value_counts())
