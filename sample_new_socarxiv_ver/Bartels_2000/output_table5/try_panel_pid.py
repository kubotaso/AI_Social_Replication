import pandas as pd
import numpy as np
from statsmodels.discrete.discrete_model import Probit
import statsmodels.api as sm

cdf = pd.read_csv('anes_cumulative.csv', low_memory=False)
panel92 = pd.read_csv('panel_1992.csv')

# panel_1992.csv has:
# pid_current (1-7 scale from V923634+1), pid_lagged (1-7 from V900320+1)
# vote_house (INCORRECTLY coded from V925701)
# V923634 (original 0-6 PID), V900320 (original 0-6 lagged PID)
# V925609 (party? 0-3), V925701 (ballot position 0,1,5)

# The CDF has VCF0707 properly coded
# But how to link panel_1992 respondents to CDF?
# panel_1992 has 1336 rows, CDF 1992 has 2485 rows
# Panel respondents in CDF are those with VCF0006a < 19920000

cdf92 = cdf[cdf['VCF0004']==1992].copy()
cdf92_panel = cdf92[cdf92['VCF0006a'] < 19920000].copy()

print(f'panel_1992.csv: N={len(panel92)}')
print(f'CDF 1992 panel: N={len(cdf92_panel)}')

# These should overlap. The panel_1992 should be a subset of CDF panel
# Let me check pid_current distribution
print(f'\npanel_1992 pid_current: {panel92["pid_current"].value_counts(dropna=False).sort_index().to_dict()}')
print(f'CDF panel VCF0301: {cdf92_panel["VCF0301"].value_counts(dropna=False).sort_index().to_dict()}')

# They should be the same 1336 respondents
# pid_current in panel_1992 = VCF0301 in CDF (both 1-7 scale)
# pid_lagged in panel_1992 = VCF0301 from 1990 CDF wave

# Can we match them by PID values to verify?
# First, let's check if panel_1992 = CDF panel exactly
print(f'\nCDF panel VCF0301 valid (1-7): {cdf92_panel["VCF0301"].isin([1,2,3,4,5,6,7]).sum()}')
print(f'panel_1992 pid_current valid (1-7): {panel92["pid_current"].isin([1,2,3,4,5,6,7]).sum()}')

# The CDF panel has 1359 respondents, panel_1992 has 1336
# Let me check if we can use pid_lagged from panel_1992 with VCF0707 from CDF
# If the respondents are in the same order...

# Actually, both datasets should have VCF0006a as the linking variable
# Check if V923001 or V900004 could be VCF0006a
print(f'\nV923634 in panel_1992: yes (same as pid_current-1)')

# Let me try a different approach: use only the CDF but check the
# VCF0301 from the 1990 wave more carefully
cdf90 = cdf[cdf['VCF0004']==1990].copy()

# Maybe the issue is in how VCF0301 is coded in 1990
# VCF0301 should be 1-7 in all years
print(f'\n1990 VCF0301 dist: {cdf90["VCF0301"].value_counts(dropna=False).sort_index().to_dict()}')
print(f'1992 panel VCF0301 dist: {cdf92_panel["VCF0301"].value_counts(dropna=False).sort_index().to_dict()}')

# Let me check if the merged lagged PID differs from panel_1992 lagged PID
merged = cdf92_panel.merge(cdf90[['VCF0006a','VCF0301']], on='VCF0006a', suffixes=('','_lag'))
print(f'\nMerged N: {len(merged)}')
print(f'Merged VCF0301_lag dist: {merged["VCF0301_lag"].value_counts(dropna=False).sort_index().to_dict()}')
print(f'panel_1992 pid_lagged dist: {panel92["pid_lagged"].value_counts(dropna=False).sort_index().to_dict()}')

# They differ! panel_1992 has 1336 all with valid pid_lagged
# CDF merge has 1359 with some NaN
# The difference is 23 respondents

# What about using the panel_1992 pid_lagged instead?
# We need to link panel_1992 rows to CDF rows
# V923634 = VCF0301 - 1 (0-6 scale vs 1-7 scale)
# V900320 = pid_lagged - 1

# Since they're in the same order (both from panel study),
# maybe we can match by row index?
# But CDF has 1359 panel respondents vs panel_1992 has 1336

# Actually, wait - maybe NOT all CDF panel respondents are the same
# as panel_1992. The CDF includes ALL ANES respondents, including
# fresh cross-section sample in 1992. The panel respondents are those
# who also appeared in 1990.

# VCF0006a for 1990 respondents
print(f'\n1990 VCF0006a range: {cdf90["VCF0006a"].min()} to {cdf90["VCF0006a"].max()}')

# The CDF 1992 panel consists of those with VCF0006a < 19920000
# These should all have been in 1990 as well
# Let's check how many merge successfully
ids_90 = set(cdf90['VCF0006a'].dropna().astype(int))
ids_92panel = set(cdf92_panel['VCF0006a'].dropna().astype(int))
print(f'1990 unique IDs: {len(ids_90)}')
print(f'1992 panel unique IDs: {len(ids_92panel)}')
print(f'Intersection: {len(ids_90 & ids_92panel)}')

# So all 1359 merge. But panel_1992.csv only has 1336
# The difference is 23 respondents - possibly missing V923634 (PID)

# Let me try: use CDF vote + CDF current PID + panel_1992 lagged PID
# We need a linking variable
# Without VCF0006a in panel_1992, we can match on VCF0301 and V923634+1
