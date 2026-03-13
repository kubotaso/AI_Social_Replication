import pandas as pd
import numpy as np
from statsmodels.discrete.discrete_model import Probit
import statsmodels.api as sm

# V925609 in the 1992 ANES might encode party of house candidates
# Let me check the codebook interpretation:
# V925609 might be: Party of House candidate #1
# 0=no candidate / NA
# 1=Democrat
# 2=Republican
# 3=Independent/other party
# 7=other
# 9=NA
#
# Then for V925701:
# 1 = voted for candidate 1 → party from V925609
# 5 = voted for candidate 2 → party from a companion variable (V925610?)
#
# But we don't have V925610 in our panel_1992.csv
# Let me try a different approach

p92 = pd.read_csv('panel_1992.csv')
cdf = pd.read_csv('anes_cumulative.csv', low_memory=False)

# Get CDF for 1992 panel respondents
cdf92 = cdf[cdf['VCF0004']==1992].copy()
cdf92_panel = cdf92[cdf92['VCF0006a'] < 19920000].copy().sort_values('VCF0006a').reset_index(drop=True)
cdf90 = cdf[cdf['VCF0004']==1990].copy()

# Check: what's the difference between CDF panel N=1359 and panel_1992 N=1336?
# 1359 - 1336 = 23 respondents
# These 23 might be respondents who are in the CDF panel but have missing data
# in the original 1992 study that caused them to be excluded from panel_1992.csv

# The panel_1992.csv pid_current has 20 NaN values
# CDF has 7 NaN values for VCF0301 among panel respondents
# Difference = 13 more NaN in panel_1992 vs CDF

# Let me check if matching by PID distribution can help identify issues
print("CDF panel N=1359, panel_1992 N=1336, diff=23")

# The CDF panel respondents include some that might not have been in the
# panel reinterview. Let me check VCF0013 (panel component indicator)
print("\nVCF0013 for 1992 CDF panel respondents:")
print(cdf92_panel['VCF0013'].value_counts().sort_index())

# VCF0013: 0=not a panel component, 1=is a panel component
# Let's filter to just those marked as panel
panel_marked = cdf92_panel[cdf92_panel['VCF0013']==1]
print(f"\nPanel-marked respondents: {len(panel_marked)}")

# Merge these with 1990
merged = panel_marked.merge(cdf90[['VCF0006a','VCF0301']], on='VCF0006a', suffixes=('','_lag'))
print(f"After merge with 1990: {len(merged)}")

# Valid sample
mask = (
    merged['VCF0707'].isin([1.0, 2.0]) &
    merged['VCF0301'].isin([1,2,3,4,5,6,7]) &
    merged['VCF0301_lag'].isin([1,2,3,4,5,6,7])
)
df = merged[mask].copy()
print(f"Valid for analysis: {len(df)}")

df['house_rep'] = (df['VCF0707'] == 2.0).astype(int)
df['strong_curr'] = np.where(df['VCF0301']==7, 1, np.where(df['VCF0301']==1, -1, 0))
df['weak_curr'] = np.where(df['VCF0301']==6, 1, np.where(df['VCF0301']==2, -1, 0))
df['lean_curr'] = np.where(df['VCF0301']==5, 1, np.where(df['VCF0301']==3, -1, 0))

X = sm.add_constant(df[['strong_curr','weak_curr','lean_curr']])
mod = Probit(df['house_rep'], X).fit(disp=0)
print(f"\nCurrent PID probit with VCF0013 filter:")
print(f"N={len(df)}, LL={mod.llf:.1f}, R2={mod.prsquared:.4f}")
print(f"Strong={mod.params['strong_curr']:.3f}, Weak={mod.params['weak_curr']:.3f}, Lean={mod.params['lean_curr']:.3f}")
print(f"Target: N=760, LL=-408.2, R2=0.20, Strong=0.975, Weak=0.627, Lean=0.472")

# Also try WITHOUT VCF0013 filter but using ALL panel respondents
print("\n\n=== Without VCF0013 filter ===")
merged2 = cdf92_panel.merge(cdf90[['VCF0006a','VCF0301']], on='VCF0006a', suffixes=('','_lag'))
mask2 = (
    merged2['VCF0707'].isin([1.0, 2.0]) &
    merged2['VCF0301'].isin([1,2,3,4,5,6,7]) &
    merged2['VCF0301_lag'].isin([1,2,3,4,5,6,7])
)
df2 = merged2[mask2].copy()
df2['house_rep'] = (df2['VCF0707'] == 2.0).astype(int)
df2['strong_curr'] = np.where(df2['VCF0301']==7, 1, np.where(df2['VCF0301']==1, -1, 0))
df2['weak_curr'] = np.where(df2['VCF0301']==6, 1, np.where(df2['VCF0301']==2, -1, 0))
df2['lean_curr'] = np.where(df2['VCF0301']==5, 1, np.where(df2['VCF0301']==3, -1, 0))

X2 = sm.add_constant(df2[['strong_curr','weak_curr','lean_curr']])
mod2 = Probit(df2['house_rep'], X2).fit(disp=0)
print(f"N={len(df2)}, LL={mod2.llf:.1f}, R2={mod2.prsquared:.4f}")
print(f"Strong={mod2.params['strong_curr']:.3f}, Weak={mod2.params['weak_curr']:.3f}, Lean={mod2.params['lean_curr']:.3f}")

# Also check 1960 and 1976 with VCF0013
print("\n\n=== 1960 with VCF0013 ===")
cdf60 = cdf[cdf['VCF0004']==1960].copy()
cdf60_panel = cdf60[cdf60['VCF0006a'] < 19600000].copy()
print(f"VCF0013 for 1960 panel: {cdf60_panel['VCF0013'].value_counts().sort_index().to_dict()}")
panel60_marked = cdf60_panel[cdf60_panel['VCF0013']==1]
cdf58 = cdf[cdf['VCF0004']==1958].copy()
m60 = panel60_marked.merge(cdf58[['VCF0006a','VCF0301']], on='VCF0006a', suffixes=('','_lag'))
mask60 = m60['VCF0707'].isin([1.0,2.0]) & m60['VCF0301'].isin([1,2,3,4,5,6,7]) & m60['VCF0301_lag'].isin([1,2,3,4,5,6,7])
print(f"Valid: {mask60.sum()} (target 911)")

print("\n=== 1976 with VCF0013 ===")
cdf76 = cdf[cdf['VCF0004']==1976].copy()
cdf76_panel = cdf76[cdf76['VCF0006a'] < 19760000].copy()
print(f"VCF0013 for 1976 panel: {cdf76_panel['VCF0013'].value_counts().sort_index().to_dict()}")
panel76_marked = cdf76_panel[cdf76_panel['VCF0013']==1]
cdf74 = cdf[cdf['VCF0004']==1974].copy()
m76 = panel76_marked.merge(cdf74[['VCF0006a','VCF0301']], on='VCF0006a', suffixes=('','_lag'))
mask76 = m76['VCF0707'].isin([1.0,2.0]) & m76['VCF0301'].isin([1,2,3,4,5,6,7]) & m76['VCF0301_lag'].isin([1,2,3,4,5,6,7])
print(f"Valid: {mask76.sum()} (target 682)")
