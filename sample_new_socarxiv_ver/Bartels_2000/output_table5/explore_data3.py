import pandas as pd
import numpy as np

cdf = pd.read_csv('anes_cumulative.csv', low_memory=False)
p92 = pd.read_csv('panel_1992.csv')

# CDF panel respondents in 1992
cdf92 = cdf[cdf['VCF0004']==1992].copy()
cdf92_panel = cdf92[cdf92['VCF0006a'] < 19920000].copy()
cdf92_panel = cdf92_panel.sort_values('VCF0006a').reset_index(drop=True)

print("CDF panel 1992:", len(cdf92_panel))
print("Panel_1992.csv:", len(p92))

# The CDF panel has 1359 respondents, panel_1992 has 1336
# Difference = 23
# The panel_1992.csv might have dropped some respondents who don't have valid 1990 PID

# Let's try matching by getting lagged PID from the CDF 1990 wave
cdf90 = cdf[cdf['VCF0004']==1990].copy()
print("CDF 1990:", len(cdf90))

# Match 1992 panel respondents to their 1990 entries
merged = cdf92_panel.merge(cdf90[['VCF0006a', 'VCF0301']], on='VCF0006a', suffixes=('_92', '_90'))
print("Matched 1992-1990:", len(merged))

# Check if VCF0301_90 (lagged PID from CDF) matches the panel_1992 pid_lagged
# We need to figure out which panel_1992 rows correspond to which CDF rows

# One approach: V923634 in panel_1992 is the raw 1992 PID (0-6 scale)
# where 0=Strong Dem, 1=Weak Dem, ..., 6=Strong Rep
# And pid_current = V923634 + 1 (mapping 0->1, 1->2, ..., 6->7)
# Same for V900320 (raw 1990 PID) and pid_lagged

# Check if V923634+1 = pid_current
print("\nV923634 mapping check:")
valid = p92[p92['V923634'].between(0,6)]
diff = valid['pid_current'] - (valid['V923634'] + 1)
print("Difference (should be all 0):", diff.abs().sum())

# V900320 mapping check
valid2 = p92[p92['V900320'].between(0,6)]
diff2 = valid2['pid_lagged'] - (valid2['V900320'] + 1)
print("V900320 mapping check (should be all 0):", diff2.abs().sum())

# Now - let's just use the CDF directly for 1992
# Get current PID from CDF 1992 (VCF0301_92) and lagged PID from CDF 1990 (VCF0301_90)
# And House vote from CDF 1992 (VCF0707)
print("\n=== Using CDF for 1992 panel ===")
print("Merged has", len(merged), "respondents with both 1992 and 1990 data")
print("VCF0301_92 (current PID):", merged['VCF0301_92'].value_counts().sort_index())
print("VCF0301_90 (lagged PID):", merged['VCF0301_90'].value_counts().sort_index())
print("VCF0707 (House vote):", merged['VCF0707'].value_counts().sort_index())

# Now let's check how many have valid House vote AND valid PIDs
valid_mask = (
    merged['VCF0707'].isin([1.0, 2.0]) &
    merged['VCF0301_92'].isin([1,2,3,4,5,6,7]) &
    merged['VCF0301_90'].isin([1,2,3,4,5,6,7])
)
print("\nValid for analysis:", valid_mask.sum())
print("Target N:", 760)

# Try a quick probit to see if results are in the right ballpark
from statsmodels.discrete.discrete_model import Probit
import statsmodels.api as sm

df = merged[valid_mask].copy()
# Dependent: House vote (1=Dem->0, 2=Rep->1)
df['house_rep'] = (df['VCF0707'] == 2.0).astype(int)

# Current PID variables
df['strong_curr'] = np.where(df['VCF0301_92']==7, 1, np.where(df['VCF0301_92']==1, -1, 0))
df['weak_curr'] = np.where(df['VCF0301_92']==6, 1, np.where(df['VCF0301_92']==2, -1, 0))
df['lean_curr'] = np.where(df['VCF0301_92']==5, 1, np.where(df['VCF0301_92']==3, -1, 0))

# Lagged PID variables
df['strong_lag'] = np.where(df['VCF0301_90']==7, 1, np.where(df['VCF0301_90']==1, -1, 0))
df['weak_lag'] = np.where(df['VCF0301_90']==6, 1, np.where(df['VCF0301_90']==2, -1, 0))
df['lean_lag'] = np.where(df['VCF0301_90']==5, 1, np.where(df['VCF0301_90']==3, -1, 0))

print("\n=== Current PID Probit ===")
X_curr = sm.add_constant(df[['strong_curr', 'weak_curr', 'lean_curr']])
mod_curr = Probit(df['house_rep'], X_curr).fit(disp=0)
print(mod_curr.summary2().tables[1])
print(f"LL: {mod_curr.llf:.1f}")
print(f"Pseudo-R2: {mod_curr.prsquared:.4f}")
print(f"N: {len(df)}")

print("\n=== Lagged PID Probit ===")
X_lag = sm.add_constant(df[['strong_lag', 'weak_lag', 'lean_lag']])
mod_lag = Probit(df['house_rep'], X_lag).fit(disp=0)
print(mod_lag.summary2().tables[1])
print(f"LL: {mod_lag.llf:.1f}")
print(f"Pseudo-R2: {mod_lag.prsquared:.4f}")
print(f"N: {len(df)}")

# Target values:
# Current: Strong=0.975(0.094), Weak=0.627(0.084), Lean=0.472(0.098), Int=-0.211(0.051), LL=-408.2, R2=0.20, N=760
# Lagged: Strong=1.061(0.100), Weak=0.404(0.077), Lean=0.519(0.101), Int=-0.168(0.051), LL=-416.2, R2=0.19, N=760
