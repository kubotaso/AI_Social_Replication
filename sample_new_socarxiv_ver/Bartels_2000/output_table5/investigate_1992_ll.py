"""Investigate 1992 LL gap: N=758 vs 760 but LL differs by ~14."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
import warnings
warnings.filterwarnings('ignore')

BASE = '/Users/administrator/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/Replication_Claude_Bartels_v2'

def construct_pid_dummies(pid_series):
    strong = pd.Series(0.0, index=pid_series.index)
    weak = pd.Series(0.0, index=pid_series.index)
    lean = pd.Series(0.0, index=pid_series.index)
    strong[pid_series == 7] = 1; strong[pid_series == 1] = -1
    weak[pid_series == 6] = 1; weak[pid_series == 2] = -1
    lean[pid_series == 5] = 1; lean[pid_series == 3] = -1
    return strong, weak, lean

# Load panel_1992
df92 = pd.read_csv(f'{BASE}/panel_1992.csv')
mask = (
    df92['vote_house'].isin([1, 2]) &
    df92['pid_current'].isin([1, 2, 3, 4, 5, 6, 7]) &
    df92['pid_lagged'].isin([1, 2, 3, 4, 5, 6, 7])
)
d92 = df92[mask].copy()
print(f"panel_1992 N: {len(d92)}")

# Check vote distribution
print(f"\nVote distribution (panel_1992):")
print(d92['vote_house'].value_counts().sort_index())
# 1=Dem, 2=Rep
vote = (d92['vote_house'] == 2).astype(int)
print(f"Dem={sum(vote==0)}, Rep={sum(vote==1)}")
print(f"Rep fraction: {sum(vote==1)/len(vote):.4f}")

# Check PID distribution
print(f"\nPID current distribution:")
print(d92['pid_current'].value_counts().sort_index())
print(f"\nPID lagged distribution:")
print(d92['pid_lagged'].value_counts().sort_index())

# Now compare with CDF
cdf = pd.read_csv(f'{BASE}/anes_cumulative.csv', low_memory=False)
cdf_curr = cdf[cdf['VCF0004'] == 1992].copy()
cdf_lag = cdf[cdf['VCF0004'] == 1990].copy()
panel = cdf_curr[cdf_curr['VCF0006a'] < 19920000].copy()
merged = panel.merge(cdf_lag[['VCF0006a', 'VCF0301']], on='VCF0006a', suffixes=('', '_lag'))
mask_cdf = merged['VCF0707'].isin([1.0, 2.0]) & merged['VCF0301'].isin(range(1,8)) & merged['VCF0301_lag'].isin(range(1,8))
d92_cdf = merged[mask_cdf].copy()
print(f"\nCDF N: {len(d92_cdf)}")
vote_cdf = (d92_cdf['VCF0707'] == 2.0).astype(int)
print(f"Dem={sum(vote_cdf==0)}, Rep={sum(vote_cdf==1)}")
print(f"Rep fraction: {sum(vote_cdf==1)/len(vote_cdf):.4f}")

print(f"\nCDF PID current distribution:")
print(d92_cdf['VCF0301'].value_counts().sort_index())
print(f"\nCDF PID lagged distribution:")
print(d92_cdf['VCF0301_lag'].value_counts().sort_index())

# Compare cross-tabulations
print("\n=== Panel_1992 vote x pid_current ===")
ct = pd.crosstab(d92['pid_current'], d92['vote_house'])
print(ct)

print("\n=== CDF vote x VCF0301 ===")
ct_cdf = pd.crosstab(d92_cdf['VCF0301'], d92_cdf['VCF0707'])
print(ct_cdf)

# Run probits and compare
s_c, w_c, l_c = construct_pid_dummies(d92['pid_current'])
X_c = pd.DataFrame({'Strong': s_c, 'Weak': w_c, 'Lean': l_c})
X_c_const = sm.add_constant(X_c)
mc_panel = Probit(vote, X_c_const).fit(disp=0)
print(f"\nPanel probit: LL={mc_panel.llf:.1f}, R2={mc_panel.prsquared:.4f}")

s_c, w_c, l_c = construct_pid_dummies(d92_cdf['VCF0301'])
X_c_cdf = pd.DataFrame({'Strong': s_c, 'Weak': w_c, 'Lean': l_c})
X_c_cdf_const = sm.add_constant(X_c_cdf)
mc_cdf = Probit(vote_cdf, X_c_cdf_const).fit(disp=0)
print(f"CDF probit:   LL={mc_cdf.llf:.1f}, R2={mc_cdf.prsquared:.4f}")
print(f"Target:       LL=-408.2, R2=0.20")

# The LL difference between our results and the target is about 14
# With N very close, this means the underlying data composition differs
# Let's see what fraction of responses differ between panel_1992 and CDF
# Check if they overlap at all
print("\n=== Checking overlap ===")
# Panel_1992 doesn't have respondent IDs, so we can't directly match
# But we can compare summary stats

# Null model LL (to understand the R2 calibration)
null_panel = Probit(vote, np.ones(len(vote))).fit(disp=0)
null_cdf = Probit(vote_cdf, np.ones(len(vote_cdf))).fit(disp=0)
print(f"\nNull LL panel: {null_panel.llf:.1f}")
print(f"Null LL CDF: {null_cdf.llf:.1f}")
print(f"Target null LL (from R2=0.20): {-408.2/(1-0.20):.1f}")

# The LL difference is really about different vote compositions
# Panel: more Dem votes than CDF
