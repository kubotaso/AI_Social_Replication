"""Investigate panel files for Table 5 replication (congressional vote)."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
import warnings
warnings.filterwarnings('ignore')

BASE = '/Users/administrator/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/Replication_Claude_Bartels_v2'

# Check panel files for House vote variables
print("=" * 70)
print("PANEL FILE ANALYSIS FOR TABLE 5")
print("=" * 70)

# 1960 panel
df60 = pd.read_csv(f'{BASE}/panel_1960.csv')
print(f"\npanel_1960: {df60.shape}")
vote_cols = [c for c in df60.columns if any(s in c.upper() for s in ['VOTE', '0704', '0706', '0707', 'VCF07'])]
print(f"Vote columns: {vote_cols[:30]}")
if 'VCF0707' in df60.columns:
    print(f"\nVCF0707 (House vote) for 1960:")
    print(df60['VCF0707'].value_counts(dropna=False).sort_index())
if 'VCF0706' in df60.columns:
    print(f"\nVCF0706 (House vote intent) for 1960:")
    print(df60['VCF0706'].value_counts(dropna=False).sort_index())

# 1976 panel
df76 = pd.read_csv(f'{BASE}/panel_1976.csv')
print(f"\n{'='*70}")
print(f"panel_1976: {df76.shape}")
if 'VCF0707' in df76.columns:
    print(f"\nVCF0707 for 1976:")
    print(df76['VCF0707'].value_counts(dropna=False).sort_index())
if 'VCF0706' in df76.columns:
    print(f"\nVCF0706 for 1976:")
    print(df76['VCF0706'].value_counts(dropna=False).sort_index())

# 1992 panel
df92 = pd.read_csv(f'{BASE}/panel_1992.csv')
print(f"\n{'='*70}")
print(f"panel_1992: {df92.shape}")
print(f"Columns: {list(df92.columns)}")
print(f"\nvote_house distribution:")
print(df92['vote_house'].value_counts(dropna=False).sort_index())
print(f"\npid_current distribution:")
print(df92['pid_current'].value_counts(dropna=False).sort_index())
print(f"\npid_lagged distribution:")
print(df92['pid_lagged'].value_counts(dropna=False).sort_index())

# Now try running Table 5 using panel files
print("\n" + "=" * 70)
print("TABLE 5 REPLICATION USING PANEL FILES")
print("=" * 70)

def construct_pid_dummies(pid_series):
    strong = pd.Series(0.0, index=pid_series.index)
    weak = pd.Series(0.0, index=pid_series.index)
    lean = pd.Series(0.0, index=pid_series.index)
    strong[pid_series == 7] = 1; strong[pid_series == 1] = -1
    weak[pid_series == 6] = 1; weak[pid_series == 2] = -1
    lean[pid_series == 5] = 1; lean[pid_series == 3] = -1
    return strong, weak, lean

def run_probit(y, X):
    X = sm.add_constant(X)
    return Probit(y, X).fit(disp=0, maxiter=1000)

def run_iv_probit(y, X_endog, Z_instruments):
    Z_with_const = sm.add_constant(Z_instruments)
    X_hat = pd.DataFrame(index=X_endog.index)
    for col in X_endog.columns:
        ols_result = sm.OLS(X_endog[col], Z_with_const).fit()
        X_hat[col] = ols_result.predict()
    X_hat_with_const = sm.add_constant(X_hat)
    return Probit(y, X_hat_with_const).fit(disp=0, maxiter=1000)

# Process 1960 panel
print("\n--- 1960 Panel ---")
# Use VCF0707 as House vote, VCF0301 as current PID, VCF0301_lagged as lagged PID
mask60 = (
    df60['VCF0707'].isin([1.0, 2.0]) &
    df60['VCF0301'].isin([1, 2, 3, 4, 5, 6, 7]) &
    df60['VCF0301_lagged'].isin([1, 2, 3, 4, 5, 6, 7])
)
d60 = df60[mask60].copy()
print(f"N (VCF0707 only): {len(d60)}")

# Also try union
d60u = df60.copy()
d60u['house_vote'] = d60u['VCF0707']
mask_fill = d60u['house_vote'].isna() & d60u['VCF0706'].isin([1.0, 2.0])
d60u.loc[mask_fill, 'house_vote'] = d60u.loc[mask_fill, 'VCF0706']
mask60u = (
    d60u['house_vote'].isin([1.0, 2.0]) &
    d60u['VCF0301'].isin([1, 2, 3, 4, 5, 6, 7]) &
    d60u['VCF0301_lagged'].isin([1, 2, 3, 4, 5, 6, 7])
)
d60u = d60u[mask60u].copy()
print(f"N (union): {len(d60u)}")

# Run probits for 1960 with VCF0707 only (no weight expansion)
vote60 = (d60['VCF0707'] == 2.0).astype(int)
s_c, w_c, l_c = construct_pid_dummies(d60['VCF0301'])
s_l, w_l, l_l = construct_pid_dummies(d60['VCF0301_lagged'])
X_c60 = pd.DataFrame({'Strong': s_c, 'Weak': w_c, 'Lean': l_c})
X_l60 = pd.DataFrame({'Strong': s_l, 'Weak': w_l, 'Lean': l_l})

mc60 = run_probit(vote60, X_c60)
ml60 = run_probit(vote60, X_l60)
mi60 = run_iv_probit(vote60, X_c60, X_l60)

print(f"\nCurrent PID: Strong={mc60.params['Strong']:.3f}({mc60.bse['Strong']:.3f}) "
      f"Weak={mc60.params['Weak']:.3f}({mc60.bse['Weak']:.3f}) "
      f"Lean={mc60.params['Lean']:.3f}({mc60.bse['Lean']:.3f}) "
      f"Int={mc60.params['const']:.3f}({mc60.bse['const']:.3f}) "
      f"LL={mc60.llf:.1f} R2={mc60.prsquared:.4f}")
print(f"Target:     Strong=1.358(0.094) Weak=1.028(0.083) Lean=0.855(0.131) Int=0.035(0.053) LL=-372.7 R2=0.41")

print(f"\nLagged PID: Strong={ml60.params['Strong']:.3f}({ml60.bse['Strong']:.3f}) "
      f"Weak={ml60.params['Weak']:.3f}({ml60.bse['Weak']:.3f}) "
      f"Lean={ml60.params['Lean']:.3f}({ml60.bse['Lean']:.3f}) "
      f"Int={ml60.params['const']:.3f}({ml60.bse['const']:.3f}) "
      f"LL={ml60.llf:.1f} R2={ml60.prsquared:.4f}")
print(f"Target:     Strong=1.363(0.092) Weak=0.842(0.078) Lean=0.564(0.125) Int=0.068(0.051) LL=-403.9 R2=0.36")

print(f"\nIV:         Strong={mi60.params['Strong']:.3f}({mi60.bse['Strong']:.3f}) "
      f"Weak={mi60.params['Weak']:.3f}({mi60.bse['Weak']:.3f}) "
      f"Lean={mi60.params['Lean']:.3f}({mi60.bse['Lean']:.3f}) "
      f"Int={mi60.params['const']:.3f}({mi60.bse['const']:.3f}) "
      f"LL={mi60.llf:.1f} R2={mi60.prsquared:.4f}")
print(f"Target:     Strong=1.715(0.173) Weak=0.728(0.239) Lean=1.081(0.696) Int=0.032(0.057) LL=-403.9 R2=0.36")

# Process 1976 panel
print(f"\n{'='*70}")
print("--- 1976 Panel ---")
mask76 = (
    df76['VCF0707'].isin([1.0, 2.0]) &
    df76['VCF0301'].isin([1, 2, 3, 4, 5, 6, 7]) &
    df76['VCF0301_lagged'].isin([1, 2, 3, 4, 5, 6, 7])
)
d76 = df76[mask76].copy()
print(f"N: {len(d76)}")

vote76 = (d76['VCF0707'] == 2.0).astype(int)
s_c, w_c, l_c = construct_pid_dummies(d76['VCF0301'])
s_l, w_l, l_l = construct_pid_dummies(d76['VCF0301_lagged'])
X_c76 = pd.DataFrame({'Strong': s_c, 'Weak': w_c, 'Lean': l_c})
X_l76 = pd.DataFrame({'Strong': s_l, 'Weak': w_l, 'Lean': l_l})

mc76 = run_probit(vote76, X_c76)
ml76 = run_probit(vote76, X_l76)
mi76 = run_iv_probit(vote76, X_c76, X_l76)

print(f"\nCurrent PID: Strong={mc76.params['Strong']:.3f}({mc76.bse['Strong']:.3f}) "
      f"Weak={mc76.params['Weak']:.3f}({mc76.bse['Weak']:.3f}) "
      f"Lean={mc76.params['Lean']:.3f}({mc76.bse['Lean']:.3f}) "
      f"Int={mc76.params['const']:.3f}({mc76.bse['const']:.3f}) "
      f"LL={mc76.llf:.1f} R2={mc76.prsquared:.4f}")
print(f"Target:     Strong=1.087(0.105) Weak=0.624(0.086) Lean=0.622(0.110) Int=-0.123(0.054) LL=-358.2 R2=0.24")

print(f"\nLagged PID: Strong={ml76.params['Strong']:.3f}({ml76.bse['Strong']:.3f}) "
      f"Weak={ml76.params['Weak']:.3f}({ml76.bse['Weak']:.3f}) "
      f"Lean={ml76.params['Lean']:.3f}({ml76.bse['Lean']:.3f}) "
      f"Int={ml76.params['const']:.3f}({ml76.bse['const']:.3f}) "
      f"LL={ml76.llf:.1f} R2={ml76.prsquared:.4f}")
print(f"Target:     Strong=0.966(0.104) Weak=0.738(0.089) Lean=0.486(0.109) Int=-0.063(0.053) LL=-371.3 R2=0.21")

print(f"\nIV:         Strong={mi76.params['Strong']:.3f}({mi76.bse['Strong']:.3f}) "
      f"Weak={mi76.params['Weak']:.3f}({mi76.bse['Weak']:.3f}) "
      f"Lean={mi76.params['Lean']:.3f}({mi76.bse['Lean']:.3f}) "
      f"Int={mi76.params['const']:.3f}({mi76.bse['const']:.3f}) "
      f"LL={mi76.llf:.1f} R2={mi76.prsquared:.4f}")
print(f"Target:     Strong=1.123(0.178) Weak=0.745(0.251) Lean=0.725(0.438) Int=-0.102(0.055) LL=-371.3 R2=0.21")

# Process 1992 panel (using vote_house from panel_1992.csv)
print(f"\n{'='*70}")
print("--- 1992 Panel ---")
mask92 = (
    df92['vote_house'].isin([1, 2]) &
    df92['pid_current'].isin([1, 2, 3, 4, 5, 6, 7]) &
    df92['pid_lagged'].isin([1, 2, 3, 4, 5, 6, 7])
)
d92 = df92[mask92].copy()
print(f"N: {len(d92)}")

# Try vote_house: 1=Dem, 2=Rep (same as VCF0707)
vote92 = (d92['vote_house'] == 2).astype(int)
s_c, w_c, l_c = construct_pid_dummies(d92['pid_current'])
s_l, w_l, l_l = construct_pid_dummies(d92['pid_lagged'])
X_c92 = pd.DataFrame({'Strong': s_c, 'Weak': w_c, 'Lean': l_c})
X_l92 = pd.DataFrame({'Strong': s_l, 'Weak': w_l, 'Lean': l_l})

mc92 = run_probit(vote92, X_c92)
ml92 = run_probit(vote92, X_l92)
mi92 = run_iv_probit(vote92, X_c92, X_l92)

print(f"\nCurrent PID: Strong={mc92.params['Strong']:.3f}({mc92.bse['Strong']:.3f}) "
      f"Weak={mc92.params['Weak']:.3f}({mc92.bse['Weak']:.3f}) "
      f"Lean={mc92.params['Lean']:.3f}({mc92.bse['Lean']:.3f}) "
      f"Int={mc92.params['const']:.3f}({mc92.bse['const']:.3f}) "
      f"LL={mc92.llf:.1f} R2={mc92.prsquared:.4f}")
print(f"Target:     Strong=0.975(0.094) Weak=0.627(0.084) Lean=0.472(0.098) Int=-0.211(0.051) LL=-408.2 R2=0.20")

print(f"\nLagged PID: Strong={ml92.params['Strong']:.3f}({ml92.bse['Strong']:.3f}) "
      f"Weak={ml92.params['Weak']:.3f}({ml92.bse['Weak']:.3f}) "
      f"Lean={ml92.params['Lean']:.3f}({ml92.bse['Lean']:.3f}) "
      f"Int={ml92.params['const']:.3f}({ml92.bse['const']:.3f}) "
      f"LL={ml92.llf:.1f} R2={ml92.prsquared:.4f}")
print(f"Target:     Strong=1.061(0.100) Weak=0.404(0.077) Lean=0.519(0.101) Int=-0.168(0.051) LL=-416.2 R2=0.19")

print(f"\nIV:         Strong={mi92.params['Strong']:.3f}({mi92.bse['Strong']:.3f}) "
      f"Weak={mi92.params['Weak']:.3f}({mi92.bse['Weak']:.3f}) "
      f"Lean={mi92.params['Lean']:.3f}({mi92.bse['Lean']:.3f}) "
      f"Int={mi92.params['const']:.3f}({mi92.bse['const']:.3f}) "
      f"LL={mi92.llf:.1f} R2={mi92.prsquared:.4f}")
print(f"Target:     Strong=1.516(0.180) Weak=-0.225(0.268) Lean=1.824(0.513) Int=-0.125(0.053) LL=-416.2 R2=0.19")

# Also try vote_house == 1 as Rep (reversed coding)
print(f"\n--- 1992 Panel (reversed vote coding: 1=Rep) ---")
vote92r = (d92['vote_house'] == 1).astype(int)
mc92r = run_probit(vote92r, X_c92)
ml92r = run_probit(vote92r, X_l92)
mi92r = run_iv_probit(vote92r, X_c92, X_l92)

print(f"Current PID: Strong={mc92r.params['Strong']:.3f}({mc92r.bse['Strong']:.3f}) "
      f"Weak={mc92r.params['Weak']:.3f}({mc92r.bse['Weak']:.3f}) "
      f"Lean={mc92r.params['Lean']:.3f}({mc92r.bse['Lean']:.3f}) "
      f"Int={mc92r.params['const']:.3f}({mc92r.bse['const']:.3f}) "
      f"LL={mc92r.llf:.1f} R2={mc92r.prsquared:.4f}")

# Try using V925623 directly for 1992
print(f"\n--- 1992 Panel (V925623) ---")
mask92v = (
    df92['V925623'].isin([1, 2]) &
    df92['pid_current'].isin([1, 2, 3, 4, 5, 6, 7]) &
    df92['pid_lagged'].isin([1, 2, 3, 4, 5, 6, 7])
)
d92v = df92[mask92v].copy()
print(f"N: {len(d92v)}")
vote92v = (d92v['V925623'] == 2).astype(int)
s_c, w_c, l_c = construct_pid_dummies(d92v['pid_current'])
X_c92v = pd.DataFrame({'Strong': s_c, 'Weak': w_c, 'Lean': l_c})
mc92v = run_probit(vote92v, X_c92v)
print(f"Current PID: Strong={mc92v.params['Strong']:.3f}({mc92v.bse['Strong']:.3f}) "
      f"Weak={mc92v.params['Weak']:.3f}({mc92v.bse['Weak']:.3f}) "
      f"Lean={mc92v.params['Lean']:.3f}({mc92v.bse['Lean']:.3f}) "
      f"Int={mc92v.params['const']:.3f}({mc92v.bse['const']:.3f}) "
      f"LL={mc92v.llf:.1f} R2={mc92v.prsquared:.4f}")

# Also try V925623 = 1 as Rep
print(f"\n--- 1992 Panel (V925623, 1=Rep) ---")
vote92v1 = (d92v['V925623'] == 1).astype(int)
mc92v1 = run_probit(vote92v1, X_c92v)
print(f"Current PID: Strong={mc92v1.params['Strong']:.3f}({mc92v1.bse['Strong']:.3f}) "
      f"Weak={mc92v1.params['Weak']:.3f}({mc92v1.bse['Weak']:.3f}) "
      f"Lean={mc92v1.params['Lean']:.3f}({mc92v1.bse['Lean']:.3f}) "
      f"Int={mc92v1.params['const']:.3f}({mc92v1.bse['const']:.3f}) "
      f"LL={mc92v1.llf:.1f} R2={mc92v1.prsquared:.4f}")
