"""Test different IV approaches for 1992 to see which gets closest to targets."""
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

df92 = pd.read_csv(f'{BASE}/panel_1992.csv')
mask = df92['vote_house'].isin([1, 2]) & df92['pid_current'].isin(range(1,8)) & df92['pid_lagged'].isin(range(1,8))
d = df92[mask].copy()
vote = (d['vote_house'] == 2).astype(int)
s_c, w_c, l_c = construct_pid_dummies(d['pid_current'])
s_l, w_l, l_l = construct_pid_dummies(d['pid_lagged'])

X_c = pd.DataFrame({'Strong': s_c, 'Weak': w_c, 'Lean': l_c})
X_l = pd.DataFrame({'Strong': s_l, 'Weak': w_l, 'Lean': l_l})

print("Target IV: Strong=1.516 Weak=-0.225 Lean=1.824 Int=-0.125")
print(f"N={len(d)}")

# Approach 1: 3-dummy directional IV (current)
Z = sm.add_constant(X_l)
X_hat = pd.DataFrame(index=d.index)
for col in X_c.columns:
    X_hat[col] = sm.OLS(X_c[col], Z).fit().predict()
X_hat_c = sm.add_constant(X_hat)
iv1 = Probit(vote, X_hat_c).fit(disp=0)
print(f"\n1. 3-dummy directional: Strong={iv1.params['Strong']:.3f} Weak={iv1.params['Weak']:.3f} Lean={iv1.params['Lean']:.3f} Int={iv1.params['const']:.3f}")

# Approach 2: 6-dummy IV (individual PID category dummies)
Z6 = pd.DataFrame(index=d.index)
for val in [1, 2, 3, 5, 6, 7]:
    Z6[f'd{val}'] = (d['pid_lagged'] == val).astype(float)
Z6c = sm.add_constant(Z6)
X_hat2 = pd.DataFrame(index=d.index)
for col in X_c.columns:
    X_hat2[col] = sm.OLS(X_c[col], Z6c).fit().predict()
X_hat2c = sm.add_constant(X_hat2)
iv2 = Probit(vote, X_hat2c).fit(disp=0)
print(f"2. 6-dummy IV:         Strong={iv2.params['Strong']:.3f} Weak={iv2.params['Weak']:.3f} Lean={iv2.params['Lean']:.3f} Int={iv2.params['const']:.3f}")

# Approach 3: Use V900320 (original lagged PID variable) as instruments
Z_orig = pd.DataFrame(index=d.index)
for val in [0, 1, 2, 3, 4, 5, 6]:
    Z_orig[f'd{val}'] = (d['V900320'] == val).astype(float)
# Drop one for identification (d4 = pure independent = ref)
Z_orig3 = Z_orig.drop(columns=['d4'])
Z_orig3c = sm.add_constant(Z_orig3)
X_hat3 = pd.DataFrame(index=d.index)
for col in X_c.columns:
    X_hat3[col] = sm.OLS(X_c[col], Z_orig3c).fit().predict()
X_hat3c = sm.add_constant(X_hat3)
try:
    iv3 = Probit(vote, X_hat3c).fit(disp=0)
    print(f"3. V900320 6-dummy:    Strong={iv3.params['Strong']:.3f} Weak={iv3.params['Weak']:.3f} Lean={iv3.params['Lean']:.3f} Int={iv3.params['const']:.3f}")
except:
    print("3. V900320 6-dummy: FAILED")

# Approach 4: Construct directional from V900320 (0-6 scale)
# V900320: 0=Strong Dem, 1=Weak Dem, 2=Lean Dem, 3=Ind, 4=Lean Rep, 5=Weak Rep, 6=Strong Rep
s_l2 = pd.Series(0.0, index=d.index)
w_l2 = pd.Series(0.0, index=d.index)
l_l2 = pd.Series(0.0, index=d.index)
s_l2[d['V900320'] == 6] = 1; s_l2[d['V900320'] == 0] = -1
w_l2[d['V900320'] == 5] = 1; w_l2[d['V900320'] == 1] = -1
l_l2[d['V900320'] == 4] = 1; l_l2[d['V900320'] == 2] = -1
X_l2 = pd.DataFrame({'Strong': s_l2, 'Weak': w_l2, 'Lean': l_l2})

Z4 = sm.add_constant(X_l2)
X_hat4 = pd.DataFrame(index=d.index)
for col in X_c.columns:
    X_hat4[col] = sm.OLS(X_c[col], Z4).fit().predict()
X_hat4c = sm.add_constant(X_hat4)
iv4 = Probit(vote, X_hat4c).fit(disp=0)
print(f"4. V900320 directional: Strong={iv4.params['Strong']:.3f} Weak={iv4.params['Weak']:.3f} Lean={iv4.params['Lean']:.3f} Int={iv4.params['const']:.3f}")

# Check the lagged PID using V900320 directional
ml_v9 = Probit(vote, sm.add_constant(X_l2)).fit(disp=0)
print(f"\nLagged (V900320): Strong={ml_v9.params['Strong']:.3f} Weak={ml_v9.params['Weak']:.3f} Lean={ml_v9.params['Lean']:.3f} Int={ml_v9.params['const']:.3f} LL={ml_v9.llf:.1f}")
print(f"Target lagged:    Strong=1.061 Weak=0.404 Lean=0.519 Int=-0.168 LL=-416.2")

# Also check pid_lagged directional
ml_pid = Probit(vote, sm.add_constant(X_l)).fit(disp=0)
print(f"Lagged (pid_lag): Strong={ml_pid.params['Strong']:.3f} Weak={ml_pid.params['Weak']:.3f} Lean={ml_pid.params['Lean']:.3f} Int={ml_pid.params['const']:.3f} LL={ml_pid.llf:.1f}")

# Check the V923634 for current PID
print(f"\nV923634 distribution:")
print(d['V923634'].value_counts().sort_index())
print(f"\npid_current distribution:")
print(d['pid_current'].value_counts().sort_index())

# Construct directional from V923634 (0-6 scale)
# V923634: 0=Strong Dem, 1=Weak Dem, 2=Lean Dem, 3=Ind, 4=Lean Rep, 5=Weak Rep, 6=Strong Rep
s_c2 = pd.Series(0.0, index=d.index)
w_c2 = pd.Series(0.0, index=d.index)
l_c2 = pd.Series(0.0, index=d.index)
s_c2[d['V923634'] == 6] = 1; s_c2[d['V923634'] == 0] = -1
w_c2[d['V923634'] == 5] = 1; w_c2[d['V923634'] == 1] = -1
l_c2[d['V923634'] == 4] = 1; l_c2[d['V923634'] == 2] = -1
X_c2 = pd.DataFrame({'Strong': s_c2, 'Weak': w_c2, 'Lean': l_c2})

mc_v9 = Probit(vote, sm.add_constant(X_c2)).fit(disp=0)
print(f"\nCurrent (V923634): Strong={mc_v9.params['Strong']:.3f} Weak={mc_v9.params['Weak']:.3f} Lean={mc_v9.params['Lean']:.3f} Int={mc_v9.params['const']:.3f} LL={mc_v9.llf:.1f}")
print(f"Target current:    Strong=0.975 Weak=0.627 Lean=0.472 Int=-0.211 LL=-408.2")

# IV with V923634 current and V900320 lagged
Z5 = sm.add_constant(X_l2)
X_hat5 = pd.DataFrame(index=d.index)
for col in X_c2.columns:
    X_hat5[col] = sm.OLS(X_c2[col], Z5).fit().predict()
X_hat5c = sm.add_constant(X_hat5)
iv5 = Probit(vote, X_hat5c).fit(disp=0)
print(f"\nIV (V923634/V900320): Strong={iv5.params['Strong']:.3f} Weak={iv5.params['Weak']:.3f} Lean={iv5.params['Lean']:.3f} Int={iv5.params['const']:.3f}")
print(f"Target IV:            Strong=1.516 Weak=-0.225 Lean=1.824 Int=-0.125")
