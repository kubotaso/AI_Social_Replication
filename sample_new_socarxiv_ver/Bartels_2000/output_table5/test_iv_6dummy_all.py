"""Test 6-dummy IV vs 3-dummy IV for all three panels."""
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

def iv_3dummy(vote, X_c, X_l):
    Z = sm.add_constant(X_l)
    X_hat = pd.DataFrame(index=vote.index)
    for col in X_c.columns:
        X_hat[col] = sm.OLS(X_c[col], Z).fit().predict()
    return Probit(vote, sm.add_constant(X_hat)).fit(disp=0)

def iv_6dummy(vote, X_c, pid_lag_series):
    Z6 = pd.DataFrame(index=vote.index)
    for val in [1, 2, 3, 5, 6, 7]:
        Z6[f'd{val}'] = (pid_lag_series == val).astype(float)
    Z6c = sm.add_constant(Z6)
    X_hat = pd.DataFrame(index=vote.index)
    for col in X_c.columns:
        X_hat[col] = sm.OLS(X_c[col], Z6c).fit().predict()
    return Probit(vote, sm.add_constant(X_hat)).fit(disp=0)

def print_iv(label, model, gt):
    p = model.params
    print(f"  {label}: Strong={p['Strong']:.3f}(d={abs(p['Strong']-gt[0]):.3f}) "
          f"Weak={p['Weak']:.3f}(d={abs(p['Weak']-gt[1]):.3f}) "
          f"Lean={p['Lean']:.3f}(d={abs(p['Lean']-gt[2]):.3f}) "
          f"Int={p['const']:.3f}(d={abs(p['const']-gt[3]):.3f}) "
          f"TotalDist={abs(p['Strong']-gt[0])+abs(p['Weak']-gt[1])+abs(p['Lean']-gt[2])+abs(p['const']-gt[3]):.3f}")

# 1960
print("=== 1960 ===")
df60 = pd.read_csv(f'{BASE}/panel_1960.csv')
mask = df60['VCF0707'].isin([1.0, 2.0]) & df60['VCF0301'].isin(range(1,8)) & df60['VCF0301_lagged'].isin(range(1,8))
d = df60[mask].copy()
vote = (d['VCF0707'] == 2.0).astype(int)
s_c, w_c, l_c = construct_pid_dummies(d['VCF0301'])
s_l, w_l, l_l = construct_pid_dummies(d['VCF0301_lagged'])
X_c = pd.DataFrame({'Strong': s_c, 'Weak': w_c, 'Lean': l_c})
X_l = pd.DataFrame({'Strong': s_l, 'Weak': w_l, 'Lean': l_l})
gt = (1.715, 0.728, 1.081, 0.032)
print(f"  Target:  Strong={gt[0]} Weak={gt[1]} Lean={gt[2]} Int={gt[3]}")
iv3 = iv_3dummy(vote, X_c, X_l)
print_iv("3-dummy", iv3, gt)
iv6 = iv_6dummy(vote, X_c, d['VCF0301_lagged'])
print_iv("6-dummy", iv6, gt)

# 1976
print("\n=== 1976 ===")
df76 = pd.read_csv(f'{BASE}/panel_1976.csv')
mask = df76['VCF0707'].isin([1.0, 2.0]) & df76['VCF0301'].isin(range(1,8)) & df76['VCF0301_lagged'].isin(range(1,8))
d = df76[mask].copy()
vote = (d['VCF0707'] == 2.0).astype(int)
s_c, w_c, l_c = construct_pid_dummies(d['VCF0301'])
s_l, w_l, l_l = construct_pid_dummies(d['VCF0301_lagged'])
X_c = pd.DataFrame({'Strong': s_c, 'Weak': w_c, 'Lean': l_c})
X_l = pd.DataFrame({'Strong': s_l, 'Weak': w_l, 'Lean': l_l})
gt = (1.123, 0.745, 0.725, -0.102)
print(f"  Target:  Strong={gt[0]} Weak={gt[1]} Lean={gt[2]} Int={gt[3]}")
iv3 = iv_3dummy(vote, X_c, X_l)
print_iv("3-dummy", iv3, gt)
iv6 = iv_6dummy(vote, X_c, d['VCF0301_lagged'])
print_iv("6-dummy", iv6, gt)

# 1992
print("\n=== 1992 ===")
df92 = pd.read_csv(f'{BASE}/panel_1992.csv')
mask = df92['vote_house'].isin([1, 2]) & df92['pid_current'].isin(range(1,8)) & df92['pid_lagged'].isin(range(1,8))
d = df92[mask].copy()
vote = (d['vote_house'] == 2).astype(int)
s_c, w_c, l_c = construct_pid_dummies(d['pid_current'])
s_l, w_l, l_l = construct_pid_dummies(d['pid_lagged'])
X_c = pd.DataFrame({'Strong': s_c, 'Weak': w_c, 'Lean': l_c})
X_l = pd.DataFrame({'Strong': s_l, 'Weak': w_l, 'Lean': l_l})
gt = (1.516, -0.225, 1.824, -0.125)
print(f"  Target:  Strong={gt[0]} Weak={gt[1]} Lean={gt[2]} Int={gt[3]}")
iv3 = iv_3dummy(vote, X_c, X_l)
print_iv("3-dummy", iv3, gt)
iv6 = iv_6dummy(vote, X_c, d['pid_lagged'])
print_iv("6-dummy", iv6, gt)
