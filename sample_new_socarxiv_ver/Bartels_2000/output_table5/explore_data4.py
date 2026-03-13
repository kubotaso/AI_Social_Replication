import pandas as pd
import numpy as np
from statsmodels.discrete.discrete_model import Probit
import statsmodels.api as sm

# Test 1960 panel
df60 = pd.read_csv('panel_1960.csv')
valid60 = df60[
    df60['VCF0707'].isin([1.0, 2.0]) &
    df60['VCF0301'].isin([1,2,3,4,5,6,7]) &
    df60['VCF0301_lagged'].isin([1,2,3,4,5,6,7])
].copy()

valid60['house_rep'] = (valid60['VCF0707'] == 2.0).astype(int)
valid60['strong_curr'] = np.where(valid60['VCF0301']==7, 1, np.where(valid60['VCF0301']==1, -1, 0))
valid60['weak_curr'] = np.where(valid60['VCF0301']==6, 1, np.where(valid60['VCF0301']==2, -1, 0))
valid60['lean_curr'] = np.where(valid60['VCF0301']==5, 1, np.where(valid60['VCF0301']==3, -1, 0))
valid60['strong_lag'] = np.where(valid60['VCF0301_lagged']==7, 1, np.where(valid60['VCF0301_lagged']==1, -1, 0))
valid60['weak_lag'] = np.where(valid60['VCF0301_lagged']==6, 1, np.where(valid60['VCF0301_lagged']==2, -1, 0))
valid60['lean_lag'] = np.where(valid60['VCF0301_lagged']==5, 1, np.where(valid60['VCF0301_lagged']==3, -1, 0))

print("=== 1960 Panel ===")
print(f"N: {len(valid60)} (target: 911)")

X = sm.add_constant(valid60[['strong_curr', 'weak_curr', 'lean_curr']])
mod = Probit(valid60['house_rep'], X).fit(disp=0)
print(f"Current PID: Strong={mod.params['strong_curr']:.3f}, Weak={mod.params['weak_curr']:.3f}, Lean={mod.params['lean_curr']:.3f}, Int={mod.params['const']:.3f}")
print(f"LL: {mod.llf:.1f}, R2: {mod.prsquared:.4f}")
# Target: Strong=1.358, Weak=1.028, Lean=0.855, Int=0.035, LL=-372.7, R2=0.41

X2 = sm.add_constant(valid60[['strong_lag', 'weak_lag', 'lean_lag']])
mod2 = Probit(valid60['house_rep'], X2).fit(disp=0)
print(f"Lagged PID: Strong={mod2.params['strong_lag']:.3f}, Weak={mod2.params['weak_lag']:.3f}, Lean={mod2.params['lean_lag']:.3f}, Int={mod2.params['const']:.3f}")
print(f"LL: {mod2.llf:.1f}, R2: {mod2.prsquared:.4f}")
# Target: Strong=1.363, Weak=0.842, Lean=0.564, Int=0.068, LL=-403.9, R2=0.36

print("\n=== 1976 Panel ===")
df76 = pd.read_csv('panel_1976.csv')
valid76 = df76[
    df76['VCF0707'].isin([1.0, 2.0]) &
    df76['VCF0301'].isin([1,2,3,4,5,6,7]) &
    df76['VCF0301_lagged'].isin([1,2,3,4,5,6,7])
].copy()

valid76['house_rep'] = (valid76['VCF0707'] == 2.0).astype(int)
valid76['strong_curr'] = np.where(valid76['VCF0301']==7, 1, np.where(valid76['VCF0301']==1, -1, 0))
valid76['weak_curr'] = np.where(valid76['VCF0301']==6, 1, np.where(valid76['VCF0301']==2, -1, 0))
valid76['lean_curr'] = np.where(valid76['VCF0301']==5, 1, np.where(valid76['VCF0301']==3, -1, 0))
valid76['strong_lag'] = np.where(valid76['VCF0301_lagged']==7, 1, np.where(valid76['VCF0301_lagged']==1, -1, 0))
valid76['weak_lag'] = np.where(valid76['VCF0301_lagged']==6, 1, np.where(valid76['VCF0301_lagged']==2, -1, 0))
valid76['lean_lag'] = np.where(valid76['VCF0301_lagged']==5, 1, np.where(valid76['VCF0301_lagged']==3, -1, 0))

print(f"N: {len(valid76)} (target: 682)")

X = sm.add_constant(valid76[['strong_curr', 'weak_curr', 'lean_curr']])
mod = Probit(valid76['house_rep'], X).fit(disp=0)
print(f"Current PID: Strong={mod.params['strong_curr']:.3f}, Weak={mod.params['weak_curr']:.3f}, Lean={mod.params['lean_curr']:.3f}, Int={mod.params['const']:.3f}")
print(f"LL: {mod.llf:.1f}, R2: {mod.prsquared:.4f}")
# Target: Strong=1.087, Weak=0.624, Lean=0.622, Int=-0.123, LL=-358.2, R2=0.24

X2 = sm.add_constant(valid76[['strong_lag', 'weak_lag', 'lean_lag']])
mod2 = Probit(valid76['house_rep'], X2).fit(disp=0)
print(f"Lagged PID: Strong={mod2.params['strong_lag']:.3f}, Weak={mod2.params['weak_lag']:.3f}, Lean={mod2.params['lean_lag']:.3f}, Int={mod2.params['const']:.3f}")
print(f"LL: {mod2.llf:.1f}, R2: {mod2.prsquared:.4f}")
# Target: Strong=0.966, Weak=0.738, Lean=0.486, Int=-0.063, LL=-371.3, R2=0.21
