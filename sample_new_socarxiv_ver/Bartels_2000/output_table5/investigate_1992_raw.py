"""Investigate 1992 panel raw variables V923634, V900320, V925609, V925623."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
import warnings
warnings.filterwarnings('ignore')

p92 = pd.read_csv('panel_1992.csv', low_memory=False)

print('Columns:', list(p92.columns))
print(f'Total: {len(p92)}')

# V923634 - might be House vote from 1992 study
print('\nV923634 distribution:')
print(p92['V923634'].value_counts().sort_index())

# V900320 - might be PID from 1990 study
print('\nV900320 distribution:')
print(p92['V900320'].value_counts().sort_index())

# V925609 - ?
print('\nV925609 distribution:')
print(p92['V925609'].value_counts().sort_index())

# V925623 - ?
print('\nV925623 distribution:')
print(p92['V925623'].value_counts().sort_index())

# Cross-check V900320 vs pid_lagged
print('\nV900320 vs pid_lagged cross-tab:')
ct = pd.crosstab(p92['V900320'], p92['pid_lagged'])
print(ct)

# Cross-check V923634 vs vote_house
print('\nV923634 vs vote_house cross-tab:')
ct2 = pd.crosstab(p92['V923634'].fillna(-1), p92['vote_house'].fillna(-1))
print(ct2)

# Check V925609 and V925623
# These might be alternative vote or PID variables
print('\nV925609 vs pid_current:')
ct3 = pd.crosstab(p92['V925609'], p92['pid_current'].fillna(-1))
print(ct3)

print('\nV925623 vs vote_house:')
ct4 = pd.crosstab(p92['V925623'], p92['vote_house'].fillna(-1))
print(ct4)

# Try using V923634 as House vote (different coding?)
# V923634: 1 and 2 values
valid_v = p92[
    p92['V923634'].isin([1, 2]) &
    p92['pid_current'].isin([1,2,3,4,5,6,7]) &
    p92['pid_lagged'].isin([1,2,3,4,5,6,7])
].copy()
print(f'\nV923634 in [1,2] + valid PIDs: N={len(valid_v)}')

for vote_code in ['1=Dem 2=Rep', '1=Rep 2=Dem']:
    v = valid_v.copy()
    if vote_code == '1=Dem 2=Rep':
        v['house_rep'] = (v['V923634'] == 2).astype(int)
    else:
        v['house_rep'] = (v['V923634'] == 1).astype(int)

    v['strong_curr'] = np.where(v['pid_current'] == 7, 1, np.where(v['pid_current'] == 1, -1, 0))
    v['weak_curr'] = np.where(v['pid_current'] == 6, 1, np.where(v['pid_current'] == 2, -1, 0))
    v['lean_curr'] = np.where(v['pid_current'] == 5, 1, np.where(v['pid_current'] == 3, -1, 0))

    X = sm.add_constant(v[['strong_curr','weak_curr','lean_curr']])
    mod = Probit(v['house_rep'].astype(float), X).fit(disp=0)
    print(f'\n  {vote_code}: N={len(v)}, LL={mod.llf:.1f}, R2={mod.prsquared:.4f}')
    print(f'    Strong={mod.params["strong_curr"]:.3f} Weak={mod.params["weak_curr"]:.3f} Lean={mod.params["lean_curr"]:.3f} Int={mod.params["const"]:.3f}')
    print(f'  Target: N=760, LL=-408.2, R2=0.20, Strong=0.975, Weak=0.627, Lean=0.472, Int=-0.211')
