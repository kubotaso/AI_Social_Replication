"""Investigate 1976 weight expansion - weights are 1.0 and 1.5, not all 1.0."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
import warnings
warnings.filterwarnings('ignore')

cdf = pd.read_csv('anes_cumulative.csv', low_memory=False)

def construct_pid_vars(df, pid_col, suffix):
    df[f'strong_{suffix}'] = np.where(df[pid_col] == 7, 1, np.where(df[pid_col] == 1, -1, 0))
    df[f'weak_{suffix}'] = np.where(df[pid_col] == 6, 1, np.where(df[pid_col] == 2, -1, 0))
    df[f'lean_{suffix}'] = np.where(df[pid_col] == 5, 1, np.where(df[pid_col] == 3, -1, 0))
    return df

cdf76 = cdf[cdf['VCF0004']==1976].copy()
cdf74 = cdf[cdf['VCF0004']==1974].copy()
panel76 = cdf76[cdf76['VCF0006a'] < 19760000].copy()
merged76 = panel76.merge(cdf74[['VCF0006a','VCF0301']], on='VCF0006a', suffixes=('','_lag'))

print(f'Panel respondents: {len(merged76)}')
wt = merged76['VCF0009x'].fillna(1.0)
print(f'Weight distribution in panel:')
print(wt.value_counts().sort_index())
print(f'Sum of weights: {wt.sum():.1f}')

# Try different weight rounding for expansion
# With int truncation: 1.5 -> 1
wt_int = wt.astype(int)
print(f'\nInt truncation: sum={wt_int.sum()} (all become 1)')

# With rounding: 1.5 -> 2
wt_round = wt.round().astype(int)
print(f'Rounding: sum={wt_round.sum()}')
# Count how many 1.5 -> 2
n_15 = (wt == 1.5).sum()
print(f'  Respondents with wt=1.5: {n_15}')
print(f'  Expected expanded N with round: {len(merged76) + n_15}')

# Try all approaches for 1976: union vs 707, expanded vs not
targets_76 = {'strong': 1.087, 'weak': 0.624, 'lean': 0.622, 'int': -0.123,
              'N': 682, 'LL': -358.2, 'R2': 0.24}
targets_76_lag = {'strong': 0.966, 'weak': 0.738, 'lean': 0.486, 'int': -0.063,
                  'LL': -371.3, 'R2': 0.21}

approaches = []

for vote_label, use_union in [('VCF0707', False), ('Union', True)]:
    for wt_label, expand in [('No expand', False), ('Expand round', True)]:
        m76 = merged76.copy()

        if expand:
            wt_r = m76['VCF0009x'].fillna(1.0).round().astype(int)
            m76 = m76.loc[m76.index.repeat(wt_r)].reset_index(drop=True)

        if use_union:
            m76['house_vote'] = m76['VCF0707']
            mask_fill = m76['house_vote'].isna() & m76['VCF0706'].isin([1.0, 2.0])
            m76.loc[mask_fill, 'house_vote'] = m76.loc[mask_fill, 'VCF0706']
        else:
            m76['house_vote'] = m76['VCF0707']

        valid = m76[
            m76['house_vote'].isin([1.0, 2.0]) &
            m76['VCF0301'].isin([1,2,3,4,5,6,7]) &
            m76['VCF0301_lag'].isin([1,2,3,4,5,6,7])
        ].copy()

        valid['house_rep'] = (valid['house_vote'] == 2.0).astype(int)
        valid = construct_pid_vars(valid, 'VCF0301', 'curr')
        valid = construct_pid_vars(valid, 'VCF0301_lag', 'lag')

        # Current PID
        X = sm.add_constant(valid[['strong_curr','weak_curr','lean_curr']])
        mod = Probit(valid['house_rep'].astype(float), X).fit(disp=0)

        # Lagged PID
        X_lag = sm.add_constant(valid[['strong_lag','weak_lag','lean_lag']])
        mod_lag = Probit(valid['house_rep'].astype(float), X_lag).fit(disp=0)

        t = targets_76
        coef_dist = abs(mod.params['strong_curr'] - t['strong']) + \
                    abs(mod.params['weak_curr'] - t['weak']) + \
                    abs(mod.params['lean_curr'] - t['lean']) + \
                    abs(mod.params['const'] - t['int'])
        n_pct = abs(len(valid) - t['N']) / t['N']
        ll_diff = abs(mod.llf - t['LL'])
        r2_diff = abs(mod.prsquared - t['R2'])

        tl = targets_76_lag
        coef_dist_lag = abs(mod_lag.params['strong_lag'] - tl['strong']) + \
                        abs(mod_lag.params['weak_lag'] - tl['weak']) + \
                        abs(mod_lag.params['lean_lag'] - tl['lean']) + \
                        abs(mod_lag.params['const'] - tl['int'])

        label = f'{vote_label} + {wt_label}'
        print(f'\n{label}: N={len(valid)}')
        print(f'  Current: Strong={mod.params["strong_curr"]:.3f} Weak={mod.params["weak_curr"]:.3f} Lean={mod.params["lean_curr"]:.3f} Int={mod.params["const"]:.3f}')
        print(f'  Current: LL={mod.llf:.1f} R2={mod.prsquared:.4f}')
        print(f'  Current coef_dist={coef_dist:.3f}, LL_diff={ll_diff:.1f}, R2_diff={r2_diff:.4f}, N_diff={n_pct:.1%}')
        print(f'  Lagged: Strong={mod_lag.params["strong_lag"]:.3f} Weak={mod_lag.params["weak_lag"]:.3f} Lean={mod_lag.params["lean_lag"]:.3f} Int={mod_lag.params["const"]:.3f}')
        print(f'  Lagged: LL={mod_lag.llf:.1f} R2={mod_lag.prsquared:.4f}')
        print(f'  Lagged coef_dist={coef_dist_lag:.3f}')

        # Combined score proxy (lower is better)
        combined = coef_dist + coef_dist_lag + n_pct * 10 + r2_diff * 10
        print(f'  Combined metric (lower=better): {combined:.3f}')
        approaches.append((label, combined, len(valid)))

print('\n=== RANKING ===')
for label, score, n in sorted(approaches, key=lambda x: x[1]):
    print(f'  {label}: {score:.3f} (N={n})')

# Also check: what N do we get with VCF0707 expanded by round weights?
print('\n=== VCF0707 + Expand round detailed ===')
m76_test = merged76.copy()
wt_r = m76_test['VCF0009x'].fillna(1.0).round().astype(int)
m76_test = m76_test.loc[m76_test.index.repeat(wt_r)].reset_index(drop=True)
v707_test = m76_test[m76_test['VCF0707'].isin([1.0, 2.0])]
print(f'VCF0707 expanded: {len(v707_test)} (before PID filter)')
v707_valid = m76_test[
    m76_test['VCF0707'].isin([1.0, 2.0]) &
    m76_test['VCF0301'].isin([1,2,3,4,5,6,7]) &
    m76_test['VCF0301_lag'].isin([1,2,3,4,5,6,7])
]
print(f'VCF0707 expanded + valid PID: {len(v707_valid)}')
print(f'Target N: 682, diff: {abs(len(v707_valid)-682)/682:.1%}')
