"""Test alternative House vote variables VCF0702, VCF0704, VCF0704a."""
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

targets = {
    '1960': {'strong': 1.358, 'weak': 1.028, 'lean': 0.855, 'int': 0.035, 'N': 911, 'LL': -372.7, 'R2': 0.41},
    '1976': {'strong': 1.087, 'weak': 0.624, 'lean': 0.622, 'int': -0.123, 'N': 682, 'LL': -358.2, 'R2': 0.24},
    '1992': {'strong': 0.975, 'weak': 0.627, 'lean': 0.472, 'int': -0.211, 'N': 760, 'LL': -408.2, 'R2': 0.20},
}

# VCF0702: Did R vote for House candidate? 1=yes, 2=no
# VCF0704: 2-party House vote: 1=Dem, 2=Rep
# VCF0704a: 2-party House vote, incl president: 1=Dem, 2=Rep

for year_curr, year_lag in [(1960, 1958), (1976, 1974), (1992, 1990)]:
    print(f'\n===== {year_curr} =====')
    cdf_curr = cdf[cdf['VCF0004']==year_curr].copy()
    cdf_lag = cdf[cdf['VCF0004']==year_lag].copy()
    panel = cdf_curr[cdf_curr['VCF0006a'] < year_curr*10000].copy()
    merged = panel.merge(cdf_lag[['VCF0006a','VCF0301']], on='VCF0006a', suffixes=('','_lag'))

    if year_curr == 1960:
        wt = merged['VCF0009x'].fillna(1.0).astype(int)
        merged = merged.loc[merged.index.repeat(wt)].reset_index(drop=True)

    t = targets[str(year_curr)]

    for vote_var in ['VCF0707', 'VCF0704', 'VCF0704a']:
        if vote_var not in merged.columns:
            continue
        valid = merged[
            merged[vote_var].isin([1.0, 2.0]) &
            merged['VCF0301'].isin([1,2,3,4,5,6,7]) &
            merged['VCF0301_lag'].isin([1,2,3,4,5,6,7]) if 'VCF0301_lag' in merged.columns else merged['VCF0301'].isin([1,2,3,4,5,6,7])
        ].copy()

        if 'VCF0301_lag' not in valid.columns:
            continue

        valid['house_rep'] = (valid[vote_var] == 2.0).astype(int)
        valid = construct_pid_vars(valid, 'VCF0301', 'curr')
        X = sm.add_constant(valid[['strong_curr','weak_curr','lean_curr']])
        try:
            mod = Probit(valid['house_rep'].astype(float), X).fit(disp=0)
            coef_dist = abs(mod.params['strong_curr']-t['strong']) + abs(mod.params['weak_curr']-t['weak']) + \
                       abs(mod.params['lean_curr']-t['lean']) + abs(mod.params['const']-t['int'])
            print(f'  {vote_var}: N={len(valid)}, LL={mod.llf:.1f}, R2={mod.prsquared:.4f}, coef_dist={coef_dist:.3f}')
            print(f'    S={mod.params["strong_curr"]:.3f} W={mod.params["weak_curr"]:.3f} L={mod.params["lean_curr"]:.3f} I={mod.params["const"]:.3f}')
        except:
            print(f'  {vote_var}: N={len(valid)} - model failed')

    # Also try union approaches with different base
    for base, fill in [('VCF0707+VCF0706', ('VCF0707', 'VCF0706')),
                        ('VCF0704+VCF0706', ('VCF0704', 'VCF0706')),
                        ('VCF0704a+VCF0707', ('VCF0704a', 'VCF0707'))]:
        b, f = fill
        if b not in merged.columns or f not in merged.columns:
            continue
        test = merged.copy()
        test['hv'] = test[b]
        mask = test['hv'].isna() & test[f].isin([1.0, 2.0])
        test.loc[mask, 'hv'] = test.loc[mask, f]
        valid = test[
            test['hv'].isin([1.0, 2.0]) &
            test['VCF0301'].isin([1,2,3,4,5,6,7]) &
            test['VCF0301_lag'].isin([1,2,3,4,5,6,7])
        ].copy()
        valid['house_rep'] = (valid['hv'] == 2.0).astype(int)
        valid = construct_pid_vars(valid, 'VCF0301', 'curr')
        X = sm.add_constant(valid[['strong_curr','weak_curr','lean_curr']])
        try:
            mod = Probit(valid['house_rep'].astype(float), X).fit(disp=0)
            coef_dist = abs(mod.params['strong_curr']-t['strong']) + abs(mod.params['weak_curr']-t['weak']) + \
                       abs(mod.params['lean_curr']-t['lean']) + abs(mod.params['const']-t['int'])
            print(f'  {base}: N={len(valid)}, LL={mod.llf:.1f}, R2={mod.prsquared:.4f}, coef_dist={coef_dist:.3f}')
            print(f'    S={mod.params["strong_curr"]:.3f} W={mod.params["weak_curr"]:.3f} L={mod.params["lean_curr"]:.3f} I={mod.params["const"]:.3f}')
        except:
            print(f'  {base}: N={len(valid)} - model failed')

    print(f'  Target: N={t["N"]}, LL={t["LL"]}, R2={t["R2"]}, S={t["strong"]} W={t["weak"]} L={t["lean"]} I={t["int"]}')
