"""Compare different approaches for each panel year to find best overall score."""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
import warnings
warnings.filterwarnings('ignore')

cdf = pd.read_csv('anes_cumulative.csv', low_memory=False)

GROUND_TRUTH = {
    '1960': {
        'N': 911,
        'current': {
            'strong': (1.358, 0.094), 'weak': (1.028, 0.083),
            'lean': (0.855, 0.131), 'intercept': (0.035, 0.053),
            'llf': -372.7, 'r2': 0.41
        },
    },
    '1976': {
        'N': 682,
        'current': {
            'strong': (1.087, 0.105), 'weak': (0.624, 0.086),
            'lean': (0.622, 0.110), 'intercept': (-0.123, 0.054),
            'llf': -358.2, 'r2': 0.24
        },
    },
    '1992': {
        'N': 760,
        'current': {
            'strong': (0.975, 0.094), 'weak': (0.627, 0.084),
            'lean': (0.472, 0.098), 'intercept': (-0.211, 0.051),
            'llf': -408.2, 'r2': 0.20
        },
    }
}

def construct_pid_vars(df, pid_col, suffix):
    df[f'strong_{suffix}'] = np.where(df[pid_col] == 7, 1, np.where(df[pid_col] == 1, -1, 0))
    df[f'weak_{suffix}'] = np.where(df[pid_col] == 6, 1, np.where(df[pid_col] == 2, -1, 0))
    df[f'lean_{suffix}'] = np.where(df[pid_col] == 5, 1, np.where(df[pid_col] == 3, -1, 0))
    return df

def prepare_panel(cdf, year_curr, year_lag, vote_var='VCF0707', use_union=False):
    cc = cdf[cdf['VCF0004']==year_curr].copy()
    cl = cdf[cdf['VCF0004']==year_lag].copy()
    panel = cc[cc['VCF0006a'] < year_curr*10000].copy()
    merged = panel.merge(cl[['VCF0006a','VCF0301']], on='VCF0006a', suffixes=('','_lag'))

    if use_union:
        merged['house_vote'] = merged['VCF0707']
        mask = merged['house_vote'].isna() & merged['VCF0706'].isin([1.0, 2.0])
        merged.loc[mask, 'house_vote'] = merged.loc[mask, 'VCF0706']
    elif vote_var == 'VCF0706':
        merged['house_vote'] = merged['VCF0706']
    else:
        merged['house_vote'] = merged['VCF0707']

    valid = merged[
        merged['house_vote'].isin([1.0, 2.0]) &
        merged['VCF0301'].isin([1,2,3,4,5,6,7]) &
        merged['VCF0301_lag'].isin([1,2,3,4,5,6,7])
    ].copy()
    valid['house_rep'] = (valid['house_vote'] == 2.0).astype(int)
    valid = construct_pid_vars(valid, 'VCF0301', 'curr')
    valid = construct_pid_vars(valid, 'VCF0301_lag', 'lag')
    return valid

def score_current_model(model, N, year):
    gt = GROUND_TRUTH[year]
    coefs = 0
    total_c = 0
    for var_key, param_name in [('strong','strong_curr'), ('weak','weak_curr'),
                                  ('lean','lean_curr'), ('intercept','const')]:
        gt_c, gt_se = gt['current'][var_key]
        gen_c = model.params[param_name]
        diff = abs(gen_c - gt_c)
        score = max(0, 1.0 - diff / 0.05) if diff <= 0.15 else 0.0
        coefs += score
        total_c += 1

    n_diff = abs(N - gt['N']) / gt['N']
    if n_diff <= 0.05: n_score = 1.0
    elif n_diff <= 0.10: n_score = 0.7
    elif n_diff <= 0.20: n_score = 0.4
    else: n_score = 0.1

    return coefs/total_c, n_score

for year, lag_year in [(1960, 1958), (1976, 1974), (1992, 1990)]:
    print(f'\n=== {year} ===')
    gt = GROUND_TRUTH[str(year)]

    for approach, kwargs in [
        ('VCF0707 only', {'vote_var': 'VCF0707', 'use_union': False}),
        ('VCF0706 only', {'vote_var': 'VCF0706', 'use_union': False}),
        ('Union', {'vote_var': 'VCF0707', 'use_union': True}),
    ]:
        df = prepare_panel(cdf, year, lag_year, **kwargs)
        if len(df) < 50:
            print(f'  {approach}: N={len(df)} (too few)')
            continue

        X = sm.add_constant(df[['strong_curr','weak_curr','lean_curr']])
        mod = Probit(df['house_rep'], X).fit(disp=0)

        coef_pct, n_score = score_current_model(mod, len(df), str(year))
        n_diff = abs(len(df) - gt['N']) / gt['N']
        ll_diff = abs(mod.llf - gt['current']['llf'])
        r2_diff = abs(mod.prsquared - gt['current']['r2'])

        print(f'  {approach}: N={len(df)} (diff={n_diff:.1%}), LL={mod.llf:.1f} (diff={ll_diff:.1f}), '
              f'R2={mod.prsquared:.4f} (diff={r2_diff:.4f})')
        print(f'    Strong={mod.params["strong_curr"]:.3f} (tgt {gt["current"]["strong"][0]:.3f}, diff={abs(mod.params["strong_curr"]-gt["current"]["strong"][0]):.3f})')
        print(f'    Weak={mod.params["weak_curr"]:.3f} (tgt {gt["current"]["weak"][0]:.3f}, diff={abs(mod.params["weak_curr"]-gt["current"]["weak"][0]):.3f})')
        print(f'    Lean={mod.params["lean_curr"]:.3f} (tgt {gt["current"]["lean"][0]:.3f}, diff={abs(mod.params["lean_curr"]-gt["current"]["lean"][0]):.3f})')
        print(f'    Int={mod.params["const"]:.3f} (tgt {gt["current"]["intercept"][0]:.3f}, diff={abs(mod.params["const"]-gt["current"]["intercept"][0]):.3f})')
        print(f'    Coef score: {coef_pct:.3f}, N score: {n_score}')
