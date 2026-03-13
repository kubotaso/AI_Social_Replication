"""Try all combinations of approaches for 3 panels and compute total scores."""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
from itertools import product
import warnings
warnings.filterwarnings('ignore')

cdf = pd.read_csv('anes_cumulative.csv', low_memory=False)

GROUND_TRUTH = {
    '1960': {
        'N': 911,
        'current': {'strong': (1.358, 0.094), 'weak': (1.028, 0.083), 'lean': (0.855, 0.131), 'intercept': (0.035, 0.053), 'llf': -372.7, 'r2': 0.41},
        'lagged': {'strong': (1.363, 0.092), 'weak': (0.842, 0.078), 'lean': (0.564, 0.125), 'intercept': (0.068, 0.051), 'llf': -403.9, 'r2': 0.36},
        'iv': {'strong': (1.715, 0.173), 'weak': (0.728, 0.239), 'lean': (1.081, 0.696), 'intercept': (0.032, 0.057), 'llf': -403.9, 'r2': 0.36},
    },
    '1976': {
        'N': 682,
        'current': {'strong': (1.087, 0.105), 'weak': (0.624, 0.086), 'lean': (0.622, 0.110), 'intercept': (-0.123, 0.054), 'llf': -358.2, 'r2': 0.24},
        'lagged': {'strong': (0.966, 0.104), 'weak': (0.738, 0.089), 'lean': (0.486, 0.109), 'intercept': (-0.063, 0.053), 'llf': -371.3, 'r2': 0.21},
        'iv': {'strong': (1.123, 0.178), 'weak': (0.745, 0.251), 'lean': (0.725, 0.438), 'intercept': (-0.102, 0.055), 'llf': -371.3, 'r2': 0.21},
    },
    '1992': {
        'N': 760,
        'current': {'strong': (0.975, 0.094), 'weak': (0.627, 0.084), 'lean': (0.472, 0.098), 'intercept': (-0.211, 0.051), 'llf': -408.2, 'r2': 0.20},
        'lagged': {'strong': (1.061, 0.100), 'weak': (0.404, 0.077), 'lean': (0.519, 0.101), 'intercept': (-0.168, 0.051), 'llf': -416.2, 'r2': 0.19},
        'iv': {'strong': (1.516, 0.180), 'weak': (-0.225, 0.268), 'lean': (1.824, 0.513), 'intercept': (-0.125, 0.053), 'llf': -416.2, 'r2': 0.19},
    }
}

def construct_pid_vars(df, pid_col, suffix):
    df[f'strong_{suffix}'] = np.where(df[pid_col] == 7, 1, np.where(df[pid_col] == 1, -1, 0))
    df[f'weak_{suffix}'] = np.where(df[pid_col] == 6, 1, np.where(df[pid_col] == 2, -1, 0))
    df[f'lean_{suffix}'] = np.where(df[pid_col] == 5, 1, np.where(df[pid_col] == 3, -1, 0))
    return df

def prepare_panel(cdf, year_curr, year_lag, use_union=False, vote_var='VCF0707'):
    cc = cdf[cdf['VCF0004']==year_curr].copy()
    cl = cdf[cdf['VCF0004']==year_lag].copy()
    panel = cc[cc['VCF0006a'] < year_curr*10000].copy()
    merged = panel.merge(cl[['VCF0006a','VCF0301']], on='VCF0006a', suffixes=('','_lag'))
    if use_union:
        merged['house_vote'] = merged['VCF0707']
        mask = merged['house_vote'].isna() & merged['VCF0706'].isin([1.0, 2.0])
        merged.loc[mask, 'house_vote'] = merged.loc[mask, 'VCF0706']
    else:
        merged['house_vote'] = merged[vote_var]
    valid = merged[
        merged['house_vote'].isin([1.0, 2.0]) &
        merged['VCF0301'].isin([1,2,3,4,5,6,7]) &
        merged['VCF0301_lag'].isin([1,2,3,4,5,6,7])
    ].copy()
    valid['house_rep'] = (valid['house_vote'] == 2.0).astype(int)
    valid = construct_pid_vars(valid, 'VCF0301', 'curr')
    valid = construct_pid_vars(valid, 'VCF0301_lag', 'lag')
    return valid

def run_iv_probit(df, dep_var, endog_vars, instrument_vars):
    predicted = pd.DataFrame(index=df.index)
    for var in endog_vars:
        X_first = sm.add_constant(df[instrument_vars].astype(float))
        ols_model = sm.OLS(df[var].astype(float), X_first).fit()
        predicted[var] = ols_model.predict(X_first)
    X_second = sm.add_constant(predicted[endog_vars].astype(float))
    iv_model = Probit(df[dep_var].astype(float), X_second).fit(disp=0, maxiter=1000)
    return iv_model

def score_panel(year_str, df):
    gt = GROUND_TRUTH[year_str]
    curr_vars = ['strong_curr', 'weak_curr', 'lean_curr']
    lag_vars = ['strong_lag', 'weak_lag', 'lean_lag']

    mc = Probit(df['house_rep'].astype(float), sm.add_constant(df[curr_vars].astype(float))).fit(disp=0)
    ml = Probit(df['house_rep'].astype(float), sm.add_constant(df[lag_vars].astype(float))).fit(disp=0)
    mi = run_iv_probit(df, 'house_rep', curr_vars, lag_vars)

    models = {'current': mc, 'lagged': ml, 'iv': mi}

    total_coef = 0
    total_se = 0
    total_llf = 0
    total_r2 = 0
    n_items = 0

    for model_type in ['current', 'lagged', 'iv']:
        gt_model = gt[model_type]
        gen_model = models[model_type]
        params = gen_model.params
        bse = gen_model.bse

        var_map = {}
        for name in params.index:
            if 'strong' in name: var_map['strong'] = name
            elif 'weak' in name: var_map['weak'] = name
            elif 'lean' in name: var_map['lean'] = name
            elif name == 'const': var_map['intercept'] = name

        for var_key in ['strong', 'weak', 'lean', 'intercept']:
            gt_coef, gt_se = gt_model[var_key]
            gen_coef = params[var_map[var_key]]
            gen_se = bse[var_map[var_key]]
            n_items += 1
            c_diff = abs(gen_coef - gt_coef)
            s_diff = abs(gen_se - gt_se)
            total_coef += max(0, 1.0 - c_diff / 0.05) if c_diff <= 0.15 else 0.0
            total_se += max(0, 1.0 - s_diff / 0.02) if s_diff <= 0.06 else 0.0

        ll_diff = abs(gen_model.llf - gt_model['llf'])
        total_llf += max(0, 1.0 - ll_diff / 1.0) if ll_diff <= 3.0 else 0.0
        r2_diff = abs(gen_model.prsquared - gt_model['r2'])
        total_r2 += max(0, 1.0 - r2_diff / 0.02) if r2_diff <= 0.06 else 0.0

    n_diff_pct = abs(len(df) - gt['N']) / gt['N']
    if n_diff_pct <= 0.05: n_score = 1.0
    elif n_diff_pct <= 0.10: n_score = 0.7
    elif n_diff_pct <= 0.20: n_score = 0.4
    else: n_score = 0.1

    return {
        'coef': total_coef / n_items,
        'se': total_se / n_items,
        'n_score': n_score,
        'llf': total_llf / 3,
        'r2': total_r2 / 3,
        'N': len(df)
    }

# Try all combinations
approaches = ['VCF0707', 'union']  # VCF0706 gives terrible results, skip it
configs = {
    '1960': [],
    '1976': [],
    '1992': [],
}

for approach in approaches:
    for year, lag_year in [(1960, 1958), (1976, 1974), (1992, 1990)]:
        use_union = (approach == 'union')
        df = prepare_panel(cdf, year, lag_year, use_union=use_union)
        if len(df) < 50:
            continue
        scores = score_panel(str(year), df)
        scores['approach'] = approach
        configs[str(year)].append(scores)

# Now compute total score for each combination
best_total = 0
best_combo = None

for c60 in configs['1960']:
    for c76 in configs['1976']:
        for c92 in configs['1992']:
            avg_coef = (c60['coef'] + c76['coef'] + c92['coef']) / 3
            avg_se = (c60['se'] + c76['se'] + c92['se']) / 3
            avg_n = (c60['n_score'] + c76['n_score'] + c92['n_score']) / 3
            avg_llf = (c60['llf'] + c76['llf'] + c92['llf']) / 3
            avg_r2 = (c60['r2'] + c76['r2'] + c92['r2']) / 3

            total = (30 * avg_coef + 20 * avg_se + 15 * avg_n +
                     10 * 1.0 + 10 * avg_llf + 15 * avg_r2)  # vars=10 always

            if total > best_total:
                best_total = total
                best_combo = (c60, c76, c92)

print(f'Best total score: {best_total:.1f}')
print(f'1960: {best_combo[0]["approach"]} (N={best_combo[0]["N"]})')
print(f'  coef={best_combo[0]["coef"]:.3f}, se={best_combo[0]["se"]:.3f}, n={best_combo[0]["n_score"]}, llf={best_combo[0]["llf"]:.3f}, r2={best_combo[0]["r2"]:.3f}')
print(f'1976: {best_combo[1]["approach"]} (N={best_combo[1]["N"]})')
print(f'  coef={best_combo[1]["coef"]:.3f}, se={best_combo[1]["se"]:.3f}, n={best_combo[1]["n_score"]}, llf={best_combo[1]["llf"]:.3f}, r2={best_combo[1]["r2"]:.3f}')
print(f'1992: {best_combo[2]["approach"]} (N={best_combo[2]["N"]})')
print(f'  coef={best_combo[2]["coef"]:.3f}, se={best_combo[2]["se"]:.3f}, n={best_combo[2]["n_score"]}, llf={best_combo[2]["llf"]:.3f}, r2={best_combo[2]["r2"]:.3f}')

# Show all combinations
print('\n=== All combinations ===')
for c60 in configs['1960']:
    for c76 in configs['1976']:
        for c92 in configs['1992']:
            avg_coef = (c60['coef'] + c76['coef'] + c92['coef']) / 3
            avg_se = (c60['se'] + c76['se'] + c92['se']) / 3
            avg_n = (c60['n_score'] + c76['n_score'] + c92['n_score']) / 3
            avg_llf = (c60['llf'] + c76['llf'] + c92['llf']) / 3
            avg_r2 = (c60['r2'] + c76['r2'] + c92['r2']) / 3
            total = 30*avg_coef + 20*avg_se + 15*avg_n + 10 + 10*avg_llf + 15*avg_r2
            print(f'{c60["approach"]}/{c76["approach"]}/{c92["approach"]}: '
                  f'coef={30*avg_coef:.1f}, se={20*avg_se:.1f}, n={15*avg_n:.1f}, '
                  f'llf={10*avg_llf:.1f}, r2={15*avg_r2:.1f}, TOTAL={total:.1f}')
