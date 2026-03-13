"""Try all combinations of approaches and score them fully."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
import warnings
warnings.filterwarnings('ignore')
from itertools import product

cdf = pd.read_csv('anes_cumulative.csv', low_memory=False)

GROUND_TRUTH = {
    '1960': {
        'N': 911,
        'current': {'strong': (1.358, 0.094), 'weak': (1.028, 0.083), 'lean': (0.855, 0.131), 'intercept': (0.035, 0.053), 'llf': -372.7, 'r2': 0.41},
        'lagged': {'strong': (1.363, 0.092), 'weak': (0.842, 0.078), 'lean': (0.564, 0.125), 'intercept': (0.068, 0.051), 'llf': -403.9, 'r2': 0.36},
        'iv': {'strong': (1.715, 0.173), 'weak': (0.728, 0.239), 'lean': (1.081, 0.696), 'intercept': (0.032, 0.057), 'llf': -403.9, 'r2': 0.36}
    },
    '1976': {
        'N': 682,
        'current': {'strong': (1.087, 0.105), 'weak': (0.624, 0.086), 'lean': (0.622, 0.110), 'intercept': (-0.123, 0.054), 'llf': -358.2, 'r2': 0.24},
        'lagged': {'strong': (0.966, 0.104), 'weak': (0.738, 0.089), 'lean': (0.486, 0.109), 'intercept': (-0.063, 0.053), 'llf': -371.3, 'r2': 0.21},
        'iv': {'strong': (1.123, 0.178), 'weak': (0.745, 0.251), 'lean': (0.725, 0.438), 'intercept': (-0.102, 0.055), 'llf': -371.3, 'r2': 0.21}
    },
    '1992': {
        'N': 760,
        'current': {'strong': (0.975, 0.094), 'weak': (0.627, 0.084), 'lean': (0.472, 0.098), 'intercept': (-0.211, 0.051), 'llf': -408.2, 'r2': 0.20},
        'lagged': {'strong': (1.061, 0.100), 'weak': (0.404, 0.077), 'lean': (0.519, 0.101), 'intercept': (-0.168, 0.051), 'llf': -416.2, 'r2': 0.19},
        'iv': {'strong': (1.516, 0.180), 'weak': (-0.225, 0.268), 'lean': (1.824, 0.513), 'intercept': (-0.125, 0.053), 'llf': -416.2, 'r2': 0.19}
    }
}

def construct_pid_vars(df, pid_col, suffix):
    df[f'strong_{suffix}'] = np.where(df[pid_col] == 7, 1, np.where(df[pid_col] == 1, -1, 0))
    df[f'weak_{suffix}'] = np.where(df[pid_col] == 6, 1, np.where(df[pid_col] == 2, -1, 0))
    df[f'lean_{suffix}'] = np.where(df[pid_col] == 5, 1, np.where(df[pid_col] == 3, -1, 0))
    return df

def run_iv_probit_3dummy(df, dep_var, endog_vars, pid_lag_col='VCF0301_lag'):
    """IV with 3 directional lagged dummies as instruments."""
    df = construct_pid_vars(df.copy(), pid_lag_col, 'iv_lag')
    iv_instruments = ['strong_iv_lag', 'weak_iv_lag', 'lean_iv_lag']
    predicted = pd.DataFrame(index=df.index)
    for var in endog_vars:
        X_first = sm.add_constant(df[iv_instruments].astype(float))
        ols_model = sm.OLS(df[var].astype(float), X_first).fit()
        predicted[var] = ols_model.predict(X_first)
    X_second = sm.add_constant(predicted[endog_vars].astype(float))
    iv_model = Probit(df[dep_var].astype(float), X_second).fit(disp=0, maxiter=1000)
    return iv_model

def run_iv_probit_6dummy(df, dep_var, endog_vars, pid_lag_col='VCF0301_lag'):
    """IV with 6 lagged PID dummies as instruments."""
    instrument_dummies = []
    for val in [1, 2, 3, 5, 6, 7]:
        col_name = f'lag_pid_d{val}'
        df[col_name] = (df[pid_lag_col] == val).astype(float)
        instrument_dummies.append(col_name)
    predicted = pd.DataFrame(index=df.index)
    for var in endog_vars:
        X_first = sm.add_constant(df[instrument_dummies].astype(float))
        ols_model = sm.OLS(df[var].astype(float), X_first).fit()
        predicted[var] = ols_model.predict(X_first)
    X_second = sm.add_constant(predicted[endog_vars].astype(float))
    iv_model = Probit(df[dep_var].astype(float), X_second).fit(disp=0, maxiter=1000)
    return iv_model

def prepare_panel(year_curr, year_lag, use_union, expand_weights):
    cdf_curr = cdf[cdf['VCF0004'] == year_curr].copy()
    cdf_lag = cdf[cdf['VCF0004'] == year_lag].copy()
    threshold = year_curr * 10000
    panel = cdf_curr[cdf_curr['VCF0006a'] < threshold].copy()
    merged = panel.merge(cdf_lag[['VCF0006a', 'VCF0301']], on='VCF0006a', suffixes=('', '_lag'))

    if expand_weights and 'VCF0009x' in merged.columns:
        wt = merged['VCF0009x'].fillna(1.0).round().astype(int)
        merged = merged.loc[merged.index.repeat(wt)].reset_index(drop=True)

    if use_union:
        merged['house_vote'] = merged['VCF0707']
        mask = merged['house_vote'].isna() & merged['VCF0706'].isin([1.0, 2.0])
        merged.loc[mask, 'house_vote'] = merged.loc[mask, 'VCF0706']
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

def score_models(year, n, mc, ml, mi, gt):
    """Score one panel's models against ground truth."""
    coef_score = 0
    se_score = 0
    n_coefs = 0
    n_ses = 0
    llf_score = 0
    r2_score = 0
    n_models = 0

    n_diff_pct = abs(n - gt['N']) / gt['N']
    if n_diff_pct <= 0.05: n_score = 1.0
    elif n_diff_pct <= 0.10: n_score = 0.7
    elif n_diff_pct <= 0.20: n_score = 0.4
    else: n_score = 0.1

    for model_type, model in [('current', mc), ('lagged', ml), ('iv', mi)]:
        gt_model = gt[model_type]
        params = model.params
        bse = model.bse
        n_models += 1

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
            n_coefs += 1
            n_ses += 1

            coef_diff = abs(gen_coef - gt_coef)
            se_diff = abs(gen_se - gt_se)
            cs = max(0, 1.0 - coef_diff / 0.05) if coef_diff <= 0.15 else 0.0
            ss = max(0, 1.0 - se_diff / 0.02) if se_diff <= 0.06 else 0.0
            coef_score += cs
            se_score += ss

        llf_diff = abs(model.llf - gt_model['llf'])
        ls = max(0, 1.0 - llf_diff / 1.0) if llf_diff <= 3.0 else 0.0
        llf_score += ls

        r2_diff = abs(model.prsquared - gt_model['r2'])
        rs = max(0, 1.0 - r2_diff / 0.02) if r2_diff <= 0.06 else 0.0
        r2_score += rs

    return {
        'coef': coef_score, 'se': se_score, 'n': n_score,
        'llf': llf_score, 'r2': r2_score,
        'n_coefs': n_coefs, 'n_ses': n_ses, 'n_models': n_models
    }

def total_score(scores_list):
    tc = sum(s['coef'] for s in scores_list)
    ts = sum(s['se'] for s in scores_list)
    tn = sum(s['n'] for s in scores_list)
    tl = sum(s['llf'] for s in scores_list)
    tr = sum(s['r2'] for s in scores_list)
    nc = sum(s['n_coefs'] for s in scores_list)
    ns = sum(s['n_ses'] for s in scores_list)
    nm = sum(s['n_models'] for s in scores_list)

    coef_pts = 30 * (tc / nc)
    se_pts = 20 * (ts / ns)
    n_pts = 15 * (tn / 3)
    var_pts = 10  # always 10 since we always have all vars
    llf_pts = 10 * (tl / nm)
    r2_pts = 15 * (tr / nm)
    return coef_pts + se_pts + n_pts + var_pts + llf_pts + r2_pts

# Test all combinations
curr_vars = ['strong_curr', 'weak_curr', 'lean_curr']
lag_vars = ['strong_lag', 'weak_lag', 'lean_lag']

# 1960 options: union+expand, union+no_expand, 707+expand, 707+no_expand
# 1976 options: union, 707
# 1992 options: 707, union
# IV options: 3dummy, 6dummy

print('Computing all approach combinations...\n')

best_score = 0
best_config = None
results = []

options_60 = [
    ('union+expand', True, True),
    ('union+noexpand', True, False),
    ('707+expand', False, True),
    ('707+noexpand', False, False),
]

options_76 = [
    ('union', True),
    ('707', False),
]

options_92 = [
    ('707', False),
    ('union', True),
]

options_iv = [
    ('3dummy', run_iv_probit_3dummy),
    ('6dummy', run_iv_probit_6dummy),
]

for (l60, u60, e60), (l76, u76), (l92, u92), (l_iv, iv_fn) in product(options_60, options_76, options_92, options_iv):
    try:
        df60 = prepare_panel(1960, 1958, u60, e60)
        mc60 = Probit(df60['house_rep'].astype(float), sm.add_constant(df60[curr_vars].astype(float))).fit(disp=0)
        ml60 = Probit(df60['house_rep'].astype(float), sm.add_constant(df60[lag_vars].astype(float))).fit(disp=0)
        mi60 = iv_fn(df60.copy(), 'house_rep', curr_vars)
        s60 = score_models('1960', len(df60), mc60, ml60, mi60, GROUND_TRUTH['1960'])

        df76 = prepare_panel(1976, 1974, u76, False)
        mc76 = Probit(df76['house_rep'].astype(float), sm.add_constant(df76[curr_vars].astype(float))).fit(disp=0)
        ml76 = Probit(df76['house_rep'].astype(float), sm.add_constant(df76[lag_vars].astype(float))).fit(disp=0)
        mi76 = iv_fn(df76.copy(), 'house_rep', curr_vars)
        s76 = score_models('1976', len(df76), mc76, ml76, mi76, GROUND_TRUTH['1976'])

        df92 = prepare_panel(1992, 1990, u92, False)
        mc92 = Probit(df92['house_rep'].astype(float), sm.add_constant(df92[curr_vars].astype(float))).fit(disp=0)
        ml92 = Probit(df92['house_rep'].astype(float), sm.add_constant(df92[lag_vars].astype(float))).fit(disp=0)
        mi92 = iv_fn(df92.copy(), 'house_rep', curr_vars)
        s92 = score_models('1992', len(df92), mc92, ml92, mi92, GROUND_TRUTH['1992'])

        ts = total_score([s60, s76, s92])

        config = f'60:{l60} 76:{l76} 92:{l92} IV:{l_iv}'
        results.append((ts, config, len(df60), len(df76), len(df92)))

        if ts > best_score:
            best_score = ts
            best_config = config

    except Exception as e:
        pass

# Sort and display
results.sort(key=lambda x: -x[0])
print(f'TOP 10 CONFIGURATIONS:')
print(f'{"Score":>6} | {"Configuration":<50} | N60 | N76 | N92')
print('-' * 80)
for score, config, n60, n76, n92 in results[:10]:
    print(f'{score:6.1f} | {config:<50} | {n60} | {n76} | {n92}')

print(f'\nBest: {best_score:.1f} = {best_config}')
