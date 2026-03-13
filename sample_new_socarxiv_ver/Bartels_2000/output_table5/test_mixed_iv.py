"""Test mixing 3-dummy and 6-dummy IV across panels to maximize score."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
import warnings
warnings.filterwarnings('ignore')

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

def run_iv_probit(df, dep_var, endog_vars, pid_lag_col, use_6dummy=True):
    df = df.copy()
    if use_6dummy:
        instrument_dummies = []
        for val in [1, 2, 3, 5, 6, 7]:
            col_name = f'lag_pid_d{val}'
            df[col_name] = (df[pid_lag_col] == val).astype(float)
            instrument_dummies.append(col_name)
    else:
        df = construct_pid_vars(df, pid_lag_col, 'iv_inst')
        instrument_dummies = ['strong_iv_inst', 'weak_iv_inst', 'lean_iv_inst']
    predicted = pd.DataFrame(index=df.index)
    for var in endog_vars:
        X_first = sm.add_constant(df[instrument_dummies].astype(float))
        ols_model = sm.OLS(df[var].astype(float), X_first).fit()
        predicted[var] = ols_model.predict(X_first)
    X_second = sm.add_constant(predicted[endog_vars].astype(float))
    iv_model = Probit(df[dep_var].astype(float), X_second).fit(disp=0, maxiter=1000)
    return iv_model

def score_one_model(gen_model, gt_model):
    """Score one model (current, lagged, or iv) against ground truth."""
    params = gen_model.params
    bse = gen_model.bse
    var_map = {}
    for name in params.index:
        if 'strong' in name: var_map['strong'] = name
        elif 'weak' in name: var_map['weak'] = name
        elif 'lean' in name: var_map['lean'] = name
        elif name == 'const': var_map['intercept'] = name

    coef_score = 0
    se_score = 0
    for var_key in ['strong', 'weak', 'lean', 'intercept']:
        gt_coef, gt_se = gt_model[var_key]
        gen_coef = params[var_map[var_key]]
        gen_se = bse[var_map[var_key]]
        cd = abs(gen_coef - gt_coef)
        sd = abs(gen_se - gt_se)
        coef_score += max(0, 1.0 - cd / 0.05) if cd <= 0.15 else 0.0
        se_score += max(0, 1.0 - sd / 0.02) if sd <= 0.06 else 0.0

    llf_diff = abs(gen_model.llf - gt_model['llf'])
    llf_score = max(0, 1.0 - llf_diff / 1.0) if llf_diff <= 3.0 else 0.0
    r2_diff = abs(gen_model.prsquared - gt_model['r2'])
    r2_score = max(0, 1.0 - r2_diff / 0.02) if r2_diff <= 0.06 else 0.0

    return coef_score, se_score, llf_score, r2_score

# Prepare data
cdf = pd.read_csv('anes_cumulative.csv', low_memory=False)
curr_vars = ['strong_curr', 'weak_curr', 'lean_curr']
lag_vars = ['strong_lag', 'weak_lag', 'lean_lag']

def prepare_1960():
    p60 = pd.read_csv('panel_1960.csv', low_memory=False)
    df = p60.copy()
    df['pid_lag'] = df['VCF0301_lagged']
    wt = df['VCF0009x'].fillna(1.0).astype(int)
    df = df.loc[df.index.repeat(wt)].reset_index(drop=True)
    df['house_vote'] = df['VCF0707']
    mask = df['house_vote'].isna() & df['VCF0706'].isin([1.0, 2.0])
    df.loc[mask, 'house_vote'] = df.loc[mask, 'VCF0706']
    valid = df[df['house_vote'].isin([1.0, 2.0]) & df['VCF0301'].isin([1,2,3,4,5,6,7]) & df['pid_lag'].isin([1,2,3,4,5,6,7])].copy()
    valid['house_rep'] = (valid['house_vote'] == 2.0).astype(int)
    valid = construct_pid_vars(valid, 'VCF0301', 'curr')
    valid = construct_pid_vars(valid, 'pid_lag', 'lag')
    return valid

def prepare_1976():
    p76 = pd.read_csv('panel_1976.csv', low_memory=False)
    df = p76.copy()
    df['pid_lag'] = df['VCF0301_lagged']
    df['house_vote'] = df['VCF0707']
    mask = df['house_vote'].isna() & df['VCF0706'].isin([1.0, 2.0])
    df.loc[mask, 'house_vote'] = df.loc[mask, 'VCF0706']
    valid = df[df['house_vote'].isin([1.0, 2.0]) & df['VCF0301'].isin([1,2,3,4,5,6,7]) & df['pid_lag'].isin([1,2,3,4,5,6,7])].copy()
    valid['house_rep'] = (valid['house_vote'] == 2.0).astype(int)
    valid = construct_pid_vars(valid, 'VCF0301', 'curr')
    valid = construct_pid_vars(valid, 'pid_lag', 'lag')
    return valid

def prepare_1992_cdf():
    cdf92 = cdf[cdf['VCF0004']==1992].copy()
    cdf90 = cdf[cdf['VCF0004']==1990].copy()
    panel = cdf92[cdf92['VCF0006a'] < 19920000].copy()
    merged = panel.merge(cdf90[['VCF0006a','VCF0301']], on='VCF0006a', suffixes=('','_lag'))
    merged['pid_lag'] = merged['VCF0301_lag']
    valid = merged[merged['VCF0707'].isin([1.0, 2.0]) & merged['VCF0301'].isin([1,2,3,4,5,6,7]) & merged['pid_lag'].isin([1,2,3,4,5,6,7])].copy()
    valid['house_rep'] = (valid['VCF0707'] == 2.0).astype(int)
    valid = construct_pid_vars(valid, 'VCF0301', 'curr')
    valid = construct_pid_vars(valid, 'pid_lag', 'lag')
    return valid

df60 = prepare_1960()
df76 = prepare_1976()
df92 = prepare_1992_cdf()

mc60 = Probit(df60['house_rep'].astype(float), sm.add_constant(df60[curr_vars].astype(float))).fit(disp=0)
ml60 = Probit(df60['house_rep'].astype(float), sm.add_constant(df60[lag_vars].astype(float))).fit(disp=0)
mc76 = Probit(df76['house_rep'].astype(float), sm.add_constant(df76[curr_vars].astype(float))).fit(disp=0)
ml76 = Probit(df76['house_rep'].astype(float), sm.add_constant(df76[lag_vars].astype(float))).fit(disp=0)
mc92 = Probit(df92['house_rep'].astype(float), sm.add_constant(df92[curr_vars].astype(float))).fit(disp=0)
ml92 = Probit(df92['house_rep'].astype(float), sm.add_constant(df92[lag_vars].astype(float))).fit(disp=0)

# Test all IV combinations: 3-dummy vs 6-dummy per panel
from itertools import product
best_score = 0
best_config = ''

for iv60, iv76, iv92 in product([True, False], repeat=3):
    mi60 = run_iv_probit(df60, 'house_rep', curr_vars, 'pid_lag', use_6dummy=iv60)
    mi76 = run_iv_probit(df76, 'house_rep', curr_vars, 'pid_lag', use_6dummy=iv76)
    mi92 = run_iv_probit(df92, 'house_rep', curr_vars, 'pid_lag', use_6dummy=iv92)

    # Calculate total score
    total_coef = 0; total_se = 0; total_llf = 0; total_r2 = 0
    n_coefs = 0; n_ses = 0; n_models = 0; total_n = 0

    for year, mc, ml, mi, df, gt in [('1960', mc60, ml60, mi60, df60, GROUND_TRUTH['1960']),
                                      ('1976', mc76, ml76, mi76, df76, GROUND_TRUTH['1976']),
                                      ('1992', mc92, ml92, mi92, df92, GROUND_TRUTH['1992'])]:
        n_diff = abs(len(df) - gt['N']) / gt['N']
        if n_diff <= 0.05: total_n += 1.0
        elif n_diff <= 0.10: total_n += 0.7
        elif n_diff <= 0.20: total_n += 0.4
        else: total_n += 0.1

        for mt, model in [('current', mc), ('lagged', ml), ('iv', mi)]:
            cs, ss, ls, rs = score_one_model(model, gt[mt])
            total_coef += cs; total_se += ss; total_llf += ls; total_r2 += rs
            n_coefs += 4; n_ses += 4; n_models += 1

    coef_pts = 30 * (total_coef / n_coefs)
    se_pts = 20 * (total_se / n_ses)
    n_pts = 15 * (total_n / 3)
    var_pts = 10
    llf_pts = 10 * (total_llf / n_models)
    r2_pts = 15 * (total_r2 / n_models)
    total = coef_pts + se_pts + n_pts + var_pts + llf_pts + r2_pts

    config = f'60:{"6d" if iv60 else "3d"} 76:{"6d" if iv76 else "3d"} 92:{"6d" if iv92 else "3d"}'
    if total > best_score:
        best_score = total
        best_config = config

    print(f'{config}: {total:.1f} (coef={coef_pts:.1f} se={se_pts:.1f} n={n_pts:.1f} var={var_pts:.1f} llf={llf_pts:.1f} r2={r2_pts:.1f})')

print(f'\nBest: {best_score:.1f} = {best_config}')
