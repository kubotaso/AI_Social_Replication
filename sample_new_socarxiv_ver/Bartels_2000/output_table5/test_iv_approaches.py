"""Test different IV probit first-stage specifications."""

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

def prepare_panel(cdf, year_curr, year_lag, use_union=False):
    cc = cdf[cdf['VCF0004']==year_curr].copy()
    cl = cdf[cdf['VCF0004']==year_lag].copy()
    panel = cc[cc['VCF0006a'] < year_curr*10000].copy()
    merged = panel.merge(cl[['VCF0006a','VCF0301']], on='VCF0006a', suffixes=('','_lag'))
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

GROUND_TRUTH_IV = {
    '1960': {'strong': (1.715, 0.173), 'weak': (0.728, 0.239), 'lean': (1.081, 0.696), 'intercept': (0.032, 0.057)},
    '1976': {'strong': (1.123, 0.178), 'weak': (0.745, 0.251), 'lean': (0.725, 0.438), 'intercept': (-0.102, 0.055)},
    '1992': {'strong': (1.516, 0.180), 'weak': (-0.225, 0.268), 'lean': (1.824, 0.513), 'intercept': (-0.125, 0.053)},
}

curr_vars = ['strong_curr', 'weak_curr', 'lean_curr']
lag_vars = ['strong_lag', 'weak_lag', 'lean_lag']

def run_iv_standard(df):
    """Standard 2SLS: each current PID regressed on all lagged PIDs."""
    predicted = pd.DataFrame(index=df.index)
    for var in curr_vars:
        X_first = sm.add_constant(df[lag_vars].astype(float))
        ols = sm.OLS(df[var].astype(float), X_first).fit()
        predicted[var] = ols.predict(X_first)
    X_second = sm.add_constant(predicted[curr_vars].astype(float))
    return Probit(df['house_rep'].astype(float), X_second).fit(disp=0, maxiter=1000)

def run_iv_diagonal(df):
    """Diagonal: each current PID regressed only on its own lag."""
    predicted = pd.DataFrame(index=df.index)
    for cv, lv in zip(curr_vars, lag_vars):
        X_first = sm.add_constant(df[[lv]].astype(float))
        ols = sm.OLS(df[cv].astype(float), X_first).fit()
        predicted[cv] = ols.predict(X_first)
    X_second = sm.add_constant(predicted[curr_vars].astype(float))
    return Probit(df['house_rep'].astype(float), X_second).fit(disp=0, maxiter=1000)

def run_iv_fullpid(df):
    """Full 7-point PID as instrument (6 dummies for lagged)."""
    # Create dummy variables for lagged PID
    for val in [1,2,3,5,6,7]:  # 4 as reference
        df[f'lag_pid_{val}'] = (df['VCF0301_lag'] == val).astype(int)
    instr = [f'lag_pid_{v}' for v in [1,2,3,5,6,7]]

    predicted = pd.DataFrame(index=df.index)
    for var in curr_vars:
        X_first = sm.add_constant(df[instr].astype(float))
        ols = sm.OLS(df[var].astype(float), X_first).fit()
        predicted[var] = ols.predict(X_first)
    X_second = sm.add_constant(predicted[curr_vars].astype(float))
    return Probit(df['house_rep'].astype(float), X_second).fit(disp=0, maxiter=1000)

def run_iv_with_lagscale(df):
    """Use lagged PID 7-point scale as single instrument."""
    predicted = pd.DataFrame(index=df.index)
    for var in curr_vars:
        X_first = sm.add_constant(df[['VCF0301_lag']].astype(float))
        ols = sm.OLS(df[var].astype(float), X_first).fit()
        predicted[var] = ols.predict(X_first)
    X_second = sm.add_constant(predicted[curr_vars].astype(float))
    return Probit(df['house_rep'].astype(float), X_second).fit(disp=0, maxiter=1000)

def print_iv_results(model, year, label):
    gt = GROUND_TRUTH_IV[year]
    params = model.params
    bse = model.bse

    var_map = {}
    for name in params.index:
        if 'strong' in name: var_map['strong'] = name
        elif 'weak' in name: var_map['weak'] = name
        elif 'lean' in name: var_map['lean'] = name
        elif name == 'const': var_map['intercept'] = name

    total_diff = 0
    for vk in ['strong', 'weak', 'lean', 'intercept']:
        gc, gs = gt[vk]
        diff = abs(params[var_map[vk]] - gc)
        total_diff += diff

    print(f'  {label}: Strong={params[var_map["strong"]]:.3f}({bse[var_map["strong"]]:.3f}) '
          f'Weak={params[var_map["weak"]]:.3f}({bse[var_map["weak"]]:.3f}) '
          f'Lean={params[var_map["lean"]]:.3f}({bse[var_map["lean"]]:.3f}) '
          f'Int={params[var_map["intercept"]]:.3f}({bse[var_map["intercept"]]:.3f}) '
          f'total_diff={total_diff:.3f}')

for year, lag_year, use_union in [(1960, 1958, True), (1976, 1974, True), (1992, 1990, False)]:
    df = prepare_panel(cdf, year, lag_year, use_union=use_union)
    year_str = str(year)
    gt = GROUND_TRUTH_IV[year_str]
    print(f'\n=== {year} (N={len(df)}) ===')
    print(f'  Target: Strong={gt["strong"][0]:.3f} Weak={gt["weak"][0]:.3f} '
          f'Lean={gt["lean"][0]:.3f} Int={gt["intercept"][0]:.3f}')

    for label, func in [('Standard 2SLS', run_iv_standard),
                         ('Diagonal', run_iv_diagonal),
                         ('Full PID dummies', run_iv_fullpid),
                         ('Single PID scale', run_iv_with_lagscale)]:
        try:
            model = func(df.copy())
            print_iv_results(model, year_str, label)
        except Exception as e:
            print(f'  {label}: ERROR - {e}')
