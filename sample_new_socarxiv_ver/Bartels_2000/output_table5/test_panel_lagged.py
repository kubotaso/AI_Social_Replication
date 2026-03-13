"""Test using VCF0301_lagged from panel files for Table 5 replication."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
import warnings
warnings.filterwarnings('ignore')

def construct_pid_vars(df, pid_col, suffix):
    df[f'strong_{suffix}'] = np.where(df[pid_col] == 7, 1, np.where(df[pid_col] == 1, -1, 0))
    df[f'weak_{suffix}'] = np.where(df[pid_col] == 6, 1, np.where(df[pid_col] == 2, -1, 0))
    df[f'lean_{suffix}'] = np.where(df[pid_col] == 5, 1, np.where(df[pid_col] == 3, -1, 0))
    return df

def run_iv_probit_6dummy(df, dep_var, endog_vars, pid_lag_col):
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

def run_iv_probit_3dummy(df, dep_var, endog_vars, pid_lag_col):
    df_c = df.copy()
    df_c = construct_pid_vars(df_c, pid_lag_col, 'iv_inst')
    iv_instruments = ['strong_iv_inst', 'weak_iv_inst', 'lean_iv_inst']
    predicted = pd.DataFrame(index=df_c.index)
    for var in endog_vars:
        X_first = sm.add_constant(df_c[iv_instruments].astype(float))
        ols_model = sm.OLS(df_c[var].astype(float), X_first).fit()
        predicted[var] = ols_model.predict(X_first)
    X_second = sm.add_constant(predicted[endog_vars].astype(float))
    iv_model = Probit(df_c[dep_var].astype(float), X_second).fit(disp=0, maxiter=1000)
    return iv_model

TARGETS = {
    '1960': {
        'N': 911,
        'current': {'strong': 1.358, 'weak': 1.028, 'lean': 0.855, 'int': 0.035, 'LL': -372.7, 'R2': 0.41},
        'lagged': {'strong': 1.363, 'weak': 0.842, 'lean': 0.564, 'int': 0.068, 'LL': -403.9, 'R2': 0.36},
        'iv': {'strong': 1.715, 'weak': 0.728, 'lean': 1.081, 'int': 0.032, 'LL': -403.9, 'R2': 0.36}
    },
    '1976': {
        'N': 682,
        'current': {'strong': 1.087, 'weak': 0.624, 'lean': 0.622, 'int': -0.123, 'LL': -358.2, 'R2': 0.24},
        'lagged': {'strong': 0.966, 'weak': 0.738, 'lean': 0.486, 'int': -0.063, 'LL': -371.3, 'R2': 0.21},
        'iv': {'strong': 1.123, 'weak': 0.745, 'lean': 0.725, 'int': -0.102, 'LL': -371.3, 'R2': 0.21}
    },
    '1992': {
        'N': 760,
        'current': {'strong': 0.975, 'weak': 0.627, 'lean': 0.472, 'int': -0.211, 'LL': -408.2, 'R2': 0.20},
        'lagged': {'strong': 1.061, 'weak': 0.404, 'lean': 0.519, 'int': -0.168, 'LL': -416.2, 'R2': 0.19},
        'iv': {'strong': 1.516, 'weak': -0.225, 'lean': 1.824, 'int': -0.125, 'LL': -416.2, 'R2': 0.19}
    }
}

curr_vars = ['strong_curr', 'weak_curr', 'lean_curr']
lag_vars = ['strong_lag', 'weak_lag', 'lean_lag']

# === 1960 ===
print('========== 1960 ==========')
p60 = pd.read_csv('panel_1960.csv', low_memory=False)
print(f'VCF0301_lagged values: {sorted(p60["VCF0301_lagged"].dropna().unique())}')
print(f'VCF0301_lagged valid: {p60["VCF0301_lagged"].notna().sum()} / {len(p60)}')

# Try multiple approaches: expanded vs not, union vs 707, CDF-lag vs panel-lag
for lag_src in ['panel_lagged', 'cdf_lagged']:
    for vote_type in ['union', 'VCF0707']:
        for expand in [True, False]:
            df = p60.copy()

            if lag_src == 'panel_lagged':
                df['pid_lag'] = df['VCF0301_lagged']
            else:
                # Merge with CDF 1958 wave
                cdf = pd.read_csv('anes_cumulative.csv', low_memory=False)
                cdf58 = cdf[cdf['VCF0004']==1958][['VCF0006a','VCF0301']].copy()
                df = df.merge(cdf58, on='VCF0006a', suffixes=('','_cdf58'))
                df['pid_lag'] = df['VCF0301_cdf58']

            if expand:
                wt = df['VCF0009x'].fillna(1.0).astype(int)
                df = df.loc[df.index.repeat(wt)].reset_index(drop=True)

            if vote_type == 'union':
                df['house_vote'] = df['VCF0707']
                mask = df['house_vote'].isna() & df['VCF0706'].isin([1.0, 2.0])
                df.loc[mask, 'house_vote'] = df.loc[mask, 'VCF0706']
            else:
                df['house_vote'] = df['VCF0707']

            valid = df[
                df['house_vote'].isin([1.0, 2.0]) &
                df['VCF0301'].isin([1,2,3,4,5,6,7]) &
                df['pid_lag'].isin([1,2,3,4,5,6,7])
            ].copy()
            valid['house_rep'] = (valid['house_vote'] == 2.0).astype(int)
            valid = construct_pid_vars(valid, 'VCF0301', 'curr')
            valid = construct_pid_vars(valid, 'pid_lag', 'lag')

            if len(valid) < 50:
                continue

            X = sm.add_constant(valid[curr_vars])
            mod = Probit(valid['house_rep'].astype(float), X).fit(disp=0)
            Xl = sm.add_constant(valid[lag_vars])
            modl = Probit(valid['house_rep'].astype(float), Xl).fit(disp=0)

            t = TARGETS['1960']
            cd_c = abs(mod.params['strong_curr']-t['current']['strong']) + abs(mod.params['weak_curr']-t['current']['weak']) + \
                   abs(mod.params['lean_curr']-t['current']['lean']) + abs(mod.params['const']-t['current']['int'])
            cd_l = abs(modl.params['strong_lag']-t['lagged']['strong']) + abs(modl.params['weak_lag']-t['lagged']['weak']) + \
                   abs(modl.params['lean_lag']-t['lagged']['lean']) + abs(modl.params['const']-t['lagged']['int'])
            ll_c = abs(mod.llf - t['current']['LL'])
            ll_l = abs(modl.llf - t['lagged']['LL'])

            label = f'{lag_src}+{vote_type}+{"expand" if expand else "noexpand"}'
            print(f'\n  {label}: N={len(valid)} (target {t["N"]})')
            print(f'    Current: LL={mod.llf:.1f} R2={mod.prsquared:.4f} coef_dist={cd_c:.3f} LL_diff={ll_c:.1f}')
            print(f'      S={mod.params["strong_curr"]:.3f} W={mod.params["weak_curr"]:.3f} L={mod.params["lean_curr"]:.3f} I={mod.params["const"]:.3f}')
            print(f'    Lagged: LL={modl.llf:.1f} R2={modl.prsquared:.4f} coef_dist={cd_l:.3f} LL_diff={ll_l:.1f}')
            print(f'      S={modl.params["strong_lag"]:.3f} W={modl.params["weak_lag"]:.3f} L={modl.params["lean_lag"]:.3f} I={modl.params["const"]:.3f}')

# === 1976 ===
print('\n\n========== 1976 ==========')
p76 = pd.read_csv('panel_1976.csv', low_memory=False)
print(f'VCF0301_lagged values: {sorted(p76["VCF0301_lagged"].dropna().unique())}')
print(f'VCF0301_lagged valid: {p76["VCF0301_lagged"].notna().sum()} / {len(p76)}')

for lag_src in ['panel_lagged', 'cdf_lagged']:
    for vote_type in ['union', 'VCF0707']:
        df = p76.copy()

        if lag_src == 'panel_lagged':
            df['pid_lag'] = df['VCF0301_lagged']
        else:
            cdf = pd.read_csv('anes_cumulative.csv', low_memory=False)
            cdf74 = cdf[cdf['VCF0004']==1974][['VCF0006a','VCF0301']].copy()
            df = df.merge(cdf74, on='VCF0006a', suffixes=('','_cdf74'))
            df['pid_lag'] = df['VCF0301_cdf74']

        if vote_type == 'union':
            df['house_vote'] = df['VCF0707']
            mask = df['house_vote'].isna() & df['VCF0706'].isin([1.0, 2.0])
            df.loc[mask, 'house_vote'] = df.loc[mask, 'VCF0706']
        else:
            df['house_vote'] = df['VCF0707']

        valid = df[
            df['house_vote'].isin([1.0, 2.0]) &
            df['VCF0301'].isin([1,2,3,4,5,6,7]) &
            df['pid_lag'].isin([1,2,3,4,5,6,7])
        ].copy()
        valid['house_rep'] = (valid['house_vote'] == 2.0).astype(int)
        valid = construct_pid_vars(valid, 'VCF0301', 'curr')
        valid = construct_pid_vars(valid, 'pid_lag', 'lag')

        if len(valid) < 50:
            continue

        X = sm.add_constant(valid[curr_vars])
        mod = Probit(valid['house_rep'].astype(float), X).fit(disp=0)
        Xl = sm.add_constant(valid[lag_vars])
        modl = Probit(valid['house_rep'].astype(float), Xl).fit(disp=0)

        t = TARGETS['1976']
        cd_c = abs(mod.params['strong_curr']-t['current']['strong']) + abs(mod.params['weak_curr']-t['current']['weak']) + \
               abs(mod.params['lean_curr']-t['current']['lean']) + abs(mod.params['const']-t['current']['int'])
        cd_l = abs(modl.params['strong_lag']-t['lagged']['strong']) + abs(modl.params['weak_lag']-t['lagged']['weak']) + \
               abs(modl.params['lean_lag']-t['lagged']['lean']) + abs(modl.params['const']-t['lagged']['int'])
        ll_c = abs(mod.llf - t['current']['LL'])
        ll_l = abs(modl.llf - t['lagged']['LL'])

        label = f'{lag_src}+{vote_type}'
        print(f'\n  {label}: N={len(valid)} (target {t["N"]})')
        print(f'    Current: LL={mod.llf:.1f} R2={mod.prsquared:.4f} coef_dist={cd_c:.3f} LL_diff={ll_c:.1f}')
        print(f'      S={mod.params["strong_curr"]:.3f} W={mod.params["weak_curr"]:.3f} L={mod.params["lean_curr"]:.3f} I={mod.params["const"]:.3f}')
        print(f'    Lagged: LL={modl.llf:.1f} R2={modl.prsquared:.4f} coef_dist={cd_l:.3f} LL_diff={ll_l:.1f}')
        print(f'      S={modl.params["strong_lag"]:.3f} W={modl.params["weak_lag"]:.3f} L={modl.params["lean_lag"]:.3f} I={modl.params["const"]:.3f}')

# === 1992 (use panel file) ===
print('\n\n========== 1992 ==========')
p92 = pd.read_csv('panel_1992.csv', low_memory=False)
valid92 = p92[
    p92['vote_house'].isin([1.0, 2.0]) &
    p92['pid_current'].isin([1,2,3,4,5,6,7]) &
    p92['pid_lagged'].isin([1,2,3,4,5,6,7])
].copy()
valid92['house_rep'] = (valid92['vote_house'] == 2.0).astype(int)
valid92['strong_curr'] = np.where(valid92['pid_current'] == 7, 1, np.where(valid92['pid_current'] == 1, -1, 0))
valid92['weak_curr'] = np.where(valid92['pid_current'] == 6, 1, np.where(valid92['pid_current'] == 2, -1, 0))
valid92['lean_curr'] = np.where(valid92['pid_current'] == 5, 1, np.where(valid92['pid_current'] == 3, -1, 0))
valid92['strong_lag'] = np.where(valid92['pid_lagged'] == 7, 1, np.where(valid92['pid_lagged'] == 1, -1, 0))
valid92['weak_lag'] = np.where(valid92['pid_lagged'] == 6, 1, np.where(valid92['pid_lagged'] == 2, -1, 0))
valid92['lean_lag'] = np.where(valid92['pid_lagged'] == 5, 1, np.where(valid92['pid_lagged'] == 3, -1, 0))

X92 = sm.add_constant(valid92[curr_vars])
mod92 = Probit(valid92['house_rep'].astype(float), X92).fit(disp=0)
Xl92 = sm.add_constant(valid92[lag_vars])
modl92 = Probit(valid92['house_rep'].astype(float), Xl92).fit(disp=0)

t = TARGETS['1992']
cd_c92 = abs(mod92.params['strong_curr']-t['current']['strong']) + abs(mod92.params['weak_curr']-t['current']['weak']) + \
         abs(mod92.params['lean_curr']-t['current']['lean']) + abs(mod92.params['const']-t['current']['int'])
cd_l92 = abs(modl92.params['strong_lag']-t['lagged']['strong']) + abs(modl92.params['weak_lag']-t['lagged']['weak']) + \
         abs(modl92.params['lean_lag']-t['lagged']['lean']) + abs(modl92.params['const']-t['lagged']['int'])

print(f'  panel_1992: N={len(valid92)} (target {t["N"]})')
print(f'    Current: LL={mod92.llf:.1f} R2={mod92.prsquared:.4f} coef_dist={cd_c92:.3f} LL_diff={abs(mod92.llf-t["current"]["LL"]):.1f}')
print(f'      S={mod92.params["strong_curr"]:.3f} W={mod92.params["weak_curr"]:.3f} L={mod92.params["lean_curr"]:.3f} I={mod92.params["const"]:.3f}')
print(f'    Lagged: LL={modl92.llf:.1f} R2={modl92.prsquared:.4f} coef_dist={cd_l92:.3f} LL_diff={abs(modl92.llf-t["lagged"]["LL"]):.1f}')
print(f'      S={modl92.params["strong_lag"]:.3f} W={modl92.params["weak_lag"]:.3f} L={modl92.params["lean_lag"]:.3f} I={modl92.params["const"]:.3f}')
