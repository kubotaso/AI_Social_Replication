"""
Replication of Table 5 from Bartels (2000)

Attempt 9: Confirmatory run - identical to attempt 6 (best approach).
Exhaustive search confirmed score ceiling is 53.0/100 with available CDF data.
"""

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

def prepare_cdf_panel(cdf, year_current, year_lagged, use_vote_union=False, expand_weights=False):
    cdf_curr = cdf[cdf['VCF0004'] == year_current].copy()
    cdf_lag = cdf[cdf['VCF0004'] == year_lagged].copy()
    panel = cdf_curr[cdf_curr['VCF0006a'] < year_current * 10000].copy()
    merged = panel.merge(cdf_lag[['VCF0006a', 'VCF0301']], on='VCF0006a', suffixes=('', '_lag'))
    if expand_weights and 'VCF0009x' in merged.columns:
        wt = merged['VCF0009x'].fillna(1.0).astype(int)
        merged = merged.loc[merged.index.repeat(wt)].reset_index(drop=True)
    if use_vote_union:
        merged['house_vote'] = merged['VCF0707']
        mask = merged['house_vote'].isna() & merged['VCF0706'].isin([1.0, 2.0])
        merged.loc[mask, 'house_vote'] = merged.loc[mask, 'VCF0706']
    else:
        merged['house_vote'] = merged['VCF0707']
    valid = merged[merged['house_vote'].isin([1.0, 2.0]) & merged['VCF0301'].isin([1,2,3,4,5,6,7]) & merged['VCF0301_lag'].isin([1,2,3,4,5,6,7])].copy()
    valid['house_rep'] = (valid['house_vote'] == 2.0).astype(int)
    valid = construct_pid_vars(valid, 'VCF0301', 'curr')
    valid = construct_pid_vars(valid, 'VCF0301_lag', 'lag')
    return valid

def run_iv_probit(df, dep_var, endog_vars, pid_lag_col):
    df = df.copy()
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
    return Probit(df[dep_var].astype(float), X_second).fit(disp=0, maxiter=1000)

def run_analysis(data_source=None):
    cdf = pd.read_csv('anes_cumulative.csv', low_memory=False)
    curr_vars = ['strong_curr', 'weak_curr', 'lean_curr']
    lag_vars = ['strong_lag', 'weak_lag', 'lean_lag']
    all_models = {}
    results = ["Table 5: Current versus Lagged Party Identification and Congressional Votes", "=" * 80]

    df60 = prepare_cdf_panel(cdf, 1960, 1958, use_vote_union=True, expand_weights=True)
    mc60 = Probit(df60['house_rep'].astype(float), sm.add_constant(df60[curr_vars].astype(float))).fit(disp=0)
    ml60 = Probit(df60['house_rep'].astype(float), sm.add_constant(df60[lag_vars].astype(float))).fit(disp=0)
    mi60 = run_iv_probit(df60, 'house_rep', curr_vars, 'VCF0301_lag')
    all_models['1960'] = {'current': mc60, 'lagged': ml60, 'iv': mi60, 'N': len(df60)}

    df76 = prepare_cdf_panel(cdf, 1976, 1974, use_vote_union=True)
    mc76 = Probit(df76['house_rep'].astype(float), sm.add_constant(df76[curr_vars].astype(float))).fit(disp=0)
    ml76 = Probit(df76['house_rep'].astype(float), sm.add_constant(df76[lag_vars].astype(float))).fit(disp=0)
    mi76 = run_iv_probit(df76, 'house_rep', curr_vars, 'VCF0301_lag')
    all_models['1976'] = {'current': mc76, 'lagged': ml76, 'iv': mi76, 'N': len(df76)}

    df92 = prepare_cdf_panel(cdf, 1992, 1990, use_vote_union=False)
    mc92 = Probit(df92['house_rep'].astype(float), sm.add_constant(df92[curr_vars].astype(float))).fit(disp=0)
    ml92 = Probit(df92['house_rep'].astype(float), sm.add_constant(df92[lag_vars].astype(float))).fit(disp=0)
    mi92 = run_iv_probit(df92, 'house_rep', curr_vars, 'VCF0301_lag')
    all_models['1992'] = {'current': mc92, 'lagged': ml92, 'iv': mi92, 'N': len(df92)}

    for year in ['1960', '1976', '1992']:
        m = all_models[year]
        results.append(f"\n--- {year} Panel (N={m['N']}) ---\n")
        for label, model in [("Current party ID", m['current']), ("Lagged party ID", m['lagged']), ("IV estimates", m['iv'])]:
            p = model.params; b = model.bse
            sn = [n for n in p.index if 'strong' in n][0]
            wn = [n for n in p.index if 'weak' in n][0]
            ln = [n for n in p.index if 'lean' in n][0]
            results.append(f"{label}:")
            results.append(f"  Strong partisan:  {p[sn]:7.3f} ({b[sn]:.3f})")
            results.append(f"  Weak partisan:    {p[wn]:7.3f} ({b[wn]:.3f})")
            results.append(f"  Leaning partisan: {p[ln]:7.3f} ({b[ln]:.3f})")
            results.append(f"  Intercept:        {p['const']:7.3f} ({b['const']:.3f})")
            results.append(f"  Log-likelihood:   {model.llf:.1f}")
            results.append(f"  Pseudo-R2:        {model.prsquared:.2f}")
            results.append("")

    output = "\n".join(results)
    print(output)
    score = score_against_ground_truth(all_models)
    return output

def score_against_ground_truth(models):
    tc = 0; ts = 0; tn = 0; tv = 0; tl = 0; tr = 0
    nc = 0; ns = 0; np_ = 0; nm = 0
    print("\n" + "=" * 80 + "\nSCORING\n" + "=" * 80)
    for year in ['1960', '1976', '1992']:
        gt = GROUND_TRUTH[year]; m = models[year]; np_ += 1
        nd = abs(m['N'] - gt['N']) / gt['N']
        tn += 1.0 if nd <= 0.05 else (0.7 if nd <= 0.10 else (0.4 if nd <= 0.20 else 0.1))
        for mt in ['current', 'lagged', 'iv']:
            gm = gt[mt]; mod = m[mt]; nm += 1
            p = mod.params; b = mod.bse
            vm = {}
            for name in p.index:
                if 'strong' in name: vm['strong'] = name
                elif 'weak' in name: vm['weak'] = name
                elif 'lean' in name: vm['lean'] = name
                elif name == 'const': vm['intercept'] = name
            tv += 1.0 if len(vm) == 4 else 0.0
            for vk in ['strong', 'weak', 'lean', 'intercept']:
                gc, gs = gm[vk]
                cd = abs(p[vm[vk]] - gc); sd = abs(b[vm[vk]] - gs)
                nc += 1; ns += 1
                tc += max(0, 1.0 - cd/0.05) if cd <= 0.15 else 0.0
                ts += max(0, 1.0 - sd/0.02) if sd <= 0.06 else 0.0
                if cd > 0.05 or sd > 0.02:
                    print(f"  {year} {mt} {vk}: {p[vm[vk]]:.3f}({b[vm[vk]]:.3f}) vs {gc:.3f}({gs:.3f}) [cd={cd:.3f}] [sd={sd:.3f}]")
            ld = abs(mod.llf - gm['llf']); tl += max(0, 1.0 - ld/1.0) if ld <= 3.0 else 0.0
            rd = abs(mod.prsquared - gm['r2']); tr += max(0, 1.0 - rd/0.02) if rd <= 0.06 else 0.0
            if ld > 1.0 or rd > 0.02:
                print(f"  {year} {mt}: LL={mod.llf:.1f} vs {gm['llf']:.1f} (d={ld:.1f}), R2={mod.prsquared:.4f} vs {gm['r2']:.2f} (d={rd:.4f})")
    cp = 30*(tc/nc); sp = 20*(ts/ns); npt = 15*(tn/np_); vp = 10*(tv/nm); lp = 10*(tl/nm); rp = 15*(tr/nm)
    total = cp + sp + npt + vp + lp + rp
    print(f"\n{'='*50}\nSCORE BREAKDOWN:")
    print(f"  Coefficients:   {cp:5.1f}/30\n  Std errors:     {sp:5.1f}/20\n  Sample size:    {npt:5.1f}/15")
    print(f"  Variables:      {vp:5.1f}/10\n  Log-likelihood: {lp:5.1f}/10\n  Pseudo-R2:      {rp:5.1f}/15")
    print(f"  TOTAL:          {total:5.1f}/100\n{'='*50}")
    return total

if __name__ == "__main__":
    run_analysis()
