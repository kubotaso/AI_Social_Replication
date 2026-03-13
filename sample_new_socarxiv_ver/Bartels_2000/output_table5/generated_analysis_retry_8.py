"""
Replication of Table 5 from Bartels (2000)
"Partisanship and Voting Behavior, 1952-1996"

Table 5: Current versus Lagged Party Identification and Congressional Votes

Attempt 8:
- Uses panel data files (panel_1960.csv, panel_1976.csv, panel_1992.csv)
  which contain VCF0301_lagged (for 1960/76) and pre-extracted variables (for 1992)
- For 1960: Expand rows by VCF0009x frequency weights + VCF0707/VCF0706 union
  Uses VCF0301_lagged from panel file (confirmed identical to CDF merge approach)
- For 1976: VCF0707/VCF0706 union, uses VCF0301_lagged from panel file
- For 1992: Uses panel_1992.csv directly with pid_current, pid_lagged, vote_house
  (gives N=758 vs CDF's 759)
- IV estimation: Uses 6 lagged PID dummies as instruments
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
import warnings
warnings.filterwarnings('ignore')


# Ground truth values
GROUND_TRUTH = {
    '1960': {
        'N': 911,
        'current': {
            'strong': (1.358, 0.094), 'weak': (1.028, 0.083),
            'lean': (0.855, 0.131), 'intercept': (0.035, 0.053),
            'llf': -372.7, 'r2': 0.41
        },
        'lagged': {
            'strong': (1.363, 0.092), 'weak': (0.842, 0.078),
            'lean': (0.564, 0.125), 'intercept': (0.068, 0.051),
            'llf': -403.9, 'r2': 0.36
        },
        'iv': {
            'strong': (1.715, 0.173), 'weak': (0.728, 0.239),
            'lean': (1.081, 0.696), 'intercept': (0.032, 0.057),
            'llf': -403.9, 'r2': 0.36
        }
    },
    '1976': {
        'N': 682,
        'current': {
            'strong': (1.087, 0.105), 'weak': (0.624, 0.086),
            'lean': (0.622, 0.110), 'intercept': (-0.123, 0.054),
            'llf': -358.2, 'r2': 0.24
        },
        'lagged': {
            'strong': (0.966, 0.104), 'weak': (0.738, 0.089),
            'lean': (0.486, 0.109), 'intercept': (-0.063, 0.053),
            'llf': -371.3, 'r2': 0.21
        },
        'iv': {
            'strong': (1.123, 0.178), 'weak': (0.745, 0.251),
            'lean': (0.725, 0.438), 'intercept': (-0.102, 0.055),
            'llf': -371.3, 'r2': 0.21
        }
    },
    '1992': {
        'N': 760,
        'current': {
            'strong': (0.975, 0.094), 'weak': (0.627, 0.084),
            'lean': (0.472, 0.098), 'intercept': (-0.211, 0.051),
            'llf': -408.2, 'r2': 0.20
        },
        'lagged': {
            'strong': (1.061, 0.100), 'weak': (0.404, 0.077),
            'lean': (0.519, 0.101), 'intercept': (-0.168, 0.051),
            'llf': -416.2, 'r2': 0.19
        },
        'iv': {
            'strong': (1.516, 0.180), 'weak': (-0.225, 0.268),
            'lean': (1.824, 0.513), 'intercept': (-0.125, 0.053),
            'llf': -416.2, 'r2': 0.19
        }
    }
}


def construct_pid_vars(df, pid_col, suffix):
    """Construct directional party ID variables from 7-point scale."""
    df[f'strong_{suffix}'] = np.where(df[pid_col] == 7, 1,
                              np.where(df[pid_col] == 1, -1, 0))
    df[f'weak_{suffix}'] = np.where(df[pid_col] == 6, 1,
                            np.where(df[pid_col] == 2, -1, 0))
    df[f'lean_{suffix}'] = np.where(df[pid_col] == 5, 1,
                            np.where(df[pid_col] == 3, -1, 0))
    return df


def run_probit(df, dep_var, indep_vars):
    """Run probit regression."""
    X = sm.add_constant(df[indep_vars].astype(float))
    model = Probit(df[dep_var].astype(float), X).fit(disp=0, maxiter=1000)
    return model


def run_iv_probit(df, dep_var, endog_vars, pid_lag_col):
    """Run IV probit using 6 lagged PID dummies as instruments."""
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
    iv_model = Probit(df[dep_var].astype(float), X_second).fit(disp=0, maxiter=1000)
    return iv_model


def prepare_panel_1960():
    """Prepare 1960 panel data using panel file with VCF0301_lagged."""
    p60 = pd.read_csv('panel_1960.csv', low_memory=False)
    df = p60.copy()
    df['pid_lag'] = df['VCF0301_lagged']

    # Expand by frequency weights
    wt = df['VCF0009x'].fillna(1.0).astype(int)
    df = df.loc[df.index.repeat(wt)].reset_index(drop=True)

    # Union vote (VCF0707 + VCF0706)
    df['house_vote'] = df['VCF0707']
    mask = df['house_vote'].isna() & df['VCF0706'].isin([1.0, 2.0])
    df.loc[mask, 'house_vote'] = df.loc[mask, 'VCF0706']

    valid = df[
        df['house_vote'].isin([1.0, 2.0]) &
        df['VCF0301'].isin([1,2,3,4,5,6,7]) &
        df['pid_lag'].isin([1,2,3,4,5,6,7])
    ].copy()
    valid['house_rep'] = (valid['house_vote'] == 2.0).astype(int)
    valid = construct_pid_vars(valid, 'VCF0301', 'curr')
    valid = construct_pid_vars(valid, 'pid_lag', 'lag')
    return valid


def prepare_panel_1976():
    """Prepare 1976 panel data using panel file with VCF0301_lagged."""
    p76 = pd.read_csv('panel_1976.csv', low_memory=False)
    df = p76.copy()
    df['pid_lag'] = df['VCF0301_lagged']

    # Union vote
    df['house_vote'] = df['VCF0707']
    mask = df['house_vote'].isna() & df['VCF0706'].isin([1.0, 2.0])
    df.loc[mask, 'house_vote'] = df.loc[mask, 'VCF0706']

    valid = df[
        df['house_vote'].isin([1.0, 2.0]) &
        df['VCF0301'].isin([1,2,3,4,5,6,7]) &
        df['pid_lag'].isin([1,2,3,4,5,6,7])
    ].copy()
    valid['house_rep'] = (valid['house_vote'] == 2.0).astype(int)
    valid = construct_pid_vars(valid, 'VCF0301', 'curr')
    valid = construct_pid_vars(valid, 'pid_lag', 'lag')
    return valid


def prepare_panel_1992():
    """Prepare 1992 panel data using dedicated panel file."""
    p92 = pd.read_csv('panel_1992.csv', low_memory=False)

    valid = p92[
        p92['vote_house'].isin([1.0, 2.0]) &
        p92['pid_current'].isin([1,2,3,4,5,6,7]) &
        p92['pid_lagged'].isin([1,2,3,4,5,6,7])
    ].copy()
    valid['house_rep'] = (valid['vote_house'] == 2.0).astype(int)

    valid['strong_curr'] = np.where(valid['pid_current'] == 7, 1,
                            np.where(valid['pid_current'] == 1, -1, 0))
    valid['weak_curr'] = np.where(valid['pid_current'] == 6, 1,
                          np.where(valid['pid_current'] == 2, -1, 0))
    valid['lean_curr'] = np.where(valid['pid_current'] == 5, 1,
                          np.where(valid['pid_current'] == 3, -1, 0))
    valid['strong_lag'] = np.where(valid['pid_lagged'] == 7, 1,
                           np.where(valid['pid_lagged'] == 1, -1, 0))
    valid['weak_lag'] = np.where(valid['pid_lagged'] == 6, 1,
                         np.where(valid['pid_lagged'] == 2, -1, 0))
    valid['lean_lag'] = np.where(valid['pid_lagged'] == 5, 1,
                         np.where(valid['pid_lagged'] == 3, -1, 0))
    valid['pid_lag'] = valid['pid_lagged']
    return valid


def format_panel_results(year, n, model_curr, model_lag, model_iv):
    """Format results for one panel year."""
    lines = [f"\n--- {year} Panel (N={n}) ---\n"]

    for label, model in [("Current party ID", model_curr),
                          ("Lagged party ID", model_lag),
                          ("IV estimates", model_iv)]:
        params = model.params
        bse = model.bse
        strong_name = [n for n in params.index if 'strong' in n][0]
        weak_name = [n for n in params.index if 'weak' in n][0]
        lean_name = [n for n in params.index if 'lean' in n][0]

        lines.append(f"{label}:")
        lines.append(f"  Strong partisan:  {params[strong_name]:7.3f} ({bse[strong_name]:.3f})")
        lines.append(f"  Weak partisan:    {params[weak_name]:7.3f} ({bse[weak_name]:.3f})")
        lines.append(f"  Leaning partisan: {params[lean_name]:7.3f} ({bse[lean_name]:.3f})")
        lines.append(f"  Intercept:        {params['const']:7.3f} ({bse['const']:.3f})")
        lines.append(f"  Log-likelihood:   {model.llf:.1f}")
        lines.append(f"  Pseudo-R2:        {model.prsquared:.2f}")
        lines.append("")

    return "\n".join(lines)


def run_analysis(data_source=None):
    """Replicate Table 5 from Bartels (2000)."""
    results = ["Table 5: Current versus Lagged Party Identification and Congressional Votes"]
    results.append("=" * 80)

    curr_vars = ['strong_curr', 'weak_curr', 'lean_curr']
    lag_vars = ['strong_lag', 'weak_lag', 'lean_lag']

    all_models = {}

    # === 1960 Panel ===
    df60 = prepare_panel_1960()
    mc60 = run_probit(df60, 'house_rep', curr_vars)
    ml60 = run_probit(df60, 'house_rep', lag_vars)
    mi60 = run_iv_probit(df60, 'house_rep', curr_vars, 'pid_lag')
    results.append(format_panel_results(1960, len(df60), mc60, ml60, mi60))
    all_models['1960'] = {'current': mc60, 'lagged': ml60, 'iv': mi60, 'N': len(df60)}

    # === 1976 Panel ===
    df76 = prepare_panel_1976()
    mc76 = run_probit(df76, 'house_rep', curr_vars)
    ml76 = run_probit(df76, 'house_rep', lag_vars)
    mi76 = run_iv_probit(df76, 'house_rep', curr_vars, 'pid_lag')
    results.append(format_panel_results(1976, len(df76), mc76, ml76, mi76))
    all_models['1976'] = {'current': mc76, 'lagged': ml76, 'iv': mi76, 'N': len(df76)}

    # === 1992 Panel ===
    df92 = prepare_panel_1992()
    mc92 = run_probit(df92, 'house_rep', curr_vars)
    ml92 = run_probit(df92, 'house_rep', lag_vars)
    mi92 = run_iv_probit(df92, 'house_rep', curr_vars, 'pid_lag')
    results.append(format_panel_results(1992, len(df92), mc92, ml92, mi92))
    all_models['1992'] = {'current': mc92, 'lagged': ml92, 'iv': mi92, 'N': len(df92)}

    output = "\n".join(results)
    print(output)

    score = score_against_ground_truth(all_models)
    return output


def score_against_ground_truth(models):
    """Score results against ground truth."""
    total_coef_score = 0
    total_se_score = 0
    total_n_score = 0
    total_var_score = 0
    total_llf_score = 0
    total_r2_score = 0
    n_coefs = 0
    n_ses = 0
    n_panels = 0
    n_models = 0

    print("\n" + "=" * 80)
    print("SCORING")
    print("=" * 80)

    for year in ['1960', '1976', '1992']:
        gt = GROUND_TRUTH[year]
        m = models[year]
        n_panels += 1

        n_diff_pct = abs(m['N'] - gt['N']) / gt['N']
        if n_diff_pct <= 0.05:
            n_score = 1.0
        elif n_diff_pct <= 0.10:
            n_score = 0.7
        elif n_diff_pct <= 0.20:
            n_score = 0.4
        else:
            n_score = 0.1
        total_n_score += n_score
        print(f"\n{year}: N={m['N']} (target {gt['N']}), diff={n_diff_pct:.1%}, N_score={n_score:.1f}")

        for model_type in ['current', 'lagged', 'iv']:
            gt_model = gt[model_type]
            gen_model = m[model_type]
            n_models += 1

            params = gen_model.params
            bse = gen_model.bse

            var_map = {}
            for name in params.index:
                if 'strong' in name: var_map['strong'] = name
                elif 'weak' in name: var_map['weak'] = name
                elif 'lean' in name: var_map['lean'] = name
                elif name == 'const': var_map['intercept'] = name

            total_var_score += 1.0 if len(var_map) == 4 else 0.0

            for var_key in ['strong', 'weak', 'lean', 'intercept']:
                gt_coef, gt_se = gt_model[var_key]
                gen_coef = params[var_map[var_key]]
                gen_se = bse[var_map[var_key]]
                n_coefs += 1
                n_ses += 1

                coef_diff = abs(gen_coef - gt_coef)
                se_diff = abs(gen_se - gt_se)
                coef_score = max(0, 1.0 - coef_diff / 0.05) if coef_diff <= 0.15 else 0.0
                se_score = max(0, 1.0 - se_diff / 0.02) if se_diff <= 0.06 else 0.0
                total_coef_score += coef_score
                total_se_score += se_score

                if coef_diff > 0.05 or se_diff > 0.02:
                    print(f"  {year} {model_type} {var_key}: coef={gen_coef:.3f}({gen_se:.3f}) "
                          f"vs {gt_coef:.3f}({gt_se:.3f}) [c_diff={coef_diff:.3f}] [se_diff={se_diff:.3f}]")

            llf_diff = abs(gen_model.llf - gt_model['llf'])
            llf_score = max(0, 1.0 - llf_diff / 1.0) if llf_diff <= 3.0 else 0.0
            total_llf_score += llf_score

            r2_diff = abs(gen_model.prsquared - gt_model['r2'])
            r2_score = max(0, 1.0 - r2_diff / 0.02) if r2_diff <= 0.06 else 0.0
            total_r2_score += r2_score

            if llf_diff > 1.0 or r2_diff > 0.02:
                print(f"  {year} {model_type}: LL={gen_model.llf:.1f} vs {gt_model['llf']:.1f} "
                      f"(diff={llf_diff:.1f}), R2={gen_model.prsquared:.4f} vs {gt_model['r2']:.2f} "
                      f"(diff={r2_diff:.4f})")

    coef_pct = total_coef_score / n_coefs if n_coefs > 0 else 0
    se_pct = total_se_score / n_ses if n_ses > 0 else 0
    n_pct = total_n_score / n_panels if n_panels > 0 else 0
    var_pct = total_var_score / n_models if n_models > 0 else 0
    llf_pct = total_llf_score / n_models if n_models > 0 else 0
    r2_pct = total_r2_score / n_models if n_models > 0 else 0

    coef_pts = 30 * coef_pct
    se_pts = 20 * se_pct
    n_pts = 15 * n_pct
    var_pts = 10 * var_pct
    llf_pts = 10 * llf_pct
    r2_pts = 15 * r2_pct

    total = coef_pts + se_pts + n_pts + var_pts + llf_pts + r2_pts

    print(f"\n{'='*50}")
    print(f"SCORE BREAKDOWN:")
    print(f"  Coefficients:   {coef_pts:5.1f}/30 ({coef_pct:.1%})")
    print(f"  Std errors:     {se_pts:5.1f}/20 ({se_pct:.1%})")
    print(f"  Sample size:    {n_pts:5.1f}/15 ({n_pct:.1%})")
    print(f"  Variables:      {var_pts:5.1f}/10 ({var_pct:.1%})")
    print(f"  Log-likelihood: {llf_pts:5.1f}/10 ({llf_pct:.1%})")
    print(f"  Pseudo-R2:      {r2_pts:5.1f}/15 ({r2_pct:.1%})")
    print(f"  TOTAL:          {total:5.1f}/100")
    print(f"{'='*50}")

    return total


if __name__ == "__main__":
    result = run_analysis()
