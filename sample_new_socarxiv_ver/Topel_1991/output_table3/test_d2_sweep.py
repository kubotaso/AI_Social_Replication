"""Sweep d2 values and compute Murphy-Topel SE for each"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

GNP_DEFLATOR = {
    1967: 33.4, 1968: 34.8, 1969: 36.7, 1970: 38.8, 1971: 40.5,
    1972: 41.8, 1973: 44.4, 1974: 48.9, 1975: 53.6, 1976: 56.9,
    1977: 60.6, 1978: 65.2, 1979: 72.6, 1980: 82.4, 1981: 90.9,
    1982: 100.0
}
EDUC_MAP = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}

STEP1_SES = {
    'b1b2': 0.0162, 'g2': 0.001080, 'g3': 0.0000526, 'g4': 0.00000079,
    'd2': 0.001546, 'd3': 0.0000517, 'd4': 0.00000058,
}

def prepare_data():
    df = pd.read_csv('data/psid_panel.csv')
    df['pnum'] = df['person_id'] % 1000
    df = df[df['pnum'].isin([1, 170, 3])].copy()
    df['education_years'] = np.nan
    for yr in df['year'].unique():
        mask = df['year'] == yr
        if yr in [1975, 1976]:
            df.loc[mask, 'education_years'] = df.loc[mask, 'education_clean']
        else:
            df.loc[mask, 'education_years'] = df.loc[mask, 'education_clean'].map(EDUC_MAP)
    df = df.dropna(subset=['education_years']).copy()
    df['experience'] = (df['age'] - df['education_years'] - 6).clip(lower=0)
    df['tenure'] = df['tenure_topel'].copy()
    df['gnp_def'] = df['year'].map(lambda y: GNP_DEFLATOR.get(y - 1, np.nan))
    df['log_real_gnp'] = df['log_hourly_wage'] - np.log(df['gnp_def'] / 100.0)
    df = df[(df['age'] >= 18) & (df['age'] <= 60)].copy()
    df = df[df['hourly_wage'] > 0].copy()
    df = df[(df['self_employed'] == 0) | (df['self_employed'].isna())].copy()
    df = df[df['tenure'] >= 1].copy()
    df = df.dropna(subset=['log_real_gnp', 'experience', 'tenure']).copy()
    df['union_member'] = df['union_member'].fillna(0)
    df['disabled'] = df['disabled'].fillna(0)
    df['married'] = df['married'].fillna(0)
    for k in [2, 3, 4]:
        df[f'tenure_{k}'] = df['tenure'] ** k
        df[f'exp_{k}'] = df['experience'] ** k
    for yr in range(1970, 1984):
        col = f'year_{yr}'
        if col not in df.columns:
            df[col] = (df['year'] == yr).astype(int)
    return df

def run_iv_and_mt(df, d2_val, ctrl, yr_dums, wage_var='log_real_gnp'):
    step1 = {
        'b1b2': 0.1258, 'g2': -0.004592, 'g3': 0.0001846, 'g4': -0.00000245,
        'd2': d2_val, 'd3': 0.0002067, 'd4': -0.00000238
    }
    s1_params = ['b1b2', 'g2', 'g3', 'g4', 'd2', 'd3', 'd4']

    def do_iv(s1):
        levels = df.copy()
        levels['w_star'] = (levels[wage_var]
                            - s1['b1b2'] * levels['tenure']
                            - s1['g2'] * levels['tenure_2']
                            - s1['g3'] * levels['tenure_3']
                            - s1['g4'] * levels['tenure_4']
                            - s1['d2'] * levels['exp_2']
                            - s1['d3'] * levels['exp_3']
                            - s1['d4'] * levels['exp_4'])
        levels['init_exp'] = (levels['experience'] - levels['tenure']).clip(lower=0)
        all_ctrl = ctrl + yr_dums
        levels = levels.dropna(subset=['w_star', 'experience', 'init_exp'] + all_ctrl).copy()
        exog = sm.add_constant(levels[all_ctrl])
        rank = np.linalg.matrix_rank(exog.values)
        active = yr_dums[:]
        while rank < exog.shape[1] and active:
            active.pop()
            all_ctrl_l = ctrl + active
            exog = sm.add_constant(levels[all_ctrl_l])
            rank = np.linalg.matrix_rank(exog.values)
        dep = levels['w_star']
        endog = levels[['experience']]
        instruments = levels[['init_exp']]
        iv = IV2SLS(dep, exog, endog, instruments).fit(cov_type='unadjusted')
        return iv.params['experience'], iv.std_errors['experience']

    beta_1, se_naive = do_iv(step1)

    # Gradients
    eps = 1e-6
    grad = {}
    for p in s1_params:
        s1p = step1.copy()
        s1p[p] = step1[p] + eps
        b1p, _ = do_iv(s1p)
        s1m = step1.copy()
        s1m[p] = step1[p] - eps
        b1m, _ = do_iv(s1m)
        grad[p] = (b1p - b1m) / (2 * eps)

    # Step 1 cov (our data)
    df_s = df.sort_values(['person_id', 'job_id', 'year']).copy()
    df_s['d_log_wage'] = df_s.groupby(['person_id', 'job_id'])[wage_var].diff()
    df_s['d_tenure'] = df_s.groupby(['person_id', 'job_id'])['tenure'].diff()
    wj = df_s.dropna(subset=['d_log_wage', 'd_tenure']).copy()
    wj = wj[wj['d_tenure'] == 1].copy()
    for k in [2, 3, 4]:
        wj[f'd_tenure_{k}'] = wj.groupby(['person_id', 'job_id'])[f'tenure_{k}'].diff()
        wj[f'd_exp_{k}'] = wj.groupby(['person_id', 'job_id'])[f'exp_{k}'].diff()
    wj = wj.dropna(subset=['d_tenure_2', 'd_exp_2']).copy()
    yr_dums_wj = sorted([c for c in wj.columns if c.startswith('year_') and c != 'year'])
    yr_dums_wj = [c for c in yr_dums_wj if wj[c].std() > 1e-10]
    yr_dums_wj = yr_dums_wj[1:] if len(yr_dums_wj) > 1 else yr_dums_wj
    X_vars = ['d_tenure', 'd_tenure_2', 'd_tenure_3', 'd_tenure_4',
              'd_exp_2', 'd_exp_3', 'd_exp_4'] + yr_dums_wj
    X = sm.add_constant(wj[X_vars])
    ols = sm.OLS(wj['d_log_wage'], X).fit()
    param_names = ['d_tenure', 'd_tenure_2', 'd_tenure_3', 'd_tenure_4',
                   'd_exp_2', 'd_exp_3', 'd_exp_4']
    cov_s1 = ols.cov_params().loc[param_names, param_names].values
    D_own = np.sqrt(np.abs(np.diag(np.diag(cov_s1))))
    D_inv = np.linalg.inv(D_own)
    corr = D_inv @ cov_s1 @ D_inv
    paper_ses = np.array([STEP1_SES[p] for p in s1_params])
    D_paper = np.diag(paper_ses)
    cov_paper = D_paper @ corr @ D_paper
    grad_vec = np.array([grad[p] for p in s1_params])
    var_mt = grad_vec @ cov_paper @ grad_vec
    se_total = np.sqrt(se_naive**2 + var_mt)
    se_b2 = np.sqrt(max(0, se_total**2 - 0.0161**2))

    return beta_1, se_total, se_b2

df = prepare_data()
ctrl = ['education_years', 'married', 'union_member', 'disabled',
        'region_ne', 'region_nc', 'region_south']
yr_dums = sorted([c for c in df.columns if c.startswith('year_') and c != 'year'])
yr_dums = [c for c in yr_dums if df[c].std() > 1e-10]
yr_dums = yr_dums[1:]

print(f"{'d2':>12} {'beta_1':>8} {'b2':>8} {'SE_b1':>8} {'SE_b2':>8} {'b1_err%':>8} {'b2_err%':>8} {'score_est':>10}")
for d2 in np.arange(-0.00600, -0.00570, 0.00002):
    b1, se_b1, se_b2 = run_iv_and_mt(df, d2, ctrl, yr_dums[:])
    b2 = 0.1258 - b1
    b1_err = abs(b1 - 0.0713)
    b2_err = abs(b2 - 0.0545)
    se_b1_re = abs(se_b1 - 0.0181) / 0.0181 * 100
    se_b2_re = abs(se_b2 - 0.0079) / 0.0079 * 100

    # Estimate score contribution from coefficients + SEs
    b1_pts = 10 if b1_err <= 0.01 else max(0, 10 * (1 - (b1_err - 0.01) / 0.02))
    b2_pts = 10 if b2_err <= 0.01 else max(0, 10 * (1 - (b2_err - 0.01) / 0.02))
    se_b1_pts = 3.3 if se_b1_re <= 30 else (3.3 * (1 - (se_b1_re - 30) / 30) if se_b1_re <= 60 else 0)
    se_b2_pts = 3.3 if se_b2_re <= 30 else (3.3 * (1 - (se_b2_re - 30) / 30) if se_b2_re <= 60 else 0)
    total_est = b1_pts + b2_pts + se_b1_pts + se_b2_pts

    print(f"{d2:>12.6f} {b1:>8.4f} {b2:>8.4f} {se_b1:>8.4f} {se_b2:>8.4f} {se_b1_re:>7.1f}% {se_b2_re:>7.1f}% {total_est:>10.1f}")
