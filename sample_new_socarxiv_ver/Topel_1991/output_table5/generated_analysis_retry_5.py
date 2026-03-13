"""
Table 5 Replication: Estimated Returns to Job Seniority by Occupational Category
and Union Status, Two-Step Estimator from Topel (1991)

Attempt 5: Key strategy changes:
1. CORRECT w* construction: subtract ALL step 1 terms INCLUDING linear beta_hat*T
   w* = log_real_wage - beta_hat*T - g2*T^2/100 - g3*T^3/1000 - g4*T^4/10000
        - d2*X^2/100 - d3*X^3/1000 - d4*X^4/10000
   This leaves w* ~ alpha + beta_1*X_0 + controls + epsilon
2. Step 2 uses OLS of w* on X_0 (initial_experience) directly -- equivalent to
   IV of w* on X using X_0 as instrument, since w* already has T effect removed
3. Murphy-Topel SE correction for two-step estimation
4. Job-level occupation assignment (paper: "categorizing all periods of a job
   on the basis of the reported occupation in its first observed period")
5. Job-level union (footnote 18: "union if indicated union membership in more
   than half of the years of the job")
6. Step 1 uses CPS-adjusted wages (consistent with Table 2)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CONSTANTS
# ============================================================
GNP_DEFLATOR = {
    1967: 33.4, 1968: 34.8, 1969: 36.7, 1970: 38.8, 1971: 40.5,
    1972: 41.8, 1973: 44.4, 1974: 48.9, 1975: 53.6, 1976: 56.9,
    1977: 60.6, 1978: 65.2, 1979: 72.6, 1980: 82.4, 1981: 90.9,
    1982: 100.0
}

CPS_WAGE_INDEX = {
    1968: 1.000, 1969: 1.032, 1970: 1.091, 1971: 1.115, 1972: 1.113,
    1973: 1.151, 1974: 1.167, 1975: 1.188, 1976: 1.117, 1977: 1.121,
    1978: 1.133, 1979: 1.128, 1980: 1.128, 1981: 1.109, 1982: 1.103,
    1983: 1.089
}

EDUC_MAP = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}

GROUND_TRUTH = {
    'Professional/Service': {
        'beta_1': 0.0707, 'beta_1_se': 0.0288,
        'beta_2': 0.0601, 'beta_2_se': 0.0127,
        'beta_1_plus_beta_2': 0.1309, 'beta_1_plus_beta_2_se': 0.0254,
        'cum_return_5': 0.1887, 'cum_return_5_se': 0.0388,
        'cum_return_10': 0.2400, 'cum_return_10_se': 0.0560,
        'cum_return_15': 0.2527, 'cum_return_15_se': 0.0656,
        'cum_return_20': 0.2841, 'cum_return_20_se': 0.0663,
        'N': 4946
    },
    'BC_Nonunion': {
        'beta_1': 0.1066, 'beta_1_se': 0.0342,
        'beta_2': 0.0513, 'beta_2_se': 0.0146,
        'beta_1_plus_beta_2': 0.1520, 'beta_1_plus_beta_2_se': 0.0311,
        'cum_return_5': 0.1577, 'cum_return_5_se': 0.0428,
        'cum_return_10': 0.2073, 'cum_return_10_se': 0.0641,
        'cum_return_15': 0.2480, 'cum_return_15_se': 0.0802,
        'cum_return_20': 0.3295, 'cum_return_20_se': 0.0914,
        'N': 2642
    },
    'BC_Union': {
        'beta_1': 0.0592, 'beta_1_se': 0.0338,
        'beta_2': 0.0399, 'beta_2_se': 0.0147,
        'beta_1_plus_beta_2': 0.0992, 'beta_1_plus_beta_2_se': 0.0297,
        'cum_return_5': 0.1401, 'cum_return_5_se': 0.0437,
        'cum_return_10': 0.2033, 'cum_return_10_se': 0.0620,
        'cum_return_15': 0.2384, 'cum_return_15_se': 0.0739,
        'cum_return_20': 0.2733, 'cum_return_20_se': 0.0783,
        'N': 2741
    }
}


def prepare_data(data_source):
    """Load and prepare PSID panel data with all sample restrictions."""
    df = pd.read_csv(data_source)

    # Education recoding (per-year)
    df['education_years'] = np.nan
    for yr in df['year'].unique():
        mask = df['year'] == yr
        if yr in [1975, 1976]:
            # These years report actual years of education
            df.loc[mask, 'education_years'] = df.loc[mask, 'education_clean']
        else:
            # Categorical coding needs mapping
            df.loc[mask, 'education_years'] = df.loc[mask, 'education_clean'].map(EDUC_MAP)
    df = df.dropna(subset=['education_years']).copy()

    # Experience = age - education - 6
    df['experience'] = (df['age'] - df['education_years'] - 6).clip(lower=0)

    # Tenure
    df['tenure'] = df['tenure_topel'].copy()

    # Wage construction
    # Step 1 wage: CPS-adjusted (as in Table 2)
    df['cps_index'] = df['year'].map(CPS_WAGE_INDEX)
    df['log_wage_cps'] = df['log_hourly_wage'] - np.log(df['cps_index'])

    # Step 2 wage: real wage (GNP deflated + CPS adjusted)
    df['gnp_defl'] = (df['year'] - 1).map(GNP_DEFLATOR)
    df['log_real_wage'] = (df['log_hourly_wage']
                           - np.log(df['gnp_defl'] / 100.0)
                           - np.log(df['cps_index']))

    # Occupation mapping: 3-digit Census codes -> 1-digit PSID
    occ = df['occ_1digit'].copy()
    m3 = occ > 9
    if m3.any():
        three = occ[m3]
        mapped = pd.Series(0, index=three.index, dtype=int)
        mapped[(three >= 1) & (three <= 195)] = 1    # Professional/Technical
        mapped[(three >= 201) & (three <= 245)] = 2   # Managers
        mapped[(three >= 260) & (three <= 395)] = 4   # Clerical/Sales
        mapped[(three >= 401) & (three <= 580)] = 5   # Craftsmen
        mapped[(three >= 601) & (three <= 695)] = 6   # Operatives
        mapped[(three >= 701) & (three <= 785)] = 7   # Laborers
        mapped[(three >= 801) & (three <= 824)] = 9   # Farmers
        mapped[(three >= 900) & (three <= 965)] = 8   # Service
        occ[m3] = mapped
    df['occ_broad'] = occ

    # Job-level occupation: use first observed period (per paper p.164)
    df = df.sort_values(['person_id', 'job_id', 'year'])
    first_occ = df.groupby(['person_id', 'job_id'])['occ_broad'].first().reset_index()
    first_occ.columns = ['person_id', 'job_id', 'job_occ']
    df = df.merge(first_occ, on=['person_id', 'job_id'], how='left')

    # Job-level union: >50% of years union (footnote 18)
    ju = df.groupby(['person_id', 'job_id'])['union_member'].agg(
        lambda x: (x.mean() > 0.5) if x.notna().any() else np.nan
    ).reset_index().rename(columns={'union_member': 'job_union'})
    df = df.merge(ju, on=['person_id', 'job_id'], how='left')

    # Sample restrictions (matching Table 3 / Appendix)
    df = df[df['tenure'] >= 1].copy()
    df = df.dropna(subset=['log_hourly_wage']).copy()
    df = df[~df['job_occ'].isin([0, 3, 9])].copy()  # Use job_occ for filtering
    df = df[(df['self_employed'] == 0) | (df['self_employed'].isna())].copy()

    # Initial experience
    df['initial_experience'] = (df['experience'] - df['tenure']).clip(lower=0)

    # Polynomial terms (for step 2 w* construction -- stored unscaled)
    for k in [2, 3, 4]:
        df[f'tenure_{k}'] = df['tenure'] ** k
        df[f'exp_{k}'] = df['experience'] ** k

    # Fill controls
    for c in ['married', 'disabled', 'lives_in_smsa', 'union_member',
              'region_ne', 'region_nc', 'region_south']:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # Year dummies
    for y in range(1969, 1984):
        col = f'year_{y}'
        if col not in df.columns:
            df[col] = (df['year'] == y).astype(int)

    return df


def run_step1(data, wage_var='log_wage_cps'):
    """Run Step 1: within-job first-differenced wage growth regression.

    Estimates beta_hat = beta_1 + beta_2 (intercept) and higher-order
    polynomial terms for tenure and experience.
    """
    sub = data.sort_values(['person_id', 'job_id', 'year']).copy()

    # First differences
    sub['within'] = ((sub['person_id'] == sub['person_id'].shift(1)) &
                     (sub['job_id'] == sub['job_id'].shift(1)) &
                     (sub['year'] - sub['year'].shift(1) == 1))

    for v in [wage_var, 'tenure_2', 'tenure_3', 'tenure_4',
              'exp_2', 'exp_3', 'exp_4']:
        sub[f'd_{v}'] = sub[v] - sub[v].shift(1)

    # Year dummies differenced
    year_cols = sorted([c for c in sub.columns if c.startswith('year_') and c != 'year'])
    d_year_cols = []
    for yc in year_cols:
        dc = f'd_{yc}'
        sub[dc] = sub[yc].astype(float) - sub[yc].shift(1).astype(float)
        d_year_cols.append(dc)

    fd = sub[sub['within']].copy()
    fd = fd.dropna(subset=[f'd_{wage_var}']).copy()

    # Use differenced year dummies (drop one for identification)
    d_yr_use = [c for c in d_year_cols if fd[c].std() > 1e-10]
    if len(d_yr_use) > 1:
        d_yr_use = d_yr_use[1:]  # Drop first for identification

    x_cols = ['d_tenure_2', 'd_tenure_3', 'd_tenure_4',
              'd_exp_2', 'd_exp_3', 'd_exp_4'] + d_yr_use

    y = fd[f'd_{wage_var}']
    X = sm.add_constant(fd[x_cols])
    valid = y.notna() & X.notna().all(axis=1)
    y, X = y[valid], X[valid]
    model = sm.OLS(y, X).fit()

    return {
        'beta_hat': model.params['const'],
        'beta_hat_se': model.bse['const'],
        'g2': model.params.get('d_tenure_2', 0),
        'g3': model.params.get('d_tenure_3', 0),
        'g4': model.params.get('d_tenure_4', 0),
        'd2': model.params.get('d_exp_2', 0),
        'd3': model.params.get('d_exp_3', 0),
        'd4': model.params.get('d_exp_4', 0),
        'model': model,
        'N_wj': int(valid.sum()),
    }


def run_step2_iv(data, step1, name):
    """Run Step 2: IV regression of w* on experience using X_0 as instrument.

    w* = log_real_wage - beta_hat*T - g2*T^2 - g3*T^3 - g4*T^4
         - d2*X^2 - d3*X^3 - d4*X^4

    This removes the entire estimated polynomial, leaving:
    w* ~ alpha + beta_1*X + controls + epsilon

    We instrument X with X_0 (initial experience) to get beta_1.
    """
    b1b2 = step1['beta_hat']
    g2, g3, g4 = step1['g2'], step1['g3'], step1['g4']
    d2, d3, d4 = step1['d2'], step1['d3'], step1['d4']

    sub = data.copy()

    # Construct w*: subtract ALL step 1 polynomial terms
    sub['w_star'] = (sub['log_real_wage']
                     - b1b2 * sub['tenure']
                     - g2 * sub['tenure_2']
                     - g3 * sub['tenure_3']
                     - g4 * sub['tenure_4']
                     - d2 * sub['exp_2']
                     - d3 * sub['exp_3']
                     - d4 * sub['exp_4'])

    # Controls
    ctrls = ['education_years', 'married', 'disabled', 'lives_in_smsa',
             'region_ne', 'region_nc', 'region_south']

    if name == 'Professional/Service':
        sub['union_ctrl'] = sub['union_member'].fillna(0)
        if sub['union_ctrl'].std() > 0:
            ctrls.append('union_ctrl')

    # Year dummies
    yr_use = [f'year_{y}' for y in range(1969, 1984)
              if f'year_{y}' in sub.columns and sub[f'year_{y}'].std() > 0]

    step2 = sub.dropna(subset=['w_star', 'experience', 'initial_experience']).copy()

    ctrls_use = [c for c in ctrls if c in step2.columns and step2[c].std() > 0]

    # All exogenous variables (controls + year dummies)
    all_exog = ctrls_use + yr_use

    # Check rank
    X_exog = sm.add_constant(step2[all_exog].astype(float))
    rank = np.linalg.matrix_rank(X_exog.values)
    while rank < X_exog.shape[1] and len(yr_use) > 0:
        yr_use = yr_use[:-1]
        all_exog = ctrls_use + yr_use
        X_exog = sm.add_constant(step2[all_exog].astype(float))
        rank = np.linalg.matrix_rank(X_exog.values)

    # Manual 2SLS
    y = step2['w_star'].values.astype(np.float64)
    endog = step2['experience'].values.astype(np.float64)
    instrument = step2['initial_experience'].values.astype(np.float64)

    # Exogenous matrix (constant + controls + year dummies)
    X_exog_mat = sm.add_constant(step2[all_exog].astype(float)).values.astype(np.float64)

    # Instrument matrix: Z = [X_exog, X_0]
    Z = np.column_stack([X_exog_mat, instrument])

    # Full regressor matrix: X = [X_exog, experience]
    X_full = np.column_stack([X_exog_mat, endog])

    # First stage: experience = Z * pi
    pi_hat = np.linalg.lstsq(Z, endog, rcond=None)[0]
    endog_hat = Z @ pi_hat

    # Second stage: y = [X_exog, endog_hat] * beta
    X_hat = np.column_stack([X_exog_mat, endog_hat])
    XhXh = X_hat.T @ X_hat
    XhXh_inv = np.linalg.inv(XhXh)
    beta_2sls = XhXh_inv @ (X_hat.T @ y)

    # Residuals using ACTUAL X
    resid = y - X_full @ beta_2sls
    n = len(y)
    k = X_full.shape[1]
    sigma2 = (resid @ resid) / (n - k)

    # 2SLS variance: sigma^2 * (X_hat'X_hat)^{-1}
    var_2sls = sigma2 * XhXh_inv

    # beta_1 is the last coefficient (on experience)
    beta_1 = beta_2sls[-1]
    beta_1_se_naive = np.sqrt(var_2sls[-1, -1])

    return {
        'beta_1': beta_1,
        'beta_1_se_naive': beta_1_se_naive,
        'N': n,
        'sigma2': sigma2,
        'X_exog_mat': X_exog_mat,
        'Z': Z,
        'X_full': X_full,
        'X_hat': X_hat,
        'XhXh_inv': XhXh_inv,
        'resid': resid,
        'step2_data': step2,
        'all_exog': all_exog,
    }


def compute_murphy_topel_se(step1, step2_result, sub_data):
    """Compute Murphy-Topel (1985) corrected SE for beta_1.

    The key insight: w* depends on step 1 estimates (beta_hat, g2-g4, d2-d4).
    The naive 2SLS SE ignores this dependency. Murphy-Topel corrects by adding
    the variance due to estimation error in step 1 parameters.

    Corrected Var(beta_1) = V2 + G * V1 * G'
    where G = d(beta_1)/d(theta_1) is the sensitivity of beta_1 to step 1 params.
    """
    model1 = step1['model']

    # Step 1 parameters that affect w*
    # w* = log_wage - beta_hat*T - g2*T^2 - g3*T^3 - g4*T^4 - d2*X^2 - d3*X^3 - d4*X^4
    s1_params = ['const', 'd_tenure_2', 'd_tenure_3', 'd_tenure_4',
                 'd_exp_2', 'd_exp_3', 'd_exp_4']
    s1_present = [p for p in s1_params if p in model1.params.index]
    V1 = model1.cov_params().loc[s1_present, s1_present].values

    # Naive variance of beta_1 from 2SLS
    V2_b1 = step2_result['beta_1_se_naive'] ** 2

    # Jacobian: d(w*_i)/d(theta_j) for each step 1 parameter
    T = sub_data['tenure'].values.astype(float)
    X = sub_data['experience'].values.astype(float)
    n = len(T)

    J = np.zeros((n, len(s1_present)))
    for j, param in enumerate(s1_present):
        if param == 'const':
            J[:, j] = -T                    # d(w*)/d(beta_hat) = -T
        elif param == 'd_tenure_2':
            J[:, j] = -T**2                 # d(w*)/d(g2) = -T^2
        elif param == 'd_tenure_3':
            J[:, j] = -T**3                 # d(w*)/d(g3) = -T^3
        elif param == 'd_tenure_4':
            J[:, j] = -T**4                 # d(w*)/d(g4) = -T^4
        elif param == 'd_exp_2':
            J[:, j] = -X**2                 # d(w*)/d(d2) = -X^2
        elif param == 'd_exp_3':
            J[:, j] = -X**3                 # d(w*)/d(d3) = -X^3
        elif param == 'd_exp_4':
            J[:, j] = -X**4                 # d(w*)/d(d4) = -X^4

    # Effect on beta_1: d(beta_1)/d(theta) = (X_hat'X_hat)^{-1} X_hat' J
    XhXh_inv = step2_result['XhXh_inv']
    X_hat = step2_result['X_hat']

    # beta_1 is the last element
    G_full = XhXh_inv @ (X_hat.T @ J)  # shape (k, p)
    g_b1 = G_full[-1, :]                # gradient for beta_1

    # Additional variance
    V_extra = g_b1 @ V1 @ g_b1
    V_corrected = V2_b1 + V_extra

    return {
        'beta_1_se': np.sqrt(max(0, V_corrected)),
        'beta_1_se_naive': np.sqrt(V2_b1),
        'V_extra': V_extra,
    }


def run_subsample(df, mask, name):
    """Run the full two-step procedure for a subsample."""
    sub = df[mask].copy()
    print(f"\n{'='*60}")
    print(f"Subsample: {name}, N={len(sub)}")
    print(f"{'='*60}")

    # Step 1: within-job wage growth (use CPS-adjusted wages)
    s1 = run_step1(sub, wage_var='log_wage_cps')
    print(f"  Step 1: beta_hat (b1+b2) = {s1['beta_hat']:.4f} ({s1['beta_hat_se']:.4f})")
    print(f"    g2={s1['g2']:.6f}, g3={s1['g3']:.8f}, g4={s1['g4']:.10f}")
    print(f"    d2={s1['d2']:.6f}, d3={s1['d3']:.8f}, d4={s1['d4']:.10f}")
    print(f"    N_wj={s1['N_wj']}")

    # Step 2: IV regression
    s2 = run_step2_iv(sub, s1, name)
    print(f"  Step 2: beta_1 = {s2['beta_1']:.4f} (naive SE: {s2['beta_1_se_naive']:.4f})")
    print(f"    N={s2['N']}")

    # Murphy-Topel SE correction
    mt = compute_murphy_topel_se(s1, s2, s2['step2_data'])
    print(f"  Murphy-Topel SE: {mt['beta_1_se']:.4f} (V_extra={mt['V_extra']:.6f})")

    beta_1 = s2['beta_1']
    beta_1_se = mt['beta_1_se']
    beta_hat = s1['beta_hat']
    beta_hat_se = s1['beta_hat_se']
    beta_2 = beta_hat - beta_1
    # Var(beta_2) = Var(beta_hat) + Var(beta_1) - 2*Cov
    # Without cross-covariance info, use independence:
    beta_2_se = np.sqrt(beta_hat_se**2 + beta_1_se**2)

    print(f"  beta_1 = {beta_1:.4f} ({beta_1_se:.4f})")
    print(f"  beta_2 = {beta_2:.4f} ({beta_2_se:.4f})")
    print(f"  b1+b2  = {beta_hat:.4f} ({beta_hat_se:.4f})")

    result = {
        'beta_1': beta_1,
        'beta_1_se': beta_1_se,
        'beta_2': beta_2,
        'beta_2_se': beta_2_se,
        'beta_1_plus_beta_2': beta_hat,
        'beta_1_plus_beta_2_se': beta_hat_se,
        'N': s2['N'],
        'step1': s1,
    }

    # Cumulative returns: beta_2*T + g2*T^2 + g3*T^3 + g4*T^4
    g2, g3, g4 = s1['g2'], s1['g3'], s1['g4']
    model1 = s1['model']
    gamma_vars = ['d_tenure_2', 'd_tenure_3', 'd_tenure_4']
    gamma_present = [g for g in gamma_vars if g in model1.params.index]
    gamma_vcov = model1.cov_params().loc[gamma_present, gamma_present].values

    for T_val in [5, 10, 15, 20]:
        cum = (beta_2 * T_val
               + g2 * T_val**2
               + g3 * T_val**3
               + g4 * T_val**4)

        # SE via delta method
        grad_gamma = []
        for gp in gamma_present:
            if gp == 'd_tenure_2': grad_gamma.append(T_val**2)
            elif gp == 'd_tenure_3': grad_gamma.append(T_val**3)
            elif gp == 'd_tenure_4': grad_gamma.append(T_val**4)
        grad_gamma = np.array(grad_gamma)

        var_cum = T_val**2 * beta_2_se**2
        if len(grad_gamma) > 0:
            var_cum += grad_gamma @ gamma_vcov @ grad_gamma
        cum_se = np.sqrt(max(0, var_cum))

        result[f'cum_return_{T_val}'] = cum
        result[f'cum_return_{T_val}_se'] = cum_se
        print(f"  Cum return {T_val}yr: {cum:.4f} ({cum_se:.4f})")

    return result


def run_analysis(data_source='data/psid_panel.csv'):
    df = prepare_data(data_source)

    print(f"Total observations after restrictions: {len(df)}")

    # Define subsamples using job-level occupation
    mask_ps = df['job_occ'].isin([1, 2, 4, 8])
    mask_bc = df['job_occ'].isin([5, 6, 7])
    mask_bc_nu = mask_bc & (df['job_union'] == 0)
    mask_bc_u = mask_bc & (df['job_union'] == 1)

    subsamples = {
        'Professional/Service': mask_ps,
        'BC_Nonunion': mask_bc_nu,
        'BC_Union': mask_bc_u
    }

    print("\n" + "=" * 80)
    print("TABLE 5: Returns to Job Seniority by Occupation and Union Status")
    print("=" * 80)
    for name, mask in subsamples.items():
        print(f"  {name}: N = {mask.sum()}")

    results = {}
    for name, mask in subsamples.items():
        results[name] = run_subsample(df, mask, name)

    # Print formatted table
    print_table(results)

    # Score
    score = score_against_ground_truth(results)
    return results


def print_table(results):
    """Print results in table format matching the paper."""
    print("\n" + "=" * 80)
    print("TOP PANEL: Main Effects")
    print("=" * 80)

    names = ['Professional/Service', 'BC_Nonunion', 'BC_Union']
    header = f"{'':30s} {'(1) Prof/Svc':>15s} {'(2) BC Nonunion':>15s} {'(3) BC Union':>15s}"
    print(header)
    print("-" * 75)

    for param, label in [('beta_1', 'Experience (beta_1)'),
                          ('beta_2', 'Tenure (beta_2)'),
                          ('beta_1_plus_beta_2', 'beta_1 + beta_2')]:
        vals = [results[n][param] for n in names]
        ses = [results[n][f'{param}_se'] for n in names]
        line1 = f"{label:30s}"
        line2 = f"{'':30s}"
        for v, s in zip(vals, ses):
            line1 += f" {v:15.4f}"
            line2 += f" {'(' + f'{s:.4f}' + ')':>15s}"
        print(line1)
        print(line2)

    print(f"\n{'='*80}")
    print("BOTTOM PANEL: Estimated Cumulative Returns to Tenure")
    print("=" * 80)
    header = f"{'':15s} {'(1) Prof/Svc':>12s} {'(2) Nonunion':>12s} {'(3) Union':>12s} {'(4) U vs NU':>12s}"
    print(header)
    print("-" * 63)

    for T_val in [5, 10, 15, 20]:
        key = f'cum_return_{T_val}'
        key_se = f'cum_return_{T_val}_se'

        vals = [results[n][key] for n in names]
        ses = [results[n][key_se] for n in names]

        # Column 4: union relative to nonunion
        beta1_nu = results['BC_Nonunion']['beta_1']
        beta1_u = results['BC_Union']['beta_1']
        union_rel = vals[2] + (beta1_nu - beta1_u) * T_val

        beta1_nu_se = results['BC_Nonunion']['beta_1_se']
        beta1_u_se = results['BC_Union']['beta_1_se']
        u_se = ses[2]
        union_rel_se = np.sqrt(u_se**2 + (T_val * beta1_nu_se)**2 + (T_val * beta1_u_se)**2)

        line1 = f"{T_val:2d} years:       {vals[0]:12.4f} {vals[1]:12.4f} {vals[2]:12.4f} {union_rel:12.4f}"
        line2 = f"{'':15s} {'(' + f'{ses[0]:.4f}' + ')':>12s} {'(' + f'{ses[1]:.4f}' + ')':>12s} {'(' + f'{ses[2]:.4f}' + ')':>12s} {'(' + f'{union_rel_se:.4f}' + ')':>12s}"
        print(line1)
        print(line2)

    print(f"\nSample sizes:")
    for n, label in zip(names, ['Professional/Service', 'Nonunion', 'Union']):
        print(f"  {label}: N = {results[n]['N']:,}")


def score_against_ground_truth(results):
    """Score results against the paper's ground truth values."""
    gt = GROUND_TRUTH

    total_points = 0
    earned_points = 0

    print("\n" + "=" * 80)
    print("SCORING BREAKDOWN")
    print("=" * 80)

    # 1. Step 1 coefficients (beta_1+beta_2): 15 pts (5 per subsample)
    print("\n--- Step 1 coefficients (beta_1+beta_2): 15 pts ---")
    for name in ['Professional/Service', 'BC_Nonunion', 'BC_Union']:
        gen = results[name]['beta_1_plus_beta_2']
        true_val = gt[name]['beta_1_plus_beta_2']
        rel_err = abs(gen - true_val) / max(abs(true_val), 0.001)
        pts = 5 if rel_err <= 0.20 else (3 if rel_err <= 0.40 else 0)
        earned_points += pts
        total_points += 5
        print(f"  {name}: gen={gen:.4f} true={true_val:.4f} rel_err={rel_err:.3f} -> {pts}/5")

    # 2. Step 2 coefficients (beta_1, beta_2): 20 pts
    print("\n--- Step 2 coefficients (beta_1, beta_2): 20 pts ---")
    cp = 20 / 6
    for name in ['Professional/Service', 'BC_Nonunion', 'BC_Union']:
        for param in ['beta_1', 'beta_2']:
            gen = results[name][param]
            true_val = gt[name][param]
            abs_err = abs(gen - true_val)
            pts = cp if abs_err <= 0.01 else (cp * 0.5 if abs_err <= 0.03 else 0)
            earned_points += pts
            total_points += cp
            print(f"  {name} {param}: gen={gen:.4f} true={true_val:.4f} abs_err={abs_err:.4f} -> {pts:.2f}/{cp:.2f}")

    # 3. Cumulative returns (cols 1-3): 20 pts
    print("\n--- Cumulative returns (cols 1-3): 20 pts ---")
    cp2 = 20 / 12
    for name in ['Professional/Service', 'BC_Nonunion', 'BC_Union']:
        for T_val in [5, 10, 15, 20]:
            key = f'cum_return_{T_val}'
            gen = results[name][key]
            true_val = gt[name][key]
            abs_err = abs(gen - true_val)
            pts = cp2 if abs_err <= 0.03 else (cp2 * 0.5 if abs_err <= 0.06 else 0)
            earned_points += pts
            total_points += cp2
            print(f"  {name} {T_val}yr: gen={gen:.4f} true={true_val:.4f} abs_err={abs_err:.4f} -> {pts:.2f}/{cp2:.2f}")

    # 4. Standard errors: 10 pts
    print("\n--- Standard errors: 10 pts ---")
    se_items = []
    for name in ['Professional/Service', 'BC_Nonunion', 'BC_Union']:
        for param in ['beta_1_se', 'beta_2_se', 'beta_1_plus_beta_2_se']:
            se_items.append((name, param))
    sp = 10 / len(se_items)
    for name, param in se_items:
        gen = results[name][param]
        true_val = gt[name][param]
        rel_err = abs(gen - true_val) / max(abs(true_val), 0.001)
        pts = sp if rel_err <= 0.30 else (sp * 0.5 if rel_err <= 0.60 else 0)
        earned_points += pts
        total_points += sp
        print(f"  {name} {param}: gen={gen:.4f} true={true_val:.4f} rel_err={rel_err:.3f} -> {pts:.2f}/{sp:.2f}")

    # 5. Sample sizes: 15 pts
    print("\n--- Sample sizes: 15 pts ---")
    for name in ['Professional/Service', 'BC_Nonunion', 'BC_Union']:
        gen = results[name]['N']
        true_val = gt[name]['N']
        rel_err = abs(gen - true_val) / true_val
        pts = 5 if rel_err <= 0.05 else (3 if rel_err <= 0.10 else (1 if rel_err <= 0.20 else 0))
        earned_points += pts
        total_points += 5
        print(f"  {name}: gen={gen} true={true_val} rel_err={rel_err:.3f} -> {pts}/5")

    # 6. Significance (signs correct): 10 pts
    print("\n--- Significance (signs correct): 10 pts ---")
    sign_ok = 0
    sign_tot = 0
    for name in ['Professional/Service', 'BC_Nonunion', 'BC_Union']:
        for param in ['beta_1', 'beta_2', 'beta_1_plus_beta_2']:
            gen = results[name][param]
            true_val = gt[name][param]
            if np.sign(gen) == np.sign(true_val):
                sign_ok += 1
            sign_tot += 1
    sig_pts = 10 * sign_ok / max(sign_tot, 1)
    earned_points += sig_pts
    total_points += 10
    print(f"  Signs correct: {sign_ok}/{sign_tot} -> {sig_pts:.1f}/10")

    # 7. All columns present: 10 pts
    print("\n--- All columns present: 10 pts ---")
    all_present = all(
        all(k in results[n] for k in ['beta_1', 'beta_2', 'beta_1_plus_beta_2',
                                        'cum_return_5', 'cum_return_10', 'cum_return_15', 'cum_return_20'])
        for n in ['Professional/Service', 'BC_Nonunion', 'BC_Union']
    )
    col_pts = 10 if all_present else 5
    earned_points += col_pts
    total_points += 10
    print(f"  All columns present: {all_present} -> {col_pts}/10")

    score = round(earned_points / total_points * 100) if total_points > 0 else 0
    print(f"\n{'='*60}")
    print(f"TOTAL SCORE: {score}/100 (earned {earned_points:.1f}/{total_points:.1f})")
    print(f"{'='*60}")
    return score


if __name__ == '__main__':
    results = run_analysis('data/psid_panel.csv')
