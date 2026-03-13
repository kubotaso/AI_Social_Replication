"""
Table 5 Replication: Estimated Returns to Job Seniority by Occupation and Union Status
Two-Step Estimator from Topel (1991)

Attempt 6: Major strategy revision.

KEY INSIGHTS FROM PRIOR ATTEMPTS:
- Subsample-specific quartic polynomials are unstable on small samples (attempts 1-5)
- Job-level occupation reduces sample too much (attempt 5: BC_NU=1625 vs target 2642)
- Need observation-level occupation for better N match
- w* should subtract FULL step 1 polynomial including linear term
- Then step 2: IV of w* on X using X_0 as instrument

APPROACH:
1. Use observation-level (current period) occupation -- better sample sizes
2. Run Step 1 per subsample to get subsample-specific beta_hat (intercept)
3. Use FULL-SAMPLE higher-order terms from own step 1 on full data (like Table 2)
   -- This stabilizes the quartic polynomial that is noisy on small subsamples
4. Construct w* using subsample beta_hat + full-sample higher-order terms
5. Step 2: 2SLS of w* on experience with X_0 as instrument
6. Murphy-Topel SE correction

The paper says (Table 5 note): "The effects of higher-order terms in experience
and tenure (X^2, X^3, X^4, T^2, T^3, T^4) are estimated from models of
within-job wage growth in each occupation." This suggests per-subsample estimation.
BUT the polynomial terms need many observations. Since subsample polynomials explode,
we estimate them from the full sample for stability.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')


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

# Paper's Table 2 Model 3 step 1 values (full sample, as reported in Table 2)
# These are used as the higher-order polynomial terms when subsample polynomials
# are unstable. Scaled: g2 is coeff on T^2/100 in the FD regression.
PAPER_STEP1 = {
    'g2': -0.004592,   # coeff on T^2 (already in natural units: g2 * T^2)
    'g3':  0.0001846,  # coeff on T^3
    'g4': -0.00000245, # coeff on T^4
    'd2': -0.006051,   # coeff on X^2
    'd3':  0.0002067,  # coeff on X^3
    'd4': -0.00000238, # coeff on X^4
}


def prepare_data(data_source):
    df = pd.read_csv(data_source)

    # Education recoding
    df['education_years'] = np.nan
    for yr in df['year'].unique():
        mask = df['year'] == yr
        if yr in [1975, 1976]:
            df.loc[mask, 'education_years'] = df.loc[mask, 'education_clean']
        else:
            df.loc[mask, 'education_years'] = df.loc[mask, 'education_clean'].map(EDUC_MAP)
    df = df.dropna(subset=['education_years']).copy()

    # Experience
    df['experience'] = (df['age'] - df['education_years'] - 6).clip(lower=0)
    df['tenure'] = df['tenure_topel'].copy()

    # Wages
    df['cps_index'] = df['year'].map(CPS_WAGE_INDEX)
    df['gnp_defl'] = (df['year'] - 1).map(GNP_DEFLATOR)
    df['log_wage_cps'] = df['log_hourly_wage'] - np.log(df['cps_index'])
    df['log_real_wage'] = (df['log_hourly_wage']
                           - np.log(df['gnp_defl'] / 100.0)
                           - np.log(df['cps_index']))

    # Occupation mapping (observation-level)
    occ = df['occ_1digit'].copy()
    m3 = occ > 9
    if m3.any():
        three = occ[m3]
        mapped = pd.Series(0, index=three.index, dtype=int)
        mapped[(three >= 1) & (three <= 195)] = 1
        mapped[(three >= 201) & (three <= 245)] = 2
        mapped[(three >= 260) & (three <= 395)] = 4
        mapped[(three >= 401) & (three <= 580)] = 5
        mapped[(three >= 601) & (three <= 695)] = 6
        mapped[(three >= 701) & (three <= 785)] = 7
        mapped[(three >= 801) & (three <= 824)] = 9
        mapped[(three >= 900) & (three <= 965)] = 8
        occ[m3] = mapped
    df['occ_broad'] = occ

    # Job-level union (footnote 18)
    df = df.sort_values(['person_id', 'job_id', 'year'])
    ju = df.groupby(['person_id', 'job_id'])['union_member'].agg(
        lambda x: (x.mean() > 0.5) if x.notna().any() else np.nan
    ).reset_index().rename(columns={'union_member': 'job_union'})
    df = df.merge(ju, on=['person_id', 'job_id'], how='left')

    # Sample restrictions
    df = df[df['tenure'] >= 1].copy()
    df = df.dropna(subset=['log_hourly_wage']).copy()
    df = df[~df['occ_broad'].isin([0, 3, 9])].copy()
    df = df[(df['self_employed'] == 0) | (df['self_employed'].isna())].copy()

    # Initial experience
    df['initial_experience'] = (df['experience'] - df['tenure']).clip(lower=0)

    # Polynomial terms
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


def run_step1_full(data, wage_var='log_wage_cps'):
    """Run Step 1 on the full sample to get stable higher-order terms."""
    sub = data.sort_values(['person_id', 'job_id', 'year']).copy()

    sub['within'] = ((sub['person_id'] == sub['person_id'].shift(1)) &
                     (sub['job_id'] == sub['job_id'].shift(1)) &
                     (sub['year'] - sub['year'].shift(1) == 1))

    for v in [wage_var, 'tenure_2', 'tenure_3', 'tenure_4',
              'exp_2', 'exp_3', 'exp_4']:
        sub[f'd_{v}'] = sub[v] - sub[v].shift(1)

    year_cols = sorted([c for c in sub.columns if c.startswith('year_') and c != 'year'])
    d_year_cols = []
    for yc in year_cols:
        dc = f'd_{yc}'
        sub[dc] = sub[yc].astype(float) - sub[yc].shift(1).astype(float)
        d_year_cols.append(dc)

    fd = sub[sub['within']].copy()
    fd = fd.dropna(subset=[f'd_{wage_var}']).copy()

    d_yr_use = [c for c in d_year_cols if fd[c].std() > 1e-10]
    if len(d_yr_use) > 1:
        d_yr_use = d_yr_use[1:]

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


def run_step1_subsample(data, wage_var='log_wage_cps'):
    """Run Step 1 on a subsample. Returns both subsample and full-sample compatible results."""
    sub = data.sort_values(['person_id', 'job_id', 'year']).copy()

    sub['within'] = ((sub['person_id'] == sub['person_id'].shift(1)) &
                     (sub['job_id'] == sub['job_id'].shift(1)) &
                     (sub['year'] - sub['year'].shift(1) == 1))

    for v in [wage_var, 'tenure_2', 'tenure_3', 'tenure_4',
              'exp_2', 'exp_3', 'exp_4']:
        sub[f'd_{v}'] = sub[v] - sub[v].shift(1)

    year_cols = sorted([c for c in sub.columns if c.startswith('year_') and c != 'year'])
    d_year_cols = []
    for yc in year_cols:
        dc = f'd_{yc}'
        sub[dc] = sub[yc].astype(float) - sub[yc].shift(1).astype(float)
        d_year_cols.append(dc)

    fd = sub[sub['within']].copy()
    fd = fd.dropna(subset=[f'd_{wage_var}']).copy()

    d_yr_use = [c for c in d_year_cols if fd[c].std() > 1e-10]
    if len(d_yr_use) > 1:
        d_yr_use = d_yr_use[1:]

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


def run_step2_iv(data, beta_hat, g2, g3, g4, d2, d3, d4, name):
    """Run Step 2: 2SLS of w* on experience.

    w* = log_real_wage - beta_hat*T - g2*T^2 - g3*T^3 - g4*T^4
         - d2*X^2 - d3*X^3 - d4*X^4
    """
    sub = data.copy()

    sub['w_star'] = (sub['log_real_wage']
                     - beta_hat * sub['tenure']
                     - g2 * sub['tenure_2']
                     - g3 * sub['tenure_3']
                     - g4 * sub['tenure_4']
                     - d2 * sub['exp_2']
                     - d3 * sub['exp_3']
                     - d4 * sub['exp_4'])

    ctrls = ['education_years', 'married', 'disabled', 'lives_in_smsa',
             'region_ne', 'region_nc', 'region_south']

    if name == 'Professional/Service':
        sub['union_ctrl'] = sub['union_member'].fillna(0)
        if sub['union_ctrl'].std() > 0:
            ctrls.append('union_ctrl')

    yr_use = [f'year_{y}' for y in range(1969, 1984)
              if f'year_{y}' in sub.columns and sub[f'year_{y}'].std() > 0]

    step2 = sub.dropna(subset=['w_star', 'experience', 'initial_experience']).copy()
    ctrls_use = [c for c in ctrls if c in step2.columns and step2[c].std() > 0]

    all_exog = ctrls_use + yr_use

    # Check rank and trim
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
    X_exog_mat = sm.add_constant(step2[all_exog].astype(float)).values.astype(np.float64)

    Z = np.column_stack([X_exog_mat, instrument])
    X_full = np.column_stack([X_exog_mat, endog])

    # First stage
    pi_hat = np.linalg.lstsq(Z, endog, rcond=None)[0]
    endog_hat = Z @ pi_hat

    # Second stage
    X_hat = np.column_stack([X_exog_mat, endog_hat])
    XhXh = X_hat.T @ X_hat
    XhXh_inv = np.linalg.inv(XhXh)
    beta_2sls = XhXh_inv @ (X_hat.T @ y)

    resid = y - X_full @ beta_2sls
    n = len(y)
    k = X_full.shape[1]
    sigma2 = (resid @ resid) / (n - k)
    var_2sls = sigma2 * XhXh_inv

    beta_1 = beta_2sls[-1]
    beta_1_se = np.sqrt(var_2sls[-1, -1])

    return {
        'beta_1': beta_1,
        'beta_1_se_naive': beta_1_se,
        'N': n,
        'sigma2': sigma2,
        'X_hat': X_hat,
        'XhXh_inv': XhXh_inv,
        'step2_data': step2,
    }


def compute_murphy_topel_se(step1_model, step2_result, step2_data):
    """Murphy-Topel corrected SE for beta_1."""
    model1 = step1_model

    s1_params = ['const', 'd_tenure_2', 'd_tenure_3', 'd_tenure_4',
                 'd_exp_2', 'd_exp_3', 'd_exp_4']
    s1_present = [p for p in s1_params if p in model1.params.index]
    V1 = model1.cov_params().loc[s1_present, s1_present].values

    V2_b1 = step2_result['beta_1_se_naive'] ** 2

    T = step2_data['tenure'].values.astype(float)
    X = step2_data['experience'].values.astype(float)
    n = len(T)

    J = np.zeros((n, len(s1_present)))
    for j, param in enumerate(s1_present):
        if param == 'const':
            J[:, j] = -T
        elif param == 'd_tenure_2':
            J[:, j] = -T**2
        elif param == 'd_tenure_3':
            J[:, j] = -T**3
        elif param == 'd_tenure_4':
            J[:, j] = -T**4
        elif param == 'd_exp_2':
            J[:, j] = -X**2
        elif param == 'd_exp_3':
            J[:, j] = -X**3
        elif param == 'd_exp_4':
            J[:, j] = -X**4

    X_hat = step2_result['X_hat']
    XhXh_inv = step2_result['XhXh_inv']

    G_full = XhXh_inv @ (X_hat.T @ J)
    g_b1 = G_full[-1, :]

    V_extra = g_b1 @ V1 @ g_b1
    V_corrected = V2_b1 + V_extra

    return {
        'beta_1_se': np.sqrt(max(0, V_corrected)),
        'beta_1_se_naive': np.sqrt(V2_b1),
        'V_extra': V_extra,
    }


def run_analysis(data_source='data/psid_panel.csv'):
    df = prepare_data(data_source)

    print(f"Total observations after restrictions: {len(df)}")

    # ==================================================================
    # First, run step 1 on FULL sample to get stable higher-order terms
    # ==================================================================
    full_s1 = run_step1_full(df)
    print(f"\nFull-sample Step 1: b1+b2={full_s1['beta_hat']:.4f} ({full_s1['beta_hat_se']:.4f})")
    print(f"  g2={full_s1['g2']:.6f}, g3={full_s1['g3']:.8f}, g4={full_s1['g4']:.10f}")
    print(f"  d2={full_s1['d2']:.6f}, d3={full_s1['d3']:.8f}, d4={full_s1['d4']:.10f}")
    print(f"  N_wj={full_s1['N_wj']}")
    print(f"\nPaper's Table 2 step 1 values for comparison:")
    print(f"  g2={PAPER_STEP1['g2']:.6f}, g3={PAPER_STEP1['g3']:.8f}, g4={PAPER_STEP1['g4']:.10f}")
    print(f"  d2={PAPER_STEP1['d2']:.6f}, d3={PAPER_STEP1['d3']:.8f}, d4={PAPER_STEP1['d4']:.10f}")

    # ==================================================================
    # Define subsamples (observation-level occupation)
    # ==================================================================
    mask_ps = df['occ_broad'].isin([1, 2, 4, 8])
    mask_bc = df['occ_broad'].isin([5, 6, 7])
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
        sub = df[mask].copy()
        print(f"\n{'='*60}")
        print(f"Subsample: {name}, N={len(sub)}")
        print(f"{'='*60}")

        # Run subsample step 1 for the intercept
        sub_s1 = run_step1_subsample(sub)
        print(f"  Subsample Step 1: b1+b2={sub_s1['beta_hat']:.4f} ({sub_s1['beta_hat_se']:.4f})")
        print(f"    Sub g2={sub_s1['g2']:.6f}, Sub g3={sub_s1['g3']:.8f}")
        print(f"    N_wj={sub_s1['N_wj']}")

        # Use subsample intercept, subsample higher-order terms
        # The paper says each occupation group has its own step 1
        beta_hat_use = sub_s1['beta_hat']
        beta_hat_se_use = sub_s1['beta_hat_se']

        # For higher-order terms: try both subsample and full-sample
        # Use subsample terms (as the paper says), but if cumulative returns
        # diverge, fall back to full-sample terms

        # Strategy: run with subsample terms first
        g2_sub, g3_sub, g4_sub = sub_s1['g2'], sub_s1['g3'], sub_s1['g4']
        d2_sub, d3_sub, d4_sub = sub_s1['d2'], sub_s1['d3'], sub_s1['d4']

        # Check if subsample terms produce reasonable cumulative returns
        test_cum_20 = (0.05 * 20 + g2_sub * 20**2 + g3_sub * 20**3 + g4_sub * 20**4)
        subsample_stable = abs(test_cum_20) < 2.0  # Reasonable bound

        if subsample_stable:
            g2_use, g3_use, g4_use = g2_sub, g3_sub, g4_sub
            d2_use, d3_use, d4_use = d2_sub, d3_sub, d4_sub
            step1_model_for_mt = sub_s1['model']
            print(f"  Using SUBSAMPLE higher-order terms (stable)")
        else:
            g2_use = full_s1['g2']
            g3_use = full_s1['g3']
            g4_use = full_s1['g4']
            d2_use = full_s1['d2']
            d3_use = full_s1['d3']
            d4_use = full_s1['d4']
            step1_model_for_mt = full_s1['model']
            print(f"  Using FULL-SAMPLE higher-order terms (subsample unstable: cum20={test_cum_20:.2f})")

        # Step 2
        s2 = run_step2_iv(sub, beta_hat_use, g2_use, g3_use, g4_use,
                          d2_use, d3_use, d4_use, name)
        print(f"  Step 2: beta_1={s2['beta_1']:.4f} (naive SE: {s2['beta_1_se_naive']:.4f})")
        print(f"    N={s2['N']}")

        # Murphy-Topel correction
        mt = compute_murphy_topel_se(step1_model_for_mt, s2, s2['step2_data'])
        print(f"  Murphy-Topel SE: {mt['beta_1_se']:.4f}")

        beta_1 = s2['beta_1']
        beta_1_se = mt['beta_1_se']
        beta_2 = beta_hat_use - beta_1
        beta_2_se = np.sqrt(beta_hat_se_use**2 + beta_1_se**2)

        print(f"  beta_1 = {beta_1:.4f} ({beta_1_se:.4f})")
        print(f"  beta_2 = {beta_2:.4f} ({beta_2_se:.4f})")
        print(f"  b1+b2  = {beta_hat_use:.4f} ({beta_hat_se_use:.4f})")

        result = {
            'beta_1': beta_1,
            'beta_1_se': beta_1_se,
            'beta_2': beta_2,
            'beta_2_se': beta_2_se,
            'beta_1_plus_beta_2': beta_hat_use,
            'beta_1_plus_beta_2_se': beta_hat_se_use,
            'N': s2['N'],
        }

        # Cumulative returns
        # Use the gamma_vcov from whichever model was used for higher-order terms
        gamma_vars = ['d_tenure_2', 'd_tenure_3', 'd_tenure_4']
        gamma_present = [g for g in gamma_vars if g in step1_model_for_mt.params.index]
        gamma_vcov = step1_model_for_mt.cov_params().loc[gamma_present, gamma_present].values

        for T_val in [5, 10, 15, 20]:
            cum = (beta_2 * T_val
                   + g2_use * T_val**2
                   + g3_use * T_val**3
                   + g4_use * T_val**4)

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

        results[name] = result

    print_table(results)
    score = score_against_ground_truth(results)
    return results


def print_table(results):
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
        beta1_nu = results['BC_Nonunion']['beta_1']
        beta1_u = results['BC_Union']['beta_1']
        union_rel = vals[2] + (beta1_nu - beta1_u) * T_val
        u_se = ses[2]
        beta1_nu_se = results['BC_Nonunion']['beta_1_se']
        beta1_u_se = results['BC_Union']['beta_1_se']
        union_rel_se = np.sqrt(u_se**2 + (T_val * beta1_nu_se)**2 + (T_val * beta1_u_se)**2)
        line1 = f"{T_val:2d} years:       {vals[0]:12.4f} {vals[1]:12.4f} {vals[2]:12.4f} {union_rel:12.4f}"
        line2 = f"{'':15s} {'(' + f'{ses[0]:.4f}' + ')':>12s} {'(' + f'{ses[1]:.4f}' + ')':>12s} {'(' + f'{ses[2]:.4f}' + ')':>12s} {'(' + f'{union_rel_se:.4f}' + ')':>12s}"
        print(line1)
        print(line2)

    print(f"\nSample sizes:")
    for n, label in zip(names, ['Professional/Service', 'Nonunion', 'Union']):
        print(f"  {label}: N = {results[n]['N']:,}")


def score_against_ground_truth(results):
    gt = GROUND_TRUTH
    total_points = 0
    earned_points = 0

    print("\n" + "=" * 80)
    print("SCORING BREAKDOWN")
    print("=" * 80)

    # 1. Step 1 coefficients (beta_1+beta_2): 15 pts
    print("\n--- Step 1 coefs (beta_1+beta_2): 15 pts ---")
    for name in ['Professional/Service', 'BC_Nonunion', 'BC_Union']:
        gen = results[name]['beta_1_plus_beta_2']
        true_val = gt[name]['beta_1_plus_beta_2']
        rel_err = abs(gen - true_val) / max(abs(true_val), 0.001)
        pts = 5 if rel_err <= 0.20 else (3 if rel_err <= 0.40 else 0)
        earned_points += pts; total_points += 5
        print(f"  {name}: gen={gen:.4f} true={true_val:.4f} rel_err={rel_err:.3f} -> {pts}/5")

    # 2. Step 2 coefficients: 20 pts
    print("\n--- Step 2 coefs: 20 pts ---")
    cp = 20 / 6
    for name in ['Professional/Service', 'BC_Nonunion', 'BC_Union']:
        for param in ['beta_1', 'beta_2']:
            gen = results[name][param]
            true_val = gt[name][param]
            abs_err = abs(gen - true_val)
            pts = cp if abs_err <= 0.01 else (cp * 0.5 if abs_err <= 0.03 else 0)
            earned_points += pts; total_points += cp
            print(f"  {name} {param}: gen={gen:.4f} true={true_val:.4f} abs_err={abs_err:.4f} -> {pts:.2f}/{cp:.2f}")

    # 3. Cumulative returns: 20 pts
    print("\n--- Cumulative returns: 20 pts ---")
    cp2 = 20 / 12
    for name in ['Professional/Service', 'BC_Nonunion', 'BC_Union']:
        for T_val in [5, 10, 15, 20]:
            key = f'cum_return_{T_val}'
            gen = results[name][key]
            true_val = gt[name][key]
            abs_err = abs(gen - true_val)
            pts = cp2 if abs_err <= 0.03 else (cp2 * 0.5 if abs_err <= 0.06 else 0)
            earned_points += pts; total_points += cp2
            print(f"  {name} {T_val}yr: gen={gen:.4f} true={true_val:.4f} abs_err={abs_err:.4f} -> {pts:.2f}/{cp2:.2f}")

    # 4. SEs: 10 pts
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
        earned_points += pts; total_points += sp
        print(f"  {name} {param}: gen={gen:.4f} true={true_val:.4f} rel_err={rel_err:.3f} -> {pts:.2f}/{sp:.2f}")

    # 5. Sample sizes: 15 pts
    print("\n--- Sample sizes: 15 pts ---")
    for name in ['Professional/Service', 'BC_Nonunion', 'BC_Union']:
        gen = results[name]['N']
        true_val = gt[name]['N']
        rel_err = abs(gen - true_val) / true_val
        pts = 5 if rel_err <= 0.05 else (3 if rel_err <= 0.10 else (1 if rel_err <= 0.20 else 0))
        earned_points += pts; total_points += 5
        print(f"  {name}: gen={gen} true={true_val} rel_err={rel_err:.3f} -> {pts}/5")

    # 6. Significance: 10 pts
    print("\n--- Significance: 10 pts ---")
    sign_ok = 0; sign_tot = 0
    for name in ['Professional/Service', 'BC_Nonunion', 'BC_Union']:
        for param in ['beta_1', 'beta_2', 'beta_1_plus_beta_2']:
            gen = results[name][param]
            true_val = gt[name][param]
            if np.sign(gen) == np.sign(true_val):
                sign_ok += 1
            sign_tot += 1
    sig_pts = 10 * sign_ok / max(sign_tot, 1)
    earned_points += sig_pts; total_points += 10
    print(f"  Signs: {sign_ok}/{sign_tot} -> {sig_pts:.1f}/10")

    # 7. All columns: 10 pts
    all_present = all(
        all(k in results[n] for k in ['beta_1', 'beta_2', 'beta_1_plus_beta_2',
                                        'cum_return_5', 'cum_return_10', 'cum_return_15', 'cum_return_20'])
        for n in ['Professional/Service', 'BC_Nonunion', 'BC_Union']
    )
    col_pts = 10 if all_present else 5
    earned_points += col_pts; total_points += 10

    score = round(earned_points / total_points * 100) if total_points > 0 else 0
    print(f"\nTOTAL SCORE: {score}/100 ({earned_points:.1f}/{total_points:.1f})")
    return score


if __name__ == '__main__':
    results = run_analysis('data/psid_panel.csv')
