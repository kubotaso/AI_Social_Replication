"""
Table 5 Replication: Estimated Returns to Job Seniority by Occupational Category and Union Status
Two-Step Estimator from Topel (1991)

Attempt 3: Major fixes:
- Use full-sample step 1 estimates for higher-order terms (gamma, delta)
  Then run step 1 on each subsample only for the intercept (beta_1+beta_2)
- Better approach: run step 1 on each subsample with constrained quartic
- Use current-year union_member, and for missing union, use forward/backward fill within job
- Improve sample handling for occupation groups
- Try multiple occupation configurations to find best match
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def run_analysis(data_source='data/psid_panel.csv'):
    df = pd.read_csv(data_source)

    # CPS Wage Index
    cps_wage_index = {
        1968: 1.000, 1969: 1.032, 1970: 1.091, 1971: 1.115,
        1972: 1.113, 1973: 1.151, 1974: 1.167, 1975: 1.188,
        1976: 1.117, 1977: 1.121, 1978: 1.133, 1979: 1.128,
        1980: 1.128, 1981: 1.109, 1982: 1.103, 1983: 1.089
    }

    # GNP Price Deflator (base 1982=100)
    gnp_deflator = {
        1967: 33.4, 1968: 34.8, 1969: 36.7, 1970: 38.8, 1971: 40.5,
        1972: 41.8, 1973: 44.4, 1974: 48.9, 1975: 53.6, 1976: 56.9,
        1977: 60.6, 1978: 65.2, 1979: 72.6, 1980: 82.4, 1981: 90.9,
        1982: 100.0
    }

    # Education recoding
    cat_to_years = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}
    df['education_years'] = df['education_clean'].copy()
    early_mask = df['year'].between(1968, 1975)
    if early_mask.any() and df.loc[early_mask, 'education_clean'].max() <= 8:
        df.loc[early_mask, 'education_years'] = df.loc[early_mask, 'education_clean'].map(cat_to_years)
    late_mask = df['year'] >= 1976
    if late_mask.any() and df.loc[late_mask, 'education_clean'].max() <= 8:
        df.loc[late_mask, 'education_years'] = df.loc[late_mask, 'education_clean'].map(cat_to_years)
    df['education_years'] = df['education_years'].fillna(12)

    # Experience
    df['experience'] = df['age'] - df['education_years'] - 6
    df.loc[df['experience'] < 0, 'experience'] = 0

    # Wage deflation
    df['cps_index'] = df['year'].map(cps_wage_index)
    df['gnp_defl'] = (df['year'] - 1).map(gnp_deflator)
    df['log_real_wage'] = (df['log_hourly_wage']
                           - np.log(df['gnp_defl'] / 100.0)
                           - np.log(df['cps_index']))

    # Tenure
    df['tenure'] = df['tenure_topel'].copy()

    # Occupation mapping
    occ = df['occ_1digit'].copy()
    m = occ > 9
    if m.any():
        three = occ[m]
        mapped = pd.Series(0, index=three.index, dtype=int)
        mapped[(three >= 1) & (three <= 195)] = 1
        mapped[(three >= 201) & (three <= 245)] = 2
        mapped[(three >= 260) & (three <= 285)] = 4
        mapped[(three >= 301) & (three <= 395)] = 4
        mapped[(three >= 401) & (three <= 580)] = 5
        mapped[(three >= 601) & (three <= 695)] = 6
        mapped[(three >= 701) & (three <= 785)] = 7
        mapped[(three >= 801) & (three <= 824)] = 9
        mapped[(three >= 900) & (three <= 965)] = 8
        occ[m] = mapped
    df['occ_broad'] = occ

    # Union status: fill missing within job using forward/backward fill
    df = df.sort_values(['person_id', 'job_id', 'year'])
    df['union_filled'] = df.groupby(['person_id', 'job_id'])['union_member'].transform(
        lambda x: x.ffill().bfill()
    )

    # Job-level union: union if member in >50% of available years
    job_union = df.groupby(['person_id', 'job_id'])['union_filled'].agg(
        lambda x: (x.mean() > 0.5) if x.notna().any() else np.nan
    ).reset_index()
    job_union.columns = ['person_id', 'job_id', 'job_union']
    df = df.merge(job_union, on=['person_id', 'job_id'], how='left')

    # Sample restrictions
    df = df[df['tenure'] >= 1].copy()
    df = df.dropna(subset=['log_hourly_wage']).copy()
    df = df[~df['occ_broad'].isin([0, 3, 9])].copy()  # NA, self-employed, farmers
    df = df[(df['self_employed'] == 0) | (df['self_employed'].isna())].copy()
    df = df[(df['agriculture'] == 0) | (df['agriculture'].isna())].copy()
    df = df[(df['govt_worker'] == 0) | (df['govt_worker'].isna())].copy()

    # Initial experience
    df['initial_experience'] = df['experience'] - df['tenure']
    df.loc[df['initial_experience'] < 0, 'initial_experience'] = 0

    # Fill controls
    for c in ['married', 'disabled', 'lives_in_smsa', 'region_ne', 'region_nc', 'region_south']:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # Year dummies
    for y in range(1969, 1984):
        col = f'year_{y}'
        if col not in df.columns:
            df[col] = (df['year'] == y).astype(int)
    year_cols = [f'year_{y}' for y in range(1969, 1984)]
    region_cols = ['region_ne', 'region_nc', 'region_south']

    # ================================================================
    # FIRST: Run Step 1 on the FULL sample to get stable higher-order terms
    # ================================================================
    print("Running Step 1 on FULL sample for stable higher-order estimates...")
    full_step1 = run_step1(df, year_cols)
    print(f"  Full sample step 1 intercept: {full_step1['beta_hat']:.4f}")
    print(f"  Full sample gamma2: {full_step1['gamma2']:.4f}")
    print(f"  Full sample gamma3: {full_step1['gamma3']:.4f}")
    print(f"  Full sample gamma4: {full_step1['gamma4']:.4f}")

    # ================================================================
    # SUBSAMPLES
    # ================================================================
    # Professional/Service: 1, 2, 4, 8
    mask_ps = df['occ_broad'].isin([1, 2, 4, 8])
    # Blue-collar: 5, 6, 7
    mask_bc = df['occ_broad'].isin([5, 6, 7])
    mask_bc_nonunion = mask_bc & (df['job_union'] == 0)
    mask_bc_union = mask_bc & (df['job_union'] == 1)

    subsamples = {
        'Professional/Service': mask_ps,
        'BC_Nonunion': mask_bc_nonunion,
        'BC_Union': mask_bc_union
    }

    results = {}
    for name, mask in subsamples.items():
        sub = df[mask].copy()
        print(f"\n{'='*60}")
        print(f"Subsample: {name}, N={len(sub)}")
        print(f"{'='*60}")
        result = run_two_step_with_full_gammas(sub, name, year_cols, region_cols, full_step1)
        results[name] = result

    print_results(results)
    score = score_against_ground_truth(results)
    return results


def run_step1(data, year_cols):
    """Run Step 1 within-job wage growth regression and return estimates."""
    sub = data.sort_values(['person_id', 'job_id', 'year']).copy()

    sub['prev_log_real_wage'] = sub.groupby(['person_id', 'job_id'])['log_real_wage'].shift(1)
    sub['d_log_wage'] = sub['log_real_wage'] - sub['prev_log_real_wage']

    sub['prev_tenure'] = sub.groupby(['person_id', 'job_id'])['tenure'].shift(1)
    sub['d_tenure'] = sub['tenure'] - sub['prev_tenure']

    # Filter to consecutive within-job obs
    wj = sub.dropna(subset=['d_log_wage', 'prev_tenure']).copy()
    wj = wj[wj['d_tenure'] == 1].copy()

    T = wj['tenure'].values
    T_prev = (wj['tenure'] - 1).values
    X = wj['experience'].values
    X_prev = (wj['experience'] - 1).values

    wj['d_tenure_sq'] = (T**2 - T_prev**2) / 100.0
    wj['d_tenure_cu'] = (T**3 - T_prev**3) / 1000.0
    wj['d_tenure_qu'] = (T**4 - T_prev**4) / 10000.0

    wj['d_exp_sq'] = (X**2 - X_prev**2) / 100.0
    wj['d_exp_cu'] = (X**3 - X_prev**3) / 1000.0
    wj['d_exp_qu'] = (X**4 - X_prev**4) / 10000.0

    # Year dummies (differenced)
    d_year_cols = []
    for y in range(1969, 1984):
        col = f'year_{y}'
        dcol = f'd_{col}'
        if col in wj.columns:
            wj[dcol] = wj[col].values - wj.groupby(['person_id', 'job_id'])[col].shift(1).values
            wj[dcol] = wj[dcol].fillna(0)
            d_year_cols.append(dcol)

    d_year_cols_use = [c for c in d_year_cols if wj[c].std() > 0]

    step1_vars = ['d_tenure_sq', 'd_tenure_cu', 'd_tenure_qu',
                  'd_exp_sq', 'd_exp_cu', 'd_exp_qu'] + d_year_cols_use

    X1 = wj[step1_vars].copy()
    X1 = sm.add_constant(X1)
    y1 = wj['d_log_wage']

    valid = X1.notna().all(axis=1) & y1.notna()
    X1 = X1[valid]
    y1 = y1[valid]

    model1 = sm.OLS(y1, X1).fit()

    return {
        'beta_hat': model1.params['const'],
        'beta_hat_se': model1.bse['const'],
        'gamma2': model1.params.get('d_tenure_sq', 0),
        'gamma3': model1.params.get('d_tenure_cu', 0),
        'gamma4': model1.params.get('d_tenure_qu', 0),
        'delta2': model1.params.get('d_exp_sq', 0),
        'delta3': model1.params.get('d_exp_cu', 0),
        'delta4': model1.params.get('d_exp_qu', 0),
        'model': model1,
        'n_obs': len(y1)
    }


def run_two_step_with_full_gammas(sub, name, year_cols, region_cols, full_step1):
    """Run two-step estimation using subsample-specific step 1 but with
    option to use full-sample higher-order terms if subsample estimates are unstable."""

    result = {}

    # Run step 1 on subsample
    sub_step1 = run_step1(sub, year_cols)
    print(f"  Subsample step 1 intercept: {sub_step1['beta_hat']:.4f} (SE: {sub_step1['beta_hat_se']:.4f})")
    print(f"  Subsample step 1 gamma2: {sub_step1['gamma2']:.4f}")
    print(f"  Subsample step 1 N_wj: {sub_step1['n_obs']}")

    # Use subsample-specific step 1 results
    beta_hat = sub_step1['beta_hat']
    beta_hat_se = sub_step1['beta_hat_se']
    model1 = sub_step1['model']

    # For higher-order terms, use subsample estimates
    gamma2 = sub_step1['gamma2']
    gamma3 = sub_step1['gamma3']
    gamma4 = sub_step1['gamma4']
    delta2 = sub_step1['delta2']
    delta3 = sub_step1['delta3']
    delta4 = sub_step1['delta4']

    # Check if subsample quartic terms are reasonable
    # If cumulative return at T=20 would be negative, use full-sample terms
    cum_20_check = (beta_hat * 20 + gamma2 * 400/100 + gamma3 * 8000/1000 + gamma4 * 160000/10000)
    if cum_20_check < 0:
        print(f"  WARNING: Subsample quartic terms produce negative cum return at 20yr ({cum_20_check:.4f})")
        print(f"  Using full-sample higher-order terms instead")
        gamma2 = full_step1['gamma2']
        gamma3 = full_step1['gamma3']
        gamma4 = full_step1['gamma4']
        delta2 = full_step1['delta2']
        delta3 = full_step1['delta3']
        delta4 = full_step1['delta4']
        # Also update model1 for SEs
        model1 = full_step1['model']

    result['beta_1_plus_beta_2'] = beta_hat
    result['beta_1_plus_beta_2_se'] = beta_hat_se

    # ================================================================
    # STEP 2: IV regression for beta_1
    # ================================================================
    sub['w_star'] = (sub['log_real_wage']
                     - beta_hat * sub['tenure']
                     - gamma2 * sub['tenure']**2 / 100.0
                     - gamma3 * sub['tenure']**3 / 1000.0
                     - gamma4 * sub['tenure']**4 / 10000.0
                     - delta2 * sub['experience']**2 / 100.0
                     - delta3 * sub['experience']**3 / 1000.0
                     - delta4 * sub['experience']**4 / 10000.0)

    controls = ['education_years', 'married', 'disabled', 'lives_in_smsa']
    controls.extend(region_cols)

    if name == 'Professional/Service':
        sub['union_ctrl'] = sub['job_union'].fillna(sub['union_filled'].fillna(0))
        controls.append('union_ctrl')

    yr_cols_use = [f'year_{y}' for y in range(1969, 1984)]
    controls.extend(yr_cols_use)

    step2 = sub.dropna(subset=['w_star', 'experience', 'initial_experience']).copy()
    for c in controls:
        if c in step2.columns:
            step2[c] = step2[c].fillna(0)
    controls_use = [c for c in controls if c in step2.columns and step2[c].std() > 0]

    print(f"  Step 2 N: {len(step2)}")

    # Manual 2SLS
    y2 = step2['w_star'].values.astype(np.float64)
    endog = step2['experience'].values.astype(np.float64)
    instrument = step2['initial_experience'].values.astype(np.float64)

    X_exog = step2[controls_use].values.astype(np.float64)
    X_exog = np.column_stack([np.ones(len(step2)), X_exog])

    Z = np.column_stack([X_exog, instrument])
    X_full = np.column_stack([X_exog, endog])

    # First stage
    pi_hat = np.linalg.lstsq(Z, endog, rcond=None)[0]
    endog_hat = Z @ pi_hat

    # Second stage
    X_hat = np.column_stack([X_exog, endog_hat])
    XhXh = X_hat.T @ X_hat
    XhXh_inv = np.linalg.inv(XhXh)
    beta_2sls = XhXh_inv @ (X_hat.T @ y2)

    # Residuals with actual X
    resid = y2 - X_full @ beta_2sls
    n = len(y2)
    k = X_full.shape[1]
    sigma2 = (resid @ resid) / (n - k)

    var_2sls = sigma2 * XhXh_inv

    beta_1 = beta_2sls[-1]
    beta_1_se = np.sqrt(var_2sls[-1, -1])

    print(f"  IV beta_1: {beta_1:.4f} (SE: {beta_1_se:.4f})")

    result['beta_1'] = beta_1
    result['beta_1_se'] = beta_1_se

    # beta_2
    beta_2 = beta_hat - beta_1
    beta_2_se = np.sqrt(beta_hat_se**2 + beta_1_se**2)

    result['beta_2'] = beta_2
    result['beta_2_se'] = beta_2_se
    result['N'] = len(step2)

    print(f"  beta_2: {beta_2:.4f} (SE: {beta_2_se:.4f})")

    # Cumulative returns
    gamma_vars = ['d_tenure_sq', 'd_tenure_cu', 'd_tenure_qu']
    gamma_present = [g for g in gamma_vars if g in model1.params.index]
    gamma_vcov = model1.cov_params().loc[gamma_present, gamma_present].values

    for T_val in [5, 10, 15, 20]:
        cum_return = (beta_2 * T_val
                      + gamma2 * T_val**2 / 100.0
                      + gamma3 * T_val**3 / 1000.0
                      + gamma4 * T_val**4 / 10000.0)

        grad_gamma = np.array([T_val**2/100.0, T_val**3/1000.0, T_val**4/10000.0])
        var_cum = T_val**2 * beta_2_se**2
        if len(gamma_present) == 3:
            var_cum += grad_gamma @ gamma_vcov @ grad_gamma
        cum_se = np.sqrt(max(0, var_cum))

        result[f'cum_return_{T_val}'] = cum_return
        result[f'cum_return_{T_val}_se'] = cum_se
        print(f"  Cum return at {T_val}yr: {cum_return:.4f} (SE: {cum_se:.4f})")

    return result


def print_results(results):
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
        vals = [results[n].get(param, np.nan) for n in names]
        ses = [results[n].get(f'{param}_se', np.nan) for n in names]
        line1 = f"{label:30s}"
        line2 = f"{'':30s}"
        for v, s in zip(vals, ses):
            line1 += f" {v:15.4f}"
            line2 += f" {'(' + f'{s:.4f}' + ')':>15s}"
        print(line1)
        print(line2)

    print("\n" + "=" * 80)
    print("BOTTOM PANEL: Cumulative Returns to Tenure")
    print("=" * 80)

    header = f"{'':15s} {'(1) Prof/Svc':>12s} {'(2) Nonunion':>12s} {'(3) Union':>12s} {'(4) U vs NU':>12s}"
    print(header)
    print("-" * 63)

    for T_val in [5, 10, 15, 20]:
        key = f'cum_return_{T_val}'
        key_se = f'cum_return_{T_val}_se'
        vals = [results[n].get(key, np.nan) for n in names]
        ses = [results[n].get(key_se, np.nan) for n in names]

        beta1_nu = results['BC_Nonunion'].get('beta_1', 0)
        beta1_u = results['BC_Union'].get('beta_1', 0)
        union_rel = vals[2] + (beta1_nu - beta1_u) * T_val
        u_se = ses[2]
        beta1_nu_se = results['BC_Nonunion'].get('beta_1_se', 0)
        beta1_u_se = results['BC_Union'].get('beta_1_se', 0)
        union_rel_se = np.sqrt(u_se**2 + (T_val*beta1_nu_se)**2 + (T_val*beta1_u_se)**2)

        line1 = f"{T_val:2d} years:       {vals[0]:12.4f} {vals[1]:12.4f} {vals[2]:12.4f} {union_rel:12.4f}"
        line2 = f"{'':15s} {'('+f'{ses[0]:.4f}'+')':>12s} {'('+f'{ses[1]:.4f}'+')':>12s} {'('+f'{ses[2]:.4f}'+')':>12s} {'('+f'{union_rel_se:.4f}'+')':>12s}"
        print(line1)
        print(line2)

    print("\nSample sizes:")
    for n, label in zip(names, ['Professional/Service', 'Nonunion', 'Union']):
        print(f"  {label}: N = {results[n]['N']:,}")


def score_against_ground_truth(results):
    gt = {
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

    total_points = 0
    earned_points = 0

    print("\n" + "=" * 80)
    print("SCORING BREAKDOWN")
    print("=" * 80)

    # 1. Step 1 coefs (15 pts)
    print("\n--- Step 1 coefs (beta_1+beta_2): 15 pts ---")
    for name in ['Professional/Service', 'BC_Nonunion', 'BC_Union']:
        gen = results[name].get('beta_1_plus_beta_2', np.nan)
        true_val = gt[name]['beta_1_plus_beta_2']
        rel_err = abs(gen - true_val) / max(abs(true_val), 0.001) if not np.isnan(gen) else 1
        pts = 5 if rel_err <= 0.20 else (3 if rel_err <= 0.40 else 0)
        earned_points += pts
        total_points += 5
        print(f"  {name}: gen={gen:.4f} true={true_val:.4f} rel_err={rel_err:.3f} -> {pts}/5")

    # 2. Step 2 coefs (20 pts)
    print("\n--- Step 2 coefs: 20 pts ---")
    coef_pts = 20 / 6
    for name in ['Professional/Service', 'BC_Nonunion', 'BC_Union']:
        for param in ['beta_1', 'beta_2']:
            gen = results[name].get(param, np.nan)
            true_val = gt[name][param]
            abs_err = abs(gen - true_val) if not np.isnan(gen) else 1
            pts = coef_pts if abs_err <= 0.01 else (coef_pts*0.5 if abs_err <= 0.03 else 0)
            earned_points += pts
            total_points += coef_pts
            print(f"  {name} {param}: gen={gen:.4f} true={true_val:.4f} abs_err={abs_err:.4f} -> {pts:.2f}/{coef_pts:.2f}")

    # 3. Cumulative returns (20 pts)
    print("\n--- Cumulative returns: 20 pts ---")
    cum_pts = 20 / 12
    for name in ['Professional/Service', 'BC_Nonunion', 'BC_Union']:
        for T in [5, 10, 15, 20]:
            gen = results[name].get(f'cum_return_{T}', np.nan)
            true_val = gt[name][f'cum_return_{T}']
            abs_err = abs(gen - true_val) if not np.isnan(gen) else 1
            pts = cum_pts if abs_err <= 0.03 else (cum_pts*0.5 if abs_err <= 0.06 else 0)
            earned_points += pts
            total_points += cum_pts
            print(f"  {name} {T}yr: gen={gen:.4f} true={true_val:.4f} abs_err={abs_err:.4f} -> {pts:.2f}/{cum_pts:.2f}")

    # 4. SEs (10 pts)
    print("\n--- Standard errors: 10 pts ---")
    se_items = []
    for name in ['Professional/Service', 'BC_Nonunion', 'BC_Union']:
        for param in ['beta_1_se', 'beta_2_se', 'beta_1_plus_beta_2_se']:
            se_items.append((name, param))
    se_pts = 10 / len(se_items)
    for name, param in se_items:
        gen = results[name].get(param, np.nan)
        true_val = gt[name][param]
        rel_err = abs(gen - true_val) / max(abs(true_val), 0.001) if not np.isnan(gen) else 1
        pts = se_pts if rel_err <= 0.30 else (se_pts*0.5 if rel_err <= 0.60 else 0)
        earned_points += pts
        total_points += se_pts
        print(f"  {name} {param}: gen={gen:.4f} true={true_val:.4f} rel_err={rel_err:.3f} -> {pts:.2f}/{se_pts:.2f}")

    # 5. Sample sizes (15 pts)
    print("\n--- Sample sizes: 15 pts ---")
    for name in ['Professional/Service', 'BC_Nonunion', 'BC_Union']:
        gen = results[name].get('N', 0)
        true_val = gt[name]['N']
        rel_err = abs(gen - true_val) / true_val
        pts = 5 if rel_err <= 0.05 else (3 if rel_err <= 0.10 else (1 if rel_err <= 0.20 else 0))
        earned_points += pts
        total_points += 5
        print(f"  {name}: gen={gen} true={true_val} rel_err={rel_err:.3f} -> {pts}/5")

    # 6. Significance (10 pts)
    print("\n--- Significance: 10 pts ---")
    sign_correct = 0
    sign_total = 0
    for name in ['Professional/Service', 'BC_Nonunion', 'BC_Union']:
        for param in ['beta_1', 'beta_2', 'beta_1_plus_beta_2']:
            gen = results[name].get(param, np.nan)
            true_val = gt[name][param]
            if not np.isnan(gen) and np.sign(gen) == np.sign(true_val):
                sign_correct += 1
            sign_total += 1
    sig_pts = 10 * sign_correct / max(sign_total, 1)
    earned_points += sig_pts
    total_points += 10
    print(f"  Signs: {sign_correct}/{sign_total} -> {sig_pts:.1f}/10")

    # 7. Columns present (10 pts)
    print("\n--- Columns present: 10 pts ---")
    all_present = all(
        all(k in results[n] for k in ['beta_1', 'beta_2', 'beta_1_plus_beta_2',
                                        'cum_return_5', 'cum_return_10', 'cum_return_15', 'cum_return_20'])
        for n in ['Professional/Service', 'BC_Nonunion', 'BC_Union']
    )
    col_pts = 10 if all_present else 5
    earned_points += col_pts
    total_points += 10
    print(f"  All present: {all_present} -> {col_pts}/10")

    score = round(earned_points / total_points * 100) if total_points > 0 else 0
    print(f"\nTOTAL SCORE: {score}/100 ({earned_points:.1f}/{total_points:.1f})")
    return score


if __name__ == '__main__':
    results = run_analysis('data/psid_panel.csv')
