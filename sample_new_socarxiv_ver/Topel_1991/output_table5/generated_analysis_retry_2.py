"""
Table 5 Replication: Estimated Returns to Job Seniority by Occupational Category and Union Status
Two-Step Estimator from Topel (1991)

Attempt 2: Key fixes:
- Use current-year union_member for blue-collar split (not job-level aggregation)
- Keep all observations for Professional/Service regardless of union availability
- Fix 2SLS standard errors using proper IV formula
- Fix sample size by proper occupation classification
- Fix higher-order term handling for cumulative returns
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def run_analysis(data_source='data/psid_panel.csv'):
    # ============================================================
    # 1. LOAD AND PREPARE DATA
    # ============================================================
    df = pd.read_csv(data_source)

    # CPS Wage Index for deflation
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

    # ============================================================
    # 2. EDUCATION RECODING
    # ============================================================
    cat_to_years = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}
    df['education_years'] = df['education_clean'].copy()
    early_mask = df['year'].between(1968, 1975)
    if early_mask.any():
        early_max = df.loc[early_mask, 'education_clean'].max()
        if early_max <= 8:
            df.loc[early_mask, 'education_years'] = df.loc[early_mask, 'education_clean'].map(cat_to_years)
    late_mask = df['year'] >= 1976
    late_max = df.loc[late_mask, 'education_clean'].max() if late_mask.any() else 0
    if late_max <= 8:
        df.loc[late_mask, 'education_years'] = df.loc[late_mask, 'education_clean'].map(cat_to_years)
    df['education_years'] = df['education_years'].fillna(12)

    # ============================================================
    # 3. EXPERIENCE CONSTRUCTION
    # ============================================================
    df['experience'] = df['age'] - df['education_years'] - 6
    df.loc[df['experience'] < 0, 'experience'] = 0

    # ============================================================
    # 4. WAGE DEFLATION
    # ============================================================
    df['cps_index'] = df['year'].map(cps_wage_index)
    df['gnp_defl'] = (df['year'] - 1).map(gnp_deflator)
    df['log_real_wage'] = (df['log_hourly_wage']
                           - np.log(df['gnp_defl'] / 100.0)
                           - np.log(df['cps_index']))

    # ============================================================
    # 5. TENURE
    # ============================================================
    df['tenure'] = df['tenure_topel'].copy()

    # ============================================================
    # 6. OCCUPATION CLASSIFICATION
    # ============================================================
    # The occ_1digit column contains PSID 1-digit codes (0-9) for most rows
    # and 3-digit Census codes for some rows
    # PSID 1-digit codes:
    # 1: Professional/Technical
    # 2: Managers/Officials
    # 3: Self-employed businessmen (exclude)
    # 4: Clerical/Sales
    # 5: Craftsmen/Foremen
    # 6: Operatives
    # 7: Laborers (non-farm)
    # 8: Service workers
    # 9: Farmers (exclude)
    # 0: NA/Not coded

    occ = df['occ_1digit'].copy()
    mask3 = occ > 9
    if mask3.any():
        three = occ[mask3]
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
        occ[mask3] = mapped
    df['occ_broad'] = occ

    # Professional/Service: codes 1, 2, 4, 8
    df['is_prof_service'] = df['occ_broad'].isin([1, 2, 4, 8]).astype(int)
    # Blue-collar: codes 5, 6, 7
    df['is_blue_collar'] = df['occ_broad'].isin([5, 6, 7]).astype(int)

    # ============================================================
    # 7. UNION STATUS
    # ============================================================
    # Union data is missing for 1973-1974 and partially for 1982-1983
    # Assign union at job level: if union member in >50% of available years
    job_union = df.groupby(['person_id', 'job_id'])['union_member'].agg(
        lambda x: (x.mean() > 0.5) if x.notna().any() else np.nan
    ).reset_index()
    job_union.columns = ['person_id', 'job_id', 'job_union']
    df = df.merge(job_union, on=['person_id', 'job_id'], how='left')

    # Also keep current-year union for fallback
    df['union_current'] = df['union_member'].copy()

    # ============================================================
    # 8. SAMPLE RESTRICTIONS (common to all subsamples)
    # ============================================================
    # Tenure >= 1
    df = df[df['tenure'] >= 1].copy()

    # Drop missing wages
    df = df.dropna(subset=['log_hourly_wage']).copy()

    # Drop self-employed (code 3)
    df = df[df['occ_broad'] != 3].copy()

    # Drop farmers (code 9)
    df = df[df['occ_broad'] != 9].copy()

    # Drop NA occupation (code 0)
    df = df[df['occ_broad'] != 0].copy()

    # Self-employed flag
    if 'self_employed' in df.columns:
        df = df[(df['self_employed'] == 0) | (df['self_employed'].isna())].copy()

    # Agriculture flag
    if 'agriculture' in df.columns:
        df = df[(df['agriculture'] == 0) | (df['agriculture'].isna())].copy()

    # Government workers
    if 'govt_worker' in df.columns:
        df = df[(df['govt_worker'] == 0) | (df['govt_worker'].isna())].copy()

    # ============================================================
    # 9. INITIAL EXPERIENCE
    # ============================================================
    df['initial_experience'] = df['experience'] - df['tenure']
    df.loc[df['initial_experience'] < 0, 'initial_experience'] = 0

    # ============================================================
    # 10. YEAR DUMMIES
    # ============================================================
    year_cols = [f'year_{y}' for y in range(1969, 1984)]
    for y in range(1969, 1984):
        col = f'year_{y}'
        if col not in df.columns:
            df[col] = (df['year'] == y).astype(int)

    # Region dummies
    region_cols = ['region_ne', 'region_nc', 'region_south']
    for col in region_cols:
        if col not in df.columns:
            df[col] = 0

    # Fill NaN controls
    for c in ['married', 'disabled', 'lives_in_smsa'] + region_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # ============================================================
    # 11. DEFINE SUBSAMPLES
    # ============================================================
    # Professional/Service: all with valid occ in (1,2,4,8)
    mask_ps = df['is_prof_service'] == 1

    # Blue-collar: codes 5,6,7 - split by union
    # For blue-collar, we need union status
    # Use job_union where available, then union_current as fallback
    df['union_for_split'] = df['job_union'].copy()
    # Where job_union is NaN, use current year
    na_mask = df['union_for_split'].isna()
    df.loc[na_mask, 'union_for_split'] = df.loc[na_mask, 'union_current']

    mask_bc_nonunion = (df['is_blue_collar'] == 1) & (df['union_for_split'] == 0)
    mask_bc_union = (df['is_blue_collar'] == 1) & (df['union_for_split'] == 1)

    subsamples = {
        'Professional/Service': mask_ps,
        'BC_Nonunion': mask_bc_nonunion,
        'BC_Union': mask_bc_union
    }

    print("=" * 80)
    print("TABLE 5: Returns to Job Seniority by Occupation and Union Status")
    print("=" * 80)

    for name, mask in subsamples.items():
        print(f"  {name}: N = {mask.sum()}")

    results = {}
    for name, mask in subsamples.items():
        sub = df[mask].copy()
        print(f"\n{'='*60}")
        print(f"Subsample: {name}")
        print(f"N (cross-sectional observations): {len(sub)}")
        print(f"{'='*60}")
        result = run_two_step(sub, name, year_cols, region_cols)
        results[name] = result

    # ============================================================
    # PRINT RESULTS IN TABLE FORMAT
    # ============================================================
    print_results(results)

    # Run scoring
    score = score_against_ground_truth(results)
    return results


def run_two_step(sub, name, year_cols, region_cols):
    """Run the two-step estimation procedure on a subsample."""
    result = {}

    # ================================================================
    # STEP 1: Within-job wage growth regression
    # ================================================================
    sub = sub.sort_values(['person_id', 'job_id', 'year']).copy()

    # Create within-job first differences
    sub['prev_log_real_wage'] = sub.groupby(['person_id', 'job_id'])['log_real_wage'].shift(1)
    sub['d_log_wage'] = sub['log_real_wage'] - sub['prev_log_real_wage']

    sub['prev_tenure'] = sub.groupby(['person_id', 'job_id'])['tenure'].shift(1)
    sub['d_tenure'] = sub['tenure'] - sub['prev_tenure']

    sub['prev_experience'] = sub.groupby(['person_id', 'job_id'])['experience'].shift(1)
    sub['d_experience'] = sub['experience'] - sub['prev_experience']

    # Filter to valid within-job obs (consecutive years, d_tenure == 1)
    wj = sub.dropna(subset=['d_log_wage', 'prev_tenure']).copy()
    wj = wj[wj['d_tenure'] == 1].copy()

    print(f"  Step 1 within-job observations: {len(wj)}")

    # Construct higher-order differenced terms
    # d(T^k) = T^k - (T-1)^k, scaled by powers of 10
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
        wj[dcol] = wj[col].values - wj.groupby(['person_id', 'job_id'])[col].shift(1).values
        wj[dcol] = wj[dcol].fillna(0)
        d_year_cols.append(dcol)

    d_year_cols_use = [c for c in d_year_cols if wj[c].std() > 0]

    # Step 1 regression
    step1_vars = ['d_tenure_sq', 'd_tenure_cu', 'd_tenure_qu',
                  'd_exp_sq', 'd_exp_cu', 'd_exp_qu'] + d_year_cols_use

    X1 = wj[step1_vars].copy()
    X1 = sm.add_constant(X1)
    y1 = wj['d_log_wage']

    valid = X1.notna().all(axis=1) & y1.notna()
    X1 = X1[valid]
    y1 = y1[valid]

    model1 = sm.OLS(y1, X1).fit()

    # Extract Step 1 estimates
    beta_hat = model1.params['const']  # beta_1 + beta_2
    beta_hat_se = model1.bse['const']

    gamma2_scaled = model1.params.get('d_tenure_sq', 0)
    gamma3_scaled = model1.params.get('d_tenure_cu', 0)
    gamma4_scaled = model1.params.get('d_tenure_qu', 0)

    delta2_scaled = model1.params.get('d_exp_sq', 0)
    delta3_scaled = model1.params.get('d_exp_cu', 0)
    delta4_scaled = model1.params.get('d_exp_qu', 0)

    print(f"  Step 1 intercept (beta_1+beta_2): {beta_hat:.4f} (SE: {beta_hat_se:.4f})")
    print(f"  gamma2: {gamma2_scaled:.4f}, gamma3: {gamma3_scaled:.4f}, gamma4: {gamma4_scaled:.4f}")

    result['beta_1_plus_beta_2'] = beta_hat
    result['beta_1_plus_beta_2_se'] = beta_hat_se

    # ================================================================
    # STEP 2: IV regression for beta_1
    # ================================================================
    # Construct adjusted wage
    sub['w_star'] = (sub['log_real_wage']
                     - beta_hat * sub['tenure']
                     - gamma2_scaled * sub['tenure']**2 / 100.0
                     - gamma3_scaled * sub['tenure']**3 / 1000.0
                     - gamma4_scaled * sub['tenure']**4 / 10000.0
                     - delta2_scaled * sub['experience']**2 / 100.0
                     - delta3_scaled * sub['experience']**3 / 1000.0
                     - delta4_scaled * sub['experience']**4 / 10000.0)

    # Controls for Step 2
    controls = ['education_years', 'married', 'disabled', 'lives_in_smsa']
    controls.extend(region_cols)

    # Year dummies
    yr_cols_use = [f'year_{y}' for y in range(1969, 1984)]
    controls.extend(yr_cols_use)

    # Union as control for Professional/Service
    if name == 'Professional/Service':
        sub['union_ctrl'] = sub['union_for_split'].fillna(0)
        controls.append('union_ctrl')

    # Prepare Step 2 data
    step2 = sub.dropna(subset=['w_star', 'experience', 'initial_experience']).copy()

    for c in controls:
        if c in step2.columns:
            step2[c] = step2[c].fillna(0)

    controls_use = [c for c in controls if c in step2.columns and step2[c].std() > 0]

    print(f"  Step 2 observations: {len(step2)}")

    # Manual IV (2SLS) with proper standard errors
    # Endogenous: experience
    # Instrument: initial_experience
    # Controls: exogenous variables

    y2 = step2['w_star'].values.astype(np.float64)
    endog = step2['experience'].values.astype(np.float64)
    instrument = step2['initial_experience'].values.astype(np.float64)

    # Build exogenous matrix (controls + constant)
    X_exog = step2[controls_use].values.astype(np.float64)
    X_exog = np.column_stack([np.ones(len(step2)), X_exog])

    # Build instrument matrix: Z = [constant, controls, initial_experience]
    Z = np.column_stack([X_exog, instrument])

    # Build regressor matrix: X = [constant, controls, experience]
    X_full = np.column_stack([X_exog, endog])

    # First stage: experience = Z * pi
    pi_hat = np.linalg.lstsq(Z, endog, rcond=None)[0]
    endog_hat = Z @ pi_hat

    # Second stage: y = X_hat * beta (replace experience with fitted)
    X_hat = np.column_stack([X_exog, endog_hat])

    # 2SLS coefficient: (X_hat'X_hat)^{-1} X_hat' y
    XhXh = X_hat.T @ X_hat
    XhXh_inv = np.linalg.inv(XhXh)
    beta_2sls = XhXh_inv @ (X_hat.T @ y2)

    # Residuals using ACTUAL X (not predicted)
    resid = y2 - X_full @ beta_2sls
    n = len(y2)
    k = X_full.shape[1]
    sigma2 = (resid @ resid) / (n - k)

    # Proper 2SLS variance: sigma^2 * (X_hat'X_hat)^{-1}
    var_2sls = sigma2 * XhXh_inv

    # beta_1 is the last coefficient (on experience)
    beta_1 = beta_2sls[-1]
    beta_1_se = np.sqrt(var_2sls[-1, -1])

    print(f"  IV beta_1 (experience): {beta_1:.4f} (SE: {beta_1_se:.4f})")

    result['beta_1'] = beta_1
    result['beta_1_se'] = beta_1_se

    # beta_2 = (beta_1 + beta_2) - beta_1
    beta_2 = beta_hat - beta_1
    # SE of beta_2: from independent estimates
    # Var(beta_2) = Var(beta_hat) + Var(beta_1) (approximately, since from different samples)
    beta_2_se = np.sqrt(beta_hat_se**2 + beta_1_se**2)

    result['beta_2'] = beta_2
    result['beta_2_se'] = beta_2_se
    result['N'] = len(step2)

    print(f"  beta_2 (tenure): {beta_2:.4f} (SE: {beta_2_se:.4f})")

    # ================================================================
    # CUMULATIVE RETURNS
    # ================================================================
    # Return at T years = beta_2*T + gamma2_scaled*T^2/100 + gamma3_scaled*T^3/1000 + gamma4_scaled*T^4/10000

    # Get variance-covariance matrix of gamma terms from step 1
    gamma_vars = ['d_tenure_sq', 'd_tenure_cu', 'd_tenure_qu']
    gamma_present = [g for g in gamma_vars if g in model1.params.index]
    gamma_vcov = model1.cov_params().loc[gamma_present, gamma_present].values

    gamma_vals = np.array([model1.params[g] for g in gamma_present])

    for T_val in [5, 10, 15, 20]:
        cum_return = (beta_2 * T_val
                      + gamma2_scaled * T_val**2 / 100.0
                      + gamma3_scaled * T_val**3 / 1000.0
                      + gamma4_scaled * T_val**4 / 10000.0)

        # SE via delta method
        # grad w.r.t. beta_2: T
        # grad w.r.t. gamma2: T^2/100
        # grad w.r.t. gamma3: T^3/1000
        # grad w.r.t. gamma4: T^4/10000
        grad_gamma = np.array([T_val**2/100.0, T_val**3/1000.0, T_val**4/10000.0])

        var_cum = T_val**2 * beta_2_se**2  # from beta_2
        # Add variance from gamma terms
        if len(gamma_present) == 3:
            var_cum += grad_gamma @ gamma_vcov @ grad_gamma
        else:
            for i, g in enumerate(gamma_present):
                var_cum += grad_gamma[i]**2 * gamma_vcov[i, i]

        cum_se = np.sqrt(max(0, var_cum))

        result[f'cum_return_{T_val}'] = cum_return
        result[f'cum_return_{T_val}_se'] = cum_se

        print(f"  Cumulative return at {T_val} years: {cum_return:.4f} (SE: {cum_se:.4f})")

    return result


def print_results(results):
    """Print results in table format."""
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
    print("BOTTOM PANEL: Estimated Cumulative Returns to Tenure")
    print("=" * 80)

    header = f"{'':15s} {'(1) Prof/Svc':>12s} {'(2) Nonunion':>12s} {'(3) Union':>12s} {'(4) U vs NU':>12s}"
    print(header)
    print("-" * 63)

    for T_val in [5, 10, 15, 20]:
        key = f'cum_return_{T_val}'
        key_se = f'cum_return_{T_val}_se'

        vals = [results[n].get(key, np.nan) for n in names]
        ses = [results[n].get(key_se, np.nan) for n in names]

        u_cum = results['BC_Union'].get(key, np.nan)
        nu_cum = results['BC_Nonunion'].get(key, np.nan)
        beta1_nu = results['BC_Nonunion'].get('beta_1', 0)
        beta1_u = results['BC_Union'].get('beta_1', 0)
        union_rel = u_cum + (beta1_nu - beta1_u) * T_val

        u_se = results['BC_Union'].get(key_se, np.nan)
        beta1_nu_se = results['BC_Nonunion'].get('beta_1_se', 0)
        beta1_u_se = results['BC_Union'].get('beta_1_se', 0)
        union_rel_se = np.sqrt(u_se**2 + (T_val * beta1_nu_se)**2 + (T_val * beta1_u_se)**2)

        line1 = f"{T_val:2d} years:       {vals[0]:12.4f} {vals[1]:12.4f} {vals[2]:12.4f} {union_rel:12.4f}"
        line2 = f"{'':15s} {'(' + f'{ses[0]:.4f}' + ')':>12s} {'(' + f'{ses[1]:.4f}' + ')':>12s} {'(' + f'{ses[2]:.4f}' + ')':>12s} {'(' + f'{union_rel_se:.4f}' + ')':>12s}"
        print(line1)
        print(line2)

    print("\nSample sizes:")
    for n, label in zip(names, ['Professional/Service', 'Nonunion', 'Union']):
        print(f"  {label}: N = {results[n]['N']:,}")


def score_against_ground_truth(results):
    """Score results against the paper's ground truth values."""
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

    # 1. Step 1 coefficients (beta_1+beta_2) - 15 pts (5 per subsample)
    print("\n--- Step 1 coefficients (beta_1+beta_2): 15 pts ---")
    for name in ['Professional/Service', 'BC_Nonunion', 'BC_Union']:
        gen = results[name].get('beta_1_plus_beta_2', np.nan)
        true_val = gt[name]['beta_1_plus_beta_2']
        rel_err = abs(gen - true_val) / max(abs(true_val), 0.001) if not np.isnan(gen) else 1
        pts = 5 if rel_err <= 0.20 else (3 if rel_err <= 0.40 else 0)
        earned_points += pts
        total_points += 5
        print(f"  {name}: gen={gen:.4f} true={true_val:.4f} rel_err={rel_err:.3f} -> {pts}/5")

    # 2. Step 2 coefficients (beta_1, beta_2) - 20 pts
    print("\n--- Step 2 coefficients (beta_1, beta_2): 20 pts ---")
    coef_pts_each = 20 / 6
    for name in ['Professional/Service', 'BC_Nonunion', 'BC_Union']:
        for param in ['beta_1', 'beta_2']:
            gen = results[name].get(param, np.nan)
            true_val = gt[name][param]
            abs_err = abs(gen - true_val) if not np.isnan(gen) else 1
            pts = coef_pts_each if abs_err <= 0.01 else (coef_pts_each * 0.5 if abs_err <= 0.03 else 0)
            earned_points += pts
            total_points += coef_pts_each
            print(f"  {name} {param}: gen={gen:.4f} true={true_val:.4f} abs_err={abs_err:.4f} -> {pts:.2f}/{coef_pts_each:.2f}")

    # 3. Cumulative returns - 20 pts
    print("\n--- Cumulative returns (cols 1-3): 20 pts ---")
    cum_pts_each = 20 / 12
    for name in ['Professional/Service', 'BC_Nonunion', 'BC_Union']:
        for T_val in [5, 10, 15, 20]:
            key = f'cum_return_{T_val}'
            gen = results[name].get(key, np.nan)
            true_val = gt[name][key]
            abs_err = abs(gen - true_val) if not np.isnan(gen) else 1
            pts = cum_pts_each if abs_err <= 0.03 else (cum_pts_each * 0.5 if abs_err <= 0.06 else 0)
            earned_points += pts
            total_points += cum_pts_each
            print(f"  {name} {T_val}yr: gen={gen:.4f} true={true_val:.4f} abs_err={abs_err:.4f} -> {pts:.2f}/{cum_pts_each:.2f}")

    # 4. Standard errors - 10 pts
    print("\n--- Standard errors: 10 pts ---")
    se_items = []
    for name in ['Professional/Service', 'BC_Nonunion', 'BC_Union']:
        for param in ['beta_1_se', 'beta_2_se', 'beta_1_plus_beta_2_se']:
            se_items.append((name, param))
    se_pts_each = 10 / len(se_items)
    for name, param in se_items:
        gen = results[name].get(param, np.nan)
        true_val = gt[name][param]
        rel_err = abs(gen - true_val) / max(abs(true_val), 0.001) if not np.isnan(gen) else 1
        pts = se_pts_each if rel_err <= 0.30 else (se_pts_each * 0.5 if rel_err <= 0.60 else 0)
        earned_points += pts
        total_points += se_pts_each
        print(f"  {name} {param}: gen={gen:.4f} true={true_val:.4f} rel_err={rel_err:.3f} -> {pts:.2f}/{se_pts_each:.2f}")

    # 5. Sample size - 15 pts
    print("\n--- Sample sizes: 15 pts ---")
    for name in ['Professional/Service', 'BC_Nonunion', 'BC_Union']:
        gen = results[name].get('N', 0)
        true_val = gt[name]['N']
        rel_err = abs(gen - true_val) / true_val
        pts = 5 if rel_err <= 0.05 else (3 if rel_err <= 0.10 else (1 if rel_err <= 0.20 else 0))
        earned_points += pts
        total_points += 5
        print(f"  {name}: gen={gen} true={true_val} rel_err={rel_err:.3f} -> {pts}/5")

    # 6. Significance levels - 10 pts
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
    sig_earned = 10 * sign_correct / max(sign_total, 1)
    earned_points += sig_earned
    total_points += 10
    print(f"  Signs correct: {sign_correct}/{sign_total} -> {sig_earned:.1f}/10")

    # 7. All columns present - 10 pts
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
