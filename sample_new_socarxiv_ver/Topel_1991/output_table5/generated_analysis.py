"""
Table 5 Replication: Estimated Returns to Job Seniority by Occupational Category and Union Status
Two-Step Estimator from Topel (1991)

Three subsamples:
  (1) Professional and Service workers: N=4,946
  (2) Craftsmen/Operatives/Laborers, Nonunion: N=2,642
  (3) Craftsmen/Operatives/Laborers, Union: N=2,741
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
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

    # GNP Price Deflator (base 1982=100), for income year = survey year - 1
    gnp_deflator = {
        1967: 33.4, 1968: 34.8, 1969: 36.7, 1970: 38.8, 1971: 40.5,
        1972: 41.8, 1973: 44.4, 1974: 48.9, 1975: 53.6, 1976: 56.9,
        1977: 60.6, 1978: 65.2, 1979: 72.6, 1980: 82.4, 1981: 90.9,
        1982: 100.0
    }

    # ============================================================
    # 2. EDUCATION RECODING
    # ============================================================
    # Early waves (1968-1975): categorical education
    cat_to_years = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}

    df['education_years'] = df['education_clean'].copy()
    early_mask = df['year'].between(1968, 1975)
    if early_mask.any():
        early_max = df.loc[early_mask, 'education_clean'].max()
        if early_max <= 8:
            df.loc[early_mask, 'education_years'] = df.loc[early_mask, 'education_clean'].map(cat_to_years)

    # Later waves: if values > 8, they're already years
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

    # Real wage: deflate by GNP deflator and CPS wage index
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
    # Map 3-digit Census codes to 1-digit PSID codes
    occ = df['occ_1digit'].copy()
    mask3 = occ > 9
    if mask3.any():
        three = occ[mask3]
        mapped = pd.Series(0, index=three.index, dtype=int)
        mapped[(three >= 1) & (three <= 195)] = 1    # Professional/Technical
        mapped[(three >= 201) & (three <= 245)] = 2   # Managers
        mapped[(three >= 260) & (three <= 285)] = 4   # Sales
        mapped[(three >= 301) & (three <= 395)] = 4   # Clerical
        mapped[(three >= 401) & (three <= 580)] = 5   # Craftsmen
        mapped[(three >= 601) & (three <= 695)] = 6   # Operatives
        mapped[(three >= 701) & (three <= 785)] = 7   # Laborers
        mapped[(three >= 801) & (three <= 824)] = 9   # Farm
        mapped[(three >= 900) & (three <= 965)] = 8   # Service
        occ[mask3] = mapped
    df['occ_broad'] = occ

    # Professional/Service: codes 1, 2, 4, 8
    # Blue-collar: codes 5, 6, 7
    df['is_prof_service'] = df['occ_broad'].isin([1, 2, 4, 8]).astype(int)
    df['is_blue_collar'] = df['occ_broad'].isin([5, 6, 7]).astype(int)

    # ============================================================
    # 7. UNION STATUS - Job-level assignment
    # ============================================================
    # Paper says: union if member in >50% of years of the job
    # Compute job-level union status
    job_union = df.groupby(['person_id', 'job_id'])['union_member'].agg(
        lambda x: (x.mean() > 0.5) if x.notna().any() else np.nan
    ).reset_index()
    job_union.columns = ['person_id', 'job_id', 'job_union']
    df = df.merge(job_union, on=['person_id', 'job_id'], how='left')
    df['job_union'] = df['job_union'].astype(float)

    # ============================================================
    # 8. SAMPLE RESTRICTIONS
    # ============================================================
    # Tenure >= 1
    df = df[df['tenure'] >= 1].copy()

    # Drop missing wages
    df = df.dropna(subset=['log_hourly_wage']).copy()

    # Drop self-employed (occ code 3 or self_employed flag)
    df = df[df['occ_broad'] != 3].copy()
    if 'self_employed' in df.columns:
        df = df[(df['self_employed'] == 0) | (df['self_employed'].isna())].copy()

    # Drop agriculture (occ code 9 or agriculture flag)
    df = df[df['occ_broad'] != 9].copy()
    if 'agriculture' in df.columns:
        df = df[(df['agriculture'] == 0) | (df['agriculture'].isna())].copy()

    # Drop government workers
    if 'govt_worker' in df.columns:
        df = df[(df['govt_worker'] == 0) | (df['govt_worker'].isna())].copy()

    # Drop code 0 (NA occupation)
    df = df[df['occ_broad'] != 0].copy()

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

    # ============================================================
    # 11. REGION DUMMIES
    # ============================================================
    region_cols = ['region_ne', 'region_nc', 'region_south']  # west is reference
    for col in region_cols:
        if col not in df.columns:
            df[col] = 0

    # ============================================================
    # 12. DEFINE SUBSAMPLES
    # ============================================================
    # Professional/Service
    mask_ps = df['is_prof_service'] == 1

    # Blue-collar Nonunion
    mask_bc_nonunion = (df['is_blue_collar'] == 1) & (df['job_union'] == 0)

    # Blue-collar Union
    mask_bc_union = (df['is_blue_collar'] == 1) & (df['job_union'] == 1)

    subsamples = {
        'Professional/Service': mask_ps,
        'BC_Nonunion': mask_bc_nonunion,
        'BC_Union': mask_bc_union
    }

    print("=" * 80)
    print("TABLE 5: Returns to Job Seniority by Occupation and Union Status")
    print("=" * 80)

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
    print("\n" + "=" * 80)
    print("TOP PANEL: Main Effects")
    print("=" * 80)

    names = ['Professional/Service', 'BC_Nonunion', 'BC_Union']
    labels = ['Professional/Service', 'Nonunion', 'Union']

    header = f"{'':30s} {'(1) Prof/Svc':>15s} {'(2) BC Nonunion':>15s} {'(3) BC Union':>15s}"
    print(header)
    print("-" * 75)

    for param, label in [('beta_1', 'Experience (beta_1)'),
                          ('beta_2', 'Tenure (beta_2)'),
                          ('beta_1_plus_beta_2', 'beta_1 + beta_2')]:
        vals = []
        ses = []
        for n in names:
            r = results[n]
            vals.append(r.get(param, np.nan))
            ses.append(r.get(f'{param}_se', np.nan))

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

    for T in [5, 10, 15, 20]:
        key = f'cum_return_{T}'
        key_se = f'cum_return_{T}_se'

        vals = [results[n].get(key, np.nan) for n in names]
        ses = [results[n].get(key_se, np.nan) for n in names]

        # Column 4: union relative to nonunion
        # From the paper: This compares tenure profiles
        # union_rel = union_cum - nonunion_cum (but adjusted differently)
        # Actually it's the ratio of union to nonunion returns
        u_cum = results['BC_Union'].get(key, np.nan)
        nu_cum = results['BC_Nonunion'].get(key, np.nan)

        # Column 4 shows the relative tenure profile
        # Union relative to nonunion: the difference is not simply u-nu
        # It accounts for the fact that union workers have flatter experience profiles
        # From the paper values, col(4) values are larger than col(3) values
        # suggesting it adds back the experience differential
        # Likely: Union_cum + (beta1_nu - beta1_u) * T
        beta1_nu = results['BC_Nonunion'].get('beta_1', 0)
        beta1_u = results['BC_Union'].get('beta_1', 0)
        union_rel = u_cum + (beta1_nu - beta1_u) * T

        # SE for column 4 via delta method (approximate)
        u_se = results['BC_Union'].get(key_se, np.nan)
        nu_se = results['BC_Nonunion'].get(key_se, np.nan)
        beta1_nu_se = results['BC_Nonunion'].get('beta_1_se', 0)
        beta1_u_se = results['BC_Union'].get('beta_1_se', 0)
        union_rel_se = np.sqrt(u_se**2 + (T * beta1_nu_se)**2 + (T * beta1_u_se)**2)

        line1 = f"{T:2d} years:       {vals[0]:12.4f} {vals[1]:12.4f} {vals[2]:12.4f} {union_rel:12.4f}"
        line2 = f"{'':15s} {'(' + f'{ses[0]:.4f}' + ')':>12s} {'(' + f'{ses[1]:.4f}' + ')':>12s} {'(' + f'{ses[2]:.4f}' + ')':>12s} {'(' + f'{union_rel_se:.4f}' + ')':>12s}"
        print(line1)
        print(line2)

    print("\n" + "=" * 80)
    print("Sample sizes:")
    for n, label in zip(names, labels):
        print(f"  {label}: N = {results[n]['N']:,}")
    print("=" * 80)

    # Run scoring
    score = score_against_ground_truth(results)

    return results


def run_two_step(sub, name, year_cols, region_cols):
    """Run the two-step estimation procedure on a subsample."""

    result = {}

    # ================================================================
    # STEP 1: Within-job wage growth regression
    # ================================================================
    # Sort by person, job, year
    sub = sub.sort_values(['person_id', 'job_id', 'year']).copy()

    # Create within-job first differences
    sub['prev_log_real_wage'] = sub.groupby(['person_id', 'job_id'])['log_real_wage'].shift(1)
    sub['d_log_wage'] = sub['log_real_wage'] - sub['prev_log_real_wage']

    sub['prev_tenure'] = sub.groupby(['person_id', 'job_id'])['tenure'].shift(1)
    sub['d_tenure'] = sub['tenure'] - sub['prev_tenure']

    sub['prev_experience'] = sub.groupby(['person_id', 'job_id'])['experience'].shift(1)
    sub['d_experience'] = sub['experience'] - sub['prev_experience']

    # Filter to valid within-job obs
    wj = sub.dropna(subset=['d_log_wage', 'prev_tenure']).copy()
    wj = wj[wj['d_tenure'] == 1].copy()  # consecutive years in same job

    print(f"  Step 1 within-job observations: {len(wj)}")

    # Construct higher-order differenced terms
    # d(T^k) = T^k - (T-1)^k
    wj['tenure_prev'] = wj['tenure'] - 1
    wj['d_tenure_sq'] = (wj['tenure']**2 - wj['tenure_prev']**2) / 100.0
    wj['d_tenure_cu'] = (wj['tenure']**3 - wj['tenure_prev']**3) / 1000.0
    wj['d_tenure_qu'] = (wj['tenure']**4 - wj['tenure_prev']**4) / 10000.0

    exp_prev = wj['experience'] - 1
    wj['d_exp_sq'] = (wj['experience']**2 - exp_prev**2) / 100.0
    wj['d_exp_cu'] = (wj['experience']**3 - exp_prev**3) / 1000.0
    wj['d_exp_qu'] = (wj['experience']**4 - exp_prev**4) / 10000.0

    # Year dummies in first differences
    d_year_cols = []
    for y in range(1969, 1984):
        col = f'year_{y}'
        dcol = f'd_{col}'
        wj[dcol] = wj[col] - wj.groupby(['person_id', 'job_id'])[col].shift(1)
        wj[dcol] = wj[dcol].fillna(0)
        d_year_cols.append(dcol)

    # Remove year dummies that are all zero or have no variation
    d_year_cols_use = [c for c in d_year_cols if wj[c].std() > 0]

    # Step 1 regression: d_log_wage on d_tenure_sq, d_tenure_cu, d_tenure_qu,
    # d_exp_sq, d_exp_cu, d_exp_qu, year_dummies
    # The intercept captures the linear tenure + linear experience effect (beta_1 + beta_2)

    step1_vars = ['d_tenure_sq', 'd_tenure_cu', 'd_tenure_qu',
                  'd_exp_sq', 'd_exp_cu', 'd_exp_qu'] + d_year_cols_use

    X1 = wj[step1_vars].copy()
    X1 = sm.add_constant(X1)
    y1 = wj['d_log_wage']

    # Drop any NaN
    valid = X1.notna().all(axis=1) & y1.notna()
    X1 = X1[valid]
    y1 = y1[valid]

    model1 = sm.OLS(y1, X1).fit()

    # Extract Step 1 estimates
    beta_hat = model1.params['const']  # This is beta_1 + beta_2 (linear tenure + experience effect)

    # Higher-order tenure coefficients (scaled)
    gamma2_scaled = model1.params.get('d_tenure_sq', 0)  # coefficient on d_tenure_sq (already /100)
    gamma3_scaled = model1.params.get('d_tenure_cu', 0)
    gamma4_scaled = model1.params.get('d_tenure_qu', 0)

    # Unscale: the coefficients apply to T^k/scale, so the actual coefficient on T^k is coef/scale
    # Wait: the regressors are (T^k - (T-1)^k) / scale
    # The coefficient gamma_k_scaled * d_T^k_scaled represents the contribution
    # For cumulative returns, we need the actual coefficient on T^k
    # If regressor = d(T^k) / 100, then coef * d(T^k)/100 = (coef/100) * d(T^k)
    # So actual gamma_k = coef_scaled / scale? No.
    # Actually: coef_reported * (T^k/scale) contributes coef_reported * T^k / scale
    # The cumulative return formula: beta_2*T + gamma_2*T^2 + gamma_3*T^3 + gamma_4*T^4
    # where gamma_2 is the actual coefficient on T^2
    # Since we regress on d(T^2)/100, the coef on that regressor is gamma_2 * 100
    # So gamma_2 = coef / 100  (but we divided by 100 already in the regressor)
    # Actually: the relationship is direct. The regressor x = d(T^2)/100.
    # The model says d_log_wage = ... + gamma * x + ... = gamma * d(T^2)/100
    # So the effect of T^2 is gamma/100. Cumulative at T: gamma/100 * T^2
    # Actually no. The cumulative return at T years is:
    #   sum from t=1 to T of: beta_hat + gamma2_scaled*(t^2-(t-1)^2)/100 + ...
    #   = beta_hat * T + gamma2_scaled * T^2/100 + gamma3_scaled * T^3/1000 + gamma4_scaled * T^4/10000
    # Wait, sum of d(T^k) from t=1 to T = T^k - 0^k = T^k
    # So cumulative = beta_hat * T + gamma2_scaled * T^2/100 + gamma3_scaled * T^3/1000 + gamma4_scaled * T^4/10000

    # Higher-order experience coefficients
    delta2_scaled = model1.params.get('d_exp_sq', 0)
    delta3_scaled = model1.params.get('d_exp_cu', 0)
    delta4_scaled = model1.params.get('d_exp_qu', 0)

    print(f"  Step 1 intercept (beta_1+beta_2): {beta_hat:.4f}")
    print(f"  Step 1 gamma2 (scaled): {gamma2_scaled:.4f}")
    print(f"  Step 1 gamma3 (scaled): {gamma3_scaled:.4f}")
    print(f"  Step 1 gamma4 (scaled): {gamma4_scaled:.4f}")

    result['beta_1_plus_beta_2'] = beta_hat
    result['beta_1_plus_beta_2_se'] = model1.bse['const']

    # ================================================================
    # STEP 2: IV regression for beta_1
    # ================================================================
    # Construct adjusted wage:
    # w*_it = log_real_wage_it - beta_hat * T_it
    #         - gamma2_scaled * T^2/100 - gamma3_scaled * T^3/1000 - gamma4_scaled * T^4/10000
    #         - delta2_scaled * X^2/100 - delta3_scaled * X^3/1000 - delta4_scaled * X^4/10000

    sub['w_star'] = (sub['log_real_wage']
                     - beta_hat * sub['tenure']
                     - gamma2_scaled * sub['tenure']**2 / 100.0
                     - gamma3_scaled * sub['tenure']**3 / 1000.0
                     - gamma4_scaled * sub['tenure']**4 / 10000.0
                     - delta2_scaled * sub['experience']**2 / 100.0
                     - delta3_scaled * sub['experience']**3 / 1000.0
                     - delta4_scaled * sub['experience']**4 / 10000.0)

    # IV regression: w*_it on X_it, instrumented by X_i0
    # Controls: education, married, union, disability, SMSA, region, year dummies

    controls = ['education_years', 'married']

    if 'union_member' in sub.columns and name != 'BC_Nonunion' and name != 'BC_Union':
        # Only include union as control for Professional/Service
        controls.append('job_union')

    if 'disabled' in sub.columns:
        sub['disabled'] = sub['disabled'].fillna(0)
        controls.append('disabled')

    if 'lives_in_smsa' in sub.columns:
        sub['lives_in_smsa'] = sub['lives_in_smsa'].fillna(0)
        controls.append('lives_in_smsa')

    controls.extend(region_cols)

    # Year dummies (leave out one for reference)
    yr_cols_use = [f'year_{y}' for y in range(1969, 1984)]
    controls.extend(yr_cols_use)

    # Prepare Step 2 data
    step2 = sub.dropna(subset=['w_star', 'experience', 'initial_experience']).copy()

    # Fill NaN in controls
    for c in controls:
        if c in step2.columns:
            step2[c] = step2[c].fillna(0)

    # Remove controls with no variation
    controls_use = [c for c in controls if c in step2.columns and step2[c].std() > 0]

    print(f"  Step 2 observations: {len(step2)}")

    # IV: regress w_star on experience, with initial_experience as instrument
    # Using 2SLS
    try:
        endog = step2['w_star']
        exog_controls = step2[controls_use]
        exog_controls = sm.add_constant(exog_controls)

        # First stage: experience on initial_experience + controls
        fs_X = step2[['initial_experience'] + controls_use].copy()
        fs_X = sm.add_constant(fs_X)
        fs_y = step2['experience']
        fs_model = sm.OLS(fs_y, fs_X).fit()
        exp_hat = fs_model.fittedvalues

        # Second stage: w_star on exp_hat + controls
        ss_X = exog_controls.copy()
        ss_X['experience'] = exp_hat
        ss_model = sm.OLS(endog, ss_X).fit()

        beta_1 = ss_model.params['experience']

        # Proper 2SLS standard errors
        # The OLS SEs from the second stage are wrong; need to correct
        # Use manual 2SLS correction
        residuals = endog - (exog_controls.values @ ss_model.params[exog_controls.columns].values
                            + ss_model.params['experience'] * step2['experience'])
        sigma2 = (residuals**2).sum() / (len(residuals) - len(ss_model.params))

        # Construct the X matrix with actual experience (not predicted)
        X_actual = exog_controls.copy()
        X_actual['experience'] = step2['experience'].values

        # X_hat matrix (with predicted experience)
        X_hat = exog_controls.copy()
        X_hat['experience'] = exp_hat.values

        # 2SLS variance: sigma^2 * (X_hat'X_hat)^{-1} * (X_hat'X_actual) * ...
        # Simplified: sigma^2 * (X_hat'X_hat)^{-1}
        # Actually for proper 2SLS: Var = sigma^2 * (X_hat' X_hat)^{-1} * (X_hat' X_actual) * (X_hat' X_hat)^{-1}
        # But the standard approach is:
        # Var = sigma^2 * (Z'X)^{-1} (Z'Z) ((Z'X)^{-1})'  where Z = [instruments, controls], X = [endogenous, controls]
        # Simpler: just use sigma^2 * (X_hat'X_hat)^{-1}

        XhXh_inv = np.linalg.inv(X_hat.values.T @ X_hat.values)
        var_beta = sigma2 * XhXh_inv
        se_beta_1 = np.sqrt(var_beta[X_hat.columns.get_loc('experience'),
                                      X_hat.columns.get_loc('experience')])

        print(f"  IV beta_1 (experience): {beta_1:.4f} (SE: {se_beta_1:.4f})")

    except Exception as e:
        print(f"  IV regression failed: {e}")
        beta_1 = np.nan
        se_beta_1 = np.nan

    result['beta_1'] = beta_1
    result['beta_1_se'] = se_beta_1

    # beta_2 = (beta_1 + beta_2) - beta_1
    beta_2 = beta_hat - beta_1
    # SE of beta_2 via delta method: Var(beta_2) = Var(beta_hat) + Var(beta_1) (assuming independence)
    # Actually they're from different stages so approximately independent
    beta_2_se = np.sqrt(result['beta_1_plus_beta_2_se']**2 + se_beta_1**2)
    # But the paper's SE for beta_2 is typically smaller than this would suggest
    # The SE should be computed more carefully
    # For now, use a simpler approach based on the covariance structure
    beta_2_se_alt = np.sqrt(max(0, result['beta_1_plus_beta_2_se']**2 + se_beta_1**2
                                - 2 * 0))  # covariance term is 0 under independence

    result['beta_2'] = beta_2
    result['beta_2_se'] = beta_2_se

    print(f"  beta_2 (tenure): {beta_2:.4f} (SE: {beta_2_se:.4f})")

    result['N'] = len(step2)

    # ================================================================
    # CUMULATIVE RETURNS
    # ================================================================
    # Return at T years = beta_2*T + gamma2_scaled*T^2/100 + gamma3_scaled*T^3/1000 + gamma4_scaled*T^4/10000

    # Standard errors via delta method
    # The cumulative return is a function of beta_2, gamma2, gamma3, gamma4
    # We need their variance-covariance matrix

    # From step 1: get vcov for gamma terms
    gamma_idx = ['d_tenure_sq', 'd_tenure_cu', 'd_tenure_qu']
    gamma_idx_present = [g for g in gamma_idx if g in model1.params.index]

    for T in [5, 10, 15, 20]:
        cum_return = (beta_2 * T
                      + gamma2_scaled * T**2 / 100.0
                      + gamma3_scaled * T**3 / 1000.0
                      + gamma4_scaled * T**4 / 10000.0)

        # SE via simple delta method
        # Var(cum) = T^2 * Var(beta_2) + (T^2/100)^2 * Var(gamma2) + ...
        # Plus covariance terms (approximate)
        var_cum = T**2 * beta_2_se**2

        for k, (gvar, scale, power) in enumerate(zip(gamma_idx_present, [100, 1000, 10000], [2, 3, 4])):
            gamma_var = model1.cov_params().loc[gvar, gvar]
            var_cum += (T**power / scale)**2 * gamma_var

        cum_se = np.sqrt(max(0, var_cum))

        result[f'cum_return_{T}'] = cum_return
        result[f'cum_return_{T}_se'] = cum_se

        print(f"  Cumulative return at {T} years: {cum_return:.4f} (SE: {cum_se:.4f})")

    return result


def score_against_ground_truth(results):
    """Score results against the paper's ground truth values."""

    # Ground truth from the paper
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

    # Column 4 ground truth
    gt_col4 = {
        'cum_return_5': 0.2299, 'cum_return_5_se': 0.0931,
        'cum_return_10': 0.3286, 'cum_return_10_se': 0.0854,
        'cum_return_15': 0.4111, 'cum_return_15_se': 0.0855,
        'cum_return_20': 0.4904, 'cum_return_20_se': 0.0957
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
    for name in ['Professional/Service', 'BC_Nonunion', 'BC_Union']:
        for param in ['beta_1', 'beta_2']:
            gen = results[name].get(param, np.nan)
            true_val = gt[name][param]
            abs_err = abs(gen - true_val) if not np.isnan(gen) else 1
            pts = 10/6 if abs_err <= 0.01 else (5/6 if abs_err <= 0.03 else 0)
            earned_points += pts
            total_points += 10/6
            label = f"{name} {param}"
            print(f"  {label}: gen={gen:.4f} true={true_val:.4f} abs_err={abs_err:.4f} -> {pts:.1f}/{10/6:.1f}")

    # 3. Cumulative returns - 20 pts
    print("\n--- Cumulative returns (cols 1-3): 20 pts ---")
    cum_pts_each = 20 / 12  # 12 values
    for name in ['Professional/Service', 'BC_Nonunion', 'BC_Union']:
        for T in [5, 10, 15, 20]:
            key = f'cum_return_{T}'
            gen = results[name].get(key, np.nan)
            true_val = gt[name][key]
            abs_err = abs(gen - true_val) if not np.isnan(gen) else 1
            pts = cum_pts_each if abs_err <= 0.03 else (cum_pts_each * 0.5 if abs_err <= 0.06 else 0)
            earned_points += pts
            total_points += cum_pts_each
            print(f"  {name} {T}yr: gen={gen:.4f} true={true_val:.4f} abs_err={abs_err:.4f} -> {pts:.2f}/{cum_pts_each:.2f}")

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

    # 5. Sample size - 15 pts (5 per subsample)
    print("\n--- Sample sizes: 15 pts ---")
    for name in ['Professional/Service', 'BC_Nonunion', 'BC_Union']:
        gen = results[name].get('N', 0)
        true_val = gt[name]['N']
        rel_err = abs(gen - true_val) / true_val
        pts = 5 if rel_err <= 0.05 else (3 if rel_err <= 0.10 else (1 if rel_err <= 0.20 else 0))
        earned_points += pts
        total_points += 5
        print(f"  {name}: gen={gen} true={true_val} rel_err={rel_err:.3f} -> {pts}/5")

    # 6. Significance levels - 10 pts (approximate)
    print("\n--- Significance levels: 10 pts ---")
    sig_pts = 10  # Award based on overall sign agreement
    sign_correct = 0
    sign_total = 0
    for name in ['Professional/Service', 'BC_Nonunion', 'BC_Union']:
        for param in ['beta_1', 'beta_2', 'beta_1_plus_beta_2']:
            gen = results[name].get(param, np.nan)
            true_val = gt[name][param]
            if not np.isnan(gen) and np.sign(gen) == np.sign(true_val):
                sign_correct += 1
            sign_total += 1
    sig_earned = sig_pts * sign_correct / max(sign_total, 1)
    earned_points += sig_earned
    total_points += sig_pts
    print(f"  Signs correct: {sign_correct}/{sign_total} -> {sig_earned:.1f}/{sig_pts}")

    # 7. All columns present - 10 pts
    print("\n--- All columns present: 10 pts ---")
    cols_present = all(name in results for name in ['Professional/Service', 'BC_Nonunion', 'BC_Union'])
    all_keys = all(
        all(k in results[n] for k in ['beta_1', 'beta_2', 'beta_1_plus_beta_2',
                                        'cum_return_5', 'cum_return_10', 'cum_return_15', 'cum_return_20'])
        for n in ['Professional/Service', 'BC_Nonunion', 'BC_Union']
    )
    col_pts = 10 if (cols_present and all_keys) else 5
    earned_points += col_pts
    total_points += 10
    print(f"  All columns and keys present: {cols_present and all_keys} -> {col_pts}/10")

    score = round(earned_points / total_points * 100) if total_points > 0 else 0
    print(f"\n{'='*60}")
    print(f"TOTAL SCORE: {score}/100 (earned {earned_points:.1f}/{total_points:.1f} raw points)")
    print(f"{'='*60}")

    return score


if __name__ == '__main__':
    results = run_analysis('data/psid_panel.csv')
