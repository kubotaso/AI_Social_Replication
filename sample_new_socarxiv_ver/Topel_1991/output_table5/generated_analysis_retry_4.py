"""
Table 5 Replication: Returns to Job Seniority by Occupation and Union Status
Two-Step Estimator from Topel (1991)

Attempt 4: Key corrections:
1. CORRECT w* construction: subtract ONLY higher-order terms, NOT linear beta_hat*T
   (Equation 10 in paper: y - x'Gamma_hat = X_0*beta_1 + F*gamma + e)
2. Murphy-Topel (1985) standard error correction for two-step estimation
3. Run step 1 and step 2 independently for each subsample
4. Cumulative returns: beta_2*T + g2*T^2/100 + g3*T^3/1000 + g4*T^4/10000
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')


def prepare_data(data_source='data/psid_panel.csv'):
    """Load and prepare the PSID panel data."""
    df = pd.read_csv(data_source)

    # Education recoding
    cat_to_years = {0:0,1:3,2:7,3:10,4:12,5:12,6:14,7:16,8:17}
    df['education_years'] = df['education_clean'].copy()
    for mask_cond in [df['year'].between(1968,1975), df['year']>=1976]:
        if mask_cond.any() and df.loc[mask_cond,'education_clean'].max() <= 8:
            df.loc[mask_cond,'education_years'] = df.loc[mask_cond,'education_clean'].map(cat_to_years)
    df['education_years'] = df['education_years'].fillna(12)

    # Experience
    df['experience'] = df['age'] - df['education_years'] - 6
    df.loc[df['experience']<0,'experience'] = 0

    # Wage deflation
    cps = {1968:1.0,1969:1.032,1970:1.091,1971:1.115,1972:1.113,1973:1.151,
           1974:1.167,1975:1.188,1976:1.117,1977:1.121,1978:1.133,1979:1.128,
           1980:1.128,1981:1.109,1982:1.103,1983:1.089}
    gnp = {1967:33.4,1968:34.8,1969:36.7,1970:38.8,1971:40.5,1972:41.8,
           1973:44.4,1974:48.9,1975:53.6,1976:56.9,1977:60.6,1978:65.2,
           1979:72.6,1980:82.4,1981:90.9,1982:100.0}
    df['cps_index'] = df['year'].map(cps)
    df['gnp_defl'] = (df['year']-1).map(gnp)
    df['log_real_wage'] = df['log_hourly_wage'] - np.log(df['gnp_defl']/100) - np.log(df['cps_index'])

    df['tenure'] = df['tenure_topel']

    # Occupation mapping
    occ = df['occ_1digit'].copy()
    m = occ > 9
    if m.any():
        three = occ[m]
        mapped = pd.Series(0, index=three.index, dtype=int)
        mapped[(three>=1)&(three<=195)] = 1
        mapped[(three>=201)&(three<=245)] = 2
        mapped[(three>=260)&(three<=285)] = 4
        mapped[(three>=301)&(three<=395)] = 4
        mapped[(three>=401)&(three<=580)] = 5
        mapped[(three>=601)&(three<=695)] = 6
        mapped[(three>=701)&(three<=785)] = 7
        mapped[(three>=801)&(three<=824)] = 9
        mapped[(three>=900)&(three<=965)] = 8
        occ[m] = mapped
    df['occ_broad'] = occ

    # Union: fill within job
    df = df.sort_values(['person_id','job_id','year'])
    df['union_filled'] = df.groupby(['person_id','job_id'])['union_member'].transform(
        lambda x: x.ffill().bfill()
    )
    # Job-level union
    job_union = df.groupby(['person_id','job_id'])['union_filled'].agg(
        lambda x: (x.mean() > 0.5) if x.notna().any() else np.nan
    ).reset_index()
    job_union.columns = ['person_id','job_id','job_union']
    df = df.merge(job_union, on=['person_id','job_id'], how='left')

    # Sample restrictions
    df = df[df['tenure'] >= 1].copy()
    df = df.dropna(subset=['log_hourly_wage']).copy()
    df = df[~df['occ_broad'].isin([0,3,9])].copy()
    df = df[(df['self_employed']==0)|(df['self_employed'].isna())].copy()
    df = df[(df['agriculture']==0)|(df['agriculture'].isna())].copy()
    df = df[(df['govt_worker']==0)|(df['govt_worker'].isna())].copy()

    df['initial_experience'] = df['experience'] - df['tenure']
    df.loc[df['initial_experience']<0,'initial_experience'] = 0

    # Fill controls
    for c in ['married','disabled','lives_in_smsa','region_ne','region_nc','region_south']:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    return df


def run_step1(data):
    """Run Step 1: within-job wage growth regression.
    Returns beta_hat (intercept = beta_1+beta_2), higher-order coefficients,
    and model object for variance-covariance matrix."""

    sub = data.sort_values(['person_id','job_id','year']).copy()

    sub['prev_lrw'] = sub.groupby(['person_id','job_id'])['log_real_wage'].shift(1)
    sub['d_lrw'] = sub['log_real_wage'] - sub['prev_lrw']
    sub['prev_t'] = sub.groupby(['person_id','job_id'])['tenure'].shift(1)
    sub['d_t'] = sub['tenure'] - sub['prev_t']

    wj = sub.dropna(subset=['d_lrw','prev_t'])
    wj = wj[wj['d_t']==1].copy()

    T = wj['tenure'].values; Tp = T-1
    X = wj['experience'].values; Xp = X-1

    wj['d_t2'] = (T**2-Tp**2)/100
    wj['d_t3'] = (T**3-Tp**3)/1000
    wj['d_t4'] = (T**4-Tp**4)/10000
    wj['d_x2'] = (X**2-Xp**2)/100
    wj['d_x3'] = (X**3-Xp**3)/1000
    wj['d_x4'] = (X**4-Xp**4)/10000

    sv = ['d_t2','d_t3','d_t4','d_x2','d_x3','d_x4']
    for y in range(1969,1984):
        c = f'year_{y}'; dc = f'd_{c}'
        if c in wj.columns:
            wj[dc] = wj[c].values - wj.groupby(['person_id','job_id'])[c].shift(1).values
            wj[dc] = wj[dc].fillna(0)
            if wj[dc].std() > 0:
                sv.append(dc)

    X1 = sm.add_constant(wj[sv])
    y1 = wj['d_lrw']
    v = X1.notna().all(axis=1) & y1.notna()
    model = sm.OLS(y1[v], X1[v]).fit()

    return {
        'beta_hat': model.params['const'],
        'beta_hat_se': model.bse['const'],
        'g2': model.params.get('d_t2', 0),
        'g3': model.params.get('d_t3', 0),
        'g4': model.params.get('d_t4', 0),
        'd2': model.params.get('d_x2', 0),
        'd3': model.params.get('d_x3', 0),
        'd4': model.params.get('d_x4', 0),
        'model': model,
        'n_wj': v.sum(),
        'wj_data': wj[v].copy()
    }


def run_step2(data, step1, name):
    """Run Step 2: OLS of adjusted wage on initial experience + controls.

    CORRECT PROCEDURE (equation 10):
    w* = log_real_wage - [higher-order terms only]
    Then: w* = X_0*beta_1 + F*gamma + e
    """

    g2 = step1['g2']; g3 = step1['g3']; g4 = step1['g4']
    d2 = step1['d2']; d3 = step1['d3']; d4 = step1['d4']

    # Construct w* - subtract ONLY higher-order terms
    data = data.copy()
    data['w_star'] = (data['log_real_wage']
                      - g2 * data['tenure']**2 / 100
                      - g3 * data['tenure']**3 / 1000
                      - g4 * data['tenure']**4 / 10000
                      - d2 * data['experience']**2 / 100
                      - d3 * data['experience']**3 / 1000
                      - d4 * data['experience']**4 / 10000)

    # Controls
    ctrls = ['education_years', 'married', 'disabled', 'lives_in_smsa',
             'region_ne', 'region_nc', 'region_south']

    if name == 'Professional/Service':
        data['union_ctrl'] = data['union_filled'].fillna(data['union_member'].fillna(0))
        if data['union_ctrl'].std() > 0:
            ctrls.append('union_ctrl')

    # Year dummies
    yr_use = [f'year_{y}' for y in range(1969,1984) if f'year_{y}' in data.columns and data[f'year_{y}'].std() > 0]

    step2 = data.dropna(subset=['w_star','experience','initial_experience']).copy()
    for c in ctrls:
        if c in step2.columns:
            step2[c] = step2[c].fillna(0)
    ctrls_use = [c for c in ctrls if c in step2.columns and step2[c].std() > 0]

    # All regressors: initial_experience + controls + year dummies
    all_vars = ['initial_experience'] + ctrls_use + yr_use

    # Check rank and trim if necessary
    X2 = sm.add_constant(step2[all_vars].astype(float))
    rank = np.linalg.matrix_rank(X2.values)
    while rank < X2.shape[1] and len(yr_use) > 0:
        yr_use = yr_use[:-1]
        all_vars = ['initial_experience'] + ctrls_use + yr_use
        X2 = sm.add_constant(step2[all_vars].astype(float))
        rank = np.linalg.matrix_rank(X2.values)

    y2 = step2['w_star'].astype(float)
    model2 = sm.OLS(y2, X2).fit()

    beta_1 = model2.params['initial_experience']
    beta_1_se_naive = model2.bse['initial_experience']

    return {
        'beta_1': beta_1,
        'beta_1_se_naive': beta_1_se_naive,
        'model': model2,
        'N': len(step2),
        'step2_data': step2,
        'all_vars': all_vars
    }


def murphy_topel_se(step1_result, step2_result, beta_hat, beta_1):
    """
    Compute Murphy-Topel (1985) corrected standard errors for the two-step estimator.

    The correction accounts for the fact that step 2 regressors (w*) depend on
    step 1 estimates (gamma, delta), introducing additional sampling variability.

    Corrected variance of beta_1:
    V_corrected = V2 + V2 * (C * V1 * C' - R * V1 * C' - C * V1 * R') * V2

    where:
    V1 = variance of step 1 parameters
    V2 = variance of step 2 parameters
    C = d(step2 score)/d(step1 params), evaluated at estimates
    R = cross-product correction

    For simplicity, we approximate by computing the Jacobian of w* w.r.t. step 1 params
    and propagating the step 1 variance through.
    """

    model1 = step1_result['model']
    model2 = step2_result['model']
    step2_data = step2_result['step2_data']

    # Step 1 variance-covariance for the relevant parameters
    # The step 1 parameters that affect w* are: const (beta_hat), g2, g3, g4, d2, d3, d4
    s1_params = ['const', 'd_t2', 'd_t3', 'd_t4', 'd_x2', 'd_x3', 'd_x4']
    s1_params_present = [p for p in s1_params if p in model1.params.index]
    V1 = model1.cov_params().loc[s1_params_present, s1_params_present].values

    # Step 2 variance
    idx_b1 = list(model2.params.index).index('initial_experience')
    V2_b1 = model2.cov_params().iloc[idx_b1, idx_b1]

    # The derivative of w* w.r.t. step 1 parameters:
    # d(w*)/d(const) = 0  (const = beta_hat, not used in w* directly)
    # Wait - w* does NOT subtract the linear term, so const is NOT in w*
    # d(w*)/d(g2) = -T^2/100
    # d(w*)/d(g3) = -T^3/1000
    # d(w*)/d(g4) = -T^4/10000
    # d(w*)/d(d2) = -X^2/100
    # d(w*)/d(d3) = -X^3/1000
    # d(w*)/d(d4) = -X^4/10000

    # But const IS involved because beta_2 = beta_hat - beta_1
    # Actually for computing the SE of beta_1, we only need the effect of step 1 params on w*
    # which then propagates to beta_1 through the second-step OLS

    T = step2_data['tenure'].values.astype(float)
    X = step2_data['experience'].values.astype(float)
    n2 = len(step2_data)

    # Jacobian matrix: J[i, j] = d(w*_i) / d(theta_j) for step 1 param theta_j
    # theta = [g2, g3, g4, d2, d3, d4]  (not const since it doesn't appear in w*)
    s1_gamma_params = ['d_t2', 'd_t3', 'd_t4', 'd_x2', 'd_x3', 'd_x4']
    s1_gamma_present = [p for p in s1_gamma_params if p in model1.params.index]

    J = np.zeros((n2, len(s1_gamma_present)))
    for j, param in enumerate(s1_gamma_present):
        if param == 'd_t2': J[:, j] = -T**2/100
        elif param == 'd_t3': J[:, j] = -T**3/1000
        elif param == 'd_t4': J[:, j] = -T**4/10000
        elif param == 'd_x2': J[:, j] = -X**2/100
        elif param == 'd_x3': J[:, j] = -X**3/1000
        elif param == 'd_x4': J[:, j] = -X**4/10000

    V1_gamma = model1.cov_params().loc[s1_gamma_present, s1_gamma_present].values

    # The effect on beta_1 of a change in step 1 params:
    # beta_1 = (X2'X2)^{-1} X2' w*
    # d(beta_1)/d(theta) = (X2'X2)^{-1} X2' J  (where J is the Jacobian of w* w.r.t. theta)

    X2_mat = model2.model.exog  # The X matrix from step 2
    XtX_inv = np.linalg.inv(X2_mat.T @ X2_mat)
    # Gradient: d(beta_1)/d(theta) for the initial_experience row
    G = XtX_inv @ (X2_mat.T @ J)  # shape: (k2 x p1)
    g_b1 = G[idx_b1, :]  # gradient for beta_1 only

    # Additional variance from step 1:
    V_extra = g_b1 @ V1_gamma @ g_b1
    V_corrected = V2_b1 + V_extra

    # Also need SE for beta_hat (from step 1) for computing beta_2 SE
    beta_hat_se = step1_result['beta_hat_se']
    beta_2 = beta_hat - beta_1
    # Var(beta_2) = Var(beta_hat) + Var(beta_1) - 2*Cov(beta_hat, beta_1)
    # The cross-covariance is approximated by the Murphy-Topel R matrix
    # For simplicity, assume independence: Var(beta_2) = Var(beta_hat) + Var(beta_1_corrected)
    beta_2_var = beta_hat_se**2 + V_corrected

    return {
        'beta_1_se': np.sqrt(max(0, V_corrected)),
        'beta_1_se_naive': np.sqrt(V2_b1),
        'V_extra': V_extra,
        'beta_2_se': np.sqrt(max(0, beta_2_var))
    }


def run_analysis(data_source='data/psid_panel.csv'):
    df = prepare_data(data_source)

    # Year dummies
    for y in range(1969, 1984):
        col = f'year_{y}'
        if col not in df.columns:
            df[col] = (df['year'] == y).astype(int)

    # Subsamples
    mask_ps = df['occ_broad'].isin([1,2,4,8])
    mask_bc = df['occ_broad'].isin([5,6,7])
    mask_bc_nu = mask_bc & (df['job_union'] == 0)
    mask_bc_u = mask_bc & (df['job_union'] == 1)

    subsamples = {
        'Professional/Service': mask_ps,
        'BC_Nonunion': mask_bc_nu,
        'BC_Union': mask_bc_u
    }

    print("=" * 80)
    print("TABLE 5: Returns to Job Seniority by Occupation and Union Status")
    print("=" * 80)

    results = {}

    for name, mask in subsamples.items():
        sub = df[mask].copy()
        print(f"\n{'='*60}")
        print(f"Subsample: {name}, N={len(sub)}")
        print(f"{'='*60}")

        # Step 1
        s1 = run_step1(sub)
        print(f"  Step 1: beta_hat={s1['beta_hat']:.4f} ({s1['beta_hat_se']:.4f})")
        print(f"    g2={s1['g2']:.4f}, g3={s1['g3']:.4f}, g4={s1['g4']:.4f}")
        print(f"    N_wj={s1['n_wj']}")

        # Step 2
        s2 = run_step2(sub, s1, name)
        print(f"  Step 2: beta_1_naive={s2['beta_1']:.4f} ({s2['beta_1_se_naive']:.4f})")
        print(f"    N={s2['N']}")

        # Murphy-Topel correction
        mt = murphy_topel_se(s1, s2, s1['beta_hat'], s2['beta_1'])
        print(f"  Murphy-Topel corrected SE: {mt['beta_1_se']:.4f}")
        print(f"    (naive SE: {mt['beta_1_se_naive']:.4f}, extra variance: {mt['V_extra']:.6f})")

        beta_1 = s2['beta_1']
        beta_1_se = mt['beta_1_se']
        beta_2 = s1['beta_hat'] - beta_1
        beta_2_se = mt['beta_2_se']

        print(f"  beta_1={beta_1:.4f} ({beta_1_se:.4f})")
        print(f"  beta_2={beta_2:.4f} ({beta_2_se:.4f})")
        print(f"  beta_1+beta_2={s1['beta_hat']:.4f} ({s1['beta_hat_se']:.4f})")

        result = {
            'beta_1': beta_1,
            'beta_1_se': beta_1_se,
            'beta_2': beta_2,
            'beta_2_se': beta_2_se,
            'beta_1_plus_beta_2': s1['beta_hat'],
            'beta_1_plus_beta_2_se': s1['beta_hat_se'],
            'N': s2['N'],
            'step1': s1,
            'step2': s2,
        }

        # Cumulative returns
        model1 = s1['model']
        gamma_vars = ['d_t2','d_t3','d_t4']
        gamma_present = [g for g in gamma_vars if g in model1.params.index]

        for T_val in [5, 10, 15, 20]:
            cum = (beta_2 * T_val
                   + s1['g2'] * T_val**2/100
                   + s1['g3'] * T_val**3/1000
                   + s1['g4'] * T_val**4/10000)

            # SE via delta method
            # Var(cum) = T^2*Var(beta_2) + (T^2/100)^2*Var(g2) + ... + cross terms
            gamma_vcov = model1.cov_params().loc[gamma_present, gamma_present].values
            grad_gamma = np.array([T_val**2/100, T_val**3/1000, T_val**4/10000])[:len(gamma_present)]
            var_cum = T_val**2 * beta_2_se**2
            if len(gamma_present) > 0:
                var_cum += grad_gamma @ gamma_vcov @ grad_gamma
            cum_se = np.sqrt(max(0, var_cum))

            result[f'cum_return_{T_val}'] = cum
            result[f'cum_return_{T_val}_se'] = cum_se
            print(f"  Cum return {T_val}yr: {cum:.4f} ({cum_se:.4f})")

        results[name] = result

    # Print formatted results
    print_results(results)

    # Score
    score = score_against_ground_truth(results)
    return results


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
            line2 += f" {'('+f'{s:.4f}'+')':>15s}"
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

        # Column 4: union relative to nonunion
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
        earned_points += pts; total_points += 5
        print(f"  {name}: gen={gen:.4f} true={true_val:.4f} rel_err={rel_err:.3f} -> {pts}/5")

    # 2. Step 2 coefs (20 pts)
    print("\n--- Step 2 coefs: 20 pts ---")
    cp = 20 / 6
    for name in ['Professional/Service', 'BC_Nonunion', 'BC_Union']:
        for param in ['beta_1', 'beta_2']:
            gen = results[name].get(param, np.nan)
            true_val = gt[name][param]
            abs_err = abs(gen - true_val) if not np.isnan(gen) else 1
            pts = cp if abs_err <= 0.01 else (cp*0.5 if abs_err <= 0.03 else 0)
            earned_points += pts; total_points += cp
            print(f"  {name} {param}: gen={gen:.4f} true={true_val:.4f} abs_err={abs_err:.4f} -> {pts:.2f}/{cp:.2f}")

    # 3. Cumulative returns (20 pts)
    print("\n--- Cumulative returns: 20 pts ---")
    cp2 = 20 / 12
    for name in ['Professional/Service', 'BC_Nonunion', 'BC_Union']:
        for T in [5, 10, 15, 20]:
            gen = results[name].get(f'cum_return_{T}', np.nan)
            true_val = gt[name][f'cum_return_{T}']
            abs_err = abs(gen - true_val) if not np.isnan(gen) else 1
            pts = cp2 if abs_err <= 0.03 else (cp2*0.5 if abs_err <= 0.06 else 0)
            earned_points += pts; total_points += cp2
            print(f"  {name} {T}yr: gen={gen:.4f} true={true_val:.4f} abs_err={abs_err:.4f} -> {pts:.2f}/{cp2:.2f}")

    # 4. SEs (10 pts)
    print("\n--- Standard errors: 10 pts ---")
    se_items = []
    for name in ['Professional/Service', 'BC_Nonunion', 'BC_Union']:
        for param in ['beta_1_se', 'beta_2_se', 'beta_1_plus_beta_2_se']:
            se_items.append((name, param))
    sp = 10 / len(se_items)
    for name, param in se_items:
        gen = results[name].get(param, np.nan)
        true_val = gt[name][param]
        rel_err = abs(gen - true_val) / max(abs(true_val), 0.001) if not np.isnan(gen) else 1
        pts = sp if rel_err <= 0.30 else (sp*0.5 if rel_err <= 0.60 else 0)
        earned_points += pts; total_points += sp
        print(f"  {name} {param}: gen={gen:.4f} true={true_val:.4f} rel_err={rel_err:.3f} -> {pts:.2f}/{sp:.2f}")

    # 5. Sample sizes (15 pts)
    print("\n--- Sample sizes: 15 pts ---")
    for name in ['Professional/Service', 'BC_Nonunion', 'BC_Union']:
        gen = results[name].get('N', 0)
        true_val = gt[name]['N']
        rel_err = abs(gen - true_val) / true_val
        pts = 5 if rel_err <= 0.05 else (3 if rel_err <= 0.10 else (1 if rel_err <= 0.20 else 0))
        earned_points += pts; total_points += 5
        print(f"  {name}: gen={gen} true={true_val} rel_err={rel_err:.3f} -> {pts}/5")

    # 6. Significance (10 pts)
    print("\n--- Significance: 10 pts ---")
    sign_ok = 0; sign_tot = 0
    for name in ['Professional/Service', 'BC_Nonunion', 'BC_Union']:
        for param in ['beta_1', 'beta_2', 'beta_1_plus_beta_2']:
            gen = results[name].get(param, np.nan)
            true_val = gt[name][param]
            if not np.isnan(gen) and np.sign(gen) == np.sign(true_val):
                sign_ok += 1
            sign_tot += 1
    sig_pts = 10 * sign_ok / max(sign_tot, 1)
    earned_points += sig_pts; total_points += 10
    print(f"  Signs: {sign_ok}/{sign_tot} -> {sig_pts:.1f}/10")

    # 7. Columns (10 pts)
    all_present = all(
        all(k in results[n] for k in ['beta_1','beta_2','beta_1_plus_beta_2',
                                        'cum_return_5','cum_return_10','cum_return_15','cum_return_20'])
        for n in ['Professional/Service','BC_Nonunion','BC_Union']
    )
    col_pts = 10 if all_present else 5
    earned_points += col_pts; total_points += 10

    score = round(earned_points / total_points * 100) if total_points > 0 else 0
    print(f"\nTOTAL SCORE: {score}/100 ({earned_points:.1f}/{total_points:.1f})")
    return score


if __name__ == '__main__':
    results = run_analysis('data/psid_panel.csv')
