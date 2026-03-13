"""
Table 5 Replication: Estimated Returns to Job Seniority by Occupational Category
and Union Status, Two-Step Estimator from Topel (1991)

Attempt 8: Use per-subsample step 1 (both intercept AND polynomial terms)
as the paper explicitly states, with OLS of w* on X_0 in step 2.

KEY INSIGHTS FROM DEBUGGING:
- OLS of w* on X_0 gives beta_1 (correct approach, not 2SLS which fails)
- The paper says each occupation group has its own step 1 polynomial
- Different polynomial terms produce different w* -> different beta_1
- Using paper's full-sample polynomials gives beta_1 ~ 0.08 for ALL groups
- Using per-subsample polynomials gives different beta_1 per group (desired)
- Subsample polynomial terms are noisy but necessary for differentiation

APPROACH:
1. Run Step 1 completely per subsample (intercept + all polynomial terms)
2. Construct w* = log_real_wage - b1b2*T - g2*T^2 - g3*T^3 - g4*T^4 - d2*X^2 - d3*X^3 - d4*X^4
3. Step 2: OLS of w* on X_0 (initial experience) + controls + year dummies
4. beta_1 = coefficient on X_0, beta_2 = b1b2 - beta_1
5. Cumulative returns: beta_2*T + g2*T^2 + g3*T^3 + g4*T^4
6. Murphy-Topel SE correction
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


def prepare_data(data_source):
    df = pd.read_csv(data_source)

    df['education_years'] = df['education_clean'].map(EDUC_MAP).fillna(12)
    df['experience'] = (df['age'] - df['education_years'] - 6).clip(lower=0)
    df['tenure'] = df['tenure_topel'].copy()

    df['cps_index'] = df['year'].map(CPS_WAGE_INDEX)
    df['gnp_defl'] = (df['year'] - 1).map(GNP_DEFLATOR)
    df['log_wage_cps'] = df['log_hourly_wage'] - np.log(df['cps_index'])
    df['log_real_wage'] = (df['log_hourly_wage']
                           - np.log(df['gnp_defl'] / 100.0)
                           - np.log(df['cps_index']))

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

    df = df.sort_values(['person_id', 'job_id', 'year'])
    ju = df.groupby(['person_id', 'job_id'])['union_member'].agg(
        lambda x: (x.mean() > 0.5) if x.notna().any() else np.nan
    ).reset_index().rename(columns={'union_member': 'job_union'})
    df = df.merge(ju, on=['person_id', 'job_id'], how='left')

    df = df[df['tenure'] >= 1].copy()
    df = df.dropna(subset=['log_hourly_wage', 'log_real_wage']).copy()
    df = df[~df['occ_broad'].isin([0, 3, 9])].copy()
    df = df[(df['self_employed'] == 0) | (df['self_employed'].isna())].copy()

    df['initial_experience'] = (df['experience'] - df['tenure']).clip(lower=0)

    for c in ['married', 'disabled', 'lives_in_smsa', 'union_member',
              'region_ne', 'region_nc', 'region_south']:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    for y in range(1969, 1984):
        col = f'year_{y}'
        if col not in df.columns:
            df[col] = (df['year'] == y).astype(int)

    return df


def run_step1(data):
    """Run Step 1: within-job FD regression on CPS-adjusted wages."""
    sub = data.sort_values(['person_id', 'job_id', 'year']).copy()

    sub['within'] = ((sub['person_id'] == sub['person_id'].shift(1)) &
                     (sub['job_id'] == sub['job_id'].shift(1)) &
                     (sub['year'] - sub['year'].shift(1) == 1))

    sub['d_lw'] = sub['log_wage_cps'] - sub['log_wage_cps'].shift(1)

    for k in [2, 3, 4]:
        sub[f'd_t{k}'] = sub['tenure']**k - (sub['tenure'] - 1)**k
        sub[f'd_x{k}'] = sub['experience']**k - (sub['experience'] - 1)**k

    yr_cols = sorted([c for c in sub.columns if c.startswith('year_') and c != 'year'])
    d_yr = []
    for yc in yr_cols:
        dc = f'd_{yc}'
        sub[dc] = sub[yc].astype(float) - sub[yc].shift(1).astype(float)
        sub[dc] = sub[dc].fillna(0)
        d_yr.append(dc)

    fd = sub[sub['within']].dropna(subset=['d_lw']).copy()

    d_yr_use = [c for c in d_yr if fd[c].std() > 1e-10]
    if len(d_yr_use) > 1:
        d_yr_use = d_yr_use[1:]

    x_cols = ['d_t2', 'd_t3', 'd_t4', 'd_x2', 'd_x3', 'd_x4'] + d_yr_use

    y = fd['d_lw']
    X = sm.add_constant(fd[x_cols])
    valid = y.notna() & X.notna().all(axis=1)
    model = sm.OLS(y[valid], X[valid]).fit()

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
        'N_wj': int(valid.sum()),
    }


def run_step2(data, s1, name):
    """Run Step 2: OLS of w* on X_0."""
    b1b2 = s1['beta_hat']
    g2, g3, g4 = s1['g2'], s1['g3'], s1['g4']
    d2, d3, d4 = s1['d2'], s1['d3'], s1['d4']

    sub = data.copy()
    sub['w_star'] = (sub['log_real_wage']
                     - b1b2 * sub['tenure']
                     - g2 * sub['tenure']**2
                     - g3 * sub['tenure']**3
                     - g4 * sub['tenure']**4
                     - d2 * sub['experience']**2
                     - d3 * sub['experience']**3
                     - d4 * sub['experience']**4)

    ctrls = ['education_years', 'married', 'disabled', 'lives_in_smsa',
             'region_ne', 'region_nc', 'region_south']
    if name == 'Professional/Service':
        sub['union_ctrl'] = sub['union_member'].fillna(0)
        if sub['union_ctrl'].std() > 0:
            ctrls.append('union_ctrl')

    yr_use = [f'year_{y}' for y in range(1969, 1984)
              if f'year_{y}' in sub.columns and sub[f'year_{y}'].std() > 0]

    step2 = sub.dropna(subset=['w_star', 'initial_experience']).copy()
    ctrls_use = [c for c in ctrls if c in step2.columns and step2[c].std() > 0]
    all_vars = ['initial_experience'] + ctrls_use + yr_use

    X = sm.add_constant(step2[all_vars].astype(float))
    rank = np.linalg.matrix_rank(X.values)
    while rank < X.shape[1] and len(yr_use) > 0:
        yr_use = yr_use[:-1]
        all_vars = ['initial_experience'] + ctrls_use + yr_use
        X = sm.add_constant(step2[all_vars].astype(float))
        rank = np.linalg.matrix_rank(X.values)

    y = step2['w_star'].astype(float)
    model = sm.OLS(y, X).fit()

    return {
        'beta_1': model.params['initial_experience'],
        'beta_1_se_naive': model.bse['initial_experience'],
        'N': len(step2),
        'model': model,
        'step2_data': step2,
    }


def murphy_topel_se(s1_model, s2_result):
    """Murphy-Topel corrected SE."""
    model2 = s2_result['model']
    data2 = s2_result['step2_data']

    s1_params = ['const', 'd_t2', 'd_t3', 'd_t4', 'd_x2', 'd_x3', 'd_x4']
    s1_present = [p for p in s1_params if p in s1_model.params.index]
    V1 = s1_model.cov_params().loc[s1_present, s1_present].values

    idx = list(model2.params.index).index('initial_experience')
    V2 = model2.cov_params().iloc[idx, idx]

    T = data2['tenure'].values.astype(float)
    X = data2['experience'].values.astype(float)
    n = len(T)

    J = np.zeros((n, len(s1_present)))
    for j, p in enumerate(s1_present):
        if p == 'const':    J[:, j] = -T
        elif p == 'd_t2':   J[:, j] = -T**2
        elif p == 'd_t3':   J[:, j] = -T**3
        elif p == 'd_t4':   J[:, j] = -T**4
        elif p == 'd_x2':   J[:, j] = -X**2
        elif p == 'd_x3':   J[:, j] = -X**3
        elif p == 'd_x4':   J[:, j] = -X**4

    X2 = model2.model.exog
    XtX_inv = np.linalg.inv(X2.T @ X2)
    G = XtX_inv @ (X2.T @ J)
    g = G[idx, :]

    V_extra = g @ V1 @ g
    return np.sqrt(max(0, V2 + V_extra)), V_extra


def run_analysis(data_source='data/psid_panel.csv'):
    df = prepare_data(data_source)
    print(f"Total observations: {len(df)}")

    mask_ps = df['occ_broad'].isin([1, 2, 4, 8])
    mask_bc_nu = df['occ_broad'].isin([5, 6, 7]) & (df['job_union'] == 0)
    mask_bc_u = df['occ_broad'].isin([5, 6, 7]) & (df['job_union'] == 1)

    subsamples = {
        'Professional/Service': mask_ps,
        'BC_Nonunion': mask_bc_nu,
        'BC_Union': mask_bc_u,
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
        print(f"{name}, N={len(sub)}")
        print(f"{'='*60}")

        s1 = run_step1(sub)
        print(f"  Step 1: b1+b2={s1['beta_hat']:.4f} ({s1['beta_hat_se']:.4f})")
        print(f"    g2={s1['g2']:.6f}, g3={s1['g3']:.8f}, g4={s1['g4']:.10f}")
        print(f"    d2={s1['d2']:.6f}, d3={s1['d3']:.8f}, d4={s1['d4']:.10f}")
        print(f"    N_wj={s1['N_wj']}")

        s2 = run_step2(sub, s1, name)
        print(f"  Step 2: beta_1={s2['beta_1']:.4f} (naive SE={s2['beta_1_se_naive']:.4f})")
        print(f"    N={s2['N']}")

        mt_se, v_extra = murphy_topel_se(s1['model'], s2)
        print(f"  Murphy-Topel SE: {mt_se:.4f}")

        beta_1 = s2['beta_1']
        beta_1_se = mt_se
        b1b2 = s1['beta_hat']
        b1b2_se = s1['beta_hat_se']
        beta_2 = b1b2 - beta_1
        beta_2_se = np.sqrt(b1b2_se**2 + beta_1_se**2)

        result = {
            'beta_1': beta_1, 'beta_1_se': beta_1_se,
            'beta_2': beta_2, 'beta_2_se': beta_2_se,
            'beta_1_plus_beta_2': b1b2, 'beta_1_plus_beta_2_se': b1b2_se,
            'N': s2['N'],
        }

        g2, g3, g4 = s1['g2'], s1['g3'], s1['g4']
        gamma_vars = ['d_t2', 'd_t3', 'd_t4']
        gamma_present = [g for g in gamma_vars if g in s1['model'].params.index]
        gamma_vcov = s1['model'].cov_params().loc[gamma_present, gamma_present].values

        for T_val in [5, 10, 15, 20]:
            cum = beta_2 * T_val + g2 * T_val**2 + g3 * T_val**3 + g4 * T_val**4

            grad = []
            for gp in gamma_present:
                if gp == 'd_t2': grad.append(T_val**2)
                elif gp == 'd_t3': grad.append(T_val**3)
                elif gp == 'd_t4': grad.append(T_val**4)
            grad = np.array(grad)

            var_cum = T_val**2 * beta_2_se**2
            if len(grad) > 0:
                var_cum += grad @ gamma_vcov @ grad
            cum_se = np.sqrt(max(0, var_cum))

            result[f'cum_return_{T_val}'] = cum
            result[f'cum_return_{T_val}_se'] = cum_se

        results[name] = result

        print(f"  beta_1={beta_1:.4f} ({beta_1_se:.4f}), beta_2={beta_2:.4f} ({beta_2_se:.4f})")
        print(f"  b1+b2={b1b2:.4f} ({b1b2_se:.4f})")
        for T_val in [5, 10, 15, 20]:
            print(f"  cum{T_val}={result[f'cum_return_{T_val}']:.4f} ({result[f'cum_return_{T_val}_se']:.4f})")

    print_table(results)
    score = score_against_ground_truth(results)
    return results


def print_table(results):
    print("\n" + "=" * 80)
    print("TABLE 5 OUTPUT")
    print("=" * 80)
    names = ['Professional/Service', 'BC_Nonunion', 'BC_Union']
    print("\nTOP PANEL: Main Effects")
    print(f"{'':30s} {'(1) Prof/Svc':>15s} {'(2) BC Nonunion':>15s} {'(3) BC Union':>15s}")
    print("-" * 75)
    for param, label in [('beta_1', 'Experience (beta_1)'),
                          ('beta_2', 'Tenure (beta_2)'),
                          ('beta_1_plus_beta_2', 'beta_1 + beta_2')]:
        vals = [results[n][param] for n in names]
        ses = [results[n][f'{param}_se'] for n in names]
        print(f"{label:30s} {vals[0]:15.4f} {vals[1]:15.4f} {vals[2]:15.4f}")
        print(f"{'':30s} {'('+f'{ses[0]:.4f}'+')':>15s} {'('+f'{ses[1]:.4f}'+')':>15s} {'('+f'{ses[2]:.4f}'+')':>15s}")

    print(f"\nBOTTOM PANEL: Cumulative Returns")
    print(f"{'':15s} {'(1) Prof/Svc':>12s} {'(2) Nonunion':>12s} {'(3) Union':>12s} {'(4) U vs NU':>12s}")
    print("-" * 63)
    for T_val in [5, 10, 15, 20]:
        key = f'cum_return_{T_val}'
        key_se = f'cum_return_{T_val}_se'
        vals = [results[n][key] for n in names]
        ses = [results[n][key_se] for n in names]
        b1_nu = results['BC_Nonunion']['beta_1']
        b1_u = results['BC_Union']['beta_1']
        u_rel = vals[2] + (b1_nu - b1_u) * T_val
        u_rel_se = np.sqrt(ses[2]**2
                           + (T_val * results['BC_Nonunion']['beta_1_se'])**2
                           + (T_val * results['BC_Union']['beta_1_se'])**2)
        print(f"{T_val:2d} years:       {vals[0]:12.4f} {vals[1]:12.4f} {vals[2]:12.4f} {u_rel:12.4f}")
        print(f"{'':15s} {'('+f'{ses[0]:.4f}'+')':>12s} {'('+f'{ses[1]:.4f}'+')':>12s} {'('+f'{ses[2]:.4f}'+')':>12s} {'('+f'{u_rel_se:.4f}'+')':>12s}")
    print(f"\nSample sizes:")
    for n, label in zip(names, ['Professional/Service', 'Nonunion', 'Union']):
        print(f"  {label}: N = {results[n]['N']:,}")


def score_against_ground_truth(results):
    gt = GROUND_TRUTH
    total_points = 0
    earned_points = 0
    print("\n" + "=" * 80)
    print("SCORING")
    print("=" * 80)

    # 1. Step 1 (15 pts)
    for name in ['Professional/Service', 'BC_Nonunion', 'BC_Union']:
        gen = results[name]['beta_1_plus_beta_2']
        true = gt[name]['beta_1_plus_beta_2']
        re = abs(gen - true) / max(abs(true), 0.001)
        pts = 5 if re <= 0.20 else (3 if re <= 0.40 else 0)
        earned_points += pts; total_points += 5
        print(f"  b1b2 {name}: {gen:.4f} vs {true:.4f} re={re:.3f} -> {pts}/5")

    # 2. Step 2 (20 pts)
    cp = 20 / 6
    for name in ['Professional/Service', 'BC_Nonunion', 'BC_Union']:
        for p in ['beta_1', 'beta_2']:
            gen = results[name][p]
            true = gt[name][p]
            ae = abs(gen - true)
            pts = cp if ae <= 0.01 else (cp * 0.5 if ae <= 0.03 else 0)
            earned_points += pts; total_points += cp
            print(f"  {p} {name}: {gen:.4f} vs {true:.4f} ae={ae:.4f} -> {pts:.1f}/{cp:.1f}")

    # 3. Cum returns (20 pts)
    cp2 = 20 / 12
    for name in ['Professional/Service', 'BC_Nonunion', 'BC_Union']:
        for T in [5, 10, 15, 20]:
            k = f'cum_return_{T}'
            gen = results[name][k]
            true = gt[name][k]
            ae = abs(gen - true)
            pts = cp2 if ae <= 0.03 else (cp2 * 0.5 if ae <= 0.06 else 0)
            earned_points += pts; total_points += cp2
            print(f"  cum{T} {name}: {gen:.4f} vs {true:.4f} ae={ae:.4f} -> {pts:.1f}/{cp2:.1f}")

    # 4. SEs (10 pts)
    items = []
    for name in ['Professional/Service', 'BC_Nonunion', 'BC_Union']:
        for p in ['beta_1_se', 'beta_2_se', 'beta_1_plus_beta_2_se']:
            items.append((name, p))
    sp = 10 / len(items)
    for name, p in items:
        gen = results[name][p]
        true = gt[name][p]
        re = abs(gen - true) / max(abs(true), 0.001)
        pts = sp if re <= 0.30 else (sp * 0.5 if re <= 0.60 else 0)
        earned_points += pts; total_points += sp
        print(f"  {p} {name}: {gen:.4f} vs {true:.4f} re={re:.3f} -> {pts:.2f}/{sp:.2f}")

    # 5. N (15 pts)
    for name in ['Professional/Service', 'BC_Nonunion', 'BC_Union']:
        gen = results[name]['N']
        true = gt[name]['N']
        re = abs(gen - true) / true
        pts = 5 if re <= 0.05 else (3 if re <= 0.10 else (1 if re <= 0.20 else 0))
        earned_points += pts; total_points += 5
        print(f"  N {name}: {gen} vs {true} re={re:.3f} -> {pts}/5")

    # 6. Signs (10 pts)
    sok = 0; stot = 0
    for name in ['Professional/Service', 'BC_Nonunion', 'BC_Union']:
        for p in ['beta_1', 'beta_2', 'beta_1_plus_beta_2']:
            if np.sign(results[name][p]) == np.sign(gt[name][p]):
                sok += 1
            stot += 1
    sig = 10 * sok / max(stot, 1)
    earned_points += sig; total_points += 10
    print(f"  Signs: {sok}/{stot} -> {sig:.1f}/10")

    # 7. Columns (10 pts)
    ap = all(all(k in results[n] for k in ['beta_1','beta_2','beta_1_plus_beta_2',
                    'cum_return_5','cum_return_10','cum_return_15','cum_return_20'])
             for n in ['Professional/Service','BC_Nonunion','BC_Union'])
    cp3 = 10 if ap else 5
    earned_points += cp3; total_points += 10

    score = round(earned_points / total_points * 100) if total_points > 0 else 0
    print(f"\nTOTAL: {score}/100 ({earned_points:.1f}/{total_points:.1f})")
    return score


if __name__ == '__main__':
    results = run_analysis('data/psid_panel.csv')
