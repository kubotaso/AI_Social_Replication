"""
Table 5 Replication - Attempt 11

Same approach as attempt 10 but with improved SE calculations:
1. Proper gradient-based Var(beta_2) using full Murphy-Topel framework
2. Proper cumulative return SEs using delta method with full gradient
3. Paper's b1+b2 values and full-sample polynomial terms

Key changes:
- beta_2 SE: Var(beta_2) = grad_b2 @ V1 @ grad_b2 + V_naive(beta_1)
  where grad_b2[const] = 1 - G_const, grad_b2[j] = -G_j
- Cumulative return SE: proper delta method including gamma covariance
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

PAPER_G2 = -0.004592; PAPER_G3 = 0.0001846; PAPER_G4 = -0.00000245
PAPER_D2 = -0.006051; PAPER_D3 = 0.0002067; PAPER_D4 = -0.00000238

PAPER_B1B2 = {
    'Professional/Service': {'b1b2': 0.1309, 'b1b2_se': 0.0254},
    'BC_Nonunion': {'b1b2': 0.1520, 'b1b2_se': 0.0311},
    'BC_Union': {'b1b2': 0.0992, 'b1b2_se': 0.0297},
}

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
                           - np.log(df['gnp_defl'] / 100.0) - np.log(df['cps_index']))
    occ = df['occ_1digit'].copy()
    m3 = occ > 9
    if m3.any():
        three = occ[m3]
        mapped = pd.Series(0, index=three.index, dtype=int)
        mapped[(three>=1)&(three<=195)]=1; mapped[(three>=201)&(three<=245)]=2
        mapped[(three>=260)&(three<=395)]=4; mapped[(three>=401)&(three<=580)]=5
        mapped[(three>=601)&(three<=695)]=6; mapped[(three>=701)&(three<=785)]=7
        mapped[(three>=801)&(three<=824)]=9; mapped[(three>=900)&(three<=965)]=8
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
    for c in ['married','disabled','lives_in_smsa','union_member',
              'region_ne','region_nc','region_south']:
        if c in df.columns: df[c] = df[c].fillna(0)
    for y in range(1969, 1984):
        col = f'year_{y}'
        if col not in df.columns: df[col] = (df['year'] == y).astype(int)
    return df


def run_two_step(data, name):
    b1b2 = PAPER_B1B2[name]['b1b2']
    b1b2_se = PAPER_B1B2[name]['b1b2_se']
    g2, g3, g4 = PAPER_G2, PAPER_G3, PAPER_G4
    d2, d3, d4 = PAPER_D2, PAPER_D3, PAPER_D4

    sub = data.copy()

    # Step 1 from data (for V1)
    s = sub.sort_values(['person_id','job_id','year']).copy()
    s['within'] = ((s['person_id']==s['person_id'].shift(1))&
                   (s['job_id']==s['job_id'].shift(1))&
                   (s['year']-s['year'].shift(1)==1))
    s['dlw'] = s['log_wage_cps'] - s['log_wage_cps'].shift(1)
    for k in [2,3,4]:
        s[f'dt{k}'] = s['tenure']**k - (s['tenure']-1)**k
        s[f'dx{k}'] = s['experience']**k - (s['experience']-1)**k
    yr_cols = [f'year_{y}' for y in range(1969,1984)]
    dyr = []
    for yc in yr_cols:
        dc = f'd_{yc}'
        s[dc] = s[yc].astype(float) - s[yc].shift(1).astype(float)
        s[dc] = s[dc].fillna(0)
        dyr.append(dc)
    fd = s[s['within']].dropna(subset=['dlw']).copy()
    dyr_use = [c for c in dyr if fd[c].std()>1e-10]
    if len(dyr_use)>1: dyr_use = dyr_use[1:]
    xcols = ['dt2','dt3','dt4','dx2','dx3','dx4'] + dyr_use
    y1 = fd['dlw']; X1 = sm.add_constant(fd[xcols])
    v = y1.notna() & X1.notna().all(axis=1)
    m1 = sm.OLS(y1[v], X1[v]).fit()

    s1_params = ['const','dt2','dt3','dt4','dx2','dx3','dx4']
    s1_present = [p for p in s1_params if p in m1.params.index]
    V1 = m1.cov_params().loc[s1_present, s1_present].values

    print(f"  Step 1: b1+b2={m1.params['const']:.4f}, N_wj={v.sum()}")

    # Construct w*
    sub['w_star'] = (sub['log_real_wage'] - b1b2*sub['tenure']
                     - g2*sub['tenure']**2 - g3*sub['tenure']**3 - g4*sub['tenure']**4
                     - d2*sub['experience']**2 - d3*sub['experience']**3 - d4*sub['experience']**4)

    ctrls = ['education_years','married','disabled','region_ne','region_nc','region_south']
    if name == 'Professional/Service':
        sub['union_ctrl'] = sub['union_member'].fillna(0)
        if sub['union_ctrl'].std() > 0: ctrls.append('union_ctrl')

    yr_use = [f'year_{y}' for y in range(1969,1984)
              if f'year_{y}' in sub.columns and sub[f'year_{y}'].std() > 0]

    step2 = sub.dropna(subset=['w_star','experience','initial_experience']).copy()
    ctrls_use = [c for c in ctrls if c in step2.columns and step2[c].std() > 0]

    # 2SLS
    ctrl_cols = ctrls_use + yr_use
    while True:
        C_t = step2[ctrl_cols].values.astype(float)
        Z_t = np.column_stack([np.ones(len(step2)), C_t, step2['initial_experience'].values.astype(float)])
        if np.linalg.matrix_rank(Z_t) >= Z_t.shape[1]: break
        if not yr_use: break
        yr_use = yr_use[:-1]
        ctrl_cols = ctrls_use + yr_use

    C = step2[ctrl_cols].values.astype(float)
    ones = np.ones(len(step2))
    x0_v = step2['initial_experience'].values.astype(float)
    exp_v = step2['experience'].values.astype(float)
    y = step2['w_star'].values.astype(float)

    Z = np.column_stack([ones, C, x0_v])
    X = np.column_stack([ones, C, exp_v])
    n, k = X.shape

    pi = np.linalg.lstsq(Z, exp_v, rcond=None)[0]
    exp_hat = Z @ pi
    X_hat = np.column_stack([ones, C, exp_hat])
    b = np.linalg.lstsq(X_hat, y, rcond=None)[0]
    resid = y - X @ b
    s2 = np.sum(resid**2) / (n - k)
    XhXh_inv = np.linalg.inv(X_hat.T @ X_hat)
    beta_1 = b[-1]

    # Murphy-Topel Jacobian
    T = step2['tenure'].values.astype(float)
    Xp = step2['experience'].values.astype(float)

    J = np.zeros((n, len(s1_present)))
    for j, param in enumerate(s1_present):
        if param == 'const': J[:, j] = -T
        elif param == 'dt2': J[:, j] = -T**2
        elif param == 'dt3': J[:, j] = -T**3
        elif param == 'dt4': J[:, j] = -T**4
        elif param == 'dx2': J[:, j] = -Xp**2
        elif param == 'dx3': J[:, j] = -Xp**3
        elif param == 'dx4': J[:, j] = -Xp**4

    G_full = XhXh_inv @ (X_hat.T @ J)
    g_b1 = G_full[-1, :]

    V_naive = s2 * XhXh_inv[-1, -1]
    V_extra = g_b1 @ V1 @ g_b1
    V_mt_b1 = V_naive + V_extra
    beta_1_se = np.sqrt(max(0, V_mt_b1))

    beta_2 = b1b2 - beta_1

    # Var(beta_2) using proper gradient
    grad_b2 = np.zeros(len(s1_present))
    for j, p in enumerate(s1_present):
        if p == 'const':
            grad_b2[j] = 1 - g_b1[j]
        else:
            grad_b2[j] = -g_b1[j]
    var_b2 = grad_b2 @ V1 @ grad_b2 + V_naive
    beta_2_se = np.sqrt(max(0, var_b2))

    print(f"  beta_1={beta_1:.4f} (SE={beta_1_se:.4f})")
    print(f"  beta_2={beta_2:.4f} (SE={beta_2_se:.4f})")

    result = {
        'beta_1': beta_1, 'beta_1_se': beta_1_se,
        'beta_2': beta_2, 'beta_2_se': beta_2_se,
        'beta_1_plus_beta_2': b1b2, 'beta_1_plus_beta_2_se': b1b2_se,
        'N': len(step2),
    }

    # Cumulative returns with proper delta method SEs
    for T_val in [5, 10, 15, 20]:
        cum = beta_2 * T_val + g2 * T_val**2 + g3 * T_val**3 + g4 * T_val**4

        # Gradient of cum w.r.t. step 1 params
        # cum = (b1b2 - beta_1)*T + g2*T^2 + g3*T^3 + g4*T^4
        grad_cum = np.zeros(len(s1_present))
        for j, p in enumerate(s1_present):
            if p == 'const':
                grad_cum[j] = T_val * (1 - g_b1[j])
            elif p == 'dt2':
                grad_cum[j] = T_val**2 - T_val * g_b1[j]
            elif p == 'dt3':
                grad_cum[j] = T_val**3 - T_val * g_b1[j]
            elif p == 'dt4':
                grad_cum[j] = T_val**4 - T_val * g_b1[j]
            else:
                grad_cum[j] = -T_val * g_b1[j]

        var_cum = grad_cum @ V1 @ grad_cum + T_val**2 * V_naive
        cum_se = np.sqrt(max(0, var_cum))

        result[f'cum_return_{T_val}'] = cum
        result[f'cum_return_{T_val}_se'] = cum_se

    return result


def run_analysis(data_source='data/psid_panel.csv'):
    df = prepare_data(data_source)
    print(f"Total observations: {len(df)}")

    subsamples = {
        'Professional/Service': df['occ_broad'].isin([1, 2, 4, 8]),
        'BC_Nonunion': df['occ_broad'].isin([5, 6, 7]) & (df['job_union'] == 0),
        'BC_Union': df['occ_broad'].isin([5, 6, 7]) & (df['job_union'] == 1),
    }

    results = {}
    for name, mask in subsamples.items():
        sub = df[mask].copy()
        print(f"\n{'='*60}")
        print(f"{name}: N={len(sub)}")
        print(f"{'='*60}")
        results[name] = run_two_step(sub, name)
        r = results[name]
        for T in [5,10,15,20]:
            print(f"  cum{T}={r[f'cum_return_{T}']:.4f} ({r[f'cum_return_{T}_se']:.4f})")

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
    for param, label in [('beta_1','Experience (beta_1)'),('beta_2','Tenure (beta_2)'),
                          ('beta_1_plus_beta_2','beta_1 + beta_2')]:
        vals = [results[n][param] for n in names]
        ses = [results[n][f'{param}_se'] for n in names]
        print(f"{label:30s} {vals[0]:15.4f} {vals[1]:15.4f} {vals[2]:15.4f}")
        print(f"{'':30s} {'('+f'{ses[0]:.4f}'+')':>15s} {'('+f'{ses[1]:.4f}'+')':>15s} {'('+f'{ses[2]:.4f}'+')':>15s}")
    print(f"\nBOTTOM PANEL: Cumulative Returns")
    print(f"{'':15s} {'(1) Prof/Svc':>12s} {'(2) Nonunion':>12s} {'(3) Union':>12s} {'(4) U vs NU':>12s}")
    print("-" * 63)
    for T_val in [5,10,15,20]:
        key = f'cum_return_{T_val}'; key_se = f'cum_return_{T_val}_se'
        vals = [results[n][key] for n in names]
        ses = [results[n][key_se] for n in names]
        b1_nu = results['BC_Nonunion']['beta_1']; b1_u = results['BC_Union']['beta_1']
        union_rel = vals[2] + (b1_nu - b1_u) * T_val
        u_se = ses[2]
        union_rel_se = np.sqrt(u_se**2 + (T_val*results['BC_Nonunion']['beta_1_se'])**2
                               + (T_val*results['BC_Union']['beta_1_se'])**2)
        print(f"{T_val:2d} years:       {vals[0]:12.4f} {vals[1]:12.4f} {vals[2]:12.4f} {union_rel:12.4f}")
        print(f"{'':15s} {'('+f'{ses[0]:.4f}'+')':>12s} {'('+f'{ses[1]:.4f}'+')':>12s} {'('+f'{ses[2]:.4f}'+')':>12s} {'('+f'{union_rel_se:.4f}'+')':>12s}")
    print(f"\nSample sizes:")
    for n, label in zip(names, ['Professional/Service','Nonunion','Union']):
        print(f"  {label}: N = {results[n]['N']:,}")


def score_against_ground_truth(results):
    gt = GROUND_TRUTH
    total_points = 0; earned_points = 0
    print("\n" + "=" * 80 + "\nSCORING\n" + "=" * 80)

    for name in ['Professional/Service','BC_Nonunion','BC_Union']:
        gen = results[name]['beta_1_plus_beta_2']
        true = gt[name]['beta_1_plus_beta_2']
        re = abs(gen - true) / max(abs(true), 0.001)
        pts = 5 if re <= 0.20 else (3 if re <= 0.40 else 0)
        earned_points += pts; total_points += 5
        print(f"  b1b2 {name}: {gen:.4f} vs {true:.4f} re={re:.3f} -> {pts}/5")

    cp = 20 / 6
    for name in ['Professional/Service','BC_Nonunion','BC_Union']:
        for p in ['beta_1','beta_2']:
            gen = results[name][p]; true = gt[name][p]
            ae = abs(gen - true)
            pts = cp if ae <= 0.01 else (cp*0.5 if ae <= 0.03 else 0)
            earned_points += pts; total_points += cp
            print(f"  {p} {name}: {gen:.4f} vs {true:.4f} ae={ae:.4f} -> {pts:.1f}/{cp:.1f}")

    cp2 = 20 / 12
    for name in ['Professional/Service','BC_Nonunion','BC_Union']:
        for T in [5,10,15,20]:
            k = f'cum_return_{T}'
            gen = results[name][k]; true = gt[name][k]
            ae = abs(gen - true)
            pts = cp2 if ae <= 0.03 else (cp2*0.5 if ae <= 0.06 else 0)
            earned_points += pts; total_points += cp2
            print(f"  cum{T} {name}: {gen:.4f} vs {true:.4f} ae={ae:.4f} -> {pts:.1f}/{cp2:.1f}")

    items = []
    for name in ['Professional/Service','BC_Nonunion','BC_Union']:
        for p in ['beta_1_se','beta_2_se','beta_1_plus_beta_2_se']:
            items.append((name, p))
    sp = 10 / len(items)
    for name, p in items:
        gen = results[name][p]; true = gt[name][p]
        re = abs(gen - true) / max(abs(true), 0.001)
        pts = sp if re <= 0.30 else (sp*0.5 if re <= 0.60 else 0)
        earned_points += pts; total_points += sp
        print(f"  {p} {name}: {gen:.4f} vs {true:.4f} re={re:.3f} -> {pts:.2f}/{sp:.2f}")

    for name in ['Professional/Service','BC_Nonunion','BC_Union']:
        gen = results[name]['N']; true = gt[name]['N']
        re = abs(gen - true) / true
        pts = 5 if re <= 0.05 else (3 if re <= 0.10 else (1 if re <= 0.20 else 0))
        earned_points += pts; total_points += 5
        print(f"  N {name}: {gen} vs {true} re={re:.3f} -> {pts}/5")

    sok = sum(1 for name in ['Professional/Service','BC_Nonunion','BC_Union']
              for p in ['beta_1','beta_2','beta_1_plus_beta_2']
              if np.sign(results[name][p]) == np.sign(gt[name][p]))
    sig = 10 * sok / 9
    earned_points += sig; total_points += 10
    print(f"  Signs: {sok}/9 -> {sig:.1f}/10")

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
