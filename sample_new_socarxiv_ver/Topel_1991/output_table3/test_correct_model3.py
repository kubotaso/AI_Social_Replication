"""Test IV with correct Model 3 experience polynomial terms"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

EDUC_MAP = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}

GNP_DEFLATOR = {
    1967: 33.4, 1968: 34.8, 1969: 36.7, 1970: 38.8, 1971: 40.5,
    1972: 41.8, 1973: 44.4, 1974: 48.9, 1975: 53.6, 1976: 56.9,
    1977: 60.6, 1978: 65.2, 1979: 72.6, 1980: 82.4, 1981: 90.9,
    1982: 100.0
}

# Correct Model 3 step 1 values from Table 2:
# Tenure polynomial:
#   b1b2 (linear): .1258 (.0162)
#   g2 (x10^2): -.4592 (.1080)  -> -0.004592
#   g3 (x10^3): .1846 (.0526)   -> 0.0001846
#   g4 (x10^4): -.0245 (.0079)  -> -0.00000245
# Experience polynomial:
#   d2 (x10^2): -.4067 (.1546)  -> -0.004067
#   d3 (x10^3): .0989 (.0517)   -> 0.0000989
#   d4 (x10^4): .0089 (.0058)   -> 0.00000089

# WRONG values that instruction_summary had:
PAPER_S1_WRONG = {
    'b1b2': 0.1258, 'b1b2_se': 0.0161,
    'g2': -0.004592, 'g3': 0.0001846, 'g4': -0.00000245,
    'd2': -0.006051, 'd3': 0.0002067, 'd4': -0.00000238  # WRONG: from Model 1 or incorrect
}

# CORRECT Model 3 values:
PAPER_S1_CORRECT = {
    'b1b2': 0.1258, 'b1b2_se': 0.0162,
    'g2': -0.004592, 'g3': 0.0001846, 'g4': -0.00000245,
    'd2': -0.004067, 'd3': 0.0000989, 'd4': 0.00000089  # CORRECT: from Model 3
}


def prepare_data(data_source, pnum_filter=None):
    df = pd.read_csv(data_source)
    if 'pnum' not in df.columns:
        df['pnum'] = df['person_id'] % 1000
    if pnum_filter is not None:
        df = df[df['pnum'].isin(pnum_filter)].copy()

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


def run_iv(df, step1, ctrl, yr_dums, wage_var='log_real_gnp'):
    b1b2 = step1['b1b2']
    g2, g3, g4 = step1['g2'], step1['g3'], step1['g4']
    d2, d3, d4 = step1['d2'], step1['d3'], step1['d4']

    levels = df.copy()
    levels['w_star'] = (levels[wage_var]
                        - b1b2 * levels['tenure']
                        - g2 * levels['tenure_2']
                        - g3 * levels['tenure_3']
                        - g4 * levels['tenure_4']
                        - d2 * levels['exp_2']
                        - d3 * levels['exp_3']
                        - d4 * levels['exp_4'])

    levels['init_exp'] = (levels['experience'] - levels['tenure']).clip(lower=0)

    all_ctrl = ctrl + yr_dums
    levels = levels.dropna(subset=['w_star', 'experience', 'init_exp'] + all_ctrl).copy()

    exog_df = sm.add_constant(levels[all_ctrl])
    rank = np.linalg.matrix_rank(exog_df.values)
    active_yr = yr_dums.copy()
    while rank < exog_df.shape[1] and active_yr:
        active_yr.pop()
        all_ctrl_temp = ctrl + active_yr
        exog_df = sm.add_constant(levels[all_ctrl_temp])
        rank = np.linalg.matrix_rank(exog_df.values)
    all_ctrl = ctrl + active_yr

    dep = levels['w_star']
    exog = sm.add_constant(levels[all_ctrl])
    endog = levels[['experience']]
    instruments = levels[['init_exp']]

    iv = IV2SLS(dep, exog, endog, instruments).fit(cov_type='unadjusted')
    return iv.params['experience'], iv.std_errors['experience'], len(levels), levels, all_ctrl


# Test with different samples and step1 values
samples = [
    ("data/psid_panel_full.csv", [1, 170, 3], "full [1,170,3]"),
    ("data/psid_panel.csv", [1, 3, 170, 171], "main [1,3,170,171]"),
    ("data/psid_panel.csv", [1, 170], "main [1,170]"),
]

ctrl = ['education_years', 'married', 'union_member', 'disabled',
        'region_ne', 'region_nc', 'region_south']

print(f"{'Sample':>25} {'S1 values':>12} {'N':>7} {'beta_1':>8} {'beta_2':>8}")
print("-" * 70)

for fname, pf, label in samples:
    df = prepare_data(fname, pf)
    yr_dums = sorted([c for c in df.columns if c.startswith('year_') and c != 'year'])
    yr_dums = [c for c in yr_dums if df[c].std() > 1e-10]
    yr_dums = yr_dums[1:]

    for s1_label, s1 in [("wrong", PAPER_S1_WRONG), ("correct_m3", PAPER_S1_CORRECT)]:
        try:
            b1, se, n, _, _ = run_iv(df, s1, ctrl, yr_dums)
            b2 = s1['b1b2'] - b1
            print(f"{label:>25} {s1_label:>12} {n:>7} {b1:>8.4f} {b2:>8.4f}")
        except Exception as e:
            print(f"{label:>25} {s1_label:>12} ERROR: {e}")

print(f"\n{'Paper':>25} {'':>12} {'10685':>7} {'0.0713':>8} {'0.0545':>8}")

# Now let's try with the correct Model 3 values AND full bootstrap (both steps)
print("\n\n=== Full bootstrap with correct Model 3 values ===")
print("Using [1,170,3] sample and correct Model 3 step 1 values")

df = prepare_data("data/psid_panel_full.csv", pnum_filter=[1, 170, 3])
yr_dums = sorted([c for c in df.columns if c.startswith('year_') and c != 'year'])
yr_dums = [c for c in yr_dums if df[c].std() > 1e-10]
yr_dums = yr_dums[1:]

# Base result
b1_base, _, n_base, _, _ = run_iv(df, PAPER_S1_CORRECT, ctrl, yr_dums)
print(f"\nBase result: beta_1 = {b1_base:.4f}, N = {n_base}")

# Full bootstrap: resample persons, run OLS step 1 and IV step 2
np.random.seed(42)
n_boot = 200
boot_betas = []
persons = df['person_id'].unique()
n_persons = len(persons)

for b in range(n_boot):
    boot_pids = np.random.choice(persons, size=n_persons, replace=True)
    frames = []
    for i, pid in enumerate(boot_pids):
        pdata = df[df['person_id'] == pid].copy()
        pdata = pdata.assign(person_id=i, job_id=pdata['job_id'].astype(str) + f'_{i}')
        frames.append(pdata)
    boot_df = pd.concat(frames, ignore_index=True)

    # Step 1: OLS on first-differenced within-job data
    boot_df = boot_df.sort_values(['person_id', 'job_id', 'year'])
    boot_df['d_log_wage'] = boot_df.groupby(['person_id', 'job_id'])['log_real_gnp'].diff()
    boot_df['d_tenure'] = boot_df.groupby(['person_id', 'job_id'])['tenure'].diff()
    wj = boot_df.dropna(subset=['d_log_wage', 'd_tenure']).copy()
    wj = wj[wj['d_tenure'] == 1].copy()  # within-job only

    if len(wj) < 100:
        continue

    # Compute changes in polynomial terms
    for k in [2, 3, 4]:
        wj[f'd_tenure_{k}'] = wj.groupby(['person_id', 'job_id'])[f'tenure_{k}'].diff()
        wj[f'd_exp_{k}'] = wj.groupby(['person_id', 'job_id'])[f'exp_{k}'].diff()
    wj = wj.dropna(subset=['d_tenure_2', 'd_exp_2']).copy()

    # Year dummies for first-diff
    yr_dums_fd = [c for c in wj.columns if c.startswith('year_') and c != 'year']
    yr_dums_fd = [c for c in yr_dums_fd if wj[c].std() > 1e-10]
    yr_dums_fd = yr_dums_fd[1:] if len(yr_dums_fd) > 1 else yr_dums_fd

    try:
        X_s1 = sm.add_constant(wj[['d_tenure', 'd_tenure_2', 'd_tenure_3', 'd_tenure_4',
                                    'd_exp_2', 'd_exp_3', 'd_exp_4'] + yr_dums_fd])
        ols_s1 = sm.OLS(wj['d_log_wage'], X_s1).fit()

        boot_s1 = {
            'b1b2': ols_s1.params['d_tenure'],
            'g2': ols_s1.params['d_tenure_2'],
            'g3': ols_s1.params['d_tenure_3'],
            'g4': ols_s1.params['d_tenure_4'],
            'd2': ols_s1.params['d_exp_2'],
            'd3': ols_s1.params['d_exp_3'],
            'd4': ols_s1.params['d_exp_4'],
        }

        # Step 2: IV with bootstrap step 1 values
        b1_b, _, _, _, _ = run_iv(boot_df, boot_s1, ctrl, yr_dums)
        boot_betas.append(b1_b)
        if (b+1) % 50 == 0:
            print(f"  Bootstrap {b+1}/{n_boot}: mean={np.mean(boot_betas):.4f}, se={np.std(boot_betas):.4f}")
    except:
        pass

if len(boot_betas) > 10:
    boot_se = np.std(boot_betas)
    boot_mean = np.mean(boot_betas)
    print(f"\nFull bootstrap results (n={len(boot_betas)}):")
    print(f"  Mean beta_1: {boot_mean:.4f}")
    print(f"  SE(beta_1): {boot_se:.4f}")
    print(f"  Paper: beta_1=0.0713, SE=0.0181")
else:
    print(f"Too few valid bootstrap samples: {len(boot_betas)}")
