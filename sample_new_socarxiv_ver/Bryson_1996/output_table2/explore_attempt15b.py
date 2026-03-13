import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv("gss1993_clean.csv")

minority_genres = ['rap', 'reggae', 'blues', 'jazz', 'gospel', 'latin']
remaining_genres = ['musicals', 'oldies', 'classicl', 'bigband', 'newage', 'opera',
                    'blugrass', 'folk', 'moodeasy', 'conrock', 'hvymetal', 'country']

for g in minority_genres + remaining_genres:
    df[g] = pd.to_numeric(df[g], errors='coerce')

minority_valid = df[minority_genres].isin([1,2,3,4,5]).all(axis=1)
df['dv_minority'] = np.nan
df.loc[minority_valid, 'dv_minority'] = (df.loc[minority_valid, minority_genres] >= 4).sum(axis=1)

remaining_valid = df[remaining_genres].isin([1,2,3,4,5]).all(axis=1)
df['dv_remaining'] = np.nan
df.loc[remaining_valid, 'dv_remaining'] = (df.loc[remaining_valid, remaining_genres] >= 4).sum(axis=1)

# Best racism scale: old 5-item, racdif3==2, min 4
for item in ['racmost', 'busing', 'racdif1', 'racdif2', 'racdif3']:
    df[item] = pd.to_numeric(df[item], errors='coerce')

df['r_racmost'] = (df['racmost'] == 1).astype(float).where(df['racmost'].notna())
df['r_busing'] = (df['busing'] == 2).astype(float).where(df['busing'].notna())
df['r_racdif1'] = (df['racdif1'] == 2).astype(float).where(df['racdif1'].notna())
df['r_racdif2'] = (df['racdif2'] == 2).astype(float).where(df['racdif2'].notna())
df['r_racdif3'] = (df['racdif3'] == 2).astype(float).where(df['racdif3'].notna())

coded = ['r_racmost', 'r_busing', 'r_racdif1', 'r_racdif2', 'r_racdif3']
racism_vals = []
for idx in df.index:
    vals = [df.loc[idx, c] for c in coded]
    n_valid = sum(1 for v in vals if not np.isnan(v))
    if n_valid >= 4:
        valid_v = [v for v in vals if not np.isnan(v)]
        pm = np.mean(valid_v)
        racism_vals.append(sum(v if not np.isnan(v) else pm for v in vals))
    else:
        racism_vals.append(np.nan)
df['racism_score'] = racism_vals

df['education'] = pd.to_numeric(df['educ'], errors='coerce')
df['realinc'] = pd.to_numeric(df['realinc'], errors='coerce')
df['hompop'] = pd.to_numeric(df['hompop'], errors='coerce')
df['income_pc'] = df['realinc'] / df['hompop']
df['occ_prestige'] = pd.to_numeric(df['prestg80'], errors='coerce')
df['female'] = (pd.to_numeric(df['sex'], errors='coerce') == 2).astype(int)
df['age_var'] = pd.to_numeric(df['age'], errors='coerce')
df['race'] = pd.to_numeric(df['race'], errors='coerce')
df['black'] = (df['race'] == 2).astype(int)
df['ethnic'] = pd.to_numeric(df['ethnic'], errors='coerce')
df['hispanic'] = (df['ethnic'].isin([17, 22, 25]) & (df['race'] != 3)).astype(int)
df['other_race'] = ((df['race'] == 3) & ~df['ethnic'].isin([17, 22, 25])).astype(int)
df['conservative_protestant'] = (pd.to_numeric(df['fund'], errors='coerce') == 1).astype(int)
df['no_religion'] = (pd.to_numeric(df['relig'], errors='coerce') == 4).astype(int)
df['southern'] = (pd.to_numeric(df['region'], errors='coerce') == 3).astype(int)

# Check available income/prestige alternatives
print("=== Available alternative variables ===")
for col in ['income91', 'rincom91', 'coninc', 'conrinc', 'sei', 'sei10', 'prestige', 'prestg80', 'prestg10']:
    if col in df.columns:
        vals = pd.to_numeric(df[col], errors='coerce')
        print(f"  {col}: {vals.notna().sum()} valid, range={vals.min():.1f}-{vals.max():.1f}, mean={vals.mean():.1f}")

# Test different income and prestige combos
predictors_base = ['racism_score', 'education', 'INCOME', 'PRESTIGE',
                   'female', 'age_var', 'black', 'hispanic', 'other_race',
                   'conservative_protestant', 'no_religion', 'southern']

income_options = {
    'realinc/hompop': df['income_pc'],
    'income91': pd.to_numeric(df['income91'], errors='coerce') if 'income91' in df.columns else None,
    'rincom91': pd.to_numeric(df['rincom91'], errors='coerce') if 'rincom91' in df.columns else None,
    'coninc': pd.to_numeric(df['coninc'], errors='coerce') if 'coninc' in df.columns else None,
    'coninc/hompop': pd.to_numeric(df['coninc'], errors='coerce') / df['hompop'] if 'coninc' in df.columns else None,
    'realinc': df['realinc'],
}

prestige_options = {
    'prestg80': df['occ_prestige'],
    'sei': pd.to_numeric(df['sei'], errors='coerce') if 'sei' in df.columns else None,
}

true_m1 = {'Racism': 0.130, 'Educ': -0.175, 'Black': -0.132, 'Age': 0.163}
true_m2 = {'Racism': 0.080, 'Educ': -0.242, 'Black': 0.042, 'Age': 0.126}

print("\n=== Testing income/prestige alternatives on M2 ===")
for inc_name, inc_col in income_options.items():
    if inc_col is None:
        continue
    for pres_name, pres_col in prestige_options.items():
        if pres_col is None:
            continue
        test_df = df.copy()
        test_df['INCOME'] = inc_col
        test_df['PRESTIGE'] = pres_col

        predictors = ['racism_score', 'education', 'INCOME', 'PRESTIGE',
                      'female', 'age_var', 'black', 'hispanic', 'other_race',
                      'conservative_protestant', 'no_religion', 'southern']

        for dv_name, label in [('dv_minority', 'M1'), ('dv_remaining', 'M2')]:
            vars_needed = [dv_name] + predictors
            model_df = test_df[vars_needed].dropna()
            N = len(model_df)
            if N < 100:
                continue
            y = model_df[dv_name].values
            X = model_df[predictors].values
            y_z = (y - y.mean()) / y.std(ddof=1)
            X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
            m = sm.OLS(y_z, sm.add_constant(X_z)).fit()

            racism_b = m.params[1]
            educ_b = m.params[2]
            black_b = m.params[7]
            age_b = m.params[6]
            r2 = sm.OLS(y, sm.add_constant(X)).fit().rsquared

            racism_p = m.pvalues[1]
            black_p = m.pvalues[7]
            age_p = m.pvalues[6]

            sig_r = "***" if racism_p < 0.001 else "**" if racism_p < 0.01 else "*" if racism_p < 0.05 else ""
            sig_b = "***" if black_p < 0.001 else "**" if black_p < 0.01 else "*" if black_p < 0.05 else ""
            sig_a = "***" if age_p < 0.001 else "**" if age_p < 0.01 else "*" if age_p < 0.05 else ""

            if label == 'M2':
                print(f"  {inc_name:20s} {pres_name:10s} {label} N={N:4d} "
                      f"Racism={racism_b:+.3f}{sig_r:3s} Black={black_b:+.3f}{sig_b:3s} "
                      f"Age={age_b:+.3f}{sig_a:3s} Educ={educ_b:+.3f} R2={r2:.3f}")

# Test: what if Other race includes Hispanic race==3?
print("\n=== Testing Hispanic/Other race coding variants on M2 ===")
hispanic_configs = {
    'non_overlap': lambda df: (
        (df['ethnic'].isin([17, 22, 25]) & (df['race'] != 3)).astype(int),
        ((df['race'] == 3) & ~df['ethnic'].isin([17, 22, 25])).astype(int)
    ),
    'overlap_hisp_priority': lambda df: (
        df['ethnic'].isin([17, 22, 25]).astype(int),
        ((df['race'] == 3) & ~df['ethnic'].isin([17, 22, 25])).astype(int)
    ),
    'hisp_race3_only': lambda df: (
        ((df['race'] == 3) & df['ethnic'].isin([17, 22, 25])).astype(int),
        ((df['race'] == 3) & ~df['ethnic'].isin([17, 22, 25])).astype(int)
    ),
    'hisp_all_other_nonhisp': lambda df: (
        df['ethnic'].isin([17, 22, 25]).astype(int),
        ((df['race'] == 3) & ~df['ethnic'].isin([17, 22, 25])).astype(int)
    ),
}

predictors = ['racism_score', 'education', 'income_pc', 'occ_prestige',
              'female', 'age_var', 'black', 'hispanic', 'other_race',
              'conservative_protestant', 'no_religion', 'southern']

for config_name, config_fn in hispanic_configs.items():
    test_df = df.copy()
    test_df['hispanic'], test_df['other_race'] = config_fn(test_df)

    for dv_name, label in [('dv_remaining', 'M2')]:
        vars_needed = [dv_name] + predictors
        model_df = test_df[vars_needed].dropna()
        N = len(model_df)
        y = model_df[dv_name].values
        X = model_df[predictors].values
        y_z = (y - y.mean()) / y.std(ddof=1)
        X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
        m = sm.OLS(y_z, sm.add_constant(X_z)).fit()

        racism_b = m.params[1]
        black_b = m.params[7]
        hisp_b = m.params[8]
        other_b = m.params[9]

        racism_p = m.pvalues[1]
        black_p = m.pvalues[7]

        sig_r = "***" if racism_p < 0.001 else "**" if racism_p < 0.01 else "*" if racism_p < 0.05 else ""
        sig_b = "***" if black_p < 0.001 else "**" if black_p < 0.01 else "*" if black_p < 0.05 else ""

        print(f"  {config_name:25s} {label} N={N} Racism={racism_b:+.3f}{sig_r:3s} "
              f"Black={black_b:+.3f}{sig_b:3s} Hisp={hisp_b:+.3f} Other={other_b:+.3f}")

# Test: what about broader ethnic codes for Hispanic?
# GSS ethnic codes: 17=Mexico, 22=Puerto Rico, 25=Other Spanish
# What about also including 2=Spanish? Or just using race==3?
print("\n=== Testing broader Hispanic ethnic codes ===")
broader_codes = {
    'eth17_22_25': [17, 22, 25],
    'eth2_17_22_25': [2, 17, 22, 25],
    'eth17_22_25_38': [17, 22, 25, 38],
    'eth17_22_25_2_38': [2, 17, 22, 25, 38],
}

for code_name, eth_codes in broader_codes.items():
    test_df = df.copy()
    test_df['hispanic'] = (test_df['ethnic'].isin(eth_codes) & (test_df['race'] != 3)).astype(int)
    test_df['other_race'] = ((test_df['race'] == 3) & ~test_df['ethnic'].isin(eth_codes)).astype(int)

    for dv_name, label in [('dv_remaining', 'M2')]:
        vars_needed = [dv_name] + predictors
        model_df = test_df[vars_needed].dropna()
        N = len(model_df)
        y = model_df[dv_name].values
        X = model_df[predictors].values
        y_z = (y - y.mean()) / y.std(ddof=1)
        X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
        m = sm.OLS(y_z, sm.add_constant(X_z)).fit()

        racism_b = m.params[1]
        black_b = m.params[7]

        racism_p = m.pvalues[1]
        black_p = m.pvalues[7]

        sig_r = "***" if racism_p < 0.001 else "**" if racism_p < 0.01 else "*" if racism_p < 0.05 else ""
        sig_b = "***" if black_p < 0.001 else "**" if black_p < 0.01 else "*" if black_p < 0.05 else ""

        n_hisp = test_df['hispanic'].sum()
        n_other = test_df['other_race'].sum()
        print(f"  {code_name:20s} n_hisp={n_hisp:3d} n_other={n_other:3d} "
              f"Racism={racism_b:+.3f}{sig_r:3s} Black={black_b:+.3f}{sig_b:3s}")

# What if age should be capped at 89?
print("\n=== Testing age treatment ===")
for age_cap in [None, 89, 85]:
    test_df = df.copy()
    if age_cap:
        test_df.loc[test_df['age_var'] > age_cap, 'age_var'] = np.nan

    vars_needed = ['dv_remaining'] + predictors
    model_df = test_df[vars_needed].dropna()
    N = len(model_df)
    y = model_df['dv_remaining'].values
    X = model_df[predictors].values
    y_z = (y - y.mean()) / y.std(ddof=1)
    X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    m = sm.OLS(y_z, sm.add_constant(X_z)).fit()

    racism_b = m.params[1]
    black_b = m.params[7]
    age_b = m.params[6]

    racism_p = m.pvalues[1]
    black_p = m.pvalues[7]
    age_p = m.pvalues[6]

    sig_r = "***" if racism_p < 0.001 else "**" if racism_p < 0.01 else "*" if racism_p < 0.05 else ""
    sig_b = "***" if black_p < 0.001 else "**" if black_p < 0.01 else "*" if black_p < 0.05 else ""
    sig_a = "***" if age_p < 0.001 else "**" if age_p < 0.01 else "*" if age_p < 0.05 else ""

    print(f"  age_cap={str(age_cap):5s} N={N} Racism={racism_b:+.3f}{sig_r:3s} "
          f"Black={black_b:+.3f}{sig_b:3s} Age={age_b:+.3f}{sig_a:3s}")

# What about using the racism scale as simple SUM (no PM imputation, require all 5)?
print("\n=== Testing simple sum vs PM imputation ===")
test_df = df.copy()
# Simple sum, all 5 required
coded = ['r_racmost', 'r_busing', 'r_racdif1', 'r_racdif2', 'r_racdif3']
for item in ['racmost', 'busing', 'racdif1', 'racdif2', 'racdif3']:
    test_df[item] = pd.to_numeric(test_df[item], errors='coerce')
test_df['r_racmost'] = (test_df['racmost'] == 1).astype(float).where(test_df['racmost'].notna())
test_df['r_busing'] = (test_df['busing'] == 2).astype(float).where(test_df['busing'].notna())
test_df['r_racdif1'] = (test_df['racdif1'] == 2).astype(float).where(test_df['racdif1'].notna())
test_df['r_racdif2'] = (test_df['racdif2'] == 2).astype(float).where(test_df['racdif2'].notna())
test_df['r_racdif3'] = (test_df['racdif3'] == 2).astype(float).where(test_df['racdif3'].notna())

# All 5 required
all5_vals = test_df[coded].dropna()
test_df['racism_all5'] = test_df[coded].sum(axis=1).where(test_df[coded].notna().all(axis=1))

# Mean instead of sum
test_df['racism_mean'] = test_df[coded].mean(axis=1).where(test_df[coded].notna().sum(axis=1) >= 4)

for scale_name, scale_col in [('all5_sum', 'racism_all5'), ('pm_min4', 'racism_score'), ('mean_min4', 'racism_mean')]:
    preds = [scale_col, 'education', 'income_pc', 'occ_prestige',
             'female', 'age_var', 'black', 'hispanic', 'other_race',
             'conservative_protestant', 'no_religion', 'southern']

    for dv_name, label in [('dv_minority', 'M1'), ('dv_remaining', 'M2')]:
        vars_needed = [dv_name] + preds
        model_df = test_df[vars_needed].dropna()
        N = len(model_df)
        if N < 100:
            continue
        y = model_df[dv_name].values
        X = model_df[preds].values
        y_z = (y - y.mean()) / y.std(ddof=1)
        X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
        m = sm.OLS(y_z, sm.add_constant(X_z)).fit()

        racism_b = m.params[1]
        black_b = m.params[7]

        racism_p = m.pvalues[1]
        black_p = m.pvalues[7]

        sig_r = "***" if racism_p < 0.001 else "**" if racism_p < 0.01 else "*" if racism_p < 0.05 else ""
        sig_b = "***" if black_p < 0.001 else "**" if black_p < 0.01 else "*" if black_p < 0.05 else ""

        print(f"  {scale_name:15s} {label} N={N} Racism={racism_b:+.3f}{sig_r:3s} Black={black_b:+.3f}{sig_b:3s}")
