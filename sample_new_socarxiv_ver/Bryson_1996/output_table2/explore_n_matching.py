import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv('gss1993_clean.csv')

minority_genres = ['rap', 'reggae', 'blues', 'jazz', 'gospel', 'latin']
remaining_genres = ['musicals', 'oldies', 'classicl', 'bigband', 'newage', 'opera',
                    'blugrass', 'folk', 'moodeasy', 'conrock', 'hvymetal', 'country']

for g in minority_genres + remaining_genres:
    df[g] = pd.to_numeric(df[g], errors='coerce')

minority_valid = df[minority_genres].isin([1,2,3,4,5]).all(axis=1)
remaining_valid = df[remaining_genres].isin([1,2,3,4,5]).all(axis=1)
df['dv_minority'] = np.nan
df.loc[minority_valid, 'dv_minority'] = (df.loc[minority_valid, minority_genres] >= 4).sum(axis=1)
df['dv_remaining'] = np.nan
df.loc[remaining_valid, 'dv_remaining'] = (df.loc[remaining_valid, remaining_genres] >= 4).sum(axis=1)

for item in ['racmost', 'busing', 'racdif1', 'racdif2', 'racdif3', 'racdif4']:
    df[item] = pd.to_numeric(df[item], errors='coerce')

df['r_racmost'] = (df['racmost'] == 1).astype(float).where(df['racmost'].notna())
df['r_busing'] = (df['busing'] == 2).astype(float).where(df['busing'].notna())
df['r_racdif1'] = (df['racdif1'] == 2).astype(float).where(df['racdif1'].notna())
df['r_racdif2'] = (df['racdif2'] == 2).astype(float).where(df['racdif2'].notna())
df['r_racdif3'] = (df['racdif3'] == 2).astype(float).where(df['racdif3'].notna())
df['r_racdif4'] = (df['racdif4'] == 1).astype(float).where(df['racdif4'].notna())

# 6-item Scale C with min 4, rescale to 0-5
coded = ['r_racmost', 'r_busing', 'r_racdif1', 'r_racdif2', 'r_racdif3', 'r_racdif4']
racism_vals = []
for idx in df.index:
    vals = [df.loc[idx, c] for c in coded]
    n_v = sum(1 for v in vals if not np.isnan(v))
    if n_v >= 4:
        valid_v = [v for v in vals if not np.isnan(v)]
        pm = np.mean(valid_v)
        racism_vals.append(sum(v if not np.isnan(v) else pm for v in vals) * 5/6)
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
df['black'] = (pd.to_numeric(df['race'], errors='coerce') == 2).astype(int)
df['ethnic'] = pd.to_numeric(df['ethnic'], errors='coerce')
df['hispanic'] = df['ethnic'].isin([17, 22, 25]).astype(int)
df['other_race'] = (pd.to_numeric(df['race'], errors='coerce') == 3).astype(int)
df['fund'] = pd.to_numeric(df['fund'], errors='coerce')
df['conservative_protestant'] = (df['fund'] == 1).astype(int)
df['no_religion'] = (pd.to_numeric(df['relig'], errors='coerce') == 4).astype(int)
df['southern'] = (pd.to_numeric(df['region'], errors='coerce') == 3).astype(int)

# What if we restrict to age < 89?
print("=== Restrict age < 89 ===")
df_test = df[df['age_var'] < 89].copy()
predictors = ['racism_score', 'education', 'income_pc', 'occ_prestige',
              'female', 'age_var', 'black', 'hispanic', 'other_race',
              'conservative_protestant', 'no_religion', 'southern']
for dv in ['dv_minority', 'dv_remaining']:
    m = df_test[[dv] + predictors].dropna()
    print(f"  {dv}: N={len(m)}")

# What if we drop ethnic == 97 (uncodeable)?
print("\n=== Drop ethnic == 97 ===")
df_test = df[df['ethnic'] != 97].copy()
for dv in ['dv_minority', 'dv_remaining']:
    m = df_test[[dv] + predictors].dropna()
    print(f"  {dv}: N={len(m)}")

# What if we exclude hompop == 1 for single-person households?
# No, that doesn't make sense.

# What if NaN ethnic is dropped (treated as missing)?
print("\n=== Drop ethnic NaN ===")
df_test = df[df['ethnic'].notna()].copy()
for dv in ['dv_minority', 'dv_remaining']:
    m = df_test[[dv] + predictors].dropna()
    print(f"  {dv}: N={len(m)}")

# What if NaN fund is dropped?
print("\n=== Drop fund NaN ===")
df_test = df[df['fund'].notna()].copy()
for dv in ['dv_minority', 'dv_remaining']:
    m = df_test[[dv] + predictors].dropna()
    print(f"  {dv}: N={len(m)}")

# What if both ethnic NaN AND fund NaN dropped?
print("\n=== Drop ethnic NaN AND fund NaN ===")
df_test = df[df['ethnic'].notna() & df['fund'].notna()].copy()
for dv in ['dv_minority', 'dv_remaining']:
    m = df_test[[dv] + predictors].dropna()
    print(f"  {dv}: N={len(m)}")

# What if age > 89 AND ethnic == 97 dropped?
print("\n=== Drop age >= 89 AND ethnic == 97 ===")
df_test = df[(df['age_var'] < 89) & (df['ethnic'] != 97)].copy()
for dv in ['dv_minority', 'dv_remaining']:
    m = df_test[[dv] + predictors].dropna()
    print(f"  {dv}: N={len(m)}")

# Check: maybe the paper uses attend (church attendance) which excludes some cases
# But attend isn't in the Table 2 model

# What about excluding people with no valid racism items at all?
# Currently min_valid=4 already handles this

# Let me try: what if we use denom for conservative protestant instead of fund?
# GSS denom: Baptist=1, Methodist=2, Lutheran=3, Presbyterian=4, Episcopalian=5, Congregationalist=6, Other=7
# Conservative Protestant = Southern Baptist and similar denominations
# In GSS, denom==1 (Baptist) could be conservative
print("\n=== Different Conservative Protestant codings ===")
df['denom'] = pd.to_numeric(df['denom'], errors='coerce')
for cp_def, label in [
    ((df['fund'] == 1), 'fund==1'),
    ((df['denom'] == 1), 'denom==1 (Baptist)'),
    ((df['denom'].isin([1, 7])), 'denom in [1,7]'),
]:
    df['conservative_protestant'] = cp_def.astype(int)
    for dv in ['dv_minority', 'dv_remaining']:
        m = df[[dv] + predictors].dropna()
        y = m[dv].values
        X = m[predictors].values
        y_z = (y - y.mean()) / y.std(ddof=1)
        X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
        model = sm.OLS(y_z, sm.add_constant(X_z)).fit()
        cp_idx = predictors.index('conservative_protestant')
        cp_beta = model.params[cp_idx + 1]
        rac_beta = model.params[1]
        blk_beta = model.params[predictors.index('black') + 1]
        r2 = sm.OLS(y, sm.add_constant(X)).fit().rsquared
        print(f"  {label:25s} {dv[:15]:15s}: N={len(m)}, cp={cp_beta:.3f}, rac={rac_beta:.3f}, blk={blk_beta:.3f}, R2={r2:.3f}")

# Reset
df['conservative_protestant'] = (df['fund'] == 1).astype(int)

# What if we use Table 1's political intolerance model approach?
# The paper's Table 1 has N=787 for the demographic model
# Table 2 should have lower N because of racism scale missing data
# Let me check: how many have valid music ratings + demographics?
print("\n=== N without racism requirement ===")
preds_no_racism = [p for p in predictors if p != 'racism_score']
for dv in ['dv_minority', 'dv_remaining']:
    m = df[[dv] + preds_no_racism].dropna()
    print(f"  {dv}: N={len(m)} (without racism)")

# What about using income91 for per capita (maybe less missing than realinc)?
print(f"\nMissing: realinc={df['realinc'].isna().sum()}, income91={df['income91'].isna().sum()}")
