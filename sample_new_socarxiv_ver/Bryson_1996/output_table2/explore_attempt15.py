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

# Also try DV as count of ratings == 5 only (strongly dislike)
df['dv_minority_5'] = np.nan
df.loc[minority_valid, 'dv_minority_5'] = (df.loc[minority_valid, minority_genres] == 5).sum(axis=1)
df['dv_remaining_5'] = np.nan
df.loc[remaining_valid, 'dv_remaining_5'] = (df.loc[remaining_valid, remaining_genres] == 5).sum(axis=1)

# Also try DV as mean rating (not count)
df['dv_minority_mean'] = np.nan
df.loc[minority_valid, 'dv_minority_mean'] = df.loc[minority_valid, minority_genres].mean(axis=1)
df['dv_remaining_mean'] = np.nan
df.loc[remaining_valid, 'dv_remaining_mean'] = df.loc[remaining_valid, remaining_genres].mean(axis=1)

# Standard controls
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

# Racism scale configs to test
for item in ['racmost', 'busing', 'racdif1', 'racdif2', 'racdif3', 'racdif4']:
    df[item] = pd.to_numeric(df[item], errors='coerce')

configs = {
    'old5_racdif3flip_min4': {
        'items': ['racmost', 'busing', 'racdif1', 'racdif2', 'racdif3'],
        'racist_vals': {
            'racmost': 1, 'busing': 2, 'racdif1': 2, 'racdif2': 2, 'racdif3': 2
        },
        'min_valid': 4
    },
    'old5_racdif3orig_min4': {
        'items': ['racmost', 'busing', 'racdif1', 'racdif2', 'racdif3'],
        'racist_vals': {
            'racmost': 1, 'busing': 2, 'racdif1': 2, 'racdif2': 2, 'racdif3': 1
        },
        'min_valid': 4
    },
    'correct5_min4': {
        'items': ['racmost', 'busing', 'racdif1', 'racdif3', 'racdif4'],
        'racist_vals': {
            'racmost': 1, 'busing': 2, 'racdif1': 2, 'racdif3': 2, 'racdif4': 1
        },
        'min_valid': 4
    },
    'hybrid_racdif2_racdif4_min4': {
        'items': ['racmost', 'busing', 'racdif1', 'racdif2', 'racdif4'],
        'racist_vals': {
            'racmost': 1, 'busing': 2, 'racdif1': 2, 'racdif2': 2, 'racdif4': 1
        },
        'min_valid': 4
    },
}

predictors = ['racism_score', 'education', 'income_pc', 'occ_prestige',
              'female', 'age_var', 'black', 'hispanic', 'other_race',
              'conservative_protestant', 'no_religion', 'southern']

true_m2 = {'Racism score': 0.080, 'Black': 0.042, 'Age': 0.126, 'Education': -0.242}

for config_name, config in configs.items():
    coded = []
    for item in config['items']:
        col = f'rc_{item}_{config_name}'
        rv = config['racist_vals'][item]
        df[col] = (df[item] == rv).astype(float).where(df[item].notna())
        coded.append(col)

    racism_vals = []
    for idx in df.index:
        vals = [df.loc[idx, c] for c in coded]
        n_valid = sum(1 for v in vals if not np.isnan(v))
        if n_valid >= config['min_valid']:
            valid_v = [v for v in vals if not np.isnan(v)]
            pm = np.mean(valid_v)
            racism_vals.append(sum(v if not np.isnan(v) else pm for v in vals))
        else:
            racism_vals.append(np.nan)
    df['racism_score'] = racism_vals

    for dv_name, dv_label in [('dv_remaining', 'M2_count>=4'),
                               ('dv_remaining_5', 'M2_count==5'),
                               ('dv_remaining_mean', 'M2_mean')]:
        vars_needed = [dv_name] + predictors
        model_df = df[vars_needed].dropna()
        N = len(model_df)
        y_raw = model_df[dv_name].values
        X_raw = model_df[predictors].values
        y_z = (y_raw - y_raw.mean()) / y_raw.std(ddof=1)
        X_z = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0, ddof=1)
        model_z = sm.OLS(y_z, sm.add_constant(X_z)).fit()
        betas = model_z.params[1:]
        pvals = model_z.pvalues[1:]

        racism_b = betas[0]
        racism_p = pvals[0]
        black_b = betas[6]
        black_p = pvals[6]
        age_b = betas[5]
        age_p = pvals[5]
        educ_b = betas[1]

        r2 = sm.OLS(y_raw, sm.add_constant(X_raw)).fit().rsquared

        racism_sig = "***" if racism_p < 0.001 else "**" if racism_p < 0.01 else "*" if racism_p < 0.05 else ""
        black_sig = "***" if black_p < 0.001 else "**" if black_p < 0.01 else "*" if black_p < 0.05 else ""
        age_sig = "***" if age_p < 0.001 else "**" if age_p < 0.01 else "*" if age_p < 0.05 else ""

        print(f"{config_name:30s} {dv_label:15s} N={N:4d} "
              f"Racism={racism_b:+.3f}{racism_sig:3s} "
              f"Black={black_b:+.3f}{black_sig:3s} "
              f"Age={age_b:+.3f}{age_sig:3s} "
              f"Educ={educ_b:+.3f} "
              f"R2={r2:.3f}")

# Now also try: what if we use racdif3==1 (original coding) in the OLD 5-item scale?
# And test with whites-only subsample
print("\n=== WHITES ONLY (race==1) ===")
df_white = df[df['race'] == 1].copy()

for item in ['racmost', 'busing', 'racdif1', 'racdif2', 'racdif3']:
    df_white[item] = pd.to_numeric(df_white[item], errors='coerce')

df_white['r_racmost'] = (df_white['racmost'] == 1).astype(float).where(df_white['racmost'].notna())
df_white['r_busing'] = (df_white['busing'] == 2).astype(float).where(df_white['busing'].notna())
df_white['r_racdif1'] = (df_white['racdif1'] == 2).astype(float).where(df_white['racdif1'].notna())
df_white['r_racdif2'] = (df_white['racdif2'] == 2).astype(float).where(df_white['racdif2'].notna())
df_white['r_racdif3'] = (df_white['racdif3'] == 2).astype(float).where(df_white['racdif3'].notna())

coded = ['r_racmost', 'r_busing', 'r_racdif1', 'r_racdif2', 'r_racdif3']
racism_vals = []
for idx in df_white.index:
    vals = [df_white.loc[idx, c] for c in coded]
    n_valid = sum(1 for v in vals if not np.isnan(v))
    if n_valid >= 4:
        valid_v = [v for v in vals if not np.isnan(v)]
        pm = np.mean(valid_v)
        racism_vals.append(sum(v if not np.isnan(v) else pm for v in vals))
    else:
        racism_vals.append(np.nan)
df_white['racism_score'] = racism_vals

for g in remaining_genres:
    df_white[g] = pd.to_numeric(df_white[g], errors='coerce')
remaining_valid_w = df_white[remaining_genres].isin([1,2,3,4,5]).all(axis=1)
df_white['dv_remaining'] = np.nan
df_white.loc[remaining_valid_w, 'dv_remaining'] = (df_white.loc[remaining_valid_w, remaining_genres] >= 4).sum(axis=1)

df_white['education'] = pd.to_numeric(df_white['educ'], errors='coerce')
df_white['income_pc'] = pd.to_numeric(df_white['realinc'], errors='coerce') / pd.to_numeric(df_white['hompop'], errors='coerce')
df_white['occ_prestige'] = pd.to_numeric(df_white['prestg80'], errors='coerce')
df_white['female'] = (pd.to_numeric(df_white['sex'], errors='coerce') == 2).astype(int)
df_white['age_var'] = pd.to_numeric(df_white['age'], errors='coerce')
df_white['conservative_protestant'] = (pd.to_numeric(df_white['fund'], errors='coerce') == 1).astype(int)
df_white['no_religion'] = (pd.to_numeric(df_white['relig'], errors='coerce') == 4).astype(int)
df_white['southern'] = (pd.to_numeric(df_white['region'], errors='coerce') == 3).astype(int)

predictors_w = ['racism_score', 'education', 'income_pc', 'occ_prestige',
                'female', 'age_var', 'conservative_protestant', 'no_religion', 'southern']
vars_needed = ['dv_remaining'] + predictors_w
model_df = df_white[vars_needed].dropna()
print(f"Whites-only M2: N={len(model_df)}")
y = model_df['dv_remaining'].values
X = model_df[predictors_w].values
y_z = (y - y.mean()) / y.std(ddof=1)
X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
m = sm.OLS(y_z, sm.add_constant(X_z)).fit()
for i, p in enumerate(predictors_w):
    sig = "***" if m.pvalues[i+1] < 0.001 else "**" if m.pvalues[i+1] < 0.01 else "*" if m.pvalues[i+1] < 0.05 else ""
    print(f"  {p:30s} {m.params[i+1]:+.3f} {sig}")

# Test: what if racdif3 is coded differently in the scale?
# In the "old" 5 items, racdif3 is coded as ==2 (racist). What if we try ==1?
print("\n=== Testing racdif3 coding direction ===")
for racdif3_val in [1, 2]:
    for item in ['racmost', 'busing', 'racdif1', 'racdif2', 'racdif3']:
        df[item] = pd.to_numeric(df[item], errors='coerce')

    df['r_racmost'] = (df['racmost'] == 1).astype(float).where(df['racmost'].notna())
    df['r_busing'] = (df['busing'] == 2).astype(float).where(df['busing'].notna())
    df['r_racdif1'] = (df['racdif1'] == 2).astype(float).where(df['racdif1'].notna())
    df['r_racdif2'] = (df['racdif2'] == 2).astype(float).where(df['racdif2'].notna())
    df['r_racdif3'] = (df['racdif3'] == racdif3_val).astype(float).where(df['racdif3'].notna())

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

    for dv_name in ['dv_minority', 'dv_remaining']:
        vars_needed = [dv_name] + predictors
        model_df = df[vars_needed].dropna()
        N = len(model_df)
        y_raw = model_df[dv_name].values
        X_raw = model_df[predictors].values
        y_z = (y_raw - y_raw.mean()) / y_raw.std(ddof=1)
        X_z = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0, ddof=1)
        model_z = sm.OLS(y_z, sm.add_constant(X_z)).fit()

        racism_b = model_z.params[1]
        racism_p = model_z.pvalues[1]
        black_b = model_z.params[7]
        black_p = model_z.pvalues[7]

        sig_r = "***" if racism_p < 0.001 else "**" if racism_p < 0.01 else "*" if racism_p < 0.05 else ""
        sig_b = "***" if black_p < 0.001 else "**" if black_p < 0.01 else "*" if black_p < 0.05 else ""

        label = "M1" if dv_name == "dv_minority" else "M2"
        print(f"racdif3=={racdif3_val} {label}: N={N} Racism={racism_b:+.3f}{sig_r:3s} Black={black_b:+.3f}{sig_b:3s}")

# Test: what if we use income91 variable instead of realinc/hompop?
print("\n=== Testing income alternatives ===")
# Check what income variables exist
for col in ['income', 'income91', 'rincom91', 'coninc', 'conrinc']:
    if col in df.columns:
        vals = pd.to_numeric(df[col], errors='coerce')
        print(f"  {col}: {vals.notna().sum()} valid, range={vals.min()}-{vals.max()}")
