import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv('gss1993_clean.csv')

minority_genres = ['rap', 'reggae', 'blues', 'jazz', 'gospel', 'latin']
remaining_genres = ['musicals', 'oldies', 'classicl', 'bigband', 'newage', 'opera',
                    'blugrass', 'folk', 'moodeasy', 'conrock', 'hvymetal', 'country']

for g in minority_genres + remaining_genres:
    df[g] = pd.to_numeric(df[g], errors='coerce')

# Test different DV threshold codings
# Main definition: >= 4 means dislike (values 4 and 5)
# Alternative: maybe >3 which is same as >= 4
# Alternative: == 5 only (strong dislike)
# Alternative: >= 3 (dislike + neutral-ish)

# Check what the actual values mean
# GSS culture module: 1=like very much, 2=like, 3=mixed, 4=dislike, 5=dislike very much
# Paper says: "dislike" + "dislike very much" = 4 and 5 - confirmed

# Let me check if there's a different count method
# Maybe it's the mean rating rather than count of dislikes?

# Test: What if DV is mean of ratings (not count of dislikes)?
# This would give continuous variable

# First check: maybe the issue is with requiring ALL 12 valid
# Try allowing 1-2 missing and using count
for min_valid in [12, 11, 10]:
    mask = df[remaining_genres].isin([1,2,3,4,5]).sum(axis=1) >= min_valid
    dv = (df.loc[mask, remaining_genres].isin([4,5])).sum(axis=1)
    print(f'DV remaining (min {min_valid} valid): n={mask.sum()}, mean={dv.mean():.3f}')

# Check if minority_genres list is correct
# Paper says: "rap, reggae, blues/R&B, jazz, gospel, and Latin music"
# GSS variables: rap, reggae, blues, jazz, gospel, latin - correct

# Check if remaining genres list is correct
# Paper says "12 remaining genres"
# All 18 genres minus 6 minority = 12 remaining
# Let me verify all 18 GSS genre variables
all_18 = minority_genres + remaining_genres
print(f'\nAll 18 genres: {all_18}')
print(f'Total: {len(all_18)}')

# Check DV remaining with different N approaches
print('\n=== DV remaining N with different approaches ===')
for item in ['racmost', 'busing', 'racdif1', 'racdif2', 'racdif3', 'racdif4']:
    df[item] = pd.to_numeric(df[item], errors='coerce')

df['r_racmost'] = (df['racmost'] == 1).astype(float).where(df['racmost'].notna())
df['r_busing'] = (df['busing'] == 2).astype(float).where(df['busing'].notna())
df['r_racdif1'] = (df['racdif1'] == 2).astype(float).where(df['racdif1'].notna())
df['r_racdif2'] = (df['racdif2'] == 2).astype(float).where(df['racdif2'].notna())
df['r_racdif3v1'] = (df['racdif3'] == 1).astype(float).where(df['racdif3'].notna())
df['r_racdif3v2'] = (df['racdif3'] == 2).astype(float).where(df['racdif3'].notna())
df['r_racdif4'] = (df['racdif4'] == 1).astype(float).where(df['racdif4'].notna())

# Build several scale versions with person-mean imputation
def build_scale_imp(df, coded_items, min_valid=4):
    vals_list = []
    for idx in df.index:
        vals = [df.loc[idx, c] for c in coded_items]
        n_v = sum(1 for v in vals if not np.isnan(v))
        if n_v >= min_valid:
            valid_v = [v for v in vals if not np.isnan(v)]
            pm = np.mean(valid_v)
            vals_list.append(sum(v if not np.isnan(v) else pm for v in vals))
        else:
            vals_list.append(np.nan)
    return pd.Series(vals_list, index=df.index)

# Scale A: original 5-item (instruction_summary) - racmost, busing, racdif1, racdif2, racdif3(1=racist)
df['scaleA'] = build_scale_imp(df, ['r_racmost', 'r_busing', 'r_racdif1', 'r_racdif2', 'r_racdif3v1'])

# Scale B: correct alpha scale - racmost, busing, racdif1, racdif3(2=racist), racdif4
df['scaleB'] = build_scale_imp(df, ['r_racmost', 'r_busing', 'r_racdif1', 'r_racdif3v2', 'r_racdif4'])

# Scale C: 6-item with racdif3 flipped, racdif4, person-mean imp, scaled to 0-5
df['scaleC'] = build_scale_imp(df, ['r_racmost', 'r_busing', 'r_racdif1', 'r_racdif2', 'r_racdif3v2', 'r_racdif4']) * (5.0/6.0)

# Scale D: 6-item with racdif3 original, racdif4, person-mean imp, scaled to 0-5
df['scaleD'] = build_scale_imp(df, ['r_racmost', 'r_busing', 'r_racdif1', 'r_racdif2', 'r_racdif3v1', 'r_racdif4']) * (5.0/6.0)

# Scale E: strict 5-item (require all 5)
coded5 = ['r_racmost', 'r_busing', 'r_racdif1', 'r_racdif2', 'r_racdif3v1']
df['scaleE'] = df[coded5].sum(axis=1, min_count=5)

# Build DVs
minority_valid = df[minority_genres].isin([1,2,3,4,5]).all(axis=1)
remaining_valid = df[remaining_genres].isin([1,2,3,4,5]).all(axis=1)

df['dv_minority'] = np.nan
df.loc[minority_valid, 'dv_minority'] = (df.loc[minority_valid, minority_genres] >= 4).sum(axis=1)
df['dv_remaining'] = np.nan
df.loc[remaining_valid, 'dv_remaining'] = (df.loc[remaining_valid, remaining_genres] >= 4).sum(axis=1)

# Build other IVs
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
df['conservative_protestant'] = (pd.to_numeric(df['fund'], errors='coerce') == 1).astype(int)
df['no_religion'] = (pd.to_numeric(df['relig'], errors='coerce') == 4).astype(int)
df['southern'] = (pd.to_numeric(df['region'], errors='coerce') == 3).astype(int)

predictors = ['racism_score', 'education', 'income_pc', 'occ_prestige',
              'female', 'age_var', 'black', 'hispanic', 'other_race',
              'conservative_protestant', 'no_religion', 'southern']
pred_no_racism = [p for p in predictors if p != 'racism_score']

# Run Model 2 with each scale
print('\n=== Model 2 results with different scales ===')
for scale_name, scale_col in [('A-5item_std', 'scaleA'), ('B-5item_alpha', 'scaleB'),
                                ('C-6item_f_05', 'scaleC'), ('D-6item_o_05', 'scaleD'),
                                ('E-5item_strict', 'scaleE')]:
    df['racism_score'] = df[scale_col]
    vars_needed = ['dv_remaining'] + predictors
    model_df = df[vars_needed].dropna()
    N = len(model_df)

    y = model_df['dv_remaining'].values
    X = model_df[predictors].values

    y_z = (y - y.mean()) / y.std(ddof=1)
    X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

    model = sm.OLS(y_z, sm.add_constant(X_z)).fit()
    racism_beta = model.params[1]
    racism_p = model.pvalues[1]
    black_beta = model.params[7]  # index for black
    black_p = model.pvalues[7]
    r2 = model.rsquared

    racism_stats = model_df['racism_score']
    print(f'{scale_name}: N={N}, racism_beta={racism_beta:.3f}(p={racism_p:.4f}), black={black_beta:.3f}(p={black_p:.4f}), R2={r2:.3f}, mean={racism_stats.mean():.2f}, SD={racism_stats.std(ddof=1):.2f}')

# Also check Model 1 quickly
print('\n=== Model 1 results with different scales ===')
for scale_name, scale_col in [('A-5item_std', 'scaleA'), ('B-5item_alpha', 'scaleB'),
                                ('C-6item_f_05', 'scaleC'), ('D-6item_o_05', 'scaleD')]:
    df['racism_score'] = df[scale_col]
    vars_needed = ['dv_minority'] + predictors
    model_df = df[vars_needed].dropna()
    N = len(model_df)

    y = model_df['dv_minority'].values
    X = model_df[predictors].values

    y_z = (y - y.mean()) / y.std(ddof=1)
    X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

    model = sm.OLS(y_z, sm.add_constant(X_z)).fit()
    racism_beta = model.params[1]
    racism_p = model.pvalues[1]
    r2 = model.rsquared

    print(f'{scale_name}: N={N}, racism_beta={racism_beta:.3f}(p={racism_p:.4f}), R2={r2:.3f}')

# Key question: can we match BOTH models?
# Let's see if using income91 changes things
print('\n=== Testing income91 instead of realinc ===')
df['income91_val'] = pd.to_numeric(df['income91'], errors='coerce')
df['income_pc_91'] = df['income91_val'] / df['hompop']

predictors_91 = ['racism_score', 'education', 'income_pc_91', 'occ_prestige',
                 'female', 'age_var', 'black', 'hispanic', 'other_race',
                 'conservative_protestant', 'no_religion', 'southern']

for scale_name, scale_col in [('C-6item_f_05', 'scaleC'), ('D-6item_o_05', 'scaleD')]:
    df['racism_score'] = df[scale_col]

    for dv, dv_name in [('dv_minority', 'M1'), ('dv_remaining', 'M2')]:
        vars_needed = [dv] + predictors_91
        model_df = df[vars_needed].dropna()
        N = len(model_df)

        y = model_df[dv].values
        X = model_df[predictors_91].values

        y_z = (y - y.mean()) / y.std(ddof=1)
        X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

        model = sm.OLS(y_z, sm.add_constant(X_z)).fit()
        racism_beta = model.params[1]
        black_beta = model.params[7]
        r2 = model.rsquared

        print(f'{scale_name} + income91, {dv_name}: N={N}, racism={racism_beta:.3f}, black={black_beta:.3f}, R2={r2:.3f}')

# Try log income
print('\n=== Testing log(income_pc) ===')
df['log_income_pc'] = np.log(df['income_pc'].clip(lower=1))
predictors_log = ['racism_score', 'education', 'log_income_pc', 'occ_prestige',
                  'female', 'age_var', 'black', 'hispanic', 'other_race',
                  'conservative_protestant', 'no_religion', 'southern']

for scale_name, scale_col in [('C-6item_f_05', 'scaleC'), ('D-6item_o_05', 'scaleD')]:
    df['racism_score'] = df[scale_col]

    for dv, dv_name in [('dv_minority', 'M1'), ('dv_remaining', 'M2')]:
        vars_needed = [dv] + predictors_log
        model_df = df[vars_needed].dropna()
        N = len(model_df)

        y = model_df[dv].values
        X = model_df[predictors_log].values

        y_z = (y - y.mean()) / y.std(ddof=1)
        X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

        model = sm.OLS(y_z, sm.add_constant(X_z)).fit()
        racism_beta = model.params[1]
        black_beta = model.params[7]
        income_beta = model.params[3]
        r2 = model.rsquared

        print(f'{scale_name} + log_inc, {dv_name}: N={N}, racism={racism_beta:.3f}, black={black_beta:.3f}, income={income_beta:.3f}, R2={r2:.3f}')

# Try winsorized income
print('\n=== Testing winsorized income_pc (1-99 percentile) ===')
p1, p99 = df['income_pc'].quantile(0.01), df['income_pc'].quantile(0.99)
df['income_pc_win'] = df['income_pc'].clip(lower=p1, upper=p99)
predictors_win = ['racism_score', 'education', 'income_pc_win', 'occ_prestige',
                  'female', 'age_var', 'black', 'hispanic', 'other_race',
                  'conservative_protestant', 'no_religion', 'southern']

df['racism_score'] = df['scaleD']
for dv, dv_name in [('dv_minority', 'M1'), ('dv_remaining', 'M2')]:
    vars_needed = [dv] + predictors_win
    model_df = df[vars_needed].dropna()
    N = len(model_df)

    y = model_df[dv].values
    X = model_df[predictors_win].values

    y_z = (y - y.mean()) / y.std(ddof=1)
    X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

    model = sm.OLS(y_z, sm.add_constant(X_z)).fit()
    racism_beta = model.params[1]
    black_beta = model.params[7]
    r2 = model.rsquared

    print(f'ScaleD + win_inc, {dv_name}: N={N}, racism={racism_beta:.3f}, black={black_beta:.3f}, R2={r2:.3f}')

# Maybe the issue is with Conservative Protestant coding
# Try using denom variable instead of fund
print('\n=== Testing denom-based Conservative Protestant ===')
df['denom'] = pd.to_numeric(df['denom'], errors='coerce')
# Conservative Protestant denominations in GSS:
# Steensland classification: Southern Baptist (10), other Baptist (11-19), etc.
# Bryson doesn't specify exactly which denominations
# fund=1 means "fundamentalist" which is close to conservative protestant

# Try: denom in specific conservative values
# GSS denom codes: 1=Baptist, 2=Methodist, etc.
# Southern Baptist Convention = denom 14 in full GSS but our data may be recoded
# Let's try fund=1 or fund=2 (moderate + fundamentalist)
df['cons_prot_fund12'] = (df['fund'].isin([1,2])).astype(int)
print(f'fund==1: n={df["conservative_protestant"].sum()}')
print(f'fund in [1,2]: n={df["cons_prot_fund12"].sum()}')

# Try with fund in [1,2]
predictors_f12 = predictors.copy()
predictors_f12[predictors_f12.index('conservative_protestant')] = 'cons_prot_fund12'

df['racism_score'] = df['scaleD']
for dv, dv_name in [('dv_minority', 'M1'), ('dv_remaining', 'M2')]:
    vars_needed = [dv] + predictors_f12
    model_df = df[vars_needed].dropna()
    N = len(model_df)

    y = model_df[dv].values
    X = model_df[predictors_f12].values

    y_z = (y - y.mean()) / y.std(ddof=1)
    X_z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

    model = sm.OLS(y_z, sm.add_constant(X_z)).fit()
    racism_beta = model.params[1]
    cons_prot_beta = model.params[10]
    r2 = model.rsquared

    print(f'ScaleD + fund12, {dv_name}: N={N}, racism={racism_beta:.3f}, cons_prot={cons_prot_beta:.3f}, R2={r2:.3f}')
