"""Further exploration of sample restrictions"""
import pandas as pd
import numpy as np

EDUC_MAP = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}

df = pd.read_csv("data/psid_panel_full.csv")
df['pnum'] = df['person_id'] % 1000 if 'pnum' not in df.columns else df['pnum']
print("Full file rows:", len(df))
print("Year 1970 rows:", len(df[df['year']==1970]))

# Check if tenure_topel is NaN for year 1970
yr1970 = df[df['year']==1970]
print("Year 1970 tenure_topel non-null:", yr1970['tenure_topel'].notna().sum())
print("Year 1970 tenure_topel >= 1:", (yr1970['tenure_topel'] >= 1).sum())

# Check if the extra rows from full are in year 1970
main = pd.read_csv("data/psid_panel.csv")
print("\nMain person_ids:", len(set(main['person_id'].unique())))
print("Full person_ids:", len(set(df['person_id'].unique())))

main_ids = set(main['person_id'].unique())
full_ids = set(df['person_id'].unique())
only_full = full_ids - main_ids
print("Only in full:", len(only_full))

# Check what years the extra persons appear in
extra = df[df['person_id'].isin(only_full)]
print("Extra persons year distribution:")
print(extra['year'].value_counts().sort_index())

# Check education_clean code 9
print("\n--- Education code 9 analysis ---")
df_full_hs = df[df['pnum'].isin([1,3,170,171])].copy()
print("Before education filter:", len(df_full_hs))

# Education
df_full_hs['education_years'] = np.nan
for yr in df_full_hs['year'].unique():
    mask = df_full_hs['year'] == yr
    if yr in [1975, 1976]:
        df_full_hs.loc[mask, 'education_years'] = df_full_hs.loc[mask, 'education_clean']
    else:
        df_full_hs.loc[mask, 'education_years'] = df_full_hs.loc[mask, 'education_clean'].map(EDUC_MAP)

lost_educ = df_full_hs['education_years'].isna().sum()
print(f"Rows lost to education mapping: {lost_educ}")
print(f"Education code distribution (non-1975/76):")
non_7576 = df_full_hs[~df_full_hs['year'].isin([1975,1976])]
print(non_7576['education_clean'].value_counts().sort_index())

# Try keeping code 9 as some value
df_full_hs2 = df[df['pnum'].isin([1,3,170,171])].copy()
df_full_hs2['education_years'] = np.nan
EDUC_MAP2 = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17, 9: 17}
for yr in df_full_hs2['year'].unique():
    mask = df_full_hs2['year'] == yr
    if yr in [1975, 1976]:
        df_full_hs2.loc[mask, 'education_years'] = df_full_hs2.loc[mask, 'education_clean']
    else:
        df_full_hs2.loc[mask, 'education_years'] = df_full_hs2.loc[mask, 'education_clean'].map(EDUC_MAP2)

df_full_hs2 = df_full_hs2.dropna(subset=['education_years'])
df_full_hs2['experience'] = (df_full_hs2['age'] - df_full_hs2['education_years'] - 6).clip(lower=0)
df_full_hs2['tenure'] = df_full_hs2['tenure_topel']
df_full_hs2 = df_full_hs2[(df_full_hs2['age'] >= 18) & (df_full_hs2['age'] <= 60)]
df_full_hs2 = df_full_hs2[df_full_hs2['hourly_wage'] > 0]
df_full_hs2 = df_full_hs2[(df_full_hs2['self_employed'] == 0) | (df_full_hs2['self_employed'].isna())]
df_full_hs2 = df_full_hs2[df_full_hs2['tenure'] >= 1]
df_full_hs2 = df_full_hs2.dropna(subset=['log_hourly_wage', 'experience', 'tenure'])
print(f"\nWith code 9 -> 17: N={len(df_full_hs2)}, persons={df_full_hs2['person_id'].nunique()}")

# Try including all pnums <= 30 (sons might have higher pnums)
for pn_max in [10, 30]:
    df_test = df[df['pnum'] <= pn_max].copy()
    df_test['education_years'] = np.nan
    for yr in df_test['year'].unique():
        mask = df_test['year'] == yr
        if yr in [1975, 1976]:
            df_test.loc[mask, 'education_years'] = df_test.loc[mask, 'education_clean']
        else:
            df_test.loc[mask, 'education_years'] = df_test.loc[mask, 'education_clean'].map(EDUC_MAP)
    df_test = df_test.dropna(subset=['education_years'])
    df_test['experience'] = (df_test['age'] - df_test['education_years'] - 6).clip(lower=0)
    df_test['tenure'] = df_test['tenure_topel']
    df_test = df_test[(df_test['age'] >= 18) & (df_test['age'] <= 60)]
    df_test = df_test[df_test['hourly_wage'] > 0]
    df_test = df_test[(df_test['self_employed'] == 0) | (df_test['self_employed'].isna())]
    df_test = df_test[df_test['tenure'] >= 1]
    df_test = df_test.dropna(subset=['log_hourly_wage', 'experience', 'tenure'])
    print(f"pnum <= {pn_max}: N={len(df_test)}, persons={df_test['person_id'].nunique()}")

# What about union and disabled - how many do we lose if we drop NaN?
df_hs = df[df['pnum'].isin([1,3,170,171])].copy()
print(f"\nunion NaN: {df_hs['union_member'].isna().sum()}")
print(f"disabled NaN: {df_hs['disabled'].isna().sum()}")
print(f"Both non-NaN: {df_hs[['union_member','disabled']].notna().all(axis=1).sum()}")
