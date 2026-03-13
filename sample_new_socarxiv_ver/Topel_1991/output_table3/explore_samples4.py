"""Fine-tune sample to get closer to N=10,685"""
import pandas as pd
import numpy as np

EDUC_MAP = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}

def get_n_detailed(fname, pnum_filter, drop_union_na=False, drop_disabled_na=False,
                   govt_filter=False, min_exp=0, max_exp=999):
    df = pd.read_csv(fname)
    if 'pnum' not in df.columns:
        df['pnum'] = df['person_id'] % 1000
    df = df[df['pnum'].isin(pnum_filter)].copy()

    df['education_years'] = np.nan
    for yr in df['year'].unique():
        mask = df['year'] == yr
        if yr in [1975, 1976]:
            df.loc[mask, 'education_years'] = df.loc[mask, 'education_clean']
        else:
            df.loc[mask, 'education_years'] = df.loc[mask, 'education_clean'].map(EDUC_MAP)
    df = df.dropna(subset=['education_years'])

    df['experience'] = (df['age'] - df['education_years'] - 6).clip(lower=0)
    df['tenure'] = df['tenure_topel']

    df = df[(df['age'] >= 18) & (df['age'] <= 60)]
    df = df[df['hourly_wage'] > 0]
    df = df[(df['self_employed'] == 0) | (df['self_employed'].isna())]
    df = df[df['tenure'] >= 1]
    df = df.dropna(subset=['log_hourly_wage', 'experience', 'tenure'])

    if drop_union_na:
        df = df[df['union_member'].notna()]
    if drop_disabled_na:
        df = df[df['disabled'].notna()]
    if govt_filter:
        df = df[(df['govt_worker'] == 0) | (df['govt_worker'].isna())]
    if min_exp > 0:
        df = df[df['experience'] >= min_exp]
    if max_exp < 999:
        df = df[df['experience'] <= max_exp]

    return len(df), df['person_id'].nunique()

fname = "data/psid_panel_full.csv"
pf = [1, 170, 3]

print(f"Base [1,170,3]: {get_n_detailed(fname, pf)}")
print(f"Drop union NaN: {get_n_detailed(fname, pf, drop_union_na=True)}")
print(f"Drop disabled NaN: {get_n_detailed(fname, pf, drop_disabled_na=True)}")
print(f"Drop both NaN: {get_n_detailed(fname, pf, drop_union_na=True, drop_disabled_na=True)}")
print(f"Govt filter: {get_n_detailed(fname, pf, govt_filter=True)}")
print(f"Exp >= 1: {get_n_detailed(fname, pf, min_exp=1)}")
print(f"Exp <= 40: {get_n_detailed(fname, pf, max_exp=40)}")

# Maybe [1,170,3] with union or disabled NaN drop gives closer to 10,685?
# Also try with main panel
fname2 = "data/psid_panel.csv"
print(f"\nMain panel [1,170,3]: {get_n_detailed(fname2, pf)}")
print(f"Main [1,170,3] drop union NaN: {get_n_detailed(fname2, pf, drop_union_na=True)}")
print(f"Main [1,170,3] drop disabled NaN: {get_n_detailed(fname2, pf, drop_disabled_na=True)}")
print(f"Main [1,170,3] drop both NaN: {get_n_detailed(fname2, pf, drop_union_na=True, drop_disabled_na=True)}")

# Try [1, 3, 4, 5] which gives ~10,240
pf2 = [1, 3, 4, 5]
print(f"\n[1,3,4,5] base: {get_n_detailed(fname, pf2)}")
print(f"[1,3,4,5] drop union NaN: {get_n_detailed(fname, pf2, drop_union_na=True)}")

# Try [1, 3, 4, 5, 6] = 10,338
pf3 = [1, 3, 4, 5, 6]
print(f"\n[1,3,4,5,6] base: {get_n_detailed(fname, pf3)}")
print(f"[1,3,4,5,6] drop union NaN: {get_n_detailed(fname, pf3, drop_union_na=True)}")

# Try [1, 170, 3] with educ code 9 -> drop vs keep
# Current: code 9 drops 90 obs. If we drop them:
# 10788 is with code 9 dropped (maps produce NaN for code 9)
# With code 9 = 17: add back ~some observations
print(f"\n--- With education code 9 -> 17 ---")
EDUC_MAP2 = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17, 9: 17}
def get_n_educ9(fname, pnum_filter):
    df = pd.read_csv(fname)
    if 'pnum' not in df.columns:
        df['pnum'] = df['person_id'] % 1000
    df = df[df['pnum'].isin(pnum_filter)].copy()
    df['education_years'] = np.nan
    for yr in df['year'].unique():
        mask = df['year'] == yr
        if yr in [1975, 1976]:
            df.loc[mask, 'education_years'] = df.loc[mask, 'education_clean']
        else:
            df.loc[mask, 'education_years'] = df.loc[mask, 'education_clean'].map(EDUC_MAP2)
    df = df.dropna(subset=['education_years'])
    df['experience'] = (df['age'] - df['education_years'] - 6).clip(lower=0)
    df['tenure'] = df['tenure_topel']
    df = df[(df['age'] >= 18) & (df['age'] <= 60)]
    df = df[df['hourly_wage'] > 0]
    df = df[(df['self_employed'] == 0) | (df['self_employed'].isna())]
    df = df[df['tenure'] >= 1]
    df = df.dropna(subset=['log_hourly_wage', 'experience', 'tenure'])
    return len(df), df['person_id'].nunique()

print(f"[1,170,3] educ9=17: {get_n_educ9(fname, [1,170,3])}")
print(f"[1,170] educ9=17: {get_n_educ9(fname, [1,170])}")
print(f"[1,3,170,171] educ9=17: {get_n_educ9(fname, [1,3,170,171])}")

# Maybe [1, 170] with educ9=17 gives a different N
# Or some age restriction difference
print("\n--- Age restrictions ---")
for age_max in [55, 60, 64, 65]:
    df = pd.read_csv(fname)
    df['pnum'] = df['person_id'] % 1000 if 'pnum' not in df.columns else df['pnum']
    df = df[df['pnum'].isin([1,170,3])]
    df['education_years'] = np.nan
    for yr in df['year'].unique():
        mask = df['year'] == yr
        if yr in [1975, 1976]:
            df.loc[mask, 'education_years'] = df.loc[mask, 'education_clean']
        else:
            df.loc[mask, 'education_years'] = df.loc[mask, 'education_clean'].map(EDUC_MAP)
    df = df.dropna(subset=['education_years'])
    df['experience'] = (df['age'] - df['education_years'] - 6).clip(lower=0)
    df['tenure'] = df['tenure_topel']
    df = df[(df['age'] >= 18) & (df['age'] <= age_max)]
    df = df[df['hourly_wage'] > 0]
    df = df[(df['self_employed'] == 0) | (df['self_employed'].isna())]
    df = df[df['tenure'] >= 1]
    df = df.dropna(subset=['log_hourly_wage', 'experience', 'tenure'])
    diff = (len(df) - 10685) / 10685 * 100
    print(f"  age <= {age_max}: N={len(df)}, persons={df['person_id'].nunique()}, diff={diff:.1f}%")
