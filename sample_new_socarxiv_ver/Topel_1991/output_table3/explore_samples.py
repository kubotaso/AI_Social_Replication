"""Explore different sample restrictions to find one giving N close to 10,685"""
import pandas as pd
import numpy as np

EDUC_MAP = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}

def prepare(fname, pnum_filter=None, min_tenure=1, age_range=(18,60)):
    df = pd.read_csv(fname)

    if 'pnum' not in df.columns:
        df['pnum'] = df['person_id'] % 1000

    if pnum_filter is not None:
        df = df[df['pnum'].isin(pnum_filter)].copy()

    # Education
    df['education_years'] = np.nan
    for yr in df['year'].unique():
        mask = df['year'] == yr
        if yr in [1975, 1976]:
            df.loc[mask, 'education_years'] = df.loc[mask, 'education_clean']
        else:
            df.loc[mask, 'education_years'] = df.loc[mask, 'education_clean'].map(EDUC_MAP)
    df = df.dropna(subset=['education_years']).copy()

    # Experience
    df['experience'] = (df['age'] - df['education_years'] - 6).clip(lower=0)
    df['tenure'] = df['tenure_topel'].copy()

    # Restrictions
    df = df[(df['age'] >= age_range[0]) & (df['age'] <= age_range[1])].copy()
    df = df[df['hourly_wage'] > 0].copy()
    df = df[(df['self_employed'] == 0) | (df['self_employed'].isna())].copy()
    df = df[df['tenure'] >= min_tenure].copy()
    df = df.dropna(subset=['log_hourly_wage', 'experience', 'tenure']).copy()

    return df

# Try various sample configs
configs = [
    # (file, pnum_filter, label)
    ("data/psid_panel.csv", [1, 170], "main_heads"),
    ("data/psid_panel.csv", [1, 3, 170, 171], "main_heads+sons"),
    ("data/psid_panel.csv", None, "main_all"),
    ("data/psid_panel_full.csv", [1, 170], "full_heads"),
    ("data/psid_panel_full.csv", [1, 3, 170, 171], "full_heads+sons"),
    ("data/psid_panel_full.csv", None, "full_all"),
    # Try broader pnum for heads - include 20/21
    ("data/psid_panel_full.csv", [1, 20, 170], "full_heads_broad"),
    ("data/psid_panel_full.csv", [1, 3, 20, 21, 170, 171], "full_heads+sons+broad"),
]

print(f"{'Label':>25} {'N':>7} {'Persons':>8} {'%diff':>7}")
print("-" * 55)
for fname, pf, label in configs:
    try:
        df = prepare(fname, pf)
        n = len(df)
        np_ = df['person_id'].nunique()
        diff = (n - 10685) / 10685 * 100
        print(f"{label:>25} {n:>7} {np_:>8} {diff:>6.1f}%")
    except Exception as e:
        print(f"{label:>25} ERROR: {e}")
