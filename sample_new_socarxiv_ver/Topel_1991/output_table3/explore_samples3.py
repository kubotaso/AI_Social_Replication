"""Try more pnum combinations to find the right sample"""
import pandas as pd
import numpy as np

EDUC_MAP = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}

def get_n(fname, pnum_filter):
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
    return len(df), df['person_id'].nunique()

fname = "data/psid_panel_full.csv"

# Try many combinations
combos = [
    [1, 170],  # heads only (SRC + Recontact)
    [1, 3, 170, 171],  # heads + sons
    [1, 2, 3, 170, 171],
    [1, 2, 3, 4, 170, 171],
    [1, 2, 3, 4, 5, 170, 171],
    [1, 2, 170, 172],
    [1, 2, 3, 170, 171, 172],
    # pnum 170+ are from later waves - 170=head, 171=wife
    # In PSID: 1=head, 2=wife, 3+=others (kids, etc.)
    # 170=moved-in head, 171=moved-in wife
    [1],  # just pnum=1 heads
    [1, 3],  # heads + first son
    [1, 3, 4],
    [1, 3, 4, 5],
    [1, 3, 4, 5, 6],
    [1, 3, 4, 5, 6, 7],
    [1, 3, 4, 5, 6, 7, 8, 9, 10],
    [170],  # just recontact heads
    [1, 170, 171],
    # Include 20/21 - could be moved-out offspring
    [1, 20, 21, 170],
    [1, 3, 20, 21, 170, 171],
]

print(f"{'Filter':>45} {'N':>7} {'P':>6} {'%diff':>7}")
print("-" * 70)
for pf in combos:
    n, p = get_n(fname, pf)
    diff = (n - 10685) / 10685 * 100
    print(f"{str(pf):>45} {n:>7} {p:>6} {diff:>6.1f}%")

# Also try with the main panel (which already has year dummies etc.)
print("\n--- Main panel ---")
fname2 = "data/psid_panel.csv"
for pf in [[1,170], [1,3,170,171], [1], [1,3]]:
    n, p = get_n(fname2, pf)
    diff = (n - 10685) / 10685 * 100
    print(f"{str(pf):>45} {n:>7} {p:>6} {diff:>6.1f}%")

# The paper says ~1540 persons and N=10,685
# Let's check what filter gives ~1540 persons
print("\n--- Looking for ~1540 persons ---")
# heads only gives 1438 persons (too few by ~100)
# Try heads + some sons
for pf in [[1, 170, 3], [1, 170, 21], [1, 170, 2]]:
    n, p = get_n(fname, pf)
    diff = (n - 10685) / 10685 * 100
    print(f"{str(pf):>45} {n:>7} {p:>6} {diff:>6.1f}%")
