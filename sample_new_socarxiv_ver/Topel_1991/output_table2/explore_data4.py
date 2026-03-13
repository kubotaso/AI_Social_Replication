#!/usr/bin/env python3
"""Deep data exploration for Table 2."""
import pandas as pd
import numpy as np

df = pd.read_csv('data/psid_panel.csv')
print('=== BASIC INFO ===')
print('Shape:', df.shape)
print('Persons:', df['person_id'].nunique())

# Check id_1968
print('\n=== ID_1968 RANGE ===')
print('min:', df['id_1968'].min(), 'max:', df['id_1968'].max())
print('id_1968 > 2930:', (df['id_1968'] > 2930).sum())
print('id_1968 > 5000:', (df['id_1968'] > 5000).sum())

# What happens if we restrict to SRC sample?
src = df[df['id_1968'] <= 2930]
print(f'\nSRC sample (id_1968 <= 2930): {len(src)} obs, {src["person_id"].nunique()} persons')

# What about age 18-60?
src_age = src[(src['age'] >= 18) & (src['age'] <= 60)]
print(f'SRC + age 18-60: {len(src_age)} obs, {src_age["person_id"].nunique()} persons')

# Check if the full panel has id_1968 info
df_full = pd.read_csv('data/psid_panel_full.csv')
print('\n=== FULL PANEL ===')
print('Shape:', df_full.shape)
print('Persons:', df_full['person_id'].nunique())
if 'id_1968' in df_full.columns:
    print('id_1968 present in full panel')
    print('id_1968 > 2930:', (df_full['id_1968'] > 2930).sum())
else:
    print('id_1968 NOT in full panel')
    # Derive it from person_id
    df_full['id_1968_derived'] = df_full['person_id'] // 1000
    print('Derived id_1968 > 2930:', (df_full['id_1968_derived'] > 2930).sum())

# Education=9 mapping issue
print('\n=== EDUCATION VALUE 9 ===')
educ9 = df[(~df['year'].isin([1975, 1976])) & (df['education_clean'] == 9)]
print(f'Count: {len(educ9)}')
# Value 9 in the PSID education bracket coding typically means NA/DK
# But it could also be a legitimate code. Let's check what persons have this
if len(educ9) > 0:
    # Check same persons in 1975-1976 when education is in actual years
    pids_9 = educ9['person_id'].unique()
    check_75_76 = df[df['person_id'].isin(pids_9) & df['year'].isin([1975, 1976])]
    if len(check_75_76) > 0:
        print(f'  Same persons in 1975-76: educ values = {sorted(check_75_76["education_clean"].unique())}')
        print(f'  Mean educ in 1975-76: {check_75_76["education_clean"].mean():.1f}')

# Check what the panel already has for tenure
print('\n=== TENURE_TOPEL ===')
print('Tenure starts at:', df['tenure_topel'].min())
print('Value counts (first 15):')
print(df['tenure_topel'].value_counts().sort_index().head(15))

# Check the key issue: years covered in the data
print('\n=== YEARS COVERED ===')
print('Years:', sorted(df['year'].unique()))

# Check number of person-year obs in 1968-1970
for y in range(1968, 1975):
    n = len(df[df['year'] == y])
    print(f'  Year {y}: {n} obs')

# Test what N we get with various restrictions
print('\n=== TESTING SAMPLE RESTRICTIONS ===')

# Start with the full dataset
test = df.copy()

# Education recode (handle value 9)
EDUC_MAP = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17, 9: np.nan}
test['education_years'] = test['education_clean'].copy()
cat_mask = ~test['year'].isin([1975, 1976])
test.loc[cat_mask, 'education_years'] = test.loc[cat_mask, 'education_clean'].map(EDUC_MAP)

# Experience
test['experience'] = test['age'] - test['education_years'] - 6
test['experience'] = test['experience'].clip(lower=0)

# Compute within-job first differences
test = test.sort_values(['person_id', 'job_id', 'year']).reset_index(drop=True)
grp = test.groupby(['person_id', 'job_id'])
test['prev_year'] = grp['year'].shift(1)
test['prev_log_wage'] = grp['log_hourly_wage'].shift(1)
test['prev_tenure'] = grp['tenure_topel'].shift(1)
test['prev_experience'] = grp['experience'].shift(1)

within = test[
    (test['prev_year'].notna()) &
    (test['year'] - test['prev_year'] == 1)
].copy()

within['d_log_wage'] = within['log_hourly_wage'] - within['prev_log_wage']

# Remove extreme outliers
within = within[within['d_log_wage'].between(-2, 2)].copy()

print(f'All within-job obs: {len(within)}, persons: {within["person_id"].nunique()}')
print(f'  Mean d_log_wage (nominal): {within["d_log_wage"].mean():.4f}')
print(f'  Std d_log_wage: {within["d_log_wage"].std():.4f}')

# Drop education=9 (missing) observations
within_clean = within[within['education_years'].notna()].copy()
print(f'After dropping educ=9 (NaN): {len(within_clean)}, persons: {within_clean["person_id"].nunique()}')

# Drop experience < 1
within_clean = within_clean[within_clean['experience'] >= 1].copy()
print(f'After experience >= 1: {len(within_clean)}, persons: {within_clean["person_id"].nunique()}')

# Drop prev_experience NaN
within_clean = within_clean[within_clean['prev_experience'].notna()].copy()
print(f'After valid prev_experience: {len(within_clean)}, persons: {within_clean["person_id"].nunique()}')

# What if we filter to tenure >= 2 (since tenure starts at 1 and first within-job obs needs lag)?
print(f'\nWithin-job tenure distribution:')
print(within_clean['tenure_topel'].value_counts().sort_index().head(15))
print(f'Min tenure in within-job: {within_clean["tenure_topel"].min()}')
print(f'Min prev_tenure in within-job: {within_clean["prev_tenure"].min()}')

# Tighter wage trim?
for lo, hi in [(0.5, 1.5), (0.5, 1.0), (-1, 1)]:
    n = len(within_clean[within_clean['d_log_wage'].between(lo, hi)])
    print(f'  d_log_wage in [{lo},{hi}]: {n}')
