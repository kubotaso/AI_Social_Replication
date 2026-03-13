#!/usr/bin/env python3
"""Debug why 1968 matching gave 0 results."""
import pandas as pd, numpy as np

df = pd.read_csv('data/psid_panel.csv')
existing = df[['person_id']].drop_duplicates()
existing['id_68'] = existing['person_id'] // 1000
existing['pn'] = existing['person_id'] % 1000

print("pn value counts:")
print(existing['pn'].value_counts().sort_index())

# The script was filtering pn < 170, but heads can have pn=1 or pn=170+
# Let's check: which pn values correspond to heads?
# In PSID, pn=1 is always the head of the family unit
heads = existing[existing['pn'] == 1]
print(f"\nHeads (pn=1): {len(heads)}")
print(f"All persons: {len(existing)}")

# Try matching heads with 1968 raw data
raw_path = 'psid_raw/fam1968/FAM1968.txt'

# Read 1968 data
# Interview number: cols 2-5 (1-indexed) = python [1:5]
# age_head: cols 283-284 (1-indexed) = python [282:284]
# sex_head: col 287 = python [286:287]
# race: col 362 = python [361:362]
# education: col 521 = python [520:521]
# marital: col 438 = python [437:438]
# labor_income: cols 183-187 = python [182:187]
# hourly_earnings: cols 608-612 = python [607:612]
# annual_hours: cols 114-117 = python [113:117]
# self_employed: col 388 = python [387:388]
# union: col 501 = python [500:501]
# disability: col 409 = python [408:409]
# occupation: cols 382-384 = python [381:384]
# industry: cols 385-387 = python [384:387]

rows = []
with open(raw_path) as f:
    for line in f:
        try:
            intno = int(line[1:5].strip())
            age = int(line[282:284].strip()) if line[282:284].strip() else 0
            sex = int(line[286:287].strip()) if line[286:287].strip() else 0
            race = int(line[361:362].strip()) if line[361:362].strip() else 0
            edu = int(line[520:521].strip()) if line[520:521].strip() else 99
            married = int(line[437:438].strip()) if line[437:438].strip() else 0
            labor_inc = int(line[182:187].strip()) if line[182:187].strip() else 0
            he = int(line[607:612].strip()) if line[607:612].strip() else 0
            hours = int(line[113:117].strip()) if line[113:117].strip() else 0
            se = int(line[387:388].strip()) if line[387:388].strip() else 0
            union = int(line[500:501].strip()) if line[500:501].strip() else 0
            disab = int(line[408:409].strip()) if line[408:409].strip() else 0
            occ = int(line[381:384].strip()) if line[381:384].strip() else 0
            ind = int(line[384:387].strip()) if line[384:387].strip() else 0
            rows.append({
                'interview_number': intno, 'age': age, 'sex': sex,
                'race': race, 'edu': edu, 'married': married,
                'labor_inc': labor_inc, 'he': he, 'hours': hours,
                'self_employed': se, 'union': union, 'disabled': disab,
                'occ': occ, 'ind': ind
            })
        except:
            pass

df_1968 = pd.DataFrame(rows)
print(f"\n1968 raw data: {len(df_1968)} families")
print(f"Interview number range: {df_1968['interview_number'].min()} - {df_1968['interview_number'].max()}")

# Match with heads
matched = heads.merge(df_1968, left_on='id_68', right_on='interview_number', how='inner')
print(f"\nMatched heads: {len(matched)}")

# Apply filters
m = matched.copy()
print(f"Before filters: {len(m)}")
m = m[m['race'] == 1]; print(f"After white: {len(m)}")
m = m[m['sex'] == 1]; print(f"After male: {len(m)}")
m = m[m['age'].between(18, 60)]; print(f"After age 18-60: {len(m)}")
m = m[~m['self_employed'].isin([2, 3])]; print(f"After not SE: {len(m)}")

# Agriculture
is_ag = (m['occ'].between(100, 199)) | (m['occ'].between(600, 699)) | (m['ind'].between(17, 29))
m = m[~is_ag]; print(f"After not ag: {len(m)}")

# Earnings
m = m[m['he'] > 0]; print(f"After positive HE: {len(m)}")
m = m[m['hours'] > 0]; print(f"After positive hours: {len(m)}")

# Education
m = m[m['edu'].notna() & ~m['edu'].isin([9, 99])]; print(f"After valid edu: {len(m)}")

# Check hourly earnings format
print(f"\nHourly earnings (he) stats:")
print(f"  mean: {m['he'].mean():.1f}")
print(f"  median: {m['he'].median():.1f}")
print(f"  min: {m['he'].min()}, max: {m['he'].max()}")
print(f"  First 10: {m['he'].head(10).tolist()}")

# HE in 1968 file: V337 is described as "HOURLY EARNINGS-NEW HD" at cols 608-612
# 5 digits. If values are like 250-500, they're probably in cents (2.50-5.00 dollars)
# If values are like 25000-50000, they're in hundredths of cents (unlikely for 5 digits)
print(f"\nHE distribution:")
for lo, hi in [(0, 100), (100, 500), (500, 1000), (1000, 5000), (5000, 10000), (10000, 99999)]:
    n = len(m[(m['he'] >= lo) & (m['he'] < hi)])
    if n > 0:
        print(f"  [{lo}, {hi}): {n}")

# Also check: government employee not available in 1968 per the build script
# That means we can't filter govt employees for 1968
print(f"\nLabor income stats:")
print(f"  mean: {m['labor_inc'].mean():.0f}")
print(f"  median: {m['labor_inc'].median():.0f}")

# Check wages/hours ratio
m['computed_he'] = m['labor_inc'] / m['hours']
print(f"\nComputed hourly earnings (labor_inc/hours):")
print(f"  mean: {m['computed_he'].mean():.2f}")
print(f"  median: {m['computed_he'].median():.2f}")
