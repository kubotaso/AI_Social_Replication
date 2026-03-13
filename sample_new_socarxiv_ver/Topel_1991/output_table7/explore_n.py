import pandas as pd, numpy as np
df = pd.read_csv('data/psid_panel.csv')
print(f"Total: {len(df)}")

# Check what restrictions bring N closer to 13128
EDUC_MAP = {0:0,1:3,2:7,3:10,4:12,5:12,6:14,7:16,8:17,9:17}
df['ed_yr'] = df['education_clean'].copy()
for yr in df['year'].unique():
    m = df['year'] == yr
    if df.loc[m, 'education_clean'].max() <= 9:
        df.loc[m, 'ed_yr'] = df.loc[m, 'education_clean'].map(EDUC_MAP)
df['exp'] = (df['age'] - df['ed_yr'] - 6).clip(lower=1)
df['union'] = df['union_member'].fillna(0)
df['dis'] = df['disabled'].fillna(0)

# Current restrictions that are already applied:
# White males, age 18-60, not self-employed, not agriculture, not govt,
# SRC sample, positive earnings, positive hours, valid age/education,
# positive experience, wage $1-200, tenure >= 1

# Additional restrictions to try:
# 1. Minimum hours
for h in [100, 250, 500, 750, 1000]:
    n = len(df[df['hours'] >= h])
    print(f"Hours >= {h}: N={n}")

# 2. Wage trimming
for p in [0.5, 1.0, 2.0, 3.0]:
    lo = df['hourly_wage'].quantile(p/100)
    hi = df['hourly_wage'].quantile(1-p/100)
    n = len(df[(df['hourly_wage'] >= lo) & (df['hourly_wage'] <= hi)])
    print(f"Wage {p}%-{100-p}%: N={n} (lo={lo:.2f}, hi={hi:.2f})")

# 3. Experience restrictions
for exp_max in [30, 35, 40, 45]:
    n = len(df[df['exp'] <= exp_max])
    print(f"Exp <= {exp_max}: N={n}")

# 4. Union/disabled must be non-missing
n_nomiss = len(df.dropna(subset=['union_member', 'disabled']))
print(f"\nUnion+disabled non-missing: N={n_nomiss}")

# 5. Combinations
s = df[df['hours'] >= 250].copy()
print(f"\nHours >= 250: {len(s)}")
s = s.dropna(subset=['union_member', 'disabled'])
print(f"+ union/disabled non-missing: {len(s)}")

# Target: 13128
