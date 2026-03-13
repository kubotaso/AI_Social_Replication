#!/usr/bin/env python3
import pandas as pd, numpy as np

df = pd.read_csv('data/psid_panel.csv')
EDUC_CAT = {0:0,1:3,2:7,3:10,4:12,5:12,6:14,7:16,8:17}
df['education_years'] = df['education_clean'].copy()
cat_mask = ~df['year'].isin([1975, 1976])
df.loc[cat_mask, 'education_years'] = df.loc[cat_mask, 'education_clean'].map(EDUC_CAT)
df = df[(df['age'] >= 18) & (df['age'] <= 60)]
df = df[df['govt_worker'] != 1]
df = df[df['self_employed'] != 1]
df = df[df['agriculture'] != 1]
df = df[df['hourly_wage'] > 0]
df = df[df['education_years'].notna()]
df = df[df['hourly_wage'] < 200]
df = df[df['tenure_topel'] >= 1]

# Extended GNP deflator including 1983
GNP = {1967:33.4,1968:34.8,1969:36.7,1970:38.8,1971:40.5,1972:41.8,1973:44.4,
       1974:48.9,1975:53.6,1976:56.9,1977:60.6,1978:65.2,1979:72.6,
       1980:82.4,1981:90.9,1982:100.0,1983:103.9}
CPS = {1968:1.000,1969:1.032,1970:1.091,1971:1.115,1972:1.113,1973:1.151,1974:1.167,
       1975:1.188,1976:1.117,1977:1.121,1978:1.133,1979:1.128,1980:1.128,
       1981:1.109,1982:1.103,1983:1.089}

# Test: use interview year GNP deflator
df['gnp_int'] = df['year'].map(GNP)
df['cps'] = df['year'].map(CPS)
df['gnp_inc'] = (df['year'] - 1).map(GNP)

# Formula: ln(wage) - ln(GNP_interview/33.4) - ln(CPS)
mask = df['gnp_int'].notna() & df['cps'].notna()
df = df[mask].copy()
df['rw_int'] = np.log(df['hourly_wage']) - np.log(df['gnp_int']/33.4) - np.log(df['cps'])
print(f"Interview year GNP (with 1983): rw mean={df['rw_int'].mean():.4f}, sd={df['rw_int'].std(ddof=0):.4f}")
print(f"N={len(df)}, persons={df['person_id'].nunique()}")

# Also try without pn filter
df2 = pd.read_csv('data/psid_panel.csv')
df2['education_years'] = df2['education_clean'].copy()
cat_mask2 = ~df2['year'].isin([1975, 1976])
df2.loc[cat_mask2, 'education_years'] = df2.loc[cat_mask2, 'education_clean'].map(EDUC_CAT)
df2 = df2[(df2['age'] >= 18) & (df2['age'] <= 60)]
df2 = df2[df2['govt_worker'] != 1]
df2 = df2[df2['self_employed'] != 1]
df2 = df2[df2['agriculture'] != 1]
df2 = df2[df2['hourly_wage'] > 0]
df2 = df2[df2['education_years'].notna()]
df2 = df2[df2['hourly_wage'] < 200]
df2 = df2[df2['tenure_topel'] >= 1]
df2['gnp_int'] = df2['year'].map(GNP)
df2['cps'] = df2['year'].map(CPS)
mask2 = df2['gnp_int'].notna() & df2['cps'].notna()
df2 = df2[mask2].copy()
df2['rw_int'] = np.log(df2['hourly_wage']) - np.log(df2['gnp_int']/33.4) - np.log(df2['cps'])
print(f"\nWithout pn filter: rw mean={df2['rw_int'].mean():.4f}, sd={df2['rw_int'].std(ddof=0):.4f}")
print(f"N={len(df2)}, persons={df2['person_id'].nunique()}")

# With pn filter
df3 = pd.read_csv('data/psid_panel.csv')
df3['pn'] = df3['person_id'] % 1000
df3 = df3[df3['pn'] < 170].copy()
df3['education_years'] = df3['education_clean'].copy()
cat_mask3 = ~df3['year'].isin([1975, 1976])
df3.loc[cat_mask3, 'education_years'] = df3.loc[cat_mask3, 'education_clean'].map(EDUC_CAT)
df3 = df3[(df3['age'] >= 18) & (df3['age'] <= 60)]
df3 = df3[df3['govt_worker'] != 1]
df3 = df3[df3['self_employed'] != 1]
df3 = df3[df3['agriculture'] != 1]
df3 = df3[df3['hourly_wage'] > 0]
df3 = df3[df3['education_years'].notna()]
df3 = df3[df3['hourly_wage'] < 200]
df3 = df3[df3['tenure_topel'] >= 1]
df3['gnp_int'] = df3['year'].map(GNP)
df3['cps'] = df3['year'].map(CPS)
mask3 = df3['gnp_int'].notna() & df3['cps'].notna()
df3 = df3[mask3].copy()
df3['rw_int'] = np.log(df3['hourly_wage']) - np.log(df3['gnp_int']/33.4) - np.log(df3['cps'])
print(f"\nWith pn<170 filter: rw mean={df3['rw_int'].mean():.4f}, sd={df3['rw_int'].std(ddof=0):.4f}")
print(f"N={len(df3)}, persons={df3['person_id'].nunique()}")

# Check tenure and experience with pn filter
print(f"\nWith pn<170:")
print(f"  Experience: mean={df3['experience'].mean():.3f} (target 20.021)")
exp = df3['age'] - df3['education_years'] - 6
exp = exp.clip(lower=0)
print(f"  Experience (recomputed): mean={exp.mean():.3f}")
print(f"  Tenure: mean={df3['tenure_topel'].mean():.3f} (target 9.978)")
print(f"  Education: mean={df3['education_years'].mean():.3f} (target 12.645)")
print(f"  Married: mean={df3['married'].mean():.3f} (target 0.925)")
print(f"  Disabled: mean={df3['disabled'].mean():.3f} (target 0.074)")
print(f"  Union: mean={df3['union_member'].mean():.3f} (target 0.344)")
