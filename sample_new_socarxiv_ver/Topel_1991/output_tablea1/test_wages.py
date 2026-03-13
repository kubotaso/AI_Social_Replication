#!/usr/bin/env python3
import pandas as pd
import numpy as np

df = pd.read_csv('data/psid_panel.csv')

EDUC_CAT = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}
df['education_years'] = df['education_clean'].copy()
cat_mask = ~df['year'].isin([1975, 1976])
df.loc[cat_mask, 'education_years'] = df.loc[cat_mask, 'education_clean'].map(EDUC_CAT)

df = df[(df['age'] >= 18) & (df['age'] <= 60)].copy()
df = df[df['govt_worker'] != 1].copy()
df = df[df['self_employed'] != 1].copy()
df = df[df['agriculture'] != 1].copy()
df = df[df['hourly_wage'] > 0].copy()
df = df[df['education_years'].notna()].copy()
df = df[df['hourly_wage'] < 200].copy()
df = df[df['tenure_topel'] >= 1].copy()

GNP = {1967:33.4,1968:34.8,1969:36.7,1970:38.8,1971:40.5,1972:41.8,1973:44.4,1974:48.9,1975:53.6,1976:56.9,1977:60.6,1978:65.2,1979:72.6,1980:82.4,1981:90.9,1982:100.0}
CPS = {1968:1.000,1969:1.032,1970:1.091,1971:1.115,1972:1.113,1973:1.151,1974:1.167,1975:1.188,1976:1.117,1977:1.121,1978:1.133,1979:1.128,1980:1.128,1981:1.109,1982:1.103,1983:1.089}

df['income_year'] = df['year'] - 1
df['gnp'] = df['income_year'].map(GNP)
df['cps'] = df['year'].map(CPS)

# Different formulas
df['rw1'] = np.log(df['hourly_wage']) - np.log(df['gnp']/33.4) - np.log(df['cps'])
df['rw2'] = np.log(df['hourly_wage']) - np.log(df['gnp']/100.0) - np.log(df['cps'])
df['rw3'] = np.log(df['hourly_wage']) - np.log(df['cps'])
df['rw4'] = np.log(df['hourly_wage']) - np.log(df['gnp']) - np.log(df['cps'])
df['rw5'] = np.log(df['hourly_wage']) - np.log(df['gnp']/33.4)
# Formula 6: use log_hourly_wage from data - ln(GNP/100)
df['rw6'] = df['log_hourly_wage'] - np.log(df['gnp']/100.0)
# Formula 7: just log_hourly_wage
df['rw7'] = df['log_hourly_wage']

mask = np.isfinite(df['rw1'])
df = df[mask]

print(f'N = {len(df)}')
print(f'Target mean: 1.131, SD: 0.497')
print(f'F1 ln(w)-ln(GNP/33.4)-ln(CPS): mean={df["rw1"].mean():.3f}, sd={df["rw1"].std(ddof=0):.3f}')
print(f'F2 ln(w)-ln(GNP/100)-ln(CPS):  mean={df["rw2"].mean():.3f}, sd={df["rw2"].std(ddof=0):.3f}')
print(f'F3 ln(w)-ln(CPS):              mean={df["rw3"].mean():.3f}, sd={df["rw3"].std(ddof=0):.3f}')
print(f'F4 ln(w)-ln(GNP)-ln(CPS):      mean={df["rw4"].mean():.3f}, sd={df["rw4"].std(ddof=0):.3f}')
print(f'F5 ln(w)-ln(GNP/33.4):         mean={df["rw5"].mean():.3f}, sd={df["rw5"].std(ddof=0):.3f}')
print(f'F6 log_hw-ln(GNP/100):         mean={df["rw6"].mean():.3f}, sd={df["rw6"].std(ddof=0):.3f}')
print(f'F7 log_hw raw:                 mean={df["rw7"].mean():.3f}, sd={df["rw7"].std(ddof=0):.3f}')

# Also try with ER30002 filter
df2 = pd.read_csv('data/psid_panel.csv')
df2['pn'] = df2['person_id'] % 1000
df2 = df2[df2['pn'] < 170].copy()
df2['education_years'] = df2['education_clean'].copy()
cat_mask2 = ~df2['year'].isin([1975, 1976])
df2.loc[cat_mask2, 'education_years'] = df2.loc[cat_mask2, 'education_clean'].map(EDUC_CAT)
df2 = df2[(df2['age'] >= 18) & (df2['age'] <= 60)].copy()
df2 = df2[df2['govt_worker'] != 1].copy()
df2 = df2[df2['self_employed'] != 1].copy()
df2 = df2[df2['agriculture'] != 1].copy()
df2 = df2[df2['hourly_wage'] > 0].copy()
df2 = df2[df2['education_years'].notna()].copy()
df2 = df2[df2['hourly_wage'] < 200].copy()
df2 = df2[df2['tenure_topel'] >= 1].copy()
df2['income_year'] = df2['year'] - 1
df2['gnp'] = df2['income_year'].map(GNP)
df2['cps'] = df2['year'].map(CPS)
df2['rw1'] = np.log(df2['hourly_wage']) - np.log(df2['gnp']/33.4) - np.log(df2['cps'])
mask2 = np.isfinite(df2['rw1'])
df2 = df2[mask2]
print(f'\nWith pn<170 filter:')
print(f'N = {len(df2)}, persons = {df2["person_id"].nunique()}')
print(f'F1 mean={df2["rw1"].mean():.3f}, sd={df2["rw1"].std(ddof=0):.3f}')
print(f'Year counts:')
print(df2['year'].value_counts().sort_index())
