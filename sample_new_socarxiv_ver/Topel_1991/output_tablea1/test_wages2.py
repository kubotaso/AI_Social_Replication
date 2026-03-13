#!/usr/bin/env python3
import pandas as pd, numpy as np

df = pd.read_csv('data/psid_panel.csv')
EDUC_CAT = {0:0,1:3,2:7,3:10,4:12,5:12,6:14,7:16,8:17}
df['education_years'] = df['education_clean'].copy()
cat_mask = ~df['year'].isin([1975, 1976])
df.loc[cat_mask, 'education_years'] = df.loc[cat_mask, 'education_clean'].map(EDUC_CAT)

# Basic filters
df = df[(df['age'] >= 18) & (df['age'] <= 60)]
df = df[df['govt_worker'] != 1]
df = df[df['self_employed'] != 1]
df = df[df['agriculture'] != 1]
df = df[df['hourly_wage'] > 0]
df = df[df['education_years'].notna()]
df = df[df['hourly_wage'] < 200]
df = df[df['tenure_topel'] >= 1]

GNP = {1967:33.4,1968:34.8,1969:36.7,1970:38.8,1971:40.5,1972:41.8,1973:44.4,1974:48.9,1975:53.6,1976:56.9,1977:60.6,1978:65.2,1979:72.6,1980:82.4,1981:90.9,1982:100.0}
CPS = {1968:1.000,1969:1.032,1970:1.091,1971:1.115,1972:1.113,1973:1.151,1974:1.167,1975:1.188,1976:1.117,1977:1.121,1978:1.133,1979:1.128,1980:1.128,1981:1.109,1982:1.103,1983:1.089}

df['income_year'] = df['year'] - 1
df['gnp'] = df['income_year'].map(GNP)
df['cps'] = df['year'].map(CPS)

# Check hourly_wage vs wages/hours
df['hw_computed'] = df['wages'] / df['hours']
mask = (df['hw_computed'] > 0) & np.isfinite(df['hw_computed'])
df_m = df[mask]
diff = (df_m['hourly_wage'] - df_m['hw_computed']).abs()
print(f"hourly_wage vs wages/hours: max_diff={diff.max():.4f}, mean_diff={diff.mean():.4f}")
print(f"  hourly_wage mean: {df_m['hourly_wage'].mean():.3f}")
print(f"  wages/hours mean: {df_m['hw_computed'].mean():.3f}")

# Try labor_inc/hours
df['hw_labor'] = df['labor_inc'] / df['hours']
mask2 = (df['hw_labor'] > 0) & np.isfinite(df['hw_labor'])
df_m2 = df[mask2]
rw_labor = np.log(df_m2['hw_labor']) - np.log(df_m2['gnp']/33.4) - np.log(df_m2['cps'])
print(f"\nUsing labor_inc/hours: rw mean={rw_labor.mean():.3f}, sd={rw_labor.std(ddof=0):.3f}")

# Try wages/hours
rw_wages = np.log(df_m['hw_computed']) - np.log(df_m['gnp']/33.4) - np.log(df_m['cps'])
print(f"Using wages/hours: rw mean={rw_wages.mean():.3f}, sd={rw_wages.std(ddof=0):.3f}")

# The paper says "average hourly earnings" - maybe that's labor_inc/hours?
# Or maybe it's the hourly_earnings from the PSID variable directly

# Check the hourly_wage distribution
print(f"\nhourly_wage stats: median={df['hourly_wage'].median():.2f}, mean={df['hourly_wage'].mean():.2f}")
print(f"wages/hours stats: median={df_m['hw_computed'].median():.2f}, mean={df_m['hw_computed'].mean():.2f}")

# Maybe the issue is that we're using nominal wages from the panel's hourly_wage column
# which may already have some adjustment applied? Check the log_hourly_wage column
print(f"\nlog_hourly_wage mean: {df['log_hourly_wage'].mean():.3f}")
print(f"ln(hourly_wage) mean: {np.log(df['hourly_wage']).mean():.3f}")
diff3 = (df['log_hourly_wage'] - np.log(df['hourly_wage'])).abs()
print(f"Diff between log_hourly_wage and ln(hourly_wage): max={diff3.max():.6f}")

# The key question: what deflation formula gives mean=1.131?
# ln(wage) - X = 1.131
# X = ln(wage) - 1.131
X_needed = np.log(df['hourly_wage']).mean() - 1.131
print(f"\nRequired deflation value: {X_needed:.3f}")
print(f"ln(GNP[Y-1]/33.4) + ln(CPS[Y]) average: {(np.log(df['gnp']/33.4) + np.log(df['cps'])).mean():.3f}")
print(f"ln(GNP[Y-1]/100) average: {np.log(df['gnp']/100.0).mean():.3f}")

# What if we use the GNP deflator for the INTERVIEW year, not income year?
df['gnp_interview'] = df['year'].map(GNP)
mask_gnp = df['gnp_interview'].notna()
rw_int = np.log(df.loc[mask_gnp, 'hourly_wage']) - np.log(df.loc[mask_gnp, 'gnp_interview']/33.4) - np.log(df.loc[mask_gnp, 'cps'])
print(f"\nUsing interview year GNP: rw mean={rw_int.mean():.3f}")

# What about using GNP for interview year without CPS?
rw_int_nocps = np.log(df.loc[mask_gnp, 'hourly_wage']) - np.log(df.loc[mask_gnp, 'gnp_interview']/33.4)
print(f"Using interview year GNP, no CPS: rw mean={rw_int_nocps.mean():.3f}")
