import pandas as pd
import numpy as np

df = pd.read_csv('data/cps_dws.csv')

m = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
    (df['DWREAS'].isin([1,2,3,4,5,6])) & (df['EMPSTAT'].isin([10,12])) & \
    (df['DWYEARS']<99) & (df['DWFULLTIME']==2) & \
    (df['DWWEEKL']>0) & (df['DWWEEKL']<9000) & \
    (df['DWWEEKC']>0) & (df['DWWEEKC']<9000) & \
    (df['DWLASTWRK']<99)
sample = df[m].copy()
sample['disp_year'] = sample['YEAR'] - sample['DWLASTWRK']

d = {1978: 72.2, 1979: 78.6, 1980: 85.7, 1981: 94.0,
     1982: 100.0, 1983: 103.9, 1984: 107.7, 1985: 110.9, 1986: 113.8}

# Standard deflation: both to base year
# log(DWWEEKC/d[YEAR]) - log(DWWEEKL/d[disp])
# = log(DWWEEKC) - log(d[YEAR]) - log(DWWEEKL) + log(d[disp])
# = log(DWWEEKC/DWWEEKL) + log(d[disp]/d[YEAR])

# "Inflate prior only (to 1984)":
# log(DWWEEKC) - log(DWWEEKL * d[1984] / d[disp])
# = log(DWWEEKC) - log(DWWEEKL) - log(d[1984]) + log(d[disp])
# = log(DWWEEKC/DWWEEKL) + log(d[disp]) - log(d[1984])

# The only difference is that the first uses log(d[disp]) - log(d[YEAR])
# and the second uses log(d[disp]) - log(d[1984])
# For 1984 survey, YEAR = 1984, so they're the same
# For 1986 survey, YEAR = 1986, difference = log(d[1986]) - log(d[1984]) = log(113.8/107.7)

# So "inflate prior to 1984" doesn't deflate 1986 current earnings
# This means the 1986 workers' current earnings are in 1986 dollars
# while 1984 workers' current earnings are in 1984 dollars
# That's inconsistent!

# Actually wait, I made an error in the exploration. Let me re-check
# what the "inflate prior only (to 1984)" actually computed
sample['def_pri'] = sample['disp_year'].map(d)
sample['inflate_prior'] = np.log(sample['DWWEEKC']) - np.log(sample['DWWEEKL'] * d[1984] / sample['def_pri'])
valid = sample.dropna(subset=['def_pri'])
print(f"'Inflate prior to 1984' mean: {valid['inflate_prior'].mean():.3f}")

# For 1984 survey workers:
y84 = valid[valid['YEAR'] == 1984]
y86 = valid[valid['YEAR'] == 1986]
print(f"  1984 workers: {y84['inflate_prior'].mean():.3f} (N={len(y84)})")
print(f"  1986 workers: {y86['inflate_prior'].mean():.3f} (N={len(y86)})")

# Standard deflation by year:
sample['def_cur'] = sample['YEAR'].map(d)
sample['standard'] = np.log(sample['DWWEEKC']/sample['def_cur']) - np.log(sample['DWWEEKL']/sample['def_pri'])
valid2 = sample.dropna(subset=['def_cur', 'def_pri'])
y84_s = valid2[valid2['YEAR'] == 1984]
y86_s = valid2[valid2['YEAR'] == 1986]
print(f"\nStandard deflation mean: {valid2['standard'].mean():.3f}")
print(f"  1984 workers: {y84_s['standard'].mean():.3f} (N={len(y84_s)})")
print(f"  1986 workers: {y86_s['standard'].mean():.3f} (N={len(y86_s)})")

# Nominal:
sample['nominal'] = np.log(sample['DWWEEKC']) - np.log(sample['DWWEEKL'])
print(f"\nNominal mean: {sample['nominal'].mean():.3f}")
print(f"  1984 workers: {sample[sample['YEAR']==1984]['nominal'].mean():.3f}")
print(f"  1986 workers: {sample[sample['YEAR']==1986]['nominal'].mean():.3f}")

# Interesting: in standard deflation, 1984 and 1986 workers should give
# similar real wage changes if the market conditions were similar
# But in nominal terms, 1986 workers would show less negative because
# more inflation occurred (boosting current wages relative to prior)

# What if the paper uses a DIFFERENT deflation approach:
# Deflate DWWEEKL to the survey year (not to base year)
# That is: real_prior = DWWEEKL * d[YEAR] / d[disp_year]
# Then: log_wc = log(DWWEEKC) - log(DWWEEKL * d[YEAR] / d[disp_year])
# = log(DWWEEKC/DWWEEKL) - log(d[YEAR]/d[disp_year])
# This is EXACTLY the same as standard deflation! So that doesn't help.

# Maybe the paper deflates by YEAR of displacement, not by survey year?
# How is displacement year determined?
# DWLASTWRK gives "years ago last worked at lost job"
# So disp_year = YEAR - DWLASTWRK

# But wait: what if DWWEEKL is the weekly wage at the time of loss,
# but the displacement event itself happened BEFORE the last date of work?
# E.g., got laid off and finished working that year
# The wage reflects the job's pay before displacement

# Actually I wonder if the paper treats prior wages as from (YEAR - DWYEARS/52) or something
# Let me try: maybe displacement year is not YEAR - DWLASTWRK but something else

# What if displacement year = YEAR - 1 for the 1984 survey?
# The surveys ask about "past 5 years"
# 1984 DWS covers 1979-1984
# 1986 DWS covers 1981-1986

# Let me check: what is the distribution of disp_year?
print(f"\nDisplacement year distribution:")
print(valid2['disp_year'].value_counts().sort_index())

# Try: assume all earnings are from the same year (current = survey year, prior = year before survey)
# This is crude but let's see
sample['adj_simple'] = sample['nominal'] - np.log(d[1984]) + np.log(d[1983])  # wrong
# Actually this makes no sense

# Let me try: scale the deflation by 0.6 (from the ratio we computed)
sample['scaled'] = sample['nominal'] + 0.6 * (np.log(sample['disp_year'].map(d)) - np.log(sample['YEAR'].map(d)))
valid3 = sample.dropna(subset=['scaled'])
print(f"\nScaled (0.6x) deflation: {valid3['scaled'].mean():.3f}")

# Try: scale to exactly match
# We need: nominal + alpha * (log(d_pri) - log(d_cur)) = -0.135
# -0.062 + alpha * (-0.1094) = -0.135
# alpha * (-0.1094) = -0.073
# alpha = 0.073 / 0.1094 = 0.667
sample['scaled2'] = sample['nominal'] + 0.667 * (np.log(sample['disp_year'].map(d)) - np.log(sample['YEAR'].map(d)))
valid4 = sample.dropna(subset=['scaled2'])
print(f"Scaled (0.667x) deflation: {valid4['scaled2'].mean():.3f}")

# What would a deflator series look like if we needed 0.667x the inflation?
# Maybe the issue is that DWWEEKL is top-coded and I'm not handling that
# Or maybe some wages are already partially adjusted

# Actually, let me reconsider: maybe we should NOT require DWFULLTIME==2
# and instead match the N differently

# What if the sample is: DWREAS 1-6, no full-time restriction, but valid DWWEEKL
# (i.e., all displaced workers with reported prior wages)
m2 = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
     (df['DWREAS'].isin([1,2,3,4,5,6])) & (df['EMPSTAT'].isin([10,12])) & \
     (df['DWYEARS']<99) & (df['DWLASTWRK']<99) & \
     (df['DWWEEKL']>0) & (df['DWWEEKL']<9000) & \
     (df['DWWEEKC']>0) & (df['DWWEEKC']<9000)
s2 = df[m2].copy()
s2['disp_year'] = s2['YEAR'] - s2['DWLASTWRK']
s2['plant_closing'] = (s2['DWREAS']==1).astype(int)
s2['def_cur'] = s2['YEAR'].map(d)
s2['def_pri'] = s2['disp_year'].map(d)
s2 = s2.dropna(subset=['def_cur', 'def_pri'])
s2['log_wc'] = np.log(s2['DWWEEKC']/s2['def_cur']) - np.log(s2['DWWEEKL']/s2['def_pri'])

print(f"\n\n=== No full-time restriction, valid wages ===")
print(f"N = {len(s2)}")
print(f"Plant closing: {s2['plant_closing'].mean():.3f}")
print(f"Log wc: {s2['log_wc'].mean():.3f}")

# Maybe the top-coding / bottom-coding of DWWEEKL matters
# Let me check the distribution of DWWEEKL
print(f"\n=== Wage distributions ===")
ws_full = sample.copy()
print(f"DWWEEKL percentiles: 1%={ws_full['DWWEEKL'].quantile(0.01):.0f}, 5%={ws_full['DWWEEKL'].quantile(0.05):.0f}, 50%={ws_full['DWWEEKL'].quantile(0.50):.0f}, 95%={ws_full['DWWEEKL'].quantile(0.95):.0f}, 99%={ws_full['DWWEEKL'].quantile(0.99):.0f}")
print(f"DWWEEKC percentiles: 1%={ws_full['DWWEEKC'].quantile(0.01):.0f}, 5%={ws_full['DWWEEKC'].quantile(0.05):.0f}, 50%={ws_full['DWWEEKC'].quantile(0.50):.0f}, 95%={ws_full['DWWEEKC'].quantile(0.95):.0f}, 99%={ws_full['DWWEEKC'].quantile(0.99):.0f}")

# Maybe the paper trims extreme wage changes?
sample_valid = sample.dropna(subset=['def_cur', 'def_pri']).copy()
sample_valid['log_wc'] = np.log(sample_valid['DWWEEKC']/sample_valid['def_cur']) - np.log(sample_valid['DWWEEKL']/sample_valid['def_pri'])
print(f"\nLog wage change percentiles:")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"  {p}th: {sample_valid['log_wc'].quantile(p/100):.3f}")

# Trim at 1st and 99th percentile
q01 = sample_valid['log_wc'].quantile(0.01)
q99 = sample_valid['log_wc'].quantile(0.99)
trimmed = sample_valid[(sample_valid['log_wc'] >= q01) & (sample_valid['log_wc'] <= q99)]
print(f"\nTrimmed (1-99%): mean = {trimmed['log_wc'].mean():.3f} (N={len(trimmed)})")

# Trim at 5th and 95th
q05 = sample_valid['log_wc'].quantile(0.05)
q95 = sample_valid['log_wc'].quantile(0.95)
trimmed2 = sample_valid[(sample_valid['log_wc'] >= q05) & (sample_valid['log_wc'] <= q95)]
print(f"Trimmed (5-95%): mean = {trimmed2['log_wc'].mean():.3f} (N={len(trimmed2)})")
