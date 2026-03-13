import pandas as pd
import numpy as np

df = pd.read_csv('data/cps_dws.csv')

deflator = {1978:72.2, 1979:78.6, 1980:85.7, 1981:94.0,
            1982:100.0, 1983:103.9, 1984:107.7, 1985:110.9, 1986:113.8}

# Sample: DWREAS 1-6, FT, valid wages
mask = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
       df['DWREAS'].isin([1,2,3,4,5,6]) & df['EMPSTAT'].isin([10,12]) & \
       (df['DWYEARS']<99) & (df['DWFULLTIME']==2) & \
       (df['DWWEEKL']>0) & (df['DWWEEKL']<9000) & \
       (df['DWWEEKC']>0) & (df['DWWEEKC']<9000)
sample = df[mask].copy()
sample['disp_year'] = sample['YEAR'] - sample['DWLASTWRK']
sample['tenure_bin'] = pd.cut(sample['DWYEARS'], bins=[-0.1,5,10,20,100],
                              labels=['0-5','6-10','11-20','21+'])

# HYPOTHESIS 1: DWWEEKL is reported in current dollars (survey year), not displacement year
# If so, no deflation of DWWEEKL is needed
# Both DWWEEKL and DWWEEKC are in the same year's dollars
# So: log_wc = log(DWWEEKC) - log(DWWEEKL) -- nominal change
# The paper says "deflated" but maybe that means the TABLE shows deflated values
# and the deflation was applied to the levels, not the change
# If both earnings are deflated to 1982 dollars, the LOG CHANGE is the same
# as nominal: log(a/P) - log(b/P) = log(a/b)
# So deflation doesn't affect the log change at all if both are in same dollars!

# The question is: are DWWEEKL and DWWEEKC in different year dollars?
# IPUMS documentation says:
# DWWEEKL: "Usual weekly earnings at the lost job" - recalled at survey time
# DWWEEKC: "Usual weekly earnings at current job" - at survey time
# These are both NOMINAL dollars from their respective time periods
# DWWEEKL = what you were earning at the lost job (in dollars of that year)
# DWWEEKC = what you're earning now (in dollars of survey year)

# So if someone was displaced in 1980 and surveyed in 1984:
# DWWEEKL is in 1980 dollars
# DWWEEKC is in 1984 dollars
# The nominal change includes inflation: log(DWWEEKC) - log(DWWEEKL) overstates real gain
# To get real change: deflate both to same year

# BUT: what if respondents report DWWEEKL in CURRENT dollars?
# I.e., they recall "I was making $300/week" but mean current equivalent?
# This would be unusual but possible for recall data
# In that case, no deflation would be needed

# Let me test: if DWWEEKL is in survey-year dollars
# Then: log_wc_nominal = log(DWWEEKC) - log(DWWEEKL) is the real change
sample['lwc_nominal'] = np.log(sample['DWWEEKC']) - np.log(sample['DWWEEKL'])
print("Nominal (DWWEEKL in survey-year dollars):")
for b in ['0-5','6-10','11-20','21+','Total']:
    if b == 'Total':
        d = sample
    else:
        d = sample[sample['tenure_bin']==b]
    m = d['lwc_nominal'].mean()
    se = d['lwc_nominal'].std() / np.sqrt(len(d))
    print(f"  {b}: {m:.3f} ({se:.3f})")

# HYPOTHESIS 2: Standard deflation (DWWEEKL in displacement-year dollars)
ws = sample[sample['DWLASTWRK']<99].copy()
ws['def_cur'] = ws['YEAR'].map(deflator)
ws['def_pri'] = ws['disp_year'].map(deflator)
ws = ws.dropna(subset=['def_cur','def_pri'])
ws['lwc_deflated'] = np.log(ws['DWWEEKC']/ws['def_cur']) - np.log(ws['DWWEEKL']/ws['def_pri'])
print("\nDeflated (standard):")
for b in ['0-5','6-10','11-20','21+','Total']:
    if b == 'Total':
        d = ws
    else:
        d = ws[ws['tenure_bin']==b]
    m = d['lwc_deflated'].mean()
    se = d['lwc_deflated'].std() / np.sqrt(len(d))
    print(f"  {b}: {m:.3f} ({se:.3f})")

# HYPOTHESIS 3: Maybe DWWEEKL is in the year BEFORE displacement
# e.g., if displaced in 1982, DWWEEKL might be 1981 earnings
# This is plausible if the question asks about "usual earnings" at the job
# and the person's last full year of earnings was the year before displacement

# HYPOTHESIS 4: What if the DWS question wording implies DWWEEKL is
# earnings AT THE TIME OF DISPLACEMENT, not annual earnings from a specific year?
# The "usual weekly earnings" would then be in dollars of the year worked
# This is what we assumed. Let me check by looking at DWWEEKL levels by DWLASTWRK

print("\n\nMean DWWEEKL by DWLASTWRK (years since displacement):")
for lw in range(0, 6):
    sub = sample[sample['DWLASTWRK']==lw]
    if len(sub) > 0:
        print(f"  DWLASTWRK={lw}: mean_DWWEEKL=${sub['DWWEEKL'].mean():.0f}, mean_DWWEEKC=${sub['DWWEEKC'].mean():.0f}, N={len(sub)}")
        # If DWWEEKL is in nominal terms from displacement year,
        # we'd expect lower DWWEEKL for longer-ago displacements (due to inflation)
        # If DWWEEKL is in current dollars, we'd expect similar levels

# The pattern should tell us: if DWWEEKL decreases with DWLASTWRK,
# it's in nominal displacement-year dollars
# If DWWEEKL is roughly constant, it might be in current dollars

# HYPOTHESIS 5: Maybe the paper applies deflation to DWWEEKL differently
# What if the paper assumes the earnings are from the year BEFORE the survey
# for everyone? Or uses a single deflator ratio?

# HYPOTHESIS 6: Maybe I should check the original CPS DWS questionnaire
# to understand what DWWEEKL actually measures

# HYPOTHESIS 7: Try DWREAS 1-3 but with nominal wage change
# Since DWREAS 1-3 gives closer N (5691) and values might be closer
print("\n\nDWREAS 1-3, nominal wage change:")
mask13 = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
         df['DWREAS'].isin([1,2,3]) & df['EMPSTAT'].isin([10,12]) & \
         (df['DWYEARS']<99) & \
         (df['DWWEEKL']>0) & (df['DWWEEKL']<9000) & \
         (df['DWWEEKC']>0) & (df['DWWEEKC']<9000)
s13 = df[mask13].copy()
s13['tenure_bin'] = pd.cut(s13['DWYEARS'], bins=[-0.1,5,10,20,100], labels=['0-5','6-10','11-20','21+'])
s13['lwc_nom'] = np.log(s13['DWWEEKC']) - np.log(s13['DWWEEKL'])
for b in ['0-5','6-10','11-20','21+','Total']:
    if b == 'Total':
        d = s13
    else:
        d = s13[s13['tenure_bin']==b]
    print(f"  {b}: {d['lwc_nom'].mean():.3f}")

print("\nPaper values for reference:")
print("  0-5: -.095  6-10: -.223  11-20: -.282  21+: -.439  Total: -.135")
