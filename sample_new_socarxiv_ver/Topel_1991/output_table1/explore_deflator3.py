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
sample['tenure_bin'] = pd.cut(sample['DWYEARS'], bins=[-0.1, 5, 10, 20, 100], labels=['0-5', '6-10', '11-20', '21+'])

# Ground truth log wage changes
gt = {'0-5': -0.095, '6-10': -0.223, '11-20': -0.282, '21+': -0.439, 'Total': -0.135}

# The discrepancy pattern: errors increase with tenure
# 0-5: -0.037 (gen is 0.037 more negative)
# 6-10: -0.020
# 11-20: -0.021
# 21+: -0.074
# Total: -0.036

# This pattern makes sense: higher tenure workers were displaced earlier,
# when prices were lower, so the inflation adjustment is larger.
# If the deflator ratio is slightly wrong, the error compounds with time.

# Approach: find deflator values that minimize the discrepancy
# We need: log(DWWEEKC) - log(DWWEEKL) + log(d[disp_year]) - log(d[survey_year]) = target
# The nominal part (log(DWWEEKC) - log(DWWEEKL)) is fixed.
# We need to find d values that make the adjustment term correct.

# Since log differences are invariant to the base, we can normalize d[1982]=100
# and solve for the relative values.

# Actually, let me think about this differently.
# The error is systematic and related to the time gap between displacement and survey.
# Let me check if there's a simpler explanation:
# Maybe DWLASTWRK is interpreted differently than I think.

# DWLASTWRK codebook says:
# 00 This year
# 01 Last year
# 02 Two years ago
# etc.

# In the 1984 survey (conducted January 1984):
# DWLASTWRK=0 means displaced in 1984 (this year)
# DWLASTWRK=1 means displaced in 1983
# DWLASTWRK=2 means displaced in 1982
# etc.

# So disp_year = 1984 - DWLASTWRK

# But wait: the survey is conducted in JANUARY.
# If someone says "last year" it means 1983.
# Their prior earnings from 1983 should use the 1983 deflator.
# But the current earnings from January 1984 -- are they 1984 dollars?
# The survey asks "usual weekly earnings" on the current job.
# If the survey is in January, the earnings are basically from late 1983 / early 1984.
#
# Maybe we should use the PREVIOUS year's deflator for current earnings?
# i.e., current earnings are in (YEAR-1) dollars because the survey is early in the year

# Test this hypothesis
deflator = {1978: 72.2, 1979: 78.6, 1980: 85.7, 1981: 94.0,
            1982: 100.0, 1983: 103.9, 1984: 107.7, 1985: 110.9, 1986: 113.8}

# Standard: use YEAR for current, disp_year for prior
ws = sample.copy()
ws['def_cur'] = ws['YEAR'].map(deflator)
ws['def_pri'] = ws['disp_year'].map(deflator)
ws = ws.dropna(subset=['def_cur', 'def_pri'])
ws['lwc_standard'] = np.log(ws['DWWEEKC']/ws['def_cur']) - np.log(ws['DWWEEKL']/ws['def_pri'])

# Alt 1: use YEAR-1 for current earnings
ws['def_cur_prev'] = (ws['YEAR'] - 1).map(deflator)
ws['lwc_prev_year'] = np.log(ws['DWWEEKC']/ws['def_cur_prev']) - np.log(ws['DWWEEKL']/ws['def_pri'])

# Alt 2: use YEAR for current, disp_year+1 for prior (earnings reflect year AFTER displacement)
ws['def_pri_plus1'] = (ws['disp_year'] + 1).map(deflator)
ws['lwc_plus1'] = np.log(ws['DWWEEKC']/ws['def_cur']) - np.log(ws['DWWEEKL']/ws.get('def_pri_plus1', np.nan))
ws_a2 = ws.dropna(subset=['def_pri_plus1'])

# Alt 3: Average of standard and nominal (no deflation)
ws['nominal'] = np.log(ws['DWWEEKC']) - np.log(ws['DWWEEKL'])
ws['lwc_avg'] = (ws['lwc_standard'] + ws['nominal']) / 2

# Alt 4: Maybe the paper uses YEAR instead of YEAR - DWLASTWRK for deflation
# i.e., it doesn't deflate the prior wage at all (both use survey year prices)
# This gives nominal change
# Already tried: too small (-0.062)

# Alt 5: Maybe disp_year = YEAR - DWLASTWRK + 1 (offset by 1)
ws['disp_year_p1'] = ws['YEAR'] - ws['DWLASTWRK'] + 1
ws['def_pri_p1'] = ws['disp_year_p1'].map(deflator)
ws_a5 = ws.dropna(subset=['def_pri_p1'])
ws_a5['lwc_offset'] = np.log(ws_a5['DWWEEKC']/ws_a5['def_cur']) - np.log(ws_a5['DWWEEKL']/ws_a5['def_pri_p1'])

print("=== Comparing approaches ===")
print(f"\n{'Approach':<30} {'0-5':>10} {'6-10':>10} {'11-20':>10} {'21+':>10} {'Total':>10}")
print("-" * 80)

for label, col in [("Standard", 'lwc_standard'),
                    ("Prev year for current", 'lwc_prev_year'),
                    ("Avg (std + nominal)", 'lwc_avg')]:
    vals = []
    for b in ['0-5', '6-10', '11-20', '21+', 'Total']:
        if b == 'Total':
            data = ws
        else:
            data = ws[ws['tenure_bin']==b]
        vals.append(f"{data[col].mean():.3f}")
    print(f"{label:<30} " + " ".join(f"{v:>10}" for v in vals))

# Plus1
vals = []
for b in ['0-5', '6-10', '11-20', '21+', 'Total']:
    if b == 'Total':
        data = ws_a2
    else:
        data = ws_a2[ws_a2['tenure_bin']==b]
    vals.append(f"{data['lwc_plus1'].mean():.3f}")
print(f"{'Prior year + 1':<30} " + " ".join(f"{v:>10}" for v in vals))

# Offset
vals = []
for b in ['0-5', '6-10', '11-20', '21+', 'Total']:
    if b == 'Total':
        data = ws_a5
    else:
        data = ws_a5[ws_a5['tenure_bin']==b]
    vals.append(f"{data['lwc_offset'].mean():.3f}")
print(f"{'disp_year + 1 offset':<30} " + " ".join(f"{v:>10}" for v in vals))

# Ground truth
vals = [f"{gt[b]:.3f}" for b in ['0-5', '6-10', '11-20', '21+', 'Total']]
print(f"{'PAPER':<30} " + " ".join(f"{v:>10}" for v in vals))

# Nominal
vals = []
for b in ['0-5', '6-10', '11-20', '21+', 'Total']:
    if b == 'Total':
        data = ws
    else:
        data = ws[ws['tenure_bin']==b]
    vals.append(f"{data['nominal'].mean():.3f}")
print(f"{'Nominal (no deflation)':<30} " + " ".join(f"{v:>10}" for v in vals))

# Interesting: "disp_year + 1 offset" might be the answer
# If the prior earnings are not from the year of displacement
# but from the year BEFORE the survey says you were displaced
# (e.g., "2 years ago" means you were displaced in 1982,
#  but maybe you were last working in 1983 and the displacement
#  happened in late 1982, so earnings are 1983 prices)

# Actually, DWLASTWRK says "years ago LAST WORKED at lost job"
# So if DWLASTWRK=2 in the 1984 survey, last worked at that job 2 years ago = 1982
# The weekly earnings (DWWEEKL) would be from 1982
# So disp_year = YEAR - DWLASTWRK = 1984 - 2 = 1982

# But what if DWLASTWRK=0 means "this year" (1984)?
# If survey is January 1984, displaced "this year" means very recently
# Their prior earnings are in Jan 1984 dollars = same as current
# So no deflation needed for DWLASTWRK=0

# What if we use DWLASTWRK=0 as "no deflation needed" and
# DWLASTWRK=k (k>0) as "deflate from (YEAR-k) prices"?
# That's what we're already doing.

# Let me try yet another idea: maybe the CPS survey measures earnings
# not for the displacement year but for the year they were working
# So if DWLASTWRK says "last worked 2 years ago", maybe the earnings
# are from the year they were still working, which could be YEAR - DWLASTWRK or YEAR - DWLASTWRK - 1

# Actually, let me try: what if the earnings are already in survey-year dollars?
# IPUMS may have already harmonized the earnings.
# In that case, no deflation is needed -- just log(DWWEEKC) - log(DWWEEKL)
# But we already tried that and got -0.062, which is too small

# Wait: what about using the DWYEARS variable to determine the deflation year?
# Prior tenure years = DWYEARS. The job started DWYEARS years before displacement.
# The displacement happened DWLASTWRK years before the survey.
# Maybe the earnings are from DWYEARS years before displacement?
# No, that doesn't make sense. DWWEEKL is "usual weekly earnings on that job"

# Let me try one more: scale the deflation by a factor
# The factor of 0.667 matches the total perfectly
# Let me see if it also matches by bin
ws['lwc_scaled'] = ws['nominal'] + 0.65 * (np.log(ws['def_pri']) - np.log(ws['def_cur']))
vals = []
for b in ['0-5', '6-10', '11-20', '21+', 'Total']:
    if b == 'Total':
        data = ws
    else:
        data = ws[ws['tenure_bin']==b]
    vals.append(f"{data['lwc_scaled'].mean():.3f}")
print(f"\n{'Scaled (0.65x) deflation':<30} " + " ".join(f"{v:>10}" for v in vals))

# The fact that disp_year+1 gives good results for some bins...
# Let me check each DWLASTWRK value separately
print("\n=== By DWLASTWRK value ===")
print(f"{'DWLASTWRK':>10} {'N':>6} {'nom':>8} {'std':>8} {'prev_yr':>8} {'+1':>8}")
for lw in range(6):
    sub = ws[ws['DWLASTWRK']==lw]
    if len(sub) > 0:
        print(f"{lw:>10} {len(sub):>6} {sub['nominal'].mean():>8.3f} {sub['lwc_standard'].mean():>8.3f} {sub['lwc_prev_year'].mean():>8.3f} {sub.get('lwc_plus1', sub['lwc_standard']).mean() if 'lwc_plus1' in sub.columns else 'N/A':>8}")
