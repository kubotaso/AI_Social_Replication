import pandas as pd
import numpy as np

df = pd.read_csv('data/cps_dws.csv')

# The paper Table 1 header says "0-5" for first bin
# But we assumed DWYEARS >= 1. What if the paper includes DWYEARS=0?
# The paper says "years of prior job seniority" starting at 0-5
# So DWYEARS=0 should be included

# Let me try without the DWYEARS>=1 restriction but with DWCLASS private
# This gives more observations

# All possible N-matching combinations:
combos = [
    # (label, mask)
    ("base (no extra filters)",
     (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) &
     (df['DWREAS'].isin([1,2,3])) & (df['EMPSTAT'].isin([10,12])) &
     (df['DWYEARS']<99)),
    ("+ full-time",
     (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) &
     (df['DWREAS'].isin([1,2,3])) & (df['EMPSTAT'].isin([10,12])) &
     (df['DWYEARS']<99) & (df['DWFULLTIME']==2)),
    ("+ full-time + DWYEARS>=1",
     (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) &
     (df['DWREAS'].isin([1,2,3])) & (df['EMPSTAT'].isin([10,12])) &
     (df['DWYEARS']>=1) & (df['DWYEARS']<99) & (df['DWFULLTIME']==2)),
    ("+ full-time + DWCLASS private + DWYEARS>=1",
     (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) &
     (df['DWREAS'].isin([1,2,3])) & (df['EMPSTAT'].isin([10,12])) &
     (df['DWYEARS']>=1) & (df['DWYEARS']<99) & (df['DWFULLTIME']==2) &
     (df['DWCLASS'].isin([1,2]))),
    ("+ DWCLASS private only (no ft, no y>=1)",
     (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) &
     (df['DWREAS'].isin([1,2,3])) & (df['EMPSTAT'].isin([10,12])) &
     (df['DWYEARS']<99) & (df['DWCLASS'].isin([1,2]))),
]

for label, mask in combos:
    s = df[mask]
    print(f"\n{label}: N = {len(s)}")

    # Quick stats for Row 2 (plant closing) and Row 1 (log wage change) to compare
    s_copy = s.copy()
    s_copy['plant_closing'] = (s_copy['DWREAS'] == 1).astype(int)
    s_copy['disp_year'] = s_copy['YEAR'] - s_copy['DWLASTWRK']

    deflator = {1978: 72.2, 1979: 78.6, 1980: 85.7, 1981: 94.0,
                1982: 100.0, 1983: 103.9, 1984: 107.7, 1985: 110.9, 1986: 113.8}

    vw = (s_copy['DWWEEKL']>0) & (s_copy['DWWEEKL']<9000) & \
         (s_copy['DWWEEKC']>0) & (s_copy['DWWEEKC']<9000) & \
         (s_copy['DWLASTWRK']<99)
    ws = s_copy[vw].copy()
    ws['deflator_current'] = ws['YEAR'].map(deflator)
    ws['deflator_prior'] = ws['disp_year'].map(deflator)
    ws = ws.dropna(subset=['deflator_current', 'deflator_prior'])
    if len(ws) > 0:
        ws['log_wc'] = np.log(ws['DWWEEKC']/ws['deflator_current']*100) - np.log(ws['DWWEEKL']/ws['deflator_prior']*100)
        print(f"  Total mean log wc: {ws['log_wc'].mean():.3f} (paper: -.135)")
        print(f"  Total mean plant closing: {s_copy['plant_closing'].mean():.3f} (paper: .390)")

    us = s_copy[s_copy['DWWKSUN']<999]
    if len(us) > 0:
        print(f"  Total mean weeks unemp: {us['DWWKSUN'].mean():.2f} (paper: 20.41)")

# The paper's values are:
# Total mean log wc: -.135
# Total plant closing: .390
# Total weeks unemp: 20.41
# N = 4367

print("\n\n=== PAPER TARGET VALUES ===")
print("Total mean log wc: -.135")
print("Total plant closing: .390")
print("Total weeks unemp: 20.41")
print("N = 4367")

# None of the filtered combinations give plant_closing close to .390
# The base sample gives .460, which is too high
# This suggests the DWREAS coding may be different, or we need a different definition

# Wait - maybe the issue is that 'plant closing' in the paper refers to DWREAS==1 only
# and the DENOMINATOR should include reasons 4-6 as well
# No - the paper says sample is restricted to economic reasons

# Another possibility: the paper says "percentage displaced by plant closing"
# Maybe this includes DWREAS==1 AND some other codes?
# In the original CPS, the codes might be different from IPUMS

# Let me check: what fraction of the base sample has DWREAS==1?
m_base = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
         (df['DWREAS'].isin([1,2,3])) & (df['EMPSTAT'].isin([10,12])) & (df['DWYEARS']<99)
base = df[m_base]
print(f"\nFraction DWREAS==1 in base: {(base['DWREAS']==1).mean():.3f}")
print(f"Fraction DWREAS==1 in ft: {(df.loc[m_base & (df['DWFULLTIME']==2), 'DWREAS']==1).mean():.3f}")

# What if we DON'T restrict to full-time workers?
# N=5691 is too high. Let me try other ways to get to ~4367

# What if the paper restricts to heads of household or a particular MARST?
print(f"\nMARST in base sample:")
print(base['MARST'].value_counts().sort_index())

# Try married only
married = base[base['MARST'].isin([1, 2])]
print(f"Married: N = {len(married)}")

# Try not married
unmarried = base[~base['MARST'].isin([1, 2])]
print(f"Not married: N = {len(unmarried)}")

# What about RACE?
print(f"\nRACE in base sample:")
print(base['RACE'].value_counts().sort_index())

# Try whites only
white = base[base['RACE'] == 100]
print(f"White: N = {len(white)}")

# Check NUMPREC (number in household)
print(f"\nNUMPREC range: {base['NUMPREC'].min()} - {base['NUMPREC'].max()}")
