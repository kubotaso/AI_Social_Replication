#!/usr/bin/env python3
"""Check data for Table 5a."""
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check Germany coding
df = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                  usecols=['S003','S024','S002VS','COUNTRY_ALPHA'], low_memory=False)

deu = df[df['COUNTRY_ALPHA']=='DEU']
print("=== Germany S024 codes ===")
for s024 in sorted(deu['S024'].unique()):
    sub = deu[deu['S024']==s024]
    print(f"  S024={s024}: n={len(sub)}, waves={sorted(sub['S002VS'].unique())}")

# In WVS: S024 region codes for Germany
# 2763 = West Germany, 2765 = Berlin, 2766 = East Germany, 2767 = unified
# Check EVS
if os.path.exists('data/EVS_1990_wvs_format.csv'):
    evs = pd.read_csv('data/EVS_1990_wvs_format.csv')
    deu_evs = evs[evs['COUNTRY_ALPHA']=='DEU']
    print(f"\n=== EVS Germany ===")
    if 'S024' in deu_evs.columns:
        for s024 in sorted(deu_evs['S024'].unique()):
            sub = deu_evs[deu_evs['S024']==s024]
            print(f"  S024={s024}: n={len(sub)}")
    else:
        print(f"  No S024 column, n={len(deu_evs)}")

# Check WB GNP data
wb = pd.read_csv('data/world_bank_indicators.csv')
gnp = wb[wb['indicator']=='NY.GNP.PCAP.PP.CD']
print("\n=== GNP per capita (PPP current) ===")
for yr in ['YR1990','YR1991','YR1992','YR1993','YR1994','YR1995']:
    cnt = gnp[yr].notna().sum()
    print(f"  {yr}: {cnt} countries")
    if cnt > 0 and yr == 'YR1995':
        vals = gnp[['economy', yr]].dropna().sort_values(yr)
        print(f"  Range: {vals[yr].min():.0f} to {vals[yr].max():.0f}")

# Education data
for ind in ['SE.PRM.ENRR','SE.SEC.ENRR','SE.TER.ENRR']:
    sub = wb[wb['indicator']==ind]
    for yr in ['YR1980','YR1985','YR1990','YR1995']:
        cnt = sub[yr].notna().sum()
        if cnt > 0:
            print(f"{ind} {yr}: {cnt} countries")
            if cnt < 15:
                print(f"  Countries: {sorted(sub[sub[yr].notna()]['economy'].tolist())}")

# Check factor score range from paper
from shared_factor_analysis import compute_nation_level_factor_scores
scores, loadings, means = compute_nation_level_factor_scores()
print(f"\n=== Factor score range ===")
print(f"  trad_secrat: {scores['trad_secrat'].min():.3f} to {scores['trad_secrat'].max():.3f}")
print(f"  surv_selfexp: {scores['surv_selfexp'].min():.3f} to {scores['surv_selfexp'].max():.3f}")
# Paper Figure 3 shows y-axis from -2.2 to 1.8 and x-axis from -2.0 to 2.0
# But these are approximate. The key issue is the SCALE.
# Paper's East Germany ~1.7, Japan ~1.5, Sweden ~1.3
# Our: DEU=3.716, JPN=2.486, SWE=3.611
# So our scale is about 2x the paper's
print(f"\n  Our DEU: {scores[scores['COUNTRY_ALPHA']=='DEU']['trad_secrat'].values[0]:.3f}")
print(f"  Our JPN: {scores[scores['COUNTRY_ALPHA']=='JPN']['trad_secrat'].values[0]:.3f}")
print(f"  Our SWE: {scores[scores['COUNTRY_ALPHA']=='SWE']['trad_secrat'].values[0]:.3f}")
print(f"  Paper approx: E.Germany~1.7, Japan~1.5, Sweden~1.3")
print(f"  Ratio: DEU={3.716/1.5:.2f}x, JPN={2.486/1.5:.2f}x, SWE={3.611/1.3:.2f}x")
