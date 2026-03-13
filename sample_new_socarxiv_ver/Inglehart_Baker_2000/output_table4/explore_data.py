#!/usr/bin/env python3
"""Explore WB data availability for Table 4."""
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

wb = pd.read_csv('data/world_bank_indicators.csv')
ind = wb[wb['indicator'] == 'SL.IND.EMPL.ZS']
srv = wb[wb['indicator'] == 'SL.SRV.EMPL.ZS']

for yr in ['YR1980','YR1985','YR1990','YR1991','YR1995']:
    ni = ind[['economy', yr]].dropna(subset=[yr]).shape[0]
    ns = srv[['economy', yr]].dropna(subset=[yr]).shape[0]
    print(f'{yr}: ind={ni}, srv={ns}')

print('\n---1991 Industrial---')
iv = ind[['economy','YR1991']].dropna(subset=['YR1991']).sort_values('economy')
for _,r in iv.iterrows():
    print(f'  {r.economy}: {r.YR1991:.1f}')

print('\n---1991 Service---')
sv = srv[['economy','YR1991']].dropna(subset=['YR1991']).sort_values('economy')
for _,r in sv.iterrows():
    print(f'  {r.economy}: {r.YR1991:.1f}')

# Check GNP PPP availability
gnp = wb[wb['indicator'] == 'NY.GNP.PCAP.PP.CD']
for yr in ['YR1990','YR1993','YR1995']:
    n = gnp[['economy', yr]].dropna(subset=[yr]).shape[0]
    print(f'\nGNP PPP {yr}: {n} countries')

# Count how many countries have ALL THREE for 1995
gnp95 = set(gnp[gnp['YR1995'].notna()]['economy'])
ind91 = set(ind[ind['YR1991'].notna()]['economy'])
srv91 = set(srv[srv['YR1991'].notna()]['economy'])
print(f'\nComplete data (GNP95+IND91+SRV91): {len(gnp95 & ind91 & srv91)}')
print(f'Countries: {sorted(gnp95 & ind91 & srv91)}')

# Factor scores
from shared_factor_analysis import compute_nation_level_factor_scores
scores, _, _ = compute_nation_level_factor_scores()
print(f'\nFactor scores for {len(scores)} countries')
fs_countries = set(scores['COUNTRY_ALPHA'])
complete = gnp95 & ind91 & srv91 & fs_countries
print(f'Complete with factor scores: {len(complete)}')
print(f'Countries: {sorted(complete)}')

# Print factor scores for all countries
print('\n--- Factor Scores ---')
for _, r in scores.sort_values('COUNTRY_ALPHA').iterrows():
    print(f'  {r.COUNTRY_ALPHA}: trad={r.trad_secrat:+.3f} surv={r.surv_selfexp:+.3f}')
