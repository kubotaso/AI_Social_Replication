#!/usr/bin/env python3
"""Diagnostic script to understand country intersection for Table 5a."""

import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load PWT data
pwt = pd.read_excel(os.path.join(BASE_DIR, 'data/pwt56_forweb.xls'), sheet_name='PWT56')
pwt80 = pwt[pwt['Year'] == 1980][['Country', 'RGDPCH']].dropna(subset=['RGDPCH'])

pwt_to_iso = {
    'U.S.A.': 'USA', 'AUSTRALIA': 'AUS', 'NEW ZEALAND': 'NZL', 'CHINA': 'CHN',
    'JAPAN': 'JPN', 'TAIWAN': 'TWN', 'KOREA, REP.': 'KOR', 'TURKEY': 'TUR',
    'BANGLADESH': 'BGD', 'INDIA': 'IND', 'PAKISTAN': 'PAK', 'PHILIPPINES': 'PHL',
    'U.K.': 'GBR', 'GERMANY, EAST': 'DEU_E', 'GERMANY, WEST': 'DEU_W',
    'SWITZERLAND': 'CHE', 'NORWAY': 'NOR', 'SWEDEN': 'SWE', 'FINLAND': 'FIN',
    'SPAIN': 'ESP', 'POLAND': 'POL', 'BULGARIA': 'BGR', 'NIGERIA': 'NGA',
    'SOUTH AFRICA': 'ZAF', 'GHANA': 'GHA', 'ARGENTINA': 'ARG', 'BRAZIL': 'BRA',
    'CHILE': 'CHL', 'COLOMBIA': 'COL', 'DOMINICAN REP.': 'DOM', 'MEXICO': 'MEX',
    'PERU': 'PER', 'PUERTO RICO': 'PRI', 'URUGUAY': 'URY', 'VENEZUELA': 'VEN',
    'CANADA': 'CAN', 'FRANCE': 'FRA', 'ITALY': 'ITA', 'PORTUGAL': 'PRT',
    'NETHERLANDS': 'NLD', 'BELGIUM': 'BEL', 'DENMARK': 'DNK', 'ICELAND': 'ISL',
    'IRELAND': 'IRL', 'AUSTRIA': 'AUT', 'HUNGARY': 'HUN', 'ROMANIA': 'ROU',
    'U.S.S.R.': 'USSR', 'YUGOSLAVIA': 'YUG', 'CZECHOSLOVAKIA': 'CSK',
}

pwt_iso = {}
for _, r in pwt80.iterrows():
    code = pwt_to_iso.get(r['Country'])
    if code:
        pwt_iso[code] = r['RGDPCH']

# Handle special cases
if 'USSR' in pwt_iso:
    pwt_iso['RUS'] = pwt_iso['USSR']
if 'YUG' in pwt_iso:
    pwt_iso['SRB'] = pwt_iso['YUG']
    pwt_iso['HRV'] = pwt_iso['YUG']  # Croatia was part of Yugoslavia
    pwt_iso['SVN'] = pwt_iso['YUG']  # Slovenia was part of Yugoslavia
    pwt_iso['BIH'] = pwt_iso['YUG']  # Bosnia was part of Yugoslavia
    pwt_iso['MKD'] = pwt_iso['YUG']  # Macedonia was part of Yugoslavia
if 'CSK' in pwt_iso:
    pwt_iso['CZE'] = pwt_iso['CSK']
    pwt_iso['SVK'] = pwt_iso['CSK']

print("PWT GDP values (in 1985 intl $) for paper countries:")
for code, val in sorted(pwt_iso.items()):
    if val is not None:
        print(f"  {code}: {val:.0f}")

print(f"\nTotal PWT countries: {len(pwt_iso)}")

# WB industry data
wb = pd.read_csv(os.path.join(BASE_DIR, 'data/world_bank_indicators.csv'))
ind = wb[wb['indicator'] == 'SL.IND.EMPL.ZS']
print(f"\nWB industry countries (N={len(ind)}):", sorted(ind['economy'].tolist()))

# Paper's 65 societies
paper_65 = [
    'ARG', 'ARM', 'AUS', 'AUT', 'AZE', 'BGD', 'BLR', 'BEL', 'BIH', 'BRA',
    'GBR', 'BGR', 'CAN', 'CHL', 'CHN', 'COL', 'HRV', 'CZE', 'DNK', 'DOM',
    'EST', 'FIN', 'FRA', 'GEO', 'DEU_E', 'DEU_W', 'GHA', 'HUN', 'ISL',
    'IND', 'IRL', 'ITA', 'JPN', 'KOR', 'LVA', 'LTU', 'MKD', 'MEX', 'MDA',
    'NLD', 'NZL', 'NGA', 'NIR', 'NOR', 'PAK', 'PER', 'PHL', 'POL', 'PRT',
    'PRI', 'ROU', 'RUS', 'SRB', 'SVK', 'SVN', 'ZAF', 'ESP', 'SWE', 'CHE',
    'TWN', 'TUR', 'UKR', 'URY', 'USA', 'VEN'
]

# WB economies (note: DEU = unified Germany, not DEU_E/DEU_W)
wb_eco = set(ind['economy'].tolist())
# Add special cases
wb_eco.add('DEU_E')  # Will use DEU as proxy
wb_eco.add('DEU_W')
wb_eco.add('TWN')    # Will impute
wb_eco.add('NIR')    # Will use GBR as proxy
wb_eco.add('SRB')    # Will impute from Yugoslavia
wb_eco.add('HRV')    # From Yugoslavia
wb_eco.add('SVN')    # From Yugoslavia

# Which paper countries have BOTH PWT and WB data?
has_pwt = set([c for c in paper_65 if c in pwt_iso and pwt_iso.get(c) is not None])
has_wb = wb_eco
has_both = has_pwt & has_wb
lacks_pwt = set(paper_65) - has_pwt
lacks_wb = set(paper_65) - has_wb

print(f"\nCountries with PWT data (N={len(has_pwt)}):", sorted(has_pwt))
print(f"\nCountries lacking PWT (N={len(lacks_pwt)}):", sorted(lacks_pwt))
print(f"\nCountries lacking WB industry (N={len(lacks_wb)}):", sorted(lacks_wb))
print(f"\nHave both (N={len(has_both)}):", sorted(has_both))

print("\n\nThe paper has N=49. Countries that likely reduce it from 65 to 49:")
print("(lacking PWT 1980 data):", sorted(set(paper_65) - has_pwt))
