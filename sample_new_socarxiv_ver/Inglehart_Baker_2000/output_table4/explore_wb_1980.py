#!/usr/bin/env python3
import pandas as pd
wb = pd.read_csv('data/world_bank_indicators.csv')
for ind_name in ['NY.GDP.PCAP.PP.KD', 'NY.GNP.PCAP.PP.CD']:
    ind = wb[wb['indicator'] == ind_name]
    print(f'\n{ind_name}:')
    for yr in ['YR1980','YR1985','YR1990','YR1991','YR1993','YR1995']:
        if yr in ind.columns:
            n = ind[yr].notna().sum()
            print(f'  {yr}: {n} countries')
        else:
            print(f'  {yr}: column missing')
    # Print 1990 values if available
    if 'YR1990' in ind.columns:
        v = ind[['economy','YR1990']].dropna(subset=['YR1990']).sort_values('economy')
        print(f'  1990 values ({len(v)} countries):')
        for _,r in v.iterrows():
            print(f'    {r.economy}: {r.YR1990:.0f}')
