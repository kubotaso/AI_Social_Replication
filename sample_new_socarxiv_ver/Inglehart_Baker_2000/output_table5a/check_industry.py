import pandas as pd
wb = pd.read_csv('data/world_bank_indicators.csv')
nvi = wb[wb['indicator']=='NV.IND.TOTL.ZS']
for yr in ['YR1975','YR1978','YR1980','YR1985','YR1990','YR1991','YR1995']:
    cnt = nvi[yr].notna().sum()
    if cnt > 0:
        print(f'Industry value added {yr}: {cnt} countries')
ind = wb[wb['indicator']=='SL.IND.EMPL.ZS']
for yr in ['YR1975','YR1978','YR1980','YR1985','YR1990','YR1991','YR1995']:
    cnt = ind[yr].notna().sum()
    if cnt > 0:
        print(f'Industry employment {yr}: {cnt} countries')
