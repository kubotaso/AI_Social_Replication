import pandas as pd
pwt = pd.read_excel('data/pwt56_forweb.xls', sheet_name='PWT56')
print('Columns:', list(pwt.columns))
usa = pwt[pwt['Country'] == 'U.S.A.']
usa80 = usa[usa['Year'] == 1980]
print('\nUSA 1980:')
for col in pwt.columns:
    if col not in ['Country', 'Year']:
        val = usa80[col].values[0] if len(usa80) > 0 else 'N/A'
        print(f'  {col}: {val}')
# Check a few other countries
for c in ['JAPAN', 'GERMANY, WEST', 'INDIA', 'CHINA', 'BRAZIL']:
    row = pwt[(pwt['Country']==c) & (pwt['Year']==1980)]
    if len(row) > 0:
        print(f'\n{c} 1980: RGDPCH={row["RGDPCH"].values[0]:.1f}')
        for col in ['RGDPL', 'CGDP', 'POP']:
            if col in pwt.columns:
                v = row[col].values[0]
                print(f'  {col}: {v}')
