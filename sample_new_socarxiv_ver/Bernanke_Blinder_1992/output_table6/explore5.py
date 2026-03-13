import pandas as pd, numpy as np
df = pd.read_csv('bb1992_data.csv', parse_dates=['date'], index_col='date')
print('CPI 1982-01:', df.loc['1982-01-01', 'cpi'])
print('CPI 1982-12:', df.loc['1982-12-01', 'cpi'])
print()
print('NBR nom 1970-01:', df.loc['1970-01-01', 'nonborrowed_reserves'])
print('NBR real 1970-01:', df.loc['1970-01-01', 'nonborrowed_reserves_real'])
print('CPI 1970-01:', df.loc['1970-01-01', 'cpi'])
print('log_NBR_real 1970-01:', df.loc['1970-01-01', 'log_nonborrowed_reserves_real'])
print('exp(log_NBR_real):', np.exp(df.loc['1970-01-01', 'log_nonborrowed_reserves_real']))
nbr_nom = df.loc['1970-01-01', 'nonborrowed_reserves']
cpi_val = df.loc['1970-01-01', 'cpi']
print('NBR_nom * 100 / CPI =', nbr_nom * 100 / cpi_val)
print('NBR_nom / (CPI/100) =', nbr_nom / (cpi_val/100))
print()
# Check what the paper's NBR values would be
# Paper uses 1982 dollars, CPI base 1982-84=100
# Our CPI in 1982 is around 96.5 (1982-84=100 base)
# Or if our CPI uses a different base...
print('All CPI values around 1982:')
for m in ['1982-01-01','1982-06-01','1982-12-01','1983-06-01','1984-01-01']:
    if m in df.index:
        print(f'  {m}: {df.loc[m, "cpi"]:.1f}')

print()
# The paper says STR82 = retail sales in 1982 dollars
# Let's check our retail_sales_real
print('Retail sales real 1982-01:', df.loc['1982-01-01', 'retail_sales_real'])
print('Retail sales nom 1982-01:', df.loc['1982-01-01', 'retail_sales_nominal'])

print()
# What if we construct NBR_real differently - using a different CPI base?
# Standard FRED CPI (CPIAUCSL) has 1982-84=100
# In our data, CPI in 1982 = ~96.5, which matches 1982-84=100
# So our deflation is: NBR_real = NBR_nom / (CPI / 100) * 100 / base_cpi?
# Let me just verify the construction
print('Verifying NBR_real construction:')
for yr in ['1965-01-01', '1970-01-01', '1975-01-01']:
    nbr_n = df.loc[yr, 'nonborrowed_reserves']
    nbr_r = df.loc[yr, 'nonborrowed_reserves_real']
    cpi = df.loc[yr, 'cpi']
    ratio = nbr_r / nbr_n
    print(f'  {yr}: NBR_nom={nbr_n:.2f}, NBR_real={nbr_r:.2f}, CPI={cpi:.1f}, ratio={ratio:.4f}, 100/CPI={100/cpi:.4f}')
