"""Check if panel_1960 and panel_1976 have additional variables for lagged PID."""
import pandas as pd
import numpy as np

p60 = pd.read_csv('panel_1960.csv', low_memory=False)
p76 = pd.read_csv('panel_1976.csv', low_memory=False)

# The CDF for these panel respondents shows VCF0004=1960 only
# But in the CDF, the lagged PID comes from matching via VCF0006a to the 1958 wave
# These panel files might already have columns for both waves

# Check all columns for data availability
print('=== Panel 1960 (939 rows, 1030 cols) ===')
# Count non-NA for each column
n_valid = p60.notna().sum()
# Show columns with > 50% valid data
good_cols = n_valid[n_valid > 0.5 * len(p60)].sort_values(ascending=False)
print(f'Columns with >50% valid data: {len(good_cols)}')

# The key question: is there a lagged PID somewhere in these 1030 columns?
# VCF0301 should be current (1960) PID
# Maybe there's a column that has 1958 PID values

# Check if there are any additional columns beyond standard CDF
# Standard CDF columns start with VCF
non_vcf = [c for c in p60.columns if not c.startswith('VCF') and c != 'Version']
print(f'\nNon-VCF columns in 1960 panel: {non_vcf}')

non_vcf76 = [c for c in p76.columns if not c.startswith('VCF') and c != 'Version']
print(f'Non-VCF columns in 1976 panel: {non_vcf76}')

# How many columns are in the standard CDF?
cdf = pd.read_csv('anes_cumulative.csv', low_memory=False, nrows=5)
print(f'\nCDF columns: {len(cdf.columns)}')
print(f'Panel 1960 columns: {len(p60.columns)}')
print(f'Panel 1976 columns: {len(p76.columns)}')

# Find columns in panel files NOT in CDF
cdf_cols = set(cdf.columns)
extra_60 = set(p60.columns) - cdf_cols
extra_76 = set(p76.columns) - cdf_cols
print(f'\nExtra columns in panel_1960 not in CDF: {extra_60}')
print(f'Extra columns in panel_1976 not in CDF: {extra_76}')

# Check if the panel files are just CDF subsets
# If so, we need to get lagged PID from the CDF's lagged wave
# But wait - the panel file for 1960 has 939 rows with VCF0004=1960
# The CDF has 939 rows with VCF0006a < 19600000 (our panel filter)
# So these are the SAME respondents

# However, maybe the panel file contains both current AND lagged variables
# encoded in the same row? Let's check VCF0301 values
print(f'\n1960 VCF0301 distribution:')
print(p60['VCF0301'].value_counts().sort_index())
print(f'Total valid VCF0301: {p60["VCF0301"].notna().sum()} / {len(p60)}')

# Check VCF0006a range
print(f'\n1960 VCF0006a range: {p60["VCF0006a"].min()} to {p60["VCF0006a"].max()}')
print(f'1976 VCF0006a range: {p76["VCF0006a"].min()} to {p76["VCF0006a"].max()}')

# The panel_1960 file has VCF0006a in the 1956xxxx range, meaning these are
# originally from the 1956-58-60 panel study
# The CDF stores each wave separately, so the 1958 wave is separate

# Let's check: does the panel_1960 file have the SAME rows as the CDF for
# VCF0004=1960 & VCF0006a < 19600000?
cdf_full = pd.read_csv('anes_cumulative.csv', low_memory=False)
cdf60_panel = cdf_full[(cdf_full['VCF0004']==1960) & (cdf_full['VCF0006a'] < 19600000)]
print(f'\nCDF 1960 panel rows: {len(cdf60_panel)}')
print(f'Panel file rows: {len(p60)}')

# Check if IDs match
ids_panel = set(p60['VCF0006a'])
ids_cdf = set(cdf60_panel['VCF0006a'])
print(f'IDs in both: {len(ids_panel & ids_cdf)}')
print(f'IDs only in panel file: {len(ids_panel - ids_cdf)}')
print(f'IDs only in CDF: {len(ids_cdf - ids_panel)}')

# Same for 1976
cdf76_panel = cdf_full[(cdf_full['VCF0004']==1976) & (cdf_full['VCF0006a'] < 19760000)]
ids_p76 = set(p76['VCF0006a'])
ids_cdf76 = set(cdf76_panel['VCF0006a'])
print(f'\n1976 CDF panel rows: {len(cdf76_panel)}')
print(f'1976 Panel file rows: {len(p76)}')
print(f'IDs in both: {len(ids_p76 & ids_cdf76)}')
print(f'IDs only in panel file: {len(ids_p76 - ids_cdf76)}')
print(f'IDs only in CDF: {len(ids_cdf76 - ids_p76)}')
