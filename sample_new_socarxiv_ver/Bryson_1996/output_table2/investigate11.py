import pandas as pd
import numpy as np

df = pd.read_csv('gss1993_clean.csv')

# Check if there are non-NA missing codes in racism items
# Maybe 0, 8, 9, etc. are in the data as valid numbers
for v in ['racmost','busing','racdif1','racdif2','racdif3']:
    s = df[v]
    print(f'{v}: dtype={s.dtype}, unique={sorted(s.dropna().unique()) if s.dtype != "object" else sorted(set(s.dropna()))}')
    # Check raw string values
    vals = s.value_counts(dropna=False)
    print(f'  Value counts: {vals.to_dict()}')
    print()

# The data was read from gss1993_clean.csv - check if NAs are truly NA or coded differently
# Let's read a few rows raw
print('\nFirst 5 rows raw for racism items:')
raw = pd.read_csv('gss1993_clean.csv', usecols=['racmost','busing','racdif1','racdif2','racdif3'],
                   dtype=str, nrows=20)
print(raw.to_string())

# Check: how was this CSV created? Were 0/8/9 already converted to NA?
# Count NAs
print('\nNA counts:')
for v in ['racmost','busing','racdif1','racdif2','racdif3']:
    print(f'  {v}: {df[v].isna().sum()} NAs, {df[v].notna().sum()} valid')
