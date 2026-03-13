#!/usr/bin/env python3
"""Parse individual file to get person-year family mappings."""
import pandas as pd, numpy as np, re

# Read .do file to find column positions
with open('psid_raw/ind2023er/IND2023ER.do') as f:
    do_content = f.read()

# Find all ER300xx variables and their positions
# Pattern in Stata .do: N varname start-end
vars_needed = {
    'ER30001': 'id_68',          # 1968 interview number
    'ER30002': 'pn',             # person number
    'ER30003': 'relhead_68',     # relation to head 1968
    'ER30020': 'interview_69',   # 1969 interview number
    'ER30021': 'relhead_69',     # sequence/relation 1969
    'ER30022': 'relhead_69b',    # relation to head 1969
    'ER30001': 'id_68',
}

# More systematically: find interview number variables for each year
# Year -> (interview_var, relhead_var) pattern:
# 1968: ER30001 (interview/id_68), ER30003 (rel to head)
# 1969: ER30020, ER30022
# 1970: ER30043, ER30045
# 1971: ER30067, ER30069
# etc.

year_vars = {}
for varname in ['ER30001', 'ER30002', 'ER30003',  # 1968
                'ER30020', 'ER30021', 'ER30022',  # 1969
                'ER30043', 'ER30044', 'ER30045',  # 1970
                'ER30067', 'ER30068', 'ER30069',  # 1971
                ]:
    match = re.search(rf'(\d+)\s+{varname}\s+(\d+)\s*-\s*(\d+)', do_content)
    if match:
        year_vars[varname] = (int(match.group(2)), int(match.group(3)))
        print(f"  {varname}: cols {match.group(2)}-{match.group(3)}")
    else:
        match2 = re.search(rf'{varname}\s+(\d+)\s*-\s*(\d+)', do_content)
        if match2:
            year_vars[varname] = (int(match2.group(1)), int(match2.group(2)))
            print(f"  {varname}: cols {match2.group(1)}-{match2.group(2)}")

# Now read the individual file with just the columns we need
ind_path = 'psid_raw/ind2023er/IND2023ER.txt'

# Convert 1-based to 0-based colspecs for pd.read_fwf
# pd.read_fwf colspecs: list of (start, end) where start is 0-based, end is exclusive
colspecs = []
col_names = []

# id_68 (ER30001): cols 2-5 -> (1, 5)
colspecs.append((1, 5)); col_names.append('id_68')
# pn (ER30002): cols 6-8 -> (5, 8)
colspecs.append((5, 8)); col_names.append('pn')
# relhead_68 (ER30003): cols 9-9 -> (8, 9)
colspecs.append((8, 9)); col_names.append('relhead_68')

# 1969 interview (ER30020): cols 44-47 -> (43, 47)
colspecs.append((43, 47)); col_names.append('interview_69')
# 1969 seq (ER30021): cols 48-49 -> (47, 49)
colspecs.append((47, 49)); col_names.append('seq_69')
# 1969 relhead (ER30022): cols 50-50 -> (49, 50)
colspecs.append((49, 50)); col_names.append('relhead_69')

# 1970 interview (ER30043): check the .do file for position
if 'ER30043' in year_vars:
    s, e = year_vars['ER30043']
    colspecs.append((s-1, e)); col_names.append('interview_70')

if 'ER30045' in year_vars:
    s, e = year_vars['ER30045']
    colspecs.append((s-1, e)); col_names.append('relhead_70')

if 'ER30067' in year_vars:
    s, e = year_vars['ER30067']
    colspecs.append((s-1, e)); col_names.append('interview_71')

if 'ER30069' in year_vars:
    s, e = year_vars['ER30069']
    colspecs.append((s-1, e)); col_names.append('relhead_71')

print(f"\nReading individual file with {len(colspecs)} columns...")
print(f"Columns: {col_names}")

# Read just first 1000 rows for testing
df_ind = pd.read_fwf(ind_path, colspecs=colspecs, names=col_names, header=None, nrows=1000)
print(f"Read {len(df_ind)} rows")
print(df_ind.head(10))
print(f"\nid_68 range: {df_ind['id_68'].min()} - {df_ind['id_68'].max()}")
print(f"pn range: {df_ind['pn'].min()} - {df_ind['pn'].max()}")

# Person_id = id_68 * 1000 + pn
df_ind['person_id'] = df_ind['id_68'] * 1000 + df_ind['pn']

# Check overlap with our panel
panel = pd.read_csv('data/psid_panel.csv')
panel_pids = set(panel['person_id'].unique())

ind_pids = set(df_ind['person_id'].unique())
overlap = panel_pids & ind_pids
print(f"\nIn first 1000 rows: {len(overlap)} persons overlap with panel")

# Now read ALL rows but only keep persons in our panel
print("\nReading full individual file...")
df_ind_full = pd.read_fwf(ind_path, colspecs=colspecs, names=col_names, header=None)
print(f"Total individuals: {len(df_ind_full)}")
df_ind_full['person_id'] = df_ind_full['id_68'] * 1000 + df_ind_full['pn']

# Keep only our panel persons
df_our = df_ind_full[df_ind_full['person_id'].isin(panel_pids)]
print(f"Our panel persons in individual file: {len(df_our)}")

# Check 1969 interview mapping
if 'interview_69' in df_our.columns:
    has_69 = df_our[df_our['interview_69'] > 0]
    print(f"\nPersons with 1969 interview: {len(has_69)}")
    print(f"1969 interview range: {has_69['interview_69'].min()} - {has_69['interview_69'].max()}")
    # How many are heads in 1969?
    if 'relhead_69' in df_our.columns:
        heads_69 = has_69[has_69['relhead_69'] == 1]
        print(f"Heads in 1969: {len(heads_69)}")

# Check 1968 head status
if 'relhead_68' in df_our.columns:
    heads_68 = df_our[df_our['relhead_68'] == 1]
    print(f"\nHeads in 1968: {len(heads_68)}")

# Check 1970 interview mapping
if 'interview_70' in df_our.columns:
    has_70 = df_our[df_our['interview_70'] > 0]
    print(f"\nPersons with 1970 interview: {len(has_70)}")
