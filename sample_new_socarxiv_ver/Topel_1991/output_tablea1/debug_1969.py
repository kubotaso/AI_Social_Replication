#!/usr/bin/env python3
"""Debug 1969 raw file and check data quality."""
import pandas as pd, numpy as np

# 1969 column specs from build_psid_panel.py FAMILY_VARS[1969]:
# 'interview_number': ('V442', 2, 5) -> python (1, 5)
# 'age_head': ('V1008', 1060, 1061) -> python (1059, 1061)
# 'sex_head': ('V1010', 1063, 1063) -> python (1062, 1063)
# 'race': ('V801', 577, 577) -> python (576, 577)
# 'education': ('V794', 570, 570) -> python (569, 570)
# 'marital_status': ('V607', 347, 347) -> python (346, 347)
# 'labor_income': ('V514', 185, 189) -> python (184, 189)
# 'hourly_earnings': ('V871', 725, 729) -> python (724, 729)
# 'annual_hours': ('V465', 62, 65) -> python (61, 65)
# 'self_employed': ('V641', 393, 393) -> python (392, 393)
# 'union': ('V766', 537, 537) -> python (536, 537)
# 'disability': ('V743', 514, 514) -> python (513, 514)

colspecs_1969 = [
    (1, 5),      # interview_number
    (1059, 1061), # age_head
    (1062, 1063), # sex_head
    (576, 577),   # race
    (569, 570),   # education
    (346, 347),   # marital_status
    (184, 189),   # labor_income
    (724, 729),   # hourly_earnings
    (61, 65),     # annual_hours
    (392, 393),   # self_employed
    (536, 537),   # union
    (513, 514),   # disability
]
names_1969 = ['interview_number', 'age_head', 'sex_head', 'race', 'education',
              'marital_status', 'labor_income', 'hourly_earnings', 'annual_hours',
              'self_employed', 'union', 'disability']

raw_path = 'psid_raw/fam1969/FAM1969.txt'
with open(raw_path) as f:
    line = f.readline()
    print(f"Line length: {len(line)}")

df = pd.read_fwf(raw_path, colspecs=colspecs_1969, names=names_1969, header=None)
print(f"1969: {len(df)} families")
print(f"interview_number range: {df['interview_number'].min()} - {df['interview_number'].max()}")
print(f"age range: {df['age_head'].min()} - {df['age_head'].max()}")
print(f"race: {df['race'].value_counts().sort_index().to_dict()}")
print(f"sex: {df['sex_head'].value_counts().sort_index().to_dict()}")

# Now, the challenge: how to map 1969 interview numbers to person_ids
# In PSID, the family composition may change year to year
# The individual file tracks each person's family assignment by year
# person_id = ER30001 * 1000 + ER30002 (1968 ID + person number)
# For each year, ER300XX gives the interview/family number that person was in

# We need to read the individual file to get the 1969 family assignment
# The individual file format has columns for each year's interview number
# Let's check the .do file for the individual file

import glob
ind_do_files = glob.glob('psid_raw/ind*/*.do')
print(f"\nIndividual file .do files: {ind_do_files}")

# Read the first ~50 lines of the .do file to find variable positions
if ind_do_files:
    with open(ind_do_files[0]) as f:
        do_content = f.read()

    # Look for ER30001 (1968 interview number), ER30002 (person number)
    # and ER30020 (1969 interview number)
    import re
    for varname in ['ER30001', 'ER30002', 'ER30020', 'ER30021', 'ER30022']:
        # Pattern: infix N varname start-end using ...
        match = re.search(rf'(\d+)\s+{varname}\s+(\d+)\s*-\s*(\d+)', do_content)
        if match:
            print(f"  {varname}: start={match.group(2)}, end={match.group(3)}")
        else:
            # Try alternative pattern
            match2 = re.search(rf'{varname}\s+(\d+)\s*-\s*(\d+)', do_content)
            if match2:
                print(f"  {varname}: start={match2.group(1)}, end={match2.group(2)}")
            else:
                # Try looking for generate statements
                match3 = re.search(rf'{varname}.*?(\d+)\s*-\s*(\d+)', do_content[:50000])
                if match3:
                    print(f"  {varname}: found near pos {match3.start()}")
