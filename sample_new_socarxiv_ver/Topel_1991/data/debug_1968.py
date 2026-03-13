import pandas as pd
import numpy as np

BASE = "/Users/administrator/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/Replication_Claude_Topel_v3/psid_raw"

# Read 1968 family file
fam68 = pd.read_fwf(f'{BASE}/fam1968/FAM1968.txt',
    colspecs=[
        (1, 5),   # interview number (cols 2-5)
        (282, 284), # age head (cols 283-284)
        (286, 287), # sex head (cols 287-287)
        (361, 362), # race (cols 362-362)
        (520, 521), # education (cols 521-521)
        (182, 187), # labor income (cols 183-187)
        (607, 612), # hourly earnings (cols 608-612)
        (113, 117), # annual hours (cols 114-117)
        (387, 388), # self employed (cols 388-388)
        (381, 384), # occupation (cols 382-384)
        (384, 387), # industry (cols 385-387)
    ],
    names=['intnum', 'age', 'sex', 'race', 'edu', 'labor_inc', 'hrly',
           'hours', 'self_emp', 'occ', 'ind'],
    header=None
)

print("1968 family file: n =", len(fam68))
print("\nRace distribution:", fam68['race'].value_counts().sort_index().to_dict())
print("Sex distribution:", fam68['sex'].value_counts().sort_index().to_dict())
print("Self-emp distribution:", fam68['self_emp'].value_counts().sort_index().to_dict())
print("Education distribution:", fam68['edu'].value_counts().sort_index().to_dict())
print()

# Check filtering step by step
white = fam68[fam68['race'] == 1]
print(f"After race=white: {len(white)}")

male = white[white['sex'] == 1]
print(f"After sex=male: {len(male)}")

# Age 18-60
age_ok = male[(male['age'] >= 18) & (male['age'] <= 60)]
print(f"After age 18-60: {len(age_ok)}")

# Not self-employed
not_se = age_ok[age_ok['self_emp'] != 1]
print(f"After not self-employed (self_emp != 1): {len(not_se)}")

# Check what self_emp values look like
print("\nSelf-emp for white males age 18-60:")
print(age_ok['self_emp'].value_counts().sort_index().to_dict())

# Not agriculture
def is_ag(row):
    occ = row['occ']
    ind = row['ind']
    if not pd.isna(ind) and 17 <= int(ind) <= 29:
        return True
    if not pd.isna(occ):
        occ = int(occ)
        if 100 <= occ <= 199 or 600 <= occ <= 699:
            return True
    return False

not_ag = not_se[~not_se.apply(is_ag, axis=1)]
print(f"After not agriculture: {len(not_ag)}")

# SRC sample only (id_68 <= 2930)
# But wait - the person_id uses 1968 interview number from INDIVIDUAL file
# Family interview number should be the same
src = not_ag[not_ag['intnum'] <= 2930]
print(f"After SRC only (intnum <= 2930): {len(src)}")
print("Interview number range:", not_ag['intnum'].min(), "-", not_ag['intnum'].max())

# Positive earnings
pos_earn = src[(src['hrly'] > 0) & (src['hrly'] < 99.98)]
print(f"After positive & non-topcoded earnings: {len(pos_earn)}")

# Valid education
val_edu = pos_earn[pos_earn['edu'] < 9]
print(f"After valid education: {len(val_edu)}")

# Positive hours
pos_hrs = val_edu[val_edu['hours'] > 0]
print(f"After positive hours: {len(pos_hrs)}")

print("\nHourly earnings distribution for final sample:")
print(pos_hrs['hrly'].describe())
