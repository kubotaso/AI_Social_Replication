#!/usr/bin/env python3
"""Debug 1968 raw file parsing."""
import pandas as pd, numpy as np

raw_path = 'psid_raw/fam1968/FAM1968.txt'

# First just read a few lines and inspect
with open(raw_path) as f:
    for i in range(3):
        line = f.readline()
        print(f"Line {i}: length={len(line)}")
        print(f"  [0:10] = '{line[0:10]}'")
        print(f"  [1:5]  = '{line[1:5]}'")
        # Try interview number extraction
        try:
            intno = int(line[1:5].strip())
            print(f"  interview_number = {intno}")
        except ValueError as e:
            print(f"  FAILED: {e}")
        # Try age_head at 283-284 (1-based)
        try:
            age = int(line[282:284].strip())
            print(f"  age = {age}")
        except:
            print(f"  age parse failed, chars=[282:284]='{line[282:284]}'")
        # Check line length
        if len(line) < 612:
            print(f"  WARNING: line only {len(line)} chars, need 612 for HE field")

# Count how many lines are long enough
short = 0
total = 0
with open(raw_path) as f:
    for line in f:
        total += 1
        if len(line) < 612:
            short += 1

print(f"\nTotal lines: {total}, short (<612): {short}")

# Now try reading more carefully - all fields
with open(raw_path) as f:
    line = f.readline()

# The build_psid_panel.py uses FAMILY_VARS[1968] which says:
# 'interview_number': ('V2', 2, 5)  -> 1-based cols 2-5 -> python [1:5]
# 'age_head': ('V117', 283, 284) -> python [282:284]
# etc.
# 'hourly_earnings': ('V337', 608, 612) -> python [607:612]

# But wait - the build script's read_fixed_width function converts
# the column specs (start_col, end_col) where start/end are 1-based inclusive
# to python (start_col-1, end_col) for pd.read_fwf

# So for V337 at (608, 612): python cols [607:612] -> 5 chars
# But our line may not be that long

# Let's use pd.read_fwf like the build script does
colspecs = {
    'interview_number': (1, 5),   # 1-based (2,5) -> 0-based (1,5)
    'age_head': (282, 284),       # (283,284) -> (282,284)
    'sex_head': (286, 287),       # (287,287) -> (286,287)
    'race': (361, 362),           # (362,362) -> (361,362)
    'education': (520, 521),      # (521,521) -> (520,521)
    'marital_status': (437, 438), # (438,438) -> (437,438)
    'labor_income': (182, 187),   # (183,187) -> (182,187)
    'hourly_earnings': (607, 612),# (608,612) -> (607,612)
    'annual_hours': (113, 117),   # (114,117) -> (113,117)
    'self_employed': (387, 388),  # (388,388) -> (387,388)
    'union': (500, 501),          # (501,501) -> (500,501)
    'disability': (408, 409),     # (409,409) -> (408,409)
}

names = list(colspecs.keys())
specs = list(colspecs.values())

try:
    df = pd.read_fwf(raw_path, colspecs=specs, names=names, header=None)
    print(f"\nread_fwf succeeded: {len(df)} rows")
    print(df.head())
    print(f"\ninterview_number range: {df['interview_number'].min()} - {df['interview_number'].max()}")
    print(f"age_head range: {df['age_head'].min()} - {df['age_head'].max()}")
    print(f"race distribution: {df['race'].value_counts().sort_index().to_dict()}")
    print(f"sex_head distribution: {df['sex_head'].value_counts().sort_index().to_dict()}")
except Exception as e:
    print(f"read_fwf failed: {e}")
