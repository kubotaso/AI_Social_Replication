#!/usr/bin/env python3
"""Check SMSA extraction from raw PSID files."""
import os

# Check the .do files for SMSA variable definitions
for year in [1977, 1978, 1976, 1971, 1972]:
    do_file = f'psid_raw/fam{year}/FAM{year}.do'
    if os.path.exists(do_file):
        with open(do_file, 'r') as f:
            content = f.read()
        # Search for SMSA
        lines = content.split('\n')
        for line in lines:
            if 'smsa' in line.lower() or 'SMSA' in line or 'metropolitan' in line.lower():
                print(f"  {year}: {line.strip()}")
    print()

# Check the build script's SMSA variable definitions
# 1977: smsa = V5206 at cols 10-10 (1-based)
# Let's check if V5206 is actually at col 10
for year in [1977]:
    do_file = f'psid_raw/fam{year}/FAM{year}.do'
    if os.path.exists(do_file):
        with open(do_file, 'r') as f:
            content = f.read()
        lines = content.split('\n')
        for line in lines:
            if 'V5206' in line:
                print(f"  {year} V5206: {line.strip()}")
            if 'V5203' in line:
                print(f"  {year} V5203: {line.strip()}")

# Now let's try reading the raw 1977 file at a wider column range
filepath = 'psid_raw/fam1977/FAM1977.txt'
with open(filepath, 'r') as f:
    for i, line in enumerate(f):
        if i >= 3: break
        # Print first 20 chars
        print(f"1977 line {i} first 20 chars: {repr(line[:20])}")
        # Check column 10 (1-based, 0-based = 9)
        print(f"  col 10 (1-based): {repr(line[9:10])}")
        # Maybe SMSA is elsewhere
        # In 1977, the state is V5203 at cols 6-7
        print(f"  state (cols 6-7): {repr(line[5:7])}")
        # V5206 at col 10 should be SMSA
        # But maybe it's a code like 1=SMSA, 2=Not
        # 0 could mean "not in sample" or "not coded"

# Check with more lines to see if any are non-zero
with open(filepath, 'r') as f:
    smsa_vals = {}
    for i, line in enumerate(f):
        v = line[9:10]
        smsa_vals[v] = smsa_vals.get(v, 0) + 1
    print(f"\n1977 col 10 distribution: {smsa_vals}")

# Check 1976 SMSA - build script says cols 337-339
filepath = 'psid_raw/fam1976/FAM1976.txt'
with open(filepath, 'r') as f:
    smsa_vals = {}
    for i, line in enumerate(f):
        v = line[336:339].strip()
        if v == '': v = 'empty'
        smsa_vals[v] = smsa_vals.get(v, 0) + 1
    print(f"\n1976 cols 337-339 distribution: {smsa_vals}")

# Check 1971 SMSA - build script says V2209 at cols 702-704
filepath = 'psid_raw/fam1971/FAM1971.txt'
with open(filepath, 'r') as f:
    smsa_vals = {}
    for i, line in enumerate(f):
        v = line[701:704].strip()
        if v == '': v = 'empty'
        smsa_vals[v] = smsa_vals.get(v, 0) + 1
    print(f"\n1971 cols 702-704 distribution: {smsa_vals}")

# Also check the variable map file
if os.path.exists('data/psid_variable_map.txt'):
    with open('data/psid_variable_map.txt', 'r') as f:
        content = f.read()
    lines = content.split('\n')
    for line in lines:
        if 'smsa' in line.lower() or 'SMSA' in line or 'metropolitan' in line.lower():
            print(f"  varmap: {line.strip()}")
