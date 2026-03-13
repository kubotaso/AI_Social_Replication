#!/usr/bin/env python3
"""Check SMSA variable more deeply."""

# For 1968: SMSA is V188 at cols 369-371 (label "NEAREST SMSA")
# For 1969: SMSA is V808 at cols 584-586
# For 1970: SMSA is V1497 at cols 685-687
# For 1971: SMSA is V2209 at cols 702-704 (label "SMSA CODE")
# For 1972: SMSA is V2835 at cols 735-737 (label "NEAREST SMSA")
# For 1977+: "SIZE LGST PLACE SMSA" at col 10 - 0 means not in SMSA

# The issue is: For 1968-1975, the "NEAREST SMSA" or "SMSA CODE" variables
# give SMSA code numbers (like specific SMSA identifiers), not binary in/not-in.
# If the code is > 0, the person is near/in an SMSA.

# For 1977+, size of largest city: 0=not SMSA, 1-9=in SMSA of various sizes
# ALL are 0, which is suspicious.

# Let me check the 1968 raw file
import pandas as pd

for year, colspec, label in [
    (1968, (368, 371), "V188"),
    (1969, (583, 586), "V808"),
    (1970, (684, 687), "V1497"),
    (1971, (701, 704), "V2209"),
    (1972, (734, 737), "V2835"),
    (1973, (474, 477), "V3250"),
    (1974, (527, 530), "V3672"),
    (1975, (336, 339), "V3933"),
]:
    filepath = f'psid_raw/fam{year}/FAM{year}.txt'
    vals = {}
    try:
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                v = line[colspec[0]:colspec[1]].strip()
                if v == '': v = 'empty'
                vals[v] = vals.get(v, 0) + 1
        total = sum(vals.values())
        # Count non-zero/non-empty
        non_zero = sum(v for k, v in vals.items() if k not in ('0', '000', 'empty', ''))
        print(f"{year} {label} (cols {colspec[0]+1}-{colspec[1]}): total={total}, non-zero/empty={non_zero}, top values:",
              dict(sorted(vals.items(), key=lambda x: -x[1])[:10]))
    except Exception as e:
        print(f"{year}: error - {e}")

# Now let's check the .do files for SMSA coding
print("\n=== Checking .do files for SMSA coding ===")
import os
for year in [1968, 1977, 1978]:
    do_file = f'psid_raw/fam{year}/FAM{year}.do'
    if os.path.exists(do_file):
        with open(do_file, 'r') as f:
            content = f.read()
        lines = content.split('\n')
        in_smsa_section = False
        for i, line in enumerate(lines):
            if 'V188' in line and year == 1968:
                for j in range(max(0,i-1), min(len(lines), i+20)):
                    print(f"  1968 line {j}: {lines[j].strip()}")
                print()
                break
            if 'V5206' in line and year == 1977:
                for j in range(max(0,i-1), min(len(lines), i+20)):
                    print(f"  1977 line {j}: {lines[j].strip()}")
                print()
                break
            if 'V5706' in line and year == 1978:
                for j in range(max(0,i-1), min(len(lines), i+20)):
                    print(f"  1978 line {j}: {lines[j].strip()}")
                print()
                break

# Check whether the beale-urban codes or MSA codes in the .do files
# indicate a different variable for metro status
for year in [1977]:
    do_file = f'psid_raw/fam{year}/FAM{year}.do'
    if os.path.exists(do_file):
        with open(do_file, 'r') as f:
            content = f.read().lower()
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'beale' in line or 'urban' in line or 'metro' in line or 'rural' in line:
                print(f"  {year} line {i}: {lines[i].strip()}")
