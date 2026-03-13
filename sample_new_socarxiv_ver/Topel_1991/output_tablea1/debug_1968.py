#!/usr/bin/env python3
"""Debug 1968 matching issue."""
import pandas as pd, numpy as np

df = pd.read_csv('data/psid_panel.csv')
ids = df['person_id'].unique()
id_68s = sorted(set(pid // 1000 for pid in ids))
pns = sorted(set(pid % 1000 for pid in ids))
print(f"id_68 range: {min(id_68s)} - {max(id_68s)}, count: {len(id_68s)}")
print(f"pn range: {min(pns)} - {max(pns)}")
print(f"First 20 id_68s: {id_68s[:20]}")
print(f"pn values: {pns[:30]}")

# Check 1968 raw file
all_int = set()
with open('psid_raw/fam1968/FAM1968.txt') as f:
    for l in f:
        try:
            all_int.add(int(l[1:5].strip()))
        except:
            pass

print(f"\n1968 file: {len(all_int)} unique interview numbers")
print(f"Range: {min(all_int)} - {max(all_int)}")

overlap = set(id_68s) & all_int
print(f"Overlap: {len(overlap)}/{len(id_68s)}")

# The issue: our person_ids might use a different encoding
# Let me check if id_1968 column exists
if 'id_1968' in df.columns:
    id68_col = df['id_1968'].unique()
    print(f"\nid_1968 column range: {min(id68_col)} - {max(id68_col)}")
    overlap2 = set(id68_col) & all_int
    print(f"Overlap with 1968 file: {len(overlap2)}/{len(id68_col)}")

# Also check if the interview_number field is at position 0-4 instead of 1-4
with open('psid_raw/fam1968/FAM1968.txt') as f:
    line = f.readline()
    print(f"\nFirst line chars 0-10: '{line[0:10]}'")
    print(f"  pos 0-3: '{line[0:4]}' -> {line[0:4].strip()}")
    print(f"  pos 1-4: '{line[1:5]}' -> {line[1:5].strip()}")
    print(f"  pos 0-4: '{line[0:5]}' -> {line[0:5].strip()}")

# Try matching with pos 0-4
all_int2 = set()
with open('psid_raw/fam1968/FAM1968.txt') as f:
    for l in f:
        try:
            all_int2.add(int(l[0:5].strip()))
        except:
            pass
overlap3 = set(id_68s) & all_int2
print(f"\nUsing pos 0-4: overlap = {len(overlap3)}/{len(id_68s)}")
print(f"1968 range (pos 0-4): {min(all_int2)} - {max(all_int2)}")
