#!/usr/bin/env python3
"""Check which variables are available in the WVS dataset for Table 2."""
import csv
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")

with open(DATA_PATH, 'r') as f:
    reader = csv.reader(f)
    header = [h.strip('"') for h in next(reader)]

vars_needed = ['A001','A003','A003B','A003P','A004','A005','A006','A008','A025','A029',
               'B008','D017','D054','E023','E033','E069_01','E114',
               'F024','F028','F034','F050','F051','F063','F119','F121','F122',
               'G007','A165']

print("=== Variable availability check ===")
for v in sorted(vars_needed):
    found = v in header
    print(f'  {v}: {"FOUND" if found else "MISSING"}')

print("\n=== Prefix searches ===")
for prefix in ['E069','A003','A025','B008','D017','D054','G007','F024','E114','A029']:
    matches = [h for h in header if h.startswith(prefix)]
    if matches:
        print(f'  {prefix}*: {matches[:10]}')
    else:
        print(f'  {prefix}*: NONE')

# Also check EVS
EVS_PATH = os.path.join(BASE, "data/EVS_1990_wvs_format.csv")
if os.path.exists(EVS_PATH):
    with open(EVS_PATH, 'r') as f:
        reader = csv.reader(f)
        evs_header = [h.strip('"') for h in next(reader)]
    print("\n=== EVS variable availability ===")
    for v in sorted(vars_needed):
        found = v in evs_header
        if found:
            print(f'  {v}: FOUND in EVS')
    print(f"\n  EVS prefixes:")
    for prefix in ['E069','A003','A025','B008','D017','D054','G007','F024','E114','A029']:
        matches = [h for h in evs_header if h.startswith(prefix)]
        if matches:
            print(f'  {prefix}*: {matches[:10]}')
