#!/usr/bin/env python3
import csv
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")

with open(DATA_PATH, 'r') as f:
    reader = csv.reader(f)
    header = [h.strip('"') for h in next(reader)]

for prefix in ['A029','A030','A032','A034','A035','G007','G00','B00','C00','D05','D06','E02']:
    matches = [h for h in header if h.startswith(prefix)]
    if matches:
        print(f'{prefix}*: {matches[:15]}')

# Full list of all A-prefix variables
print("\nAll A-prefix vars:")
a_vars = sorted([h for h in header if h.startswith('A0')])
print(a_vars)

# Full list of all G-prefix variables
print("\nAll G-prefix vars:")
g_vars = sorted([h for h in header if h.startswith('G')])
print(g_vars)

# Full list of all B-prefix variables
print("\nAll B-prefix vars:")
b_vars = sorted([h for h in header if h.startswith('B')])
print(b_vars)

# Full list of all C-prefix variables
print("\nAll C-prefix vars:")
c_vars = sorted([h for h in header if h.startswith('C')])
print(c_vars)

# Full D054-D065
print("\nD054-D065:")
d_vars = sorted([h for h in header if h.startswith('D05') or h.startswith('D06')])
print(d_vars)
