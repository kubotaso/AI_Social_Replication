import pandas as pd
import numpy as np

df = pd.read_csv('anes_cumulative.csv', low_memory=False)

# Check 1952 proportions
year_df = df[df['VCF0004'] == 1952].copy()
valid_pid = year_df[year_df['VCF0301'].isin([1,2,3,4,5,6,7])].copy()
n_valid = len(valid_pid)
print(f'1952 total valid PID: {n_valid}')

n_strong = len(valid_pid[valid_pid['VCF0301'].isin([1,7])])
n_weak = len(valid_pid[valid_pid['VCF0301'].isin([2,6])])
n_lean = len(valid_pid[valid_pid['VCF0301'].isin([3,5])])
n_pure_ind = len(valid_pid[valid_pid['VCF0301'] == 4])

print(f'Strong: {n_strong} ({n_strong/n_valid:.3f})')
print(f'Weak: {n_weak} ({n_weak/n_valid:.3f})')
print(f'Leaners: {n_lean} ({n_lean/n_valid:.3f})')
print(f'Pure Ind: {n_pure_ind} ({n_pure_ind/n_valid:.3f})')
print()
print(f'Paper proportions: strong=0.391, weak=0.376, leaners=0.176')
print(f'Paper says sum is: {0.391+0.376+0.176:.3f}')

# Try excluding pure independents
n_partisan = n_strong + n_weak + n_lean
print(f'\nPartisan only (excl pure ind): {n_partisan}')
if n_partisan > 0:
    print(f'Strong: {n_strong/n_partisan:.3f}')
    print(f'Weak: {n_weak/n_partisan:.3f}')
    print(f'Leaners: {n_lean/n_partisan:.3f}')

# Paper example: 1.600*0.391 + 0.928*0.376 + 0.902*0.176
print(f'\nPaper example: 1.600*0.391 + 0.928*0.376 + 0.902*0.176 = {1.600*0.391 + 0.928*0.376 + 0.902*0.176:.3f}')

# Try with ALL valid PID including pure ind
prop_s_all = n_strong / n_valid
prop_w_all = n_weak / n_valid
prop_l_all = n_lean / n_valid
print(f'\nWith all valid PID: 1.600*{prop_s_all:.3f} + 0.928*{prop_w_all:.3f} + 0.902*{prop_l_all:.3f} = {1.600*prop_s_all + 0.928*prop_w_all + 0.902*prop_l_all:.3f}')

# Try excluding pure ind
if n_partisan > 0:
    prop_s_part = n_strong / n_partisan
    prop_w_part = n_weak / n_partisan
    prop_l_part = n_lean / n_partisan
    print(f'Excl pure ind: 1.600*{prop_s_part:.3f} + 0.928*{prop_w_part:.3f} + 0.902*{prop_l_part:.3f} = {1.600*prop_s_part + 0.928*prop_w_part + 0.902*prop_l_part:.3f}')

# Check proportions for several years
print('\n\n=== ALL YEARS ===')
pres_years = [1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996]
for year in pres_years:
    year_df = df[df['VCF0004'] == year].copy()
    valid_pid = year_df[year_df['VCF0301'].isin([1,2,3,4,5,6,7])].copy()
    n_valid = len(valid_pid)

    n_strong = len(valid_pid[valid_pid['VCF0301'].isin([1,7])])
    n_weak = len(valid_pid[valid_pid['VCF0301'].isin([2,6])])
    n_lean = len(valid_pid[valid_pid['VCF0301'].isin([3,5])])
    n_pure_ind = len(valid_pid[valid_pid['VCF0301'] == 4])

    # All valid
    ps = n_strong / n_valid if n_valid > 0 else 0
    pw = n_weak / n_valid if n_valid > 0 else 0
    pl = n_lean / n_valid if n_valid > 0 else 0

    # Excl pure ind
    n_part = n_strong + n_weak + n_lean
    ps2 = n_strong / n_part if n_part > 0 else 0
    pw2 = n_weak / n_part if n_part > 0 else 0
    pl2 = n_lean / n_part if n_part > 0 else 0

    print(f'{year}: N={n_valid}, strong={ps:.3f}/{ps2:.3f}, weak={pw:.3f}/{pw2:.3f}, lean={pl:.3f}/{pl2:.3f}, pureInd={n_pure_ind}({n_pure_ind/n_valid:.3f})')
