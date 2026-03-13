"""Quick analysis of cell-by-cell mismatches."""

gt_A = {
    'IP':  {'M1': 0.92, 'M2': 0.10, 'BILL': 0.071, 'BOND': 0.26, 'FUNDS': 0.017},
    'CU':  {'M1': 0.74, 'M2': 0.22, 'BILL': 0.16,  'BOND': 0.40, 'FUNDS': 0.031},
    'EMP': {'M1': 0.45, 'M2': 0.27, 'BILL': 0.0040,'BOND': 0.085,'FUNDS': 0.0004},
    'UR':  {'M1': 0.96, 'M2': 0.37, 'BILL': 0.0005,'BOND': 0.024,'FUNDS': 0.0001},
    'HS':  {'M1': 0.50, 'M2': 0.32, 'BILL': 0.52,  'BOND': 0.014,'FUNDS': 0.22},
    'PI':  {'M1': 0.38, 'M2': 0.24, 'BILL': 0.35,  'BOND': 0.59, 'FUNDS': 0.049},
    'RS':  {'M1': 0.64, 'M2': 0.036,'BILL': 0.33,  'BOND': 0.74, 'FUNDS': 0.014},
    'CON': {'M1': 0.96, 'M2': 0.11, 'BILL': 0.12,  'BOND': 0.46, 'FUNDS': 0.0052},
}
gen_A = {
    'IP':  {'M1': 0.6679, 'M2': 0.0001, 'BILL': 0.0014, 'BOND': 0.8877, 'FUNDS': 0.0125},
    'CU':  {'M1': 0.6535, 'M2': 0.1486, 'BILL': 0.0307, 'BOND': 0.3715, 'FUNDS': 0.0142},
    'EMP': {'M1': 0.5908, 'M2': 0.0007, 'BILL': 0.0039, 'BOND': 0.9099, 'FUNDS': 0.0036},
    'UR':  {'M1': 0.4147, 'M2': 0.0073, 'BILL': 0.0004, 'BOND': 0.1090, 'FUNDS': 0.0001},
    'HS':  {'M1': 0.2625, 'M2': 0.2080, 'BILL': 0.4373, 'BOND': 0.0074, 'FUNDS': 0.2065},
    'PI':  {'M1': 0.2637, 'M2': 0.0028, 'BILL': 0.4378, 'BOND': 0.0947, 'FUNDS': 0.0430},
    'RS':  {'M1': 0.8145, 'M2': 0.0009, 'BILL': 0.5134, 'BOND': 0.6137, 'FUNDS': 0.0516},
    'CON': {'M1': 0.9692, 'M2': 0.0039, 'BILL': 0.2974, 'BOND': 0.3505, 'FUNDS': 0.0082},
}
gt_B = {
    'IP':  {'M1': 0.99, 'M2': 0.084,'BILL': 0.0092,'BOND': 0.61, 'FUNDS': 0.0001},
    'CU':  {'M1': 0.96, 'M2': 0.40, 'BILL': 0.025, 'BOND': 0.18, 'FUNDS': 0.0003},
    'EMP': {'M1': 0.57, 'M2': 0.41, 'BILL': 0.0005,'BOND': 0.15, 'FUNDS': 0.0004},
    'UR':  {'M1': 0.56, 'M2': 0.88, 'BILL': 0.0006,'BOND': 0.13, 'FUNDS': 0.0000},
    'HS':  {'M1': 0.34, 'M2': 0.17, 'BILL': 0.73,  'BOND': 0.72, 'FUNDS': 0.11},
    'PI':  {'M1': 0.43, 'M2': 0.095,'BILL': 0.20,  'BOND': 0.91, 'FUNDS': 0.037},
    'RS':  {'M1': 0.96, 'M2': 0.86, 'BILL': 0.27,  'BOND': 0.050,'FUNDS': 0.061},
    'CON': {'M1': 0.79, 'M2': 0.017,'BILL': 0.010, 'BOND': 0.050,'FUNDS': 0.0000},
}
gen_B = {
    'IP':  {'M1': 0.7008, 'M2': 0.0005, 'BILL': 0.0051, 'BOND': 0.6421, 'FUNDS': 0.0027},
    'CU':  {'M1': 0.9896, 'M2': 0.4311, 'BILL': 0.0052, 'BOND': 0.0272, 'FUNDS': 0.0013},
    'EMP': {'M1': 0.2516, 'M2': 0.1189, 'BILL': 0.0026, 'BOND': 0.2769, 'FUNDS': 0.0003},
    'UR':  {'M1': 0.4344, 'M2': 0.7373, 'BILL': 0.0001, 'BOND': 0.0847, 'FUNDS': 0.0000},
    'HS':  {'M1': 0.4700, 'M2': 0.1329, 'BILL': 0.6012, 'BOND': 0.6867, 'FUNDS': 0.1733},
    'PI':  {'M1': 0.2161, 'M2': 0.0151, 'BILL': 0.5358, 'BOND': 0.4178, 'FUNDS': 0.0466},
    'RS':  {'M1': 0.6928, 'M2': 0.4811, 'BILL': 0.2896, 'BOND': 0.1495, 'FUNDS': 0.0072},
    'CON': {'M1': 0.2745, 'M2': 0.0001, 'BILL': 0.0447, 'BOND': 0.0007, 'FUNDS': 0.0011},
}

def s(p):
    if p <= 0.01: return 3
    if p <= 0.05: return 2
    if p <= 0.10: return 1
    return 0

b = {0: 'NS', 1: '10%', 2: '5%', 3: '1%'}

print("=== Panel A mismatches ===")
mis_a = 0
for dep in gt_A:
    for tv in ['M1', 'M2', 'BILL', 'BOND', 'FUNDS']:
        ts = s(gt_A[dep][tv])
        gs = s(gen_A[dep][tv])
        if ts != gs:
            mis_a += 1
            print(f'  {dep:4s} {tv:5s} paper={gt_A[dep][tv]:.4f}({b[ts]:>3s}) ours={gen_A[dep][tv]:.4f}({b[gs]:>3s})')
print(f'Panel A: {mis_a}/40 mismatches\n')

print("=== Panel B mismatches ===")
mis_b = 0
for dep in gt_B:
    for tv in ['M1', 'M2', 'BILL', 'BOND', 'FUNDS']:
        ts = s(gt_B[dep][tv])
        gs = s(gen_B[dep][tv])
        if ts != gs:
            mis_b += 1
            print(f'  {dep:4s} {tv:5s} paper={gt_B[dep][tv]:.4f}({b[ts]:>3s}) ours={gen_B[dep][tv]:.4f}({b[gs]:>3s})')
print(f'Panel B: {mis_b}/40 mismatches')
print(f'\nTotal: {mis_a + mis_b}/80 mismatches')

# Count by variable
print('\nBy variable:')
for tv in ['M1', 'M2', 'BILL', 'BOND', 'FUNDS']:
    n = 0
    for dep in gt_A:
        if s(gt_A[dep][tv]) != s(gen_A[dep][tv]):
            n += 1
    for dep in gt_B:
        if s(gt_B[dep][tv]) != s(gen_B[dep][tv]):
            n += 1
    print(f'  {tv}: {n}/16')
