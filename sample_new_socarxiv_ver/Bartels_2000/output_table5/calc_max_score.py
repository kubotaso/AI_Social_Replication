"""Calculate theoretical maximum achievable score with current data."""

# Current best results (attempt 4):
# 1960: N=736 (target 911) - 19.2% off → N_score=0.4
# 1976: N=656 (target 682) - 3.8% off → N_score=1.0
# 1992: N=759 (target 760) - 0.1% off → N_score=1.0

# N points: (0.4 + 1.0 + 1.0) / 3 * 15 = 12.0/15 ✓

# Variables: All present → 10/10 ✓

# Log-likelihood: All off by > 10 → 0/10
# Can we improve? No - LL scales with N and sample composition

# R2: Mixed results
# Let me recalculate based on current values

import numpy as np

# Current R2 values:
r2_results = {
    '1960_current': (0.3827, 0.41),   # diff=0.0273
    '1960_lagged': (0.3428, 0.36),    # diff=0.0172
    '1960_iv': (0.3428, 0.36),        # diff=0.0172
    '1976_current': (0.2551, 0.24),   # diff=0.0151
    '1976_lagged': (0.2348, 0.21),    # diff=0.0248
    '1976_iv': (0.2348, 0.21),        # diff=0.0248
    '1992_current': (0.2226, 0.20),   # diff=0.0226
    '1992_lagged': (0.2028, 0.19),    # diff=0.0128
    '1992_iv': (0.2028, 0.19),        # diff=0.0128
}

r2_total = 0
for name, (gen, tgt) in r2_results.items():
    diff = abs(gen - tgt)
    score = max(0, 1.0 - diff / 0.02) if diff <= 0.06 else 0.0
    r2_total += score
    print(f'{name}: gen={gen:.4f}, tgt={tgt:.2f}, diff={diff:.4f}, score={score:.3f}')

r2_pts = r2_total / 9 * 15
print(f'\nR2 total: {r2_total:.3f}/9 = {r2_pts:.1f}/15')

# Current coefficient scores (from output):
# coef_pts = 7.0/30
# se_pts = 12.8/20

# Coefficients that are within 0.05 of target:
coef_results = {
    # 1960 current
    '60c_strong': (1.454, 1.358),  # diff=0.096 > 0.05
    '60c_weak': (0.775, 1.028),    # diff=0.253 > 0.05
    '60c_lean': (0.912, 0.855),    # diff=0.057 > 0.05
    '60c_int': (0.101, 0.035),     # diff=0.066 > 0.05
    # 1960 lagged
    '60l_strong': (1.339, 1.363),  # diff=0.024 ✓
    '60l_weak': (0.750, 0.842),    # diff=0.092 > 0.05
    '60l_lean': (0.742, 0.564),    # diff=0.178 > 0.05
    '60l_int': (0.114, 0.068),     # diff=0.046 ✓
    # 1960 IV
    '60i_strong': (1.827, 1.715),  # diff=0.112 > 0.05
    '60i_weak': (0.155, 0.728),    # diff=0.573 > 0.05
    '60i_lean': (1.888, 1.081),    # diff=0.807 > 0.05
    '60i_int': (0.063, 0.032),     # diff=0.031 ✓
    # 1976 current
    '76c_strong': (1.131, 1.087),  # diff=0.044 ✓
    '76c_weak': (0.681, 0.624),    # diff=0.057 > 0.05
    '76c_lean': (0.655, 0.622),    # diff=0.033 ✓
    '76c_int': (-0.107, -0.123),   # diff=0.016 ✓
    # 1976 lagged
    '76l_strong': (1.118, 0.966),  # diff=0.152 > 0.05
    '76l_weak': (0.693, 0.738),    # diff=0.045 ✓
    '76l_lean': (0.527, 0.486),    # diff=0.041 ✓
    '76l_int': (-0.049, -0.063),   # diff=0.014 ✓
    # 1976 IV
    '76i_strong': (1.351, 1.123),  # diff=0.228 > 0.05
    '76i_weak': (0.602, 0.745),    # diff=0.143 > 0.05
    '76i_lean': (0.833, 0.725),    # diff=0.108 > 0.05
    '76i_int': (-0.087, -0.102),   # diff=0.015 ✓
    # 1992 current
    '92c_strong': (0.999, 0.975),  # diff=0.024 ✓
    '92c_weak': (0.687, 0.627),    # diff=0.060 > 0.05
    '92c_lean': (0.490, 0.472),    # diff=0.018 ✓
    '92c_int': (-0.239, -0.211),   # diff=0.028 ✓
    # 1992 lagged
    '92l_strong': (1.089, 1.061),  # diff=0.028 ✓
    '92l_weak': (0.449, 0.404),    # diff=0.045 ✓
    '92l_lean': (0.490, 0.519),    # diff=0.029 ✓
    '92l_int': (-0.195, -0.168),   # diff=0.027 ✓
    # 1992 IV
    '92i_strong': (1.514, 1.516),  # diff=0.002 ✓
    '92i_weak': (-0.060, -0.225),  # diff=0.165 > 0.05
    '92i_lean': (1.493, 1.824),    # diff=0.331 > 0.05
    '92i_int': (-0.148, -0.125),   # diff=0.023 ✓
}

coef_total = 0
for name, (gen, tgt) in coef_results.items():
    diff = abs(gen - tgt)
    score = max(0, 1.0 - diff / 0.05) if diff <= 0.15 else 0.0
    coef_total += score
    if diff > 0.05:
        print(f'{name}: diff={diff:.3f}, score={score:.3f}')

print(f'\nCoefficient total: {coef_total:.3f}/{len(coef_results)} = {coef_total/len(coef_results)*30:.1f}/30')

# Count within-tolerance items
within_005 = sum(1 for (g,t) in coef_results.values() if abs(g-t) <= 0.05)
print(f'Coefficients within 0.05: {within_005}/{len(coef_results)}')

# SE scores
se_results = {
    '60c_strong': (0.107, 0.094),  # diff=0.013
    '60c_weak': (0.085, 0.083),    # diff=0.002
    '60c_lean': (0.156, 0.131),    # diff=0.025
    '60c_int': (0.059, 0.053),     # diff=0.006
    '60l_strong': (0.101, 0.092),  # diff=0.009
    '60l_weak': (0.081, 0.078),    # diff=0.003
    '60l_lean': (0.151, 0.125),    # diff=0.026
    '60l_int': (0.057, 0.051),     # diff=0.006
    '60i_strong': (0.196, 0.173),  # diff=0.023
    '60i_weak': (0.264, 0.239),    # diff=0.025
    '60i_lean': (0.771, 0.696),    # diff=0.075
    '60i_int': (0.060, 0.057),     # diff=0.003
    '76c_strong': (0.108, 0.105),  # diff=0.003
    '76c_weak': (0.089, 0.086),    # diff=0.003
    '76c_lean': (0.118, 0.110),    # diff=0.008
    '76c_int': (0.056, 0.054),     # diff=0.002
    '76l_strong': (0.109, 0.104),  # diff=0.005
    '76l_weak': (0.091, 0.089),    # diff=0.002
    '76l_lean': (0.113, 0.109),    # diff=0.004
    '76l_int': (0.055, 0.053),     # diff=0.002
    '76i_strong': (0.188, 0.178),  # diff=0.010
    '76i_weak': (0.253, 0.251),    # diff=0.002
    '76i_lean': (0.375, 0.438),    # diff=0.063
    '76i_int': (0.056, 0.055),     # diff=0.001
    '92c_strong': (0.093, 0.094),  # diff=0.001
    '92c_weak': (0.089, 0.084),    # diff=0.005
    '92c_lean': (0.099, 0.098),    # diff=0.001
    '92c_int': (0.051, 0.051),     # diff=0.000
    '92l_strong': (0.099, 0.100),  # diff=0.001
    '92l_weak': (0.079, 0.077),    # diff=0.002
    '92l_lean': (0.102, 0.101),    # diff=0.001
    '92l_int': (0.051, 0.051),     # diff=0.000
    '92i_strong': (0.176, 0.180),  # diff=0.004
    '92i_weak': (0.291, 0.268),    # diff=0.023
    '92i_lean': (0.489, 0.513),    # diff=0.024
    '92i_int': (0.056, 0.053),     # diff=0.003
}

se_total = 0
for name, (gen, tgt) in se_results.items():
    diff = abs(gen - tgt)
    score = max(0, 1.0 - diff / 0.02) if diff <= 0.06 else 0.0
    se_total += score

print(f'\nSE total: {se_total:.3f}/{len(se_results)} = {se_total/len(se_results)*20:.1f}/20')
within_002 = sum(1 for (g,t) in se_results.values() if abs(g-t) <= 0.02)
print(f'SEs within 0.02: {within_002}/{len(se_results)}')

# Total
total_score = (coef_total/len(coef_results)*30 + se_total/len(se_results)*20 +
               12.0 + 10.0 + 0.0 + r2_pts)
print(f'\n=== ESTIMATED TOTAL: {total_score:.1f}/100 ===')
