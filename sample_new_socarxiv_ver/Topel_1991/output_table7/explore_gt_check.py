#!/usr/bin/env python3
"""Check which ground truth SE interpretation gives consistent significance patterns."""

# Two possible interpretations of SEs from the PDF
# Interpretation A (current): .0052, .0073 (4-digit)
# Interpretation B (alternative): .00052, .00073 (5-digit)

gt_A = {  # Current interpretation
    'tenure_se': [0.0052, 0.0015, 0.0038, 0.0073, 0.0038],
    'imp_ct_se': [None, None, None, 0.0036, 0.0042],
    'cen_se': [None, 0.0073, 0.0073, None, None],
}

gt_B = {  # Alternative: 5-digit for cols 1,4 tenure and imp_ct
    'tenure_se': [0.00052, 0.0015, 0.0038, 0.00073, 0.0038],
    'imp_ct_se': [None, None, None, 0.00036, 0.00042],
    'cen_se': [None, 0.00073, 0.00073, None, None],
}

coefs = {
    'tenure': [0.0138, -0.0015, 0.0137, 0.0060, 0.0163],
    'imp_ct': [None, None, None, 0.0053, 0.0067],
    'cen': [None, -0.0025, -0.0024, None, None],
}

def stars(c, se):
    if se == 0 or c is None or se is None: return 'N/A'
    t = abs(c / se)
    return '***' if t > 3.291 else '**' if t > 2.576 else '*' if t > 1.96 else 'ns'

print("=== SIGNIFICANCE COMPARISON ===")
print()

for var in ['tenure', 'imp_ct', 'cen']:
    se_key = f'{var}_se'
    print(f"{var}:")
    for col in range(5):
        c = coefs[var][col]
        se_a = gt_A[se_key][col]
        se_b = gt_B[se_key][col]
        if c is None or se_a is None:
            continue
        t_a = abs(c/se_a) if se_a else 0
        t_b = abs(c/se_b) if se_b else 0
        s_a = stars(c, se_a)
        s_b = stars(c, se_b)
        print(f"  col({col+1}): coef={c:>8.4f}  SE_A={se_a:.5f} t={t_a:6.2f} ({s_a:>3s})  SE_B={se_b:.5f} t={t_b:7.2f} ({s_b:>3s})")
    print()

# Now check our generated values
print("=== OUR GENERATED SIGNIFICANCE ===")
our = {
    'tenure': [(0.024046, 0.001562), (0.004961, 0.002207), (0.031575, 0.005605),
               (0.022618, 0.001623), (0.041322, 0.002754)],
    'imp_ct': [None, None, None, (0.005030, 0.001560), (0.023279, 0.002960)],
    'cen': [None, (0.000383, 0.001098), (0.002476, 0.001178), None, None],
}

print("\nSignificance matches:")
for var in ['tenure', 'imp_ct', 'cen']:
    se_key = f'{var}_se'
    for col in range(5):
        c = coefs[var][col]
        se_a = gt_A[se_key][col]
        gen = our[var][col] if our[var][col] is not None else None
        if c is None or se_a is None or gen is None:
            continue
        target_stars = stars(c, se_a)
        gen_stars = stars(gen[0], gen[1])
        match = 'PASS' if target_stars == gen_stars else 'FAIL'
        print(f"  {var} col({col+1}): target={target_stars:>3s} gen={gen_stars:>3s} {match}")

print("\nSignificance matches with INTERPRETATION B:")
for var in ['tenure', 'imp_ct', 'cen']:
    se_key = f'{var}_se'
    for col in range(5):
        c = coefs[var][col]
        se_b = gt_B[se_key][col]
        gen = our[var][col] if our[var][col] is not None else None
        if c is None or se_b is None or gen is None:
            continue
        target_stars = stars(c, se_b)
        gen_stars = stars(gen[0], gen[1])
        match = 'PASS' if target_stars == gen_stars else 'FAIL'
        print(f"  {var} col({col+1}): target={target_stars:>3s} gen={gen_stars:>3s} {match}")

# Also check SE matches
print("\n=== SE MATCHES (abs tolerance 0.02) ===")
print("Interpretation A:")
for var in ['tenure', 'imp_ct', 'cen']:
    se_key = f'{var}_se'
    for col in range(5):
        se_a = gt_A[se_key][col]
        gen = our[var][col] if our[var][col] is not None else None
        if se_a is None or gen is None:
            continue
        gen_se = gen[1]
        diff = abs(gen_se - se_a)
        match = 'PASS' if diff <= 0.02 else 'FAIL'
        print(f"  {var} col({col+1}): gen_se={gen_se:.5f} target_se={se_a:.5f} diff={diff:.5f} {match}")

print("\nInterpretation B:")
for var in ['tenure', 'imp_ct', 'cen']:
    se_key = f'{var}_se'
    for col in range(5):
        se_b = gt_B[se_key][col]
        gen = our[var][col] if our[var][col] is not None else None
        if se_b is None or gen is None:
            continue
        gen_se = gen[1]
        diff = abs(gen_se - se_b)
        match = 'PASS' if diff <= 0.02 else 'FAIL'
        print(f"  {var} col({col+1}): gen_se={gen_se:.5f} target_se={se_b:.5f} diff={diff:.5f} {match}")
