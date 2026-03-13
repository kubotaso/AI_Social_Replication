"""
Diagnostic 21: Reverse-engineer what b1+b2 the paper used for each threshold,
then test which step 1 sample gives b1+b2 closest to those values.

Paper's implied b1+b2 for each threshold:
  >=0: 0.0713 + 0.0545 = 0.1258
  >=1: 0.0792 + 0.0546 = 0.1338
  >=3: 0.0716 + 0.0559 = 0.1275
  >=5: 0.0607 + 0.0584 = 0.1191

Also reverse-engineer gamma terms from cumulative returns.
Cumret(T) = beta_2*T + g2*T^2 + g3*T^3 + g4*T^4
So: Cumret(T) - beta_2*T = g2*T^2 + g3*T^3 + g4*T^4

For >=0: we know beta_2=0.0545, cumrets at 5,10,15,20
So: 0.1793 - 0.0545*5 = 0.1793 - 0.2725 = -0.0932
    0.2459 - 0.0545*10 = 0.2459 - 0.545 = -0.2991
    0.2832 - 0.0545*15 = 0.2832 - 0.8175 = -0.5343
    0.3375 - 0.0545*20 = 0.3375 - 1.09 = -0.7525

These should equal g2*T^2 + g3*T^3 + g4*T^4
With paper's g2=-0.004592, g3=0.0001846, g4=-0.00000245:
  T=5: -0.004592*25 + 0.0001846*125 - 0.00000245*625 = -0.1148 + 0.02308 - 0.00153 = -0.0932  EXACT!
  T=10: -0.004592*100 + 0.0001846*1000 - 0.00000245*10000 = -0.4592 + 0.1846 - 0.0245 = -0.2991  EXACT!

Good, so the paper's gamma terms work perfectly for >=0.

Now for other thresholds, the gamma terms MUST be different because the cumulative
returns have different curvature.

Let's back out gamma terms from the cumulative returns for each threshold.
"""

import numpy as np

# Paper values
beta_2 = {0: 0.0545, 1: 0.0546, 3: 0.0559, 5: 0.0584}
cumrets = {
    0: {5: 0.1793, 10: 0.2459, 15: 0.2832, 20: 0.3375},
    1: {5: 0.1725, 10: 0.2235, 15: 0.2439, 20: 0.2865},
    3: {5: 0.1703, 10: 0.2181, 15: 0.2503, 20: 0.3232},
    5: {5: 0.1815, 10: 0.2330, 15: 0.2565, 20: 0.3066},
}

# For each threshold, solve for g2, g3, g4 from the 4 cumulative return equations
# cumret(T) - beta_2*T = g2*T^2 + g3*T^3 + g4*T^4
# We have 4 equations and 3 unknowns, so use least squares

for thresh in [0, 1, 3, 5]:
    b2 = beta_2[thresh]
    cr = cumrets[thresh]

    T_vals = [5, 10, 15, 20]
    y = np.array([cr[T] - b2*T for T in T_vals])
    A = np.array([[T**2, T**3, T**4] for T in T_vals])

    # Least squares
    result = np.linalg.lstsq(A, y, rcond=None)
    g2, g3, g4 = result[0]

    # Check residuals
    resid = y - A @ result[0]

    print(f"\n>={thresh}:")
    print(f"  beta_2 = {b2:.4f}")
    print(f"  g2 = {g2:.10f} (x100 = {g2*100:.4f})")
    print(f"  g3 = {g3:.10f} (x1000 = {g3*1000:.4f})")
    print(f"  g4 = {g4:.10f} (x10000 = {g4*10000:.4f})")
    print(f"  Residuals: {resid}")

    # Verify cumrets
    for T in T_vals:
        pred = b2*T + g2*T**2 + g3*T**3 + g4*T**4
        print(f"    T={T}: pred={pred:.4f} vs paper={cr[T]:.4f} (err={abs(pred-cr[T]):.4f})")

print("\n\nComparison of gamma terms across thresholds:")
print(f"{'':>8} {'g2*100':>10} {'g3*1000':>10} {'g4*10000':>10}")
for thresh in [0, 1, 3, 5]:
    b2 = beta_2[thresh]
    cr = cumrets[thresh]
    T_vals = [5, 10, 15, 20]
    y = np.array([cr[T] - b2*T for T in T_vals])
    A = np.array([[T**2, T**3, T**4] for T in T_vals])
    result = np.linalg.lstsq(A, y, rcond=None)
    g2, g3, g4 = result[0]
    print(f"  >={thresh}: {g2*100:>10.4f} {g3*1000:>10.4f} {g4*10000:>10.4f}")

print(f"\nPaper baseline: {-0.4592:>10.4f} {0.1846:>10.4f} {-0.0245:>10.4f}")
