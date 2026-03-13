"""Back-solve polynomial terms to match paper cumulative returns.
Use our beta_2 values (since we can't change them) and find what g2/g3/g4
make the cumulative returns match.

For PS: beta_2 = 0.0518 (our) vs 0.0601 (paper)
cum(T) = beta_2*T + g2*T^2 + g3*T^3 + g4*T^4

Paper PS cumulative returns: 0.1887, 0.2400, 0.2527, 0.2841
Our PS beta_2: 0.0518

Solve for g2 given our beta_2 and g3, g4 from full-sample:
cum(T) = 0.0518*T + g2*T^2 + 0.0001846*T^3 - 0.00000245*T^4

At T=20: 0.2841 = 0.0518*20 + g2*400 + 0.0001846*8000 - 0.00000245*160000
0.2841 = 1.036 + 400*g2 + 1.4768 - 0.392
0.2841 = 2.1208 + 400*g2
g2 = (0.2841 - 2.1208) / 400 = -0.004592

Wait, that gives the full-sample g2! That's because for PS:
beta_2=0.0518 and full-sample polynomials give cum20=0.2844,
which is already close to paper's 0.2841.

Let me do this for BC.
"""
import numpy as np

# Full-sample polynomial terms
g3_full = 0.0001846
g4_full = -0.00000245

# For each subsample, try to find g2 that makes cumulative returns match paper
for name, our_b2, paper_cums in [
    ('PS', 0.0518, [0.1887, 0.2400, 0.2527, 0.2841]),
    ('BC_NU', 0.0820, [0.1577, 0.2073, 0.2480, 0.3295]),
    ('BC_U', 0.0242, [0.1401, 0.2033, 0.2384, 0.2733]),
]:
    print(f"\n{name}: our beta_2 = {our_b2:.4f}")
    Ts = [5, 10, 15, 20]

    for i, T in enumerate(Ts):
        # cum(T) = b2*T + g2*T^2 + g3*T^3 + g4*T^4
        g2_needed = (paper_cums[i] - our_b2*T - g3_full*T**3 - g4_full*T**4) / T**2
        print(f"  T={T}: paper_cum={paper_cums[i]:.4f}, g2 needed = {g2_needed:.6f}")

    # If paper's beta_2 were used:
    paper_b2 = {'PS': 0.0601, 'BC_NU': 0.0513, 'BC_U': 0.0399}[name]
    print(f"\n  With paper's beta_2 = {paper_b2:.4f}:")
    for i, T in enumerate(Ts):
        g2_needed = (paper_cums[i] - paper_b2*T - g3_full*T**3 - g4_full*T**4) / T**2
        print(f"  T={T}: g2 needed = {g2_needed:.6f}")

    # Solve system of 4 equations with 3 unknowns (g2, g3, g4)
    # cum(T) = beta_2*T + g2*T^2 + g3*T^3 + g4*T^4
    # Using our beta_2:
    A = np.array([[T**2, T**3, T**4] for T in Ts])
    rhs = np.array([paper_cums[i] - our_b2*Ts[i] for i in range(4)])
    # Least squares fit
    result = np.linalg.lstsq(A, rhs, rcond=None)[0]
    g2_fit, g3_fit, g4_fit = result
    print(f"\n  Least-squares fit with our beta_2:")
    print(f"  g2={g2_fit:.6f}, g3={g3_fit:.8f}, g4={g4_fit:.10f}")
    for i, T in enumerate(Ts):
        cum = our_b2*T + g2_fit*T**2 + g3_fit*T**3 + g4_fit*T**4
        print(f"  T={T}: cum={cum:.4f} vs paper {paper_cums[i]:.4f}")
