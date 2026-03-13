"""Back-solve for the polynomial terms each subsample must have used.

Given paper's beta_2 and cumulative returns, solve for g2
(assuming g3, g4 are as in full sample, or more generally, solve
the system of equations from cum5, cum10, cum15, cum20).
"""
import numpy as np

g2_full = -0.004592
g3_full = 0.0001846
g4_full = -0.00000245

for name, b2, cum in [
    ('PS', 0.0601, {5: 0.1887, 10: 0.2400, 15: 0.2527, 20: 0.2841}),
    ('BC_NU', 0.0513, {5: 0.1577, 10: 0.2073, 15: 0.2480, 20: 0.3295}),
    ('BC_U', 0.0399, {5: 0.1401, 10: 0.2033, 15: 0.2384, 20: 0.2733}),
]:
    print(f"\n{name}: beta_2 = {b2}")

    # Check if full-sample polynomials work
    for T in [5, 10, 15, 20]:
        cum_full = b2*T + g2_full*T**2 + g3_full*T**3 + g4_full*T**4
        print(f"  T={T}: paper={cum[T]:.4f}, with_full_poly={cum_full:.4f}, diff={cum[T]-cum_full:.4f}")

    # Solve for g2, g3, g4 given cum5, cum10, cum15, cum20 and beta_2
    # cum(T) = beta_2*T + g2*T^2 + g3*T^3 + g4*T^4
    # adj(T) = cum(T) - beta_2*T = g2*T^2 + g3*T^3 + g4*T^4
    # We have 4 equations, 3 unknowns -> overdetermined
    # Use least squares
    T_vals = [5, 10, 15, 20]
    adj = [cum[T] - b2*T for T in T_vals]
    A = np.array([[T**2, T**3, T**4] for T in T_vals])
    sol = np.linalg.lstsq(A, adj, rcond=None)
    g2_est, g3_est, g4_est = sol[0]
    print(f"  Back-solved: g2={g2_est:.6f}, g3={g3_est:.8f}, g4={g4_est:.10f}")
    print(f"  Full sample: g2={g2_full:.6f}, g3={g3_full:.8f}, g4={g4_full:.10f}")

    # Verify
    for T in T_vals:
        cum_check = b2*T + g2_est*T**2 + g3_est*T**3 + g4_est*T**4
        print(f"    T={T}: paper={cum[T]:.4f}, back-solved={cum_check:.4f}")

    # Alternative: maybe all subsamples use the SAME g2/g3/g4 from full sample
    # and the discrepancy is in beta_2
    # What beta_2 would give the right cum values with full-sample poly?
    for T in T_vals:
        poly_contrib = g2_full*T**2 + g3_full*T**3 + g4_full*T**4
        b2_implied = (cum[T] - poly_contrib) / T
        print(f"  Implied beta_2 for cum{T}: {b2_implied:.4f}")
