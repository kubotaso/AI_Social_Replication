"""Debug SE computation for beta_2.

The paper's beta_2_se (0.0127 for PS) is SMALLER than b1+b2_se (0.0254).
Since beta_2 = b1+b2 - beta_1, this means Cov(b1+b2, beta_1) must be large positive.

The Murphy-Topel SE for beta_1 ALREADY includes step 1 uncertainty.
So: Var_MT(beta_1) = Var_naive(beta_1) + G * V1 * G'
where G is the gradient of beta_1 w.r.t. step 1 params.

For beta_2 = b1+b2 - beta_1:
Var(beta_2) = Var(b1+b2) + Var_MT(beta_1) - 2*Cov(b1+b2, beta_1_MT)

The key: Cov(b1+b2, beta_1_MT) = G_const * V1(const, const) where G_const
is the gradient of beta_1 w.r.t. the step 1 intercept (which IS b1+b2).

So Cov = G_const * Var(b1+b2) where G_const = d(beta_1)/d(b1+b2).

If G_const is close to 1, then Cov ≈ Var(b1+b2), and
Var(beta_2) = Var(b1+b2) + Var_MT(beta_1) - 2*Var(b1+b2) = Var_MT(beta_1) - Var(b1+b2)
"""
import numpy as np

# Paper values for PS
b1b2_se = 0.0254
beta_1_se_mt = 0.0289  # our Murphy-Topel SE for PS beta_1
beta_2_se_paper = 0.0127

# If gradient d(beta_1)/d(b1+b2) = G:
# Var(beta_2) = Var(b1+b2) + Var_MT(beta_1) - 2*G*Var(b1+b2)

# What G gives the paper's beta_2_se?
# 0.0127^2 = 0.0254^2 + 0.0289^2 - 2*G*0.0254^2
# 0.000161 = 0.000645 + 0.000835 - G*0.001290
# G = (0.000645 + 0.000835 - 0.000161) / 0.001290
G_needed = (b1b2_se**2 + beta_1_se_mt**2 - beta_2_se_paper**2) / (2 * b1b2_se**2)
print(f"G needed for paper beta_2_se: {G_needed:.4f}")
print(f"(This means d(beta_1)/d(b1+b2) should be about {G_needed:.4f})")
print()

# What does our numerical gradient give?
# From attempt 10: gradient = -0.0366
# That would give:
G_actual = -0.0366
var_b2 = b1b2_se**2 + beta_1_se_mt**2 - 2*G_actual*b1b2_se**2
print(f"With G={G_actual}: beta_2_se = {np.sqrt(var_b2):.4f}")
print(f"Paper: beta_2_se = {beta_2_se_paper}")
print()

# The problem: our numerical gradient is -0.04, but it should be ~1.0
# Why? Because the Murphy-Topel correction already incorporated
# the step 1 uncertainty. The gradient G should be computed
# from the Murphy-Topel Jacobian, not numerically.

# From the Murphy-Topel framework:
# Var_MT(beta_1) = V_naive(beta_1) + G * V1 * G'
# where G = (X_hat'X_hat)^{-1} X_hat' J (Jacobian)
#
# The gradient of beta_1 w.r.t. step 1 const (b1+b2) is:
# G_const = G[beta_1_row, const_col]
# This is the key quantity for computing Cov(b1+b2, beta_1_MT)

# Actually:
# Cov(b1+b2, beta_1_MT) = G_const * Var(const) = G_const * Var(b1+b2)
# AND Var_MT(beta_1) = V_naive + G * V1 * G'
# So: Var(beta_2) = Var(b1+b2) + V_naive + G*V1*G' - 2*G_const*Var(b1+b2)

# If Var_MT = V_naive + G*V1*G', then:
# Var(beta_2) = Var(b1+b2) + V_naive + G*V1*G' - 2*G_const*Var(b1+b2)
# = V_naive + G*V1*G' + Var(b1+b2)*(1 - 2*G_const)

# For the PS, if G_const = 1.02 (needed value):
# Var(beta_2) = V_naive + G*V1*G' + Var(b1+b2)*(1 - 2*1.02)
# = V_naive + G*V1*G' - 1.04*Var(b1+b2)
# = 0.0289^2 - 1.04*0.0254^2
# = 0.000835 - 0.000671 = 0.000164
# sqrt = 0.0128 ≈ paper's 0.0127!

# So the key is to properly compute G_const from the Murphy-Topel Jacobian.
# G_const should be close to 1 because:
# When b1+b2 increases by delta, w* = lrw - b1b2*T - ... decreases by delta*T
# In the 2SLS, beta_1 = coef on exp. The Jacobian term for const is -T.
# G_const = (X_hat'X_hat)^{-1} X_hat' (-T) at the exp row.
# If T is correlated with exp, this could be close to 1.

# Let me verify with specific numbers
print("The correct formula for beta_2_se:")
print("Var(beta_2) = V_naive_b1 + G*V1*G' + Var(b1b2)*(1 - 2*G_const)")
print("= Var_MT(beta_1) + Var(b1b2) - 2*G_const*Var(b1b2)")
print()
for Gc in [-0.04, 0.5, 0.8, 0.9, 1.0, 1.02]:
    var_b2 = beta_1_se_mt**2 + b1b2_se**2 - 2*Gc*b1b2_se**2
    se = np.sqrt(max(0, var_b2))
    print(f"  G_const={Gc:.2f}: beta_2_se = {se:.4f}")
