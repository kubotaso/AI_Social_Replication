"""
Test different IV probit approaches to see which gives closest results to paper.
The paper's IV procedure shares LL with the lagged probit - this is key.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
import warnings
warnings.filterwarnings('ignore')

def construct_pid_dummies(pid_series):
    strong = pd.Series(0, index=pid_series.index, dtype=float)
    weak = pd.Series(0, index=pid_series.index, dtype=float)
    lean = pd.Series(0, index=pid_series.index, dtype=float)
    strong[pid_series == 7] = 1; strong[pid_series == 1] = -1
    weak[pid_series == 6] = 1; weak[pid_series == 2] = -1
    lean[pid_series == 5] = 1; lean[pid_series == 3] = -1
    return pd.DataFrame({'Strong': strong, 'Weak': weak, 'Lean': lean})

# Load 1992 panel
df = pd.read_csv('panel_1992.csv')
mask = (
    df['vote_pres'].isin([1, 2]) &
    df['pid_current'].isin([1, 2, 3, 4, 5, 6, 7]) &
    df['pid_lagged'].isin([1, 2, 3, 4, 5, 6, 7])
)
df_valid = df[mask].copy()
df_valid['vote'] = (df_valid['vote_pres'] == 1).astype(int)

current_pid = construct_pid_dummies(df_valid['pid_current'])
current_pid.columns = ['Strong_current', 'Weak_current', 'Lean_current']
lagged_pid = construct_pid_dummies(df_valid['pid_lagged'])
lagged_pid.columns = ['Strong_lagged', 'Weak_lagged', 'Lean_lagged']

data = pd.concat([df_valid[['vote']].reset_index(drop=True),
                  current_pid.reset_index(drop=True),
                  lagged_pid.reset_index(drop=True)], axis=1)

y = data['vote']
X_c = data[['Strong_current', 'Weak_current', 'Lean_current']]
X_l = data[['Strong_lagged', 'Weak_lagged', 'Lean_lagged']]

# Standard probit results
current_res = Probit(y, sm.add_constant(X_c)).fit(disp=0)
lagged_res = Probit(y, sm.add_constant(X_l)).fit(disp=0)

print("=== CURRENT PID PROBIT ===")
print(f"N={len(y)}")
print(current_res.summary2().tables[1])
print(f"LL: {current_res.llf:.1f}, R2: {current_res.prsquared:.2f}")
print()
print("=== LAGGED PID PROBIT ===")
print(lagged_res.summary2().tables[1])
print(f"LL: {lagged_res.llf:.1f}, R2: {lagged_res.prsquared:.2f}")
print()

# Approach 1: Standard 2-stage IV (OLS first stage, Probit second stage)
print("=== IV APPROACH 1: OLS 1st stage, Probit 2nd stage ===")
Z = sm.add_constant(X_l)
X_hat = pd.DataFrame(index=X_c.index)
for col in X_c.columns:
    ols = sm.OLS(X_c[col], Z).fit()
    X_hat[col] = ols.predict()
iv_res1 = Probit(y, sm.add_constant(X_hat)).fit(disp=0)
print(iv_res1.summary2().tables[1])
print(f"LL: {iv_res1.llf:.1f}, R2: {iv_res1.prsquared:.2f}")
print()

# Approach 2: Use the reduced form probit directly
# In the reduced form, vote = f(lagged PID dummies)
# Then transform coefficients to get "IV" estimates
# This is the Rivers-Vuong or control function approach
print("=== IV APPROACH 2: Reduced form transformation ===")
# The reduced form coefficients from the lagged probit
# The first stage OLS: Current = gamma * Lagged
# IV_beta = beta_reduced / gamma
# But this requires careful matrix algebra

# Actually, let's think about this differently.
# The paper says LL for IV = LL for lagged. This means the IV probit
# is essentially a transformation of the lagged probit.
# The reduced form is: Vote = Phi(pi * Z) where Z = lagged dummies
# The structural form is: Vote = Phi(beta * X) where X = current dummies
# With X = Gamma * Z + v, and the exclusion restriction
# Since there are exactly as many instruments as endogenous vars (3 each),
# this is exactly identified.
# The IV estimates should be: beta_IV = Gamma_inv * pi
# where Gamma maps lagged to current PID

# Get Gamma matrix (3x3 from first-stage OLS)
Gamma = np.zeros((3, 3))
Z_no_const = X_l.values
Z_const = sm.add_constant(Z_no_const)
gamma_intercepts = np.zeros(3)
for i, col in enumerate(X_c.columns):
    ols = sm.OLS(X_c[col].values, Z_const).fit()
    Gamma[i, :] = ols.params[1:]  # exclude intercept
    gamma_intercepts[i] = ols.params[0]

# Get pi (reduced form probit coefficients)
pi = lagged_res.params[1:]  # exclude intercept
pi0 = lagged_res.params[0]  # intercept

print("Gamma matrix (first stage coefficients):")
print(Gamma)
print(f"Gamma intercepts: {gamma_intercepts}")
print()

# IV coefficients: beta = Gamma_inv * pi
Gamma_inv = np.linalg.inv(Gamma)
beta_iv = Gamma_inv @ pi.values
print(f"IV coefficients via matrix: Strong={beta_iv[0]:.3f}, Weak={beta_iv[1]:.3f}, Lean={beta_iv[2]:.3f}")
print()

# Paper ground truth for 1992 IV:
print("Paper:  Strong=1.622, Weak=0.745, Lean=1.092")
print(f"Ours:   Strong={beta_iv[0]:.3f}, Weak={beta_iv[1]:.3f}, Lean={beta_iv[2]:.3f}")
print()

# Compare with approach 1
print("Approach 1: Strong={:.3f}, Weak={:.3f}, Lean={:.3f}".format(
    iv_res1.params.iloc[1], iv_res1.params.iloc[2], iv_res1.params.iloc[3]))
print()

# Approach 3: Different first stage - use probit for first stage too
print("=== IV APPROACH 3: Probit 1st stage ===")
# For each current PID dummy, run probit on lagged PID dummies
# Then use predicted probabilities in second stage
X_hat_probit = pd.DataFrame(index=X_c.index)
for col in X_c.columns:
    # Current PID dummies are -1, 0, 1 so this needs to be handled carefully
    # Shift to 0/1 for probit
    y_temp = X_c[col].copy()
    # Only use obs where the variable is 0 or 1 (exclude -1)
    # Actually, probit can't handle a 3-valued DV. Skip this.
    pass

# Approach 4: Try Newey's efficient IV probit
# This uses the minimum distance estimator
# Actually, let's try the amemiya GLS estimator

print("=== IV APPROACH 4: Amemiya GLS two-step ===")
# Step 1: Reduced form probit (already done above = lagged probit)
# Step 2: GLS on the reduced form coefficients
# In the just-identified case, this simplifies to the matrix inversion above

# Let's also compute the IV intercept via the matrix approach
# The structural intercept: pi0 = beta0 + beta * gamma_intercepts
# So: beta0 = pi0 - beta_iv @ gamma_intercepts
beta0_iv = pi0 - beta_iv @ gamma_intercepts
print(f"IV intercept via matrix: {beta0_iv:.3f}")
print(f"Paper IV intercept: -0.045")
print()

# Now let's try to get standard errors for the matrix approach
# The SE of beta_IV = Gamma_inv * Var(pi) * Gamma_inv'
# where Var(pi) is the covariance matrix of the reduced form coefficients
cov_pi = lagged_res.cov_params().values[1:, 1:]  # 3x3 cov of lagged PID coefficients
cov_beta_iv = Gamma_inv @ cov_pi @ Gamma_inv.T
se_beta_iv = np.sqrt(np.diag(cov_beta_iv))
print(f"IV SEs via matrix: Strong={se_beta_iv[0]:.3f}, Weak={se_beta_iv[1]:.3f}, Lean={se_beta_iv[2]:.3f}")
print(f"Paper IV SEs:      Strong=0.176,        Weak=0.284,        Lean=0.499")
print(f"Approach 1 SEs:    Strong={iv_res1.bse.iloc[1]:.3f}, Weak={iv_res1.bse.iloc[2]:.3f}, Lean={iv_res1.bse.iloc[3]:.3f}")
print()

# The matrix approach should give the same coefficients as Approach 1
# since both are just-identified IV. Let's verify.
print("Comparison of IV coefficients:")
print(f"  Matrix: {beta_iv[0]:.4f} {beta_iv[1]:.4f} {beta_iv[2]:.4f}")
print(f"  2SLS:   {iv_res1.params.iloc[1]:.4f} {iv_res1.params.iloc[2]:.4f} {iv_res1.params.iloc[3]:.4f}")
print(f"  Paper:  1.622 0.745 1.092")
