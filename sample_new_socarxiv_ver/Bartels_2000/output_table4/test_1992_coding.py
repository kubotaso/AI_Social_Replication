import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('panel_1992.csv')

mask = (
    df['vote_pres'].isin([1, 2]) &
    df['pid_current'].isin([1, 2, 3, 4, 5, 6, 7]) &
    df['pid_lagged'].isin([1, 2, 3, 4, 5, 6, 7])
)
df_v = df[mask].copy()
print(f"N = {len(df_v)}")

# Approach A: Reversed PID (8-pid), vote=(vote_pres==2)
# (This is attempt 3 - gives positive coefs, positive intercept)
pid_std = 8 - df_v['pid_current']
strong = pd.Series(0, index=pid_std.index, dtype=float)
weak = pd.Series(0, index=pid_std.index, dtype=float)
lean = pd.Series(0, index=pid_std.index, dtype=float)
strong[pid_std == 7] = 1; strong[pid_std == 1] = -1
weak[pid_std == 6] = 1; weak[pid_std == 2] = -1
lean[pid_std == 5] = 1; lean[pid_std == 3] = -1
X_A = sm.add_constant(pd.DataFrame({'S': strong, 'W': weak, 'L': lean}))
y_A = (df_v['vote_pres'] == 2).astype(int)
res_A = Probit(y_A, X_A).fit(disp=0)
print("\nApproach A: reversed PID, vote=Rep(vote_pres==2)")
print(f"Coeffs: S={res_A.params['S']:.3f} W={res_A.params['W']:.3f} L={res_A.params['L']:.3f} c={res_A.params['const']:.3f}")
print(f"LL={res_A.llf:.1f} R2={res_A.prsquared:.2f}")

# Approach B: Original PID (no reversal), vote=(vote_pres==1)
# PID: 1=StRep...7=StDem. Standard dummies: 7->+1(StDem), 1->-1(StRep).
# vote=1 for vote_pres=1 (Dem).
# StDem(+1) -> vote Dem(1) -> POSITIVE coefficient
# 290/725 vote Dem -> negative intercept
pid_orig = df_v['pid_current']
strong2 = pd.Series(0, index=pid_orig.index, dtype=float)
weak2 = pd.Series(0, index=pid_orig.index, dtype=float)
lean2 = pd.Series(0, index=pid_orig.index, dtype=float)
strong2[pid_orig == 7] = 1; strong2[pid_orig == 1] = -1
weak2[pid_orig == 6] = 1; weak2[pid_orig == 2] = -1
lean2[pid_orig == 5] = 1; lean2[pid_orig == 3] = -1
X_B = sm.add_constant(pd.DataFrame({'S': strong2, 'W': weak2, 'L': lean2}))
y_B = (df_v['vote_pres'] == 1).astype(int)
res_B = Probit(y_B, X_B).fit(disp=0)
print("\nApproach B: original PID, vote=Dem(vote_pres==1)")
print(f"Coeffs: S={res_B.params['S']:.3f} W={res_B.params['W']:.3f} L={res_B.params['L']:.3f} c={res_B.params['const']:.3f}")
print(f"LL={res_B.llf:.1f} R2={res_B.prsquared:.2f}")

# Approach C: Reversed PID (8-pid), vote=(vote_pres==1)
# (This is attempt 2 - gives negative coefs, negative intercept)
pid_std2 = 8 - df_v['pid_current']
strong3 = pd.Series(0, index=pid_std2.index, dtype=float)
weak3 = pd.Series(0, index=pid_std2.index, dtype=float)
lean3 = pd.Series(0, index=pid_std2.index, dtype=float)
strong3[pid_std2 == 7] = 1; strong3[pid_std2 == 1] = -1
weak3[pid_std2 == 6] = 1; weak3[pid_std2 == 2] = -1
lean3[pid_std2 == 5] = 1; lean3[pid_std2 == 3] = -1
X_C = sm.add_constant(pd.DataFrame({'S': strong3, 'W': weak3, 'L': lean3}))
y_C = (df_v['vote_pres'] == 1).astype(int)
res_C = Probit(y_C, X_C).fit(disp=0)
print("\nApproach C: reversed PID, vote=Dem(vote_pres==1)")
print(f"Coeffs: S={res_C.params['S']:.3f} W={res_C.params['W']:.3f} L={res_C.params['L']:.3f} c={res_C.params['const']:.3f}")
print(f"LL={res_C.llf:.1f} R2={res_C.prsquared:.2f}")

# Approach D: Original PID, vote=(vote_pres==2)
# (This is attempt 1)
X_D = sm.add_constant(pd.DataFrame({'S': strong2, 'W': weak2, 'L': lean2}))
y_D = (df_v['vote_pres'] == 2).astype(int)
res_D = Probit(y_D, X_D).fit(disp=0)
print("\nApproach D: original PID, vote=Rep(vote_pres==2)")
print(f"Coeffs: S={res_D.params['S']:.3f} W={res_D.params['W']:.3f} L={res_D.params['L']:.3f} c={res_D.params['const']:.3f}")
print(f"LL={res_D.llf:.1f} R2={res_D.prsquared:.2f}")

print("\n\nGround truth: S=1.853 W=0.948 L=1.117 c=-0.073 LL=-236.9 R2=0.52")
