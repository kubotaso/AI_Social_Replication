import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("anes_cumulative.csv", low_memory=False)

# Check for weight variables
weight_cols = [c for c in df.columns if 'weight' in c.lower() or 'wt' in c.lower() or c.startswith('VCF0009')]
print("Potential weight columns:", weight_cols)

# VCF0009a, VCF0009b, VCF0009x are weight variables
for col in ['VCF0009a', 'VCF0009b', 'VCF0009x', 'VCF0009z']:
    if col in df.columns:
        print(f"\n{col}:")
        print(df[col].describe())
        print(f"  Unique values (first 20): {sorted(df[col].dropna().unique())[:20]}")

# For 1988, test with and without weights
year = 1988
ydf = df[df['VCF0004'] == year].copy()

# Check weights for 1988
for col in ['VCF0009a', 'VCF0009b', 'VCF0009x', 'VCF0009z']:
    if col in df.columns:
        print(f"\n1988 {col}:")
        vals = ydf[col].dropna()
        print(f"  N non-null: {len(vals)}, unique: {vals.nunique()}")
        print(f"  Values: {sorted(vals.unique())[:20]}")

# Try weighted probit for 1988
white_ns_voters = ydf[
    (ydf['VCF0105a']==1) & (ydf['VCF0113']==2) &
    (ydf['VCF0704a'].isin([1,2])) & (ydf['VCF0301'].isin([1,2,3,4,5,6,7]))
].copy()

white_ns_voters['vote_rep'] = (white_ns_voters['VCF0704a'] == 2).astype(int)
white_ns_voters['strong'] = 0
white_ns_voters.loc[white_ns_voters['VCF0301'] == 7, 'strong'] = 1
white_ns_voters.loc[white_ns_voters['VCF0301'] == 1, 'strong'] = -1
white_ns_voters['weak'] = 0
white_ns_voters.loc[white_ns_voters['VCF0301'] == 6, 'weak'] = 1
white_ns_voters.loc[white_ns_voters['VCF0301'] == 2, 'weak'] = -1
white_ns_voters['leaning'] = 0
white_ns_voters.loc[white_ns_voters['VCF0301'] == 5, 'leaning'] = 1
white_ns_voters.loc[white_ns_voters['VCF0301'] == 3, 'leaning'] = -1

y = white_ns_voters['vote_rep']
X = sm.add_constant(white_ns_voters[['strong', 'weak', 'leaning']])

# Unweighted
res = Probit(y, X).fit(disp=0)
print(f"\n1988 NS Unweighted: s={res.params['strong']:.4f} w={res.params['weak']:.4f} l={res.params['leaning']:.4f}")

# Weighted if possible
for col in ['VCF0009a', 'VCF0009b', 'VCF0009x', 'VCF0009z']:
    if col in ydf.columns:
        w = white_ns_voters[col]
        if w.notna().sum() > 100:
            try:
                # Use frequency weights
                w_clean = w.fillna(1.0)
                # Statsmodels Probit doesn't have freq_weights in the same way
                # Use WLS-like approach or GLM
                from statsmodels.genmod.generalized_linear_model import GLM
                from statsmodels.genmod.families import Binomial
                from statsmodels.genmod.families.links import Probit as ProbitLink

                res_w = GLM(y, X, family=Binomial(link=ProbitLink()), freq_weights=w_clean).fit()
                print(f"  1988 NS Weighted ({col}): s={res_w.params['strong']:.4f} w={res_w.params['weak']:.4f} l={res_w.params['leaning']:.4f}")
            except Exception as e:
                print(f"  Weight {col} failed: {e}")

# Also check: what if Bartels used VCF0301 recoded differently?
# In the paper, the 7-point scale is:
# 1=Strong Democrat, 2=Weak Democrat, 3=Independent-leaning Democrat
# 4=Independent, 5=Independent-leaning Republican, 6=Weak Republican, 7=Strong Republican
#
# Our coding: strong = {1:-1, 7:+1}, weak = {2:-1, 6:+1}, leaning = {3:-1, 5:+1}
# This is correct for the symmetric coding Bartels describes.

# Let me also check: what if the issue is the number of categories?
# What if Bartels used the full 7-point PID as a SINGLE variable (1-7 linear)?
print("\n\n--- Testing alternative: linear PID variable ---")
for year in [1988, 1952, 1956]:
    ydf = df[df['VCF0004'] == year].copy()
    white_ns_voters = ydf[
        (ydf['VCF0105a']==1) & (ydf['VCF0113']==2) &
        (ydf['VCF0704a'].isin([1,2])) & (ydf['VCF0301'].isin([1,2,3,4,5,6,7]))
    ].copy()

    white_ns_voters['vote_rep'] = (white_ns_voters['VCF0704a'] == 2).astype(int)

    # Linear PID: rescale 1-7 to -3 to +3
    white_ns_voters['pid_linear'] = white_ns_voters['VCF0301'] - 4  # -3 to +3

    y = white_ns_voters['vote_rep']
    X = sm.add_constant(white_ns_voters[['pid_linear']])
    res = Probit(y, X).fit(disp=0)

    # Average coefficient = coefficient * mean absolute PID
    mean_abs_pid = white_ns_voters['pid_linear'].abs().mean()
    print(f"  {year} NS linear: coef={res.params['pid_linear']:.4f}, mean_abs_pid={mean_abs_pid:.3f}, product={res.params['pid_linear']*mean_abs_pid:.4f}")
