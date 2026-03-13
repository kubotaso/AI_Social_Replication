import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit

df = pd.read_csv('anes_cumulative.csv', usecols=['VCF0004','VCF0301','VCF0707','VCF0902'], low_memory=False)

dem_inc = [12, 13, 14, 19]
rep_inc = [21, 23, 24, 29]

def run_probit(df, year, null_inc_val=0):
    """Run probit for a year, assigning null_inc_val to missing VCF0902 cases."""
    sub = df[(df['VCF0004']==year) & (df['VCF0707'].isin([1,2]))].copy()
    sub['vote_rep'] = (sub['VCF0707'] == 2).astype(int)
    sub['strong'] = 0; sub['weak'] = 0; sub['leaner'] = 0
    sub.loc[sub['VCF0301'] == 7, 'strong'] = 1
    sub.loc[sub['VCF0301'] == 1, 'strong'] = -1
    sub.loc[sub['VCF0301'] == 6, 'weak'] = 1
    sub.loc[sub['VCF0301'] == 2, 'weak'] = -1
    sub.loc[sub['VCF0301'] == 5, 'leaner'] = 1
    sub.loc[sub['VCF0301'] == 3, 'leaner'] = -1
    sub['incumbency'] = null_inc_val  # default for null and non-standard codes
    sub.loc[sub['VCF0902'].isin(dem_inc), 'incumbency'] = -1
    sub.loc[sub['VCF0902'].isin(rep_inc), 'incumbency'] = 1
    # Anything not dem/rep inc gets null_inc_val (open seat or whatever)
    # But for non-null standard codes that aren't dem/rep, keep as 0
    open_mask = sub['VCF0902'].notna() & ~sub['VCF0902'].isin(dem_inc) & ~sub['VCF0902'].isin(rep_inc)
    sub.loc[open_mask, 'incumbency'] = 0

    y = sub['vote_rep']
    X = sm.add_constant(sub[['strong', 'weak', 'leaner', 'incumbency']])
    result = Probit(y, X).fit(disp=0)
    return len(sub), result

# 1974: try assigning null VCF0902 cases as Dem incumbent (-1)
# Rationale: 45 of 59 null cases voted Dem, suggesting Dem incumbent districts
print("=== 1974 variations ===")
print("Ground truth: N=798, strong=1.138, LL=-355.2, R2=0.33")
print()

for label, val in [("null=open(0)", 0), ("null=Dem_inc(-1)", -1), ("null=Rep_inc(+1)", 1)]:
    n, r = run_probit(df, 1974, null_inc_val=val)
    print(f"  {label}: N={n}, LL={r.llf:.1f}, R2={r.prsquared:.4f}")
    print(f"    strong={r.params['strong']:.3f}({r.bse['strong']:.3f}), "
          f"weak={r.params['weak']:.3f}({r.bse['weak']:.3f}), "
          f"leaner={r.params['leaner']:.3f}({r.bse['leaner']:.3f})")
    print(f"    inc={r.params['incumbency']:.3f}({r.bse['incumbency']:.3f}), "
          f"const={r.params['const']:.3f}({r.bse['const']:.3f})")
    print()

# Also try for 1996: maybe the null PID cases should be excluded
print("=== 1996 variations ===")
print("Ground truth: N=1031, strong=1.503, weak=0.865, leaner=0.874, inc=0.742, const=0.142")
print()

# Standard approach (no PID filter)
n, r = run_probit(df, 1996, null_inc_val=0)
print(f"  No PID filter: N={n}, LL={r.llf:.1f}")
print(f"    strong={r.params['strong']:.3f}, weak={r.params['weak']:.3f}, "
      f"leaner={r.params['leaner']:.3f}, inc={r.params['incumbency']:.3f}")
print()

# With PID filter
sub96 = df[(df['VCF0004']==1996) & (df['VCF0707'].isin([1,2])) & (df['VCF0301'].isin([1,2,3,4,5,6,7]))].copy()
sub96['vote_rep'] = (sub96['VCF0707'] == 2).astype(int)
sub96['strong'] = 0; sub96['weak'] = 0; sub96['leaner'] = 0
sub96.loc[sub96['VCF0301'] == 7, 'strong'] = 1
sub96.loc[sub96['VCF0301'] == 1, 'strong'] = -1
sub96.loc[sub96['VCF0301'] == 6, 'weak'] = 1
sub96.loc[sub96['VCF0301'] == 2, 'weak'] = -1
sub96.loc[sub96['VCF0301'] == 5, 'leaner'] = 1
sub96.loc[sub96['VCF0301'] == 3, 'leaner'] = -1
sub96['incumbency'] = 0
sub96.loc[sub96['VCF0902'].isin(dem_inc), 'incumbency'] = -1
sub96.loc[sub96['VCF0902'].isin(rep_inc), 'incumbency'] = 1
y = sub96['vote_rep']
X = sm.add_constant(sub96[['strong', 'weak', 'leaner', 'incumbency']])
r2 = Probit(y, X).fit(disp=0)
print(f"  With PID filter: N={len(sub96)}, LL={r2.llf:.1f}")
print(f"    strong={r2.params['strong']:.3f}, weak={r2.params['weak']:.3f}, "
      f"leaner={r2.params['leaner']:.3f}, inc={r2.params['incumbency']:.3f}")
print()

# Check: for 1996, try excluding code 55 cases and re-running
# (treating them differently)
print("=== 1996: try excluding code 55 ===")
sub96b = df[(df['VCF0004']==1996) & (df['VCF0707'].isin([1,2])) & (df['VCF0902']!=55)].copy()
sub96b['vote_rep'] = (sub96b['VCF0707'] == 2).astype(int)
sub96b['strong'] = 0; sub96b['weak'] = 0; sub96b['leaner'] = 0
sub96b.loc[sub96b['VCF0301'] == 7, 'strong'] = 1
sub96b.loc[sub96b['VCF0301'] == 1, 'strong'] = -1
sub96b.loc[sub96b['VCF0301'] == 6, 'weak'] = 1
sub96b.loc[sub96b['VCF0301'] == 2, 'weak'] = -1
sub96b.loc[sub96b['VCF0301'] == 5, 'leaner'] = 1
sub96b.loc[sub96b['VCF0301'] == 3, 'leaner'] = -1
sub96b['incumbency'] = 0
sub96b.loc[sub96b['VCF0902'].isin(dem_inc), 'incumbency'] = -1
sub96b.loc[sub96b['VCF0902'].isin(rep_inc), 'incumbency'] = 1
y = sub96b['vote_rep']
X = sm.add_constant(sub96b[['strong', 'weak', 'leaner', 'incumbency']])
r3 = Probit(y, X).fit(disp=0)
print(f"  Excluding code 55: N={len(sub96b)}, LL={r3.llf:.1f}")
print(f"    strong={r3.params['strong']:.3f}, weak={r3.params['weak']:.3f}, "
      f"leaner={r3.params['leaner']:.3f}, inc={r3.params['incumbency']:.3f}")

# For 1974 specifically, let me check how many of the null cases
# are in strong PID categories
print("\n=== 1974 null VCF0902 case analysis ===")
sub74 = df[(df['VCF0004']==1974) & (df['VCF0707'].isin([1,2]))].copy()
null74 = sub74[sub74['VCF0902'].isna()]
print("PID distribution of null VCF0902 cases:")
print(null74['VCF0301'].value_counts().sort_index())
print("Vote distribution of null VCF0902 cases:")
print(null74['VCF0707'].value_counts().sort_index())
