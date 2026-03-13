import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("anes_cumulative.csv", low_memory=False)
pres_years = [1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996]

gt_ns = {1952: 1.19, 1956: 1.33, 1960: 1.30, 1964: 1.08, 1968: 1.26, 1972: 0.79,
         1976: 1.03, 1980: 0.97, 1984: 1.20, 1988: 1.38, 1992: 1.33, 1996: 1.35}
gt_s = {1952: 0.99, 1956: 1.19, 1960: 0.95, 1964: 0.97, 1968: 1.05, 1972: 0.64,
        1976: 0.82, 1980: 0.96, 1984: 0.96, 1988: 1.14, 1992: 1.20, 1996: 1.31}

def run_probit(voters_df):
    if len(voters_df) < 20:
        return None
    v = voters_df.copy()
    v['vote_rep'] = (v['VCF0704a'] == 2).astype(int)
    v['strong'] = 0
    v.loc[v['VCF0301'] == 7, 'strong'] = 1
    v.loc[v['VCF0301'] == 1, 'strong'] = -1
    v['weak'] = 0
    v.loc[v['VCF0301'] == 6, 'weak'] = 1
    v.loc[v['VCF0301'] == 2, 'weak'] = -1
    v['leaning'] = 0
    v.loc[v['VCF0301'] == 5, 'leaning'] = 1
    v.loc[v['VCF0301'] == 3, 'leaning'] = -1
    y = v['vote_rep']
    X = sm.add_constant(v[['strong', 'weak', 'leaning']])
    try:
        model = Probit(y, X)
        res = model.fit(disp=0, method='newton', maxiter=100)
        return res.params['strong'], res.params['weak'], res.params['leaning']
    except:
        return None

def get_props(subset, excl_ind=False):
    if excl_ind:
        base = subset[subset['VCF0301'].isin([1,2,3,5,6,7])]
    else:
        base = subset[subset['VCF0301'].isin([1,2,3,4,5,6,7])]
    n = len(base)
    if n < 10:
        return None
    ps = len(base[base['VCF0301'].isin([1,7])]) / n
    pw = len(base[base['VCF0301'].isin([2,6])]) / n
    pl = len(base[base['VCF0301'].isin([3,5])]) / n
    return ps, pw, pl

def avg_coef(coefs, props):
    return coefs[0]*props[0] + coefs[1]*props[1] + coefs[2]*props[2]

# Key question: What does Bartels actually do for Figure 4?
# From Bartels p.40: "Figure 4 disaggregates these results for white southerners and
# white non-southerners"
# From Bartels p.39: "I summarize the results of these probit analyses by calculating
# an average probit coefficient for each election year"
#
# The most natural reading: run the SAME probit specification as Table 1 / Figure 3
# (on ALL major-party voters), then compute the average coefficient using
# REGION-SPECIFIC proportions.
#
# But another reading: run SEPARATE probits for each region's white voters.
#
# Let me also try: White-only voters probit (not region-specific, but white only)
# with region-specific proportions.

print("APPROACH COMPARISON WITH DETAILED YEAR-BY-YEAR ANALYSIS")
print("=" * 100)

# Test the hypothesis that Bartels used ALL-voter probit (Table 1 coefficients)
# with region-specific white proportions from the full PID sample

for year in pres_years:
    ydf = df[df['VCF0004'] == year].copy()

    # All voters probit (Table 1 specification)
    all_voters = ydf[(ydf['VCF0704a'].isin([1,2])) & (ydf['VCF0301'].isin([1,2,3,4,5,6,7]))]
    c_all = run_probit(all_voters)

    # White-only voters probit
    white_voters = ydf[(ydf['VCF0105a']==1) & (ydf['VCF0704a'].isin([1,2])) & (ydf['VCF0301'].isin([1,2,3,4,5,6,7]))]
    c_white = run_probit(white_voters)

    # Separate probits
    ns_voters = ydf[(ydf['VCF0105a']==1) & (ydf['VCF0113']==2) & (ydf['VCF0704a'].isin([1,2])) & (ydf['VCF0301'].isin([1,2,3,4,5,6,7]))]
    s_voters = ydf[(ydf['VCF0105a']==1) & (ydf['VCF0113']==1) & (ydf['VCF0704a'].isin([1,2])) & (ydf['VCF0301'].isin([1,2,3,4,5,6,7]))]
    c_ns = run_probit(ns_voters)
    c_s = run_probit(s_voters)

    # Region-specific white full PID (excl ind) proportions
    ns_all_pid = ydf[(ydf['VCF0105a']==1) & (ydf['VCF0113']==2) & (ydf['VCF0301'].isin([1,2,3,5,6,7]))]
    s_all_pid = ydf[(ydf['VCF0105a']==1) & (ydf['VCF0113']==1) & (ydf['VCF0301'].isin([1,2,3,5,6,7]))]
    p_ns = get_props(ydf[(ydf['VCF0105a']==1) & (ydf['VCF0113']==2)], excl_ind=True)
    p_s = get_props(ydf[(ydf['VCF0105a']==1) & (ydf['VCF0113']==1)], excl_ind=True)

    # Region-specific white voter proportions (incl ind)
    pv_ns = get_props(ns_voters, excl_ind=False)
    pv_s = get_props(s_voters, excl_ind=False)

    # Region-specific white voter proportions (excl ind)
    pv_ns_x = get_props(ns_voters, excl_ind=True)
    pv_s_x = get_props(s_voters, excl_ind=True)

    # Full PID incl ind
    p_ns_i = get_props(ydf[(ydf['VCF0105a']==1) & (ydf['VCF0113']==2)], excl_ind=False)
    p_s_i = get_props(ydf[(ydf['VCF0105a']==1) & (ydf['VCF0113']==1)], excl_ind=False)

    print(f"\n{year}: GT_NS={gt_ns[year]:.2f} GT_S={gt_s[year]:.2f}")
    print(f"  Probit coefs - All: s={c_all[0]:.3f} w={c_all[1]:.3f} l={c_all[2]:.3f}")
    if c_white:
        print(f"  Probit coefs - White: s={c_white[0]:.3f} w={c_white[1]:.3f} l={c_white[2]:.3f}")
    if c_ns:
        print(f"  Probit coefs - NS: s={c_ns[0]:.3f} w={c_ns[1]:.3f} l={c_ns[2]:.3f}")
    if c_s:
        print(f"  Probit coefs - S:  s={c_s[0]:.3f} w={c_s[1]:.3f} l={c_s[2]:.3f}")

    print(f"  Props full excl_ind - NS: {p_ns}, S: {p_s}")
    print(f"  Props voters incl  - NS: {pv_ns}, S: {pv_s}")

    # Compute for each combo
    combos = []
    for c_name, c_vals in [('All', c_all), ('White', c_white), ('NS_sep', c_ns)]:
        if c_vals is None:
            continue
        for p_name, (p_n, p_so) in [('full_excl', (p_ns, p_s)), ('full_incl', (p_ns_i, p_s_i)),
                                      ('voter_incl', (pv_ns, pv_s)), ('voter_excl', (pv_ns_x, pv_s_x))]:
            if p_n and p_so:
                ns_val = avg_coef(c_vals, p_n)
                combos.append((f"{c_name}+{p_name}", ns_val, 'NS'))

    # For South, use separate probit if available
    for c_name, c_vals in [('All', c_all), ('White', c_white), ('S_sep', c_s)]:
        if c_vals is None:
            continue
        for p_name, p_so in [('full_excl', p_s), ('full_incl', p_s_i),
                              ('voter_incl', pv_s), ('voter_excl', pv_s_x)]:
            if p_so:
                s_val = avg_coef(c_vals, p_so)
                combos.append((f"{c_name}+{p_name}", s_val, 'S'))

    # Find best NS and S combos
    ns_combos = [(n, v) for n, v, r in combos if r == 'NS']
    s_combos = [(n, v) for n, v, r in combos if r == 'S']

    ns_combos.sort(key=lambda x: abs(x[1] - gt_ns[year]))
    s_combos.sort(key=lambda x: abs(x[1] - gt_s[year]))

    print(f"  Best NS matches:")
    for n, v in ns_combos[:5]:
        print(f"    {n:<25} = {v:.4f} (diff={abs(v-gt_ns[year]):.4f})")
    print(f"  Best S matches:")
    for n, v in s_combos[:5]:
        print(f"    {n:<25} = {v:.4f} (diff={abs(v-gt_s[year]):.4f})")
