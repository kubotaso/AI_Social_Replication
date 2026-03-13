import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("anes_cumulative.csv", low_memory=False)
pres_years = [1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996]

# Ground truth from careful figure reading
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

# Test all approaches
approaches = {}

for year in pres_years:
    ydf = df[df['VCF0004'] == year].copy()

    # Subsets
    white_ns_all = ydf[(ydf['VCF0105a']==1) & (ydf['VCF0113']==2) & (ydf['VCF0301'].isin([1,2,3,4,5,6,7]))]
    white_s_all = ydf[(ydf['VCF0105a']==1) & (ydf['VCF0113']==1) & (ydf['VCF0301'].isin([1,2,3,4,5,6,7]))]
    white_ns_voters = white_ns_all[white_ns_all['VCF0704a'].isin([1,2])]
    white_s_voters = white_s_all[white_s_all['VCF0704a'].isin([1,2])]
    all_voters = ydf[(ydf['VCF0704a'].isin([1,2])) & (ydf['VCF0301'].isin([1,2,3,4,5,6,7]))]
    all_white_voters = ydf[(ydf['VCF0105a']==1) & (ydf['VCF0704a'].isin([1,2])) & (ydf['VCF0301'].isin([1,2,3,4,5,6,7]))]

    # Probit coefficients
    c_ns_sep = run_probit(white_ns_voters)
    c_s_sep = run_probit(white_s_voters)
    c_all = run_probit(all_voters)
    c_all_white = run_probit(all_white_voters)

    # Proportion methods
    # 1. Full PID sample (all with valid PID, incl pure indep) from that region
    p_ns_full = get_props(white_ns_all, excl_ind=False)
    p_s_full = get_props(white_s_all, excl_ind=False)
    # 2. Full PID sample, excluding pure independents
    p_ns_excl = get_props(white_ns_all, excl_ind=True)
    p_s_excl = get_props(white_s_all, excl_ind=True)
    # 3. Voter PID sample (incl pure indep)
    p_ns_voter = get_props(white_ns_voters, excl_ind=False)
    p_s_voter = get_props(white_s_voters, excl_ind=False)
    # 4. Voter PID sample, excluding pure independents
    p_ns_voter_excl = get_props(white_ns_voters, excl_ind=True)
    p_s_voter_excl = get_props(white_s_voters, excl_ind=True)

    # Compute all combinations
    combos = {}

    # Separate probit + full PID props (incl ind)
    if c_ns_sep and p_ns_full:
        combos[('sep','full_incl','ns')] = avg_coef(c_ns_sep, p_ns_full)
    if c_s_sep and p_s_full:
        combos[('sep','full_incl','s')] = avg_coef(c_s_sep, p_s_full)

    # Separate probit + full PID props (excl ind)
    if c_ns_sep and p_ns_excl:
        combos[('sep','full_excl','ns')] = avg_coef(c_ns_sep, p_ns_excl)
    if c_s_sep and p_s_excl:
        combos[('sep','full_excl','s')] = avg_coef(c_s_sep, p_s_excl)

    # Separate probit + voter props (incl ind)
    if c_ns_sep and p_ns_voter:
        combos[('sep','voter_incl','ns')] = avg_coef(c_ns_sep, p_ns_voter)
    if c_s_sep and p_s_voter:
        combos[('sep','voter_incl','s')] = avg_coef(c_s_sep, p_s_voter)

    # Separate probit + voter props (excl ind)
    if c_ns_sep and p_ns_voter_excl:
        combos[('sep','voter_excl','ns')] = avg_coef(c_ns_sep, p_ns_voter_excl)
    if c_s_sep and p_s_voter_excl:
        combos[('sep','voter_excl','s')] = avg_coef(c_s_sep, p_s_voter_excl)

    # All-voter probit + full PID props (incl ind)
    if c_all and p_ns_full:
        combos[('all','full_incl','ns')] = avg_coef(c_all, p_ns_full)
    if c_all and p_s_full:
        combos[('all','full_incl','s')] = avg_coef(c_all, p_s_full)

    # All-voter probit + full PID props (excl ind)
    if c_all and p_ns_excl:
        combos[('all','full_excl','ns')] = avg_coef(c_all, p_ns_excl)
    if c_all and p_s_excl:
        combos[('all','full_excl','s')] = avg_coef(c_all, p_s_excl)

    # All-white probit + full PID props (incl ind)
    if c_all_white and p_ns_full:
        combos[('allwhite','full_incl','ns')] = avg_coef(c_all_white, p_ns_full)
    if c_all_white and p_s_full:
        combos[('allwhite','full_incl','s')] = avg_coef(c_all_white, p_s_full)

    # All-white probit + full PID props (excl ind)
    if c_all_white and p_ns_excl:
        combos[('allwhite','full_excl','ns')] = avg_coef(c_all_white, p_ns_excl)
    if c_all_white and p_s_excl:
        combos[('allwhite','full_excl','s')] = avg_coef(c_all_white, p_s_excl)

    # All-white probit + voter props (incl ind)
    if c_all_white and p_ns_voter:
        combos[('allwhite','voter_incl','ns')] = avg_coef(c_all_white, p_ns_voter)
    if c_all_white and p_s_voter:
        combos[('allwhite','voter_incl','s')] = avg_coef(c_all_white, p_s_voter)

    # All-white probit + voter props (excl ind)
    if c_all_white and p_ns_voter_excl:
        combos[('allwhite','voter_excl','ns')] = avg_coef(c_all_white, p_ns_voter_excl)
    if c_all_white and p_s_voter_excl:
        combos[('allwhite','voter_excl','s')] = avg_coef(c_all_white, p_s_voter_excl)

    # All-voter probit + voter props
    if c_all and p_ns_voter:
        combos[('all','voter_incl','ns')] = avg_coef(c_all, p_ns_voter)
    if c_all and p_s_voter:
        combos[('all','voter_incl','s')] = avg_coef(c_all, p_s_voter)
    if c_all and p_ns_voter_excl:
        combos[('all','voter_excl','ns')] = avg_coef(c_all, p_ns_voter_excl)
    if c_all and p_s_voter_excl:
        combos[('all','voter_excl','s')] = avg_coef(c_all, p_s_voter_excl)

    for k, v in combos.items():
        approach_name = f"{k[0]}_{k[1]}"
        region = k[2]
        if approach_name not in approaches:
            approaches[approach_name] = {'ns': {}, 's': {}}
        approaches[approach_name][region][year] = v

# Score each approach
def score(ns_dict, s_dict):
    total = 0
    for gt, gen in [(gt_ns, ns_dict), (gt_s, s_dict)]:
        ds = 0
        for yr, gv in gt.items():
            if yr in gen:
                d = abs(gen[yr] - gv)
                if d < 0.05: ds += 1.0
                elif d < 0.10: ds += 0.75
                elif d < 0.15: ds += 0.5
                elif d < 0.20: ds += 0.25
                elif d < 0.30: ds += 0.1
        total += 20 * (ds / len(gt))
    return round(total, 1)

print(f"{'Approach':<25} {'DataScore':<10} {'Details'}")
print("=" * 120)

results = []
for name in sorted(approaches.keys()):
    ns = approaches[name]['ns']
    s = approaches[name]['s']
    sc = score(ns, s)
    results.append((name, sc, ns, s))

results.sort(key=lambda x: -x[1])

for name, sc, ns, s in results:
    print(f"\n{name}: Data Score = {sc}/40")
    print(f"  {'Year':<6} {'NS_gen':<10} {'NS_gt':<10} {'NS_diff':<10} {'S_gen':<10} {'S_gt':<10} {'S_diff':<10}")
    for yr in pres_years:
        ns_v = ns.get(yr, float('nan'))
        s_v = s.get(yr, float('nan'))
        ns_g = gt_ns[yr]
        s_g = gt_s[yr]
        ns_d = abs(ns_v - ns_g) if not np.isnan(ns_v) else float('nan')
        s_d = abs(s_v - s_g) if not np.isnan(s_v) else float('nan')
        print(f"  {yr:<6} {ns_v:<10.4f} {ns_g:<10.2f} {ns_d:<10.4f} {s_v:<10.4f} {s_g:<10.2f} {s_d:<10.4f}")
