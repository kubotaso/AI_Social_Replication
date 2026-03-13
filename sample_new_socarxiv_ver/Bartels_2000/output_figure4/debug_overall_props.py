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
    v['strong'] = 0; v.loc[v['VCF0301'] == 7, 'strong'] = 1; v.loc[v['VCF0301'] == 1, 'strong'] = -1
    v['weak'] = 0; v.loc[v['VCF0301'] == 6, 'weak'] = 1; v.loc[v['VCF0301'] == 2, 'weak'] = -1
    v['leaning'] = 0; v.loc[v['VCF0301'] == 5, 'leaning'] = 1; v.loc[v['VCF0301'] == 3, 'leaning'] = -1
    y = v['vote_rep']
    X = sm.add_constant(v[['strong', 'weak', 'leaning']])
    try:
        res = Probit(y, X).fit(disp=0, method='newton', maxiter=100)
        return res.params['strong'], res.params['weak'], res.params['leaning']
    except:
        return None

# Approach: Separate white regional probits + OVERALL voter proportions (same as Figure 3)
print("APPROACH: Separate white regional probits + OVERALL voter proportions")
print("=" * 90)

ns_ov = {}
s_ov = {}

for year in pres_years:
    ydf = df[df['VCF0004'] == year].copy()

    # Overall voter proportions (same as Figure 3)
    all_voters = ydf[(ydf['VCF0704a'].isin([1,2])) & (ydf['VCF0301'].isin([1,2,3,4,5,6,7]))]
    n_all = len(all_voters)
    ps_all = len(all_voters[all_voters['VCF0301'].isin([1,7])]) / n_all
    pw_all = len(all_voters[all_voters['VCF0301'].isin([2,6])]) / n_all
    pl_all = len(all_voters[all_voters['VCF0301'].isin([3,5])]) / n_all

    for region, out_dict in [(2, ns_ov), (1, s_ov)]:
        voters = ydf[(ydf['VCF0105a']==1) & (ydf['VCF0113']==region) &
                      (ydf['VCF0704a'].isin([1,2])) & (ydf['VCF0301'].isin([1,2,3,4,5,6,7]))].copy()
        c = run_probit(voters)
        if c:
            out_dict[year] = c[0]*ps_all + c[1]*pw_all + c[2]*pl_all

for year in pres_years:
    ns_v = ns_ov.get(year, float('nan'))
    s_v = s_ov.get(year, float('nan'))
    print(f"{year}: NS={ns_v:.4f} (gt={gt_ns[year]:.2f} diff={abs(ns_v-gt_ns[year]):.4f})  "
          f"S={s_v:.4f} (gt={gt_s[year]:.2f} diff={abs(s_v-gt_s[year]):.4f})")


# Approach: White regional probits + WHITE regional voter props EXCLUDING pure independents
print("\n\nAPPROACH: White regional probits + WHITE voter props (excl ind)")
print("=" * 90)

ns_wve = {}
s_wve = {}

for year in pres_years:
    ydf = df[df['VCF0004'] == year].copy()

    for region, out_dict in [(2, ns_wve), (1, s_wve)]:
        voters = ydf[(ydf['VCF0105a']==1) & (ydf['VCF0113']==region) &
                      (ydf['VCF0704a'].isin([1,2])) & (ydf['VCF0301'].isin([1,2,3,4,5,6,7]))].copy()
        c = run_probit(voters)
        if c:
            partisans = voters[voters['VCF0301'] != 4]
            n = len(partisans)
            if n < 10:
                continue
            ps = len(partisans[partisans['VCF0301'].isin([1,7])]) / n
            pw = len(partisans[partisans['VCF0301'].isin([2,6])]) / n
            pl = len(partisans[partisans['VCF0301'].isin([3,5])]) / n
            out_dict[year] = c[0]*ps + c[1]*pw + c[2]*pl

for year in pres_years:
    ns_v = ns_wve.get(year, float('nan'))
    s_v = s_wve.get(year, float('nan'))
    print(f"{year}: NS={ns_v:.4f} (gt={gt_ns[year]:.2f} diff={abs(ns_v-gt_ns[year]):.4f})  "
          f"S={s_v:.4f} (gt={gt_s[year]:.2f} diff={abs(s_v-gt_s[year]):.4f})")


# Approach: All-voter probit + WHITE regional voter props (excl ind)
print("\n\nAPPROACH: All-voter probit + WHITE voter props (excl ind)")
print("=" * 90)

ns_ave = {}
s_ave = {}

for year in pres_years:
    ydf = df[df['VCF0004'] == year].copy()
    all_voters = ydf[(ydf['VCF0704a'].isin([1,2])) & (ydf['VCF0301'].isin([1,2,3,4,5,6,7]))]
    c = run_probit(all_voters)
    if not c:
        continue

    for region, out_dict in [(2, ns_ave), (1, s_ave)]:
        voters = ydf[(ydf['VCF0105a']==1) & (ydf['VCF0113']==region) &
                      (ydf['VCF0704a'].isin([1,2])) & (ydf['VCF0301'].isin([1,2,3,4,5,6,7]))]
        partisans = voters[voters['VCF0301'] != 4]
        n = len(partisans)
        if n < 10:
            continue
        ps = len(partisans[partisans['VCF0301'].isin([1,7])]) / n
        pw = len(partisans[partisans['VCF0301'].isin([2,6])]) / n
        pl = len(partisans[partisans['VCF0301'].isin([3,5])]) / n
        out_dict[year] = c[0]*ps + c[1]*pw + c[2]*pl

for year in pres_years:
    ns_v = ns_ave.get(year, float('nan'))
    s_v = s_ave.get(year, float('nan'))
    print(f"{year}: NS={ns_v:.4f} (gt={gt_ns[year]:.2f} diff={abs(ns_v-gt_ns[year]):.4f})  "
          f"S={s_v:.4f} (gt={gt_s[year]:.2f} diff={abs(s_v-gt_s[year]):.4f})")


# Score all
def data_score(ns, s):
    total = 0
    for gt, gen in [(gt_ns, ns), (gt_s, s)]:
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

print(f"\nSeparate probit + overall voter props: {data_score(ns_ov, s_ov)}/40")
print(f"Separate probit + white voter excl ind props: {data_score(ns_wve, s_wve)}/40")
print(f"All-voter probit + white voter excl ind props: {data_score(ns_ave, s_ave)}/40")
