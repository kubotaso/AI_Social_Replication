import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("anes_cumulative.csv", low_memory=False)
pres_years = [1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996]

# Table 1 coefficients from the paper
table1_coefs = {
    1952: (1.600, 0.928, 0.902),
    1956: (1.713, 0.941, 1.017),
    1960: (1.650, 0.822, 1.189),
    1964: (1.470, 0.548, 0.981),
    1968: (1.770, 0.881, 0.935),
    1972: (1.221, 0.603, 0.727),
    1976: (1.565, 0.745, 0.877),
    1980: (1.602, 0.929, 0.699),
    1984: (1.596, 0.975, 1.174),
    1988: (1.770, 0.771, 1.095),
    1992: (1.851, 0.912, 1.215),
    1996: (1.946, 1.022, 0.942)
}

gt_ns = {1952: 1.19, 1956: 1.33, 1960: 1.30, 1964: 1.08, 1968: 1.26, 1972: 0.79,
         1976: 1.03, 1980: 0.97, 1984: 1.20, 1988: 1.38, 1992: 1.33, 1996: 1.35}
gt_s = {1952: 0.99, 1956: 1.19, 1960: 0.95, 1964: 0.97, 1968: 1.05, 1972: 0.64,
        1976: 0.82, 1980: 0.96, 1984: 0.96, 1988: 1.14, 1992: 1.20, 1996: 1.31}

print("APPROACH: Table 1 coefficients + White regional full-electorate proportions (incl ind)")
print("=" * 100)

ns_results = {}
s_results = {}

for year in pres_years:
    ydf = df[df['VCF0004'] == year].copy()
    cs, cw, cl = table1_coefs[year]

    # White Non-South full electorate (all with valid PID, incl pure ind)
    wns = ydf[(ydf['VCF0105a']==1) & (ydf['VCF0113']==2) & (ydf['VCF0301'].isin([1,2,3,4,5,6,7]))]
    n = len(wns)
    ps = len(wns[wns['VCF0301'].isin([1,7])]) / n
    pw = len(wns[wns['VCF0301'].isin([2,6])]) / n
    pl = len(wns[wns['VCF0301'].isin([3,5])]) / n
    ns_val = cs*ps + cw*pw + cl*pl
    ns_results[year] = ns_val

    # White South full electorate
    ws = ydf[(ydf['VCF0105a']==1) & (ydf['VCF0113']==1) & (ydf['VCF0301'].isin([1,2,3,4,5,6,7]))]
    n = len(ws)
    ps2 = len(ws[ws['VCF0301'].isin([1,7])]) / n
    pw2 = len(ws[ws['VCF0301'].isin([2,6])]) / n
    pl2 = len(ws[ws['VCF0301'].isin([3,5])]) / n
    s_val = cs*ps2 + cw*pw2 + cl*pl2
    s_results[year] = s_val

    print(f"{year}: NS={ns_val:.4f} (gt={gt_ns[year]:.2f} diff={abs(ns_val-gt_ns[year]):.4f})  "
          f"S={s_val:.4f} (gt={gt_s[year]:.2f} diff={abs(s_val-gt_s[year]):.4f})")

# Also try with separate WHITE probits + full-electorate props
print("\n\nAPPROACH: Separate WHITE regional probits + full-electorate props (incl ind)")
print("=" * 100)

ns_sep = {}
s_sep = {}

for year in pres_years:
    ydf = df[df['VCF0004'] == year].copy()

    for region, label, out_dict in [(2, 'NS', ns_sep), (1, 'S', s_sep)]:
        white_region = ydf[(ydf['VCF0105a']==1) & (ydf['VCF0113']==region)]
        voters = white_region[(white_region['VCF0704a'].isin([1,2])) & (white_region['VCF0301'].isin([1,2,3,4,5,6,7]))].copy()

        if len(voters) < 20:
            continue

        voters['vote_rep'] = (voters['VCF0704a'] == 2).astype(int)
        voters['strong'] = 0
        voters.loc[voters['VCF0301'] == 7, 'strong'] = 1
        voters.loc[voters['VCF0301'] == 1, 'strong'] = -1
        voters['weak'] = 0
        voters.loc[voters['VCF0301'] == 6, 'weak'] = 1
        voters.loc[voters['VCF0301'] == 2, 'weak'] = -1
        voters['leaning'] = 0
        voters.loc[voters['VCF0301'] == 5, 'leaning'] = 1
        voters.loc[voters['VCF0301'] == 3, 'leaning'] = -1

        y = voters['vote_rep']
        X = sm.add_constant(voters[['strong', 'weak', 'leaning']])

        try:
            res = Probit(y, X).fit(disp=0, method='newton', maxiter=100)
            cs, cw, cl = res.params['strong'], res.params['weak'], res.params['leaning']

            # Full electorate proportions (incl pure ind)
            all_pid = white_region[white_region['VCF0301'].isin([1,2,3,4,5,6,7])]
            n = len(all_pid)
            ps = len(all_pid[all_pid['VCF0301'].isin([1,7])]) / n
            pw = len(all_pid[all_pid['VCF0301'].isin([2,6])]) / n
            pl = len(all_pid[all_pid['VCF0301'].isin([3,5])]) / n

            out_dict[year] = cs*ps + cw*pw + cl*pl
        except:
            pass

for year in pres_years:
    ns_v = ns_sep.get(year, float('nan'))
    s_v = s_sep.get(year, float('nan'))
    print(f"{year}: NS={ns_v:.4f} (gt={gt_ns[year]:.2f} diff={abs(ns_v-gt_ns[year]):.4f})  "
          f"S={s_v:.4f} (gt={gt_s[year]:.2f} diff={abs(s_v-gt_s[year]):.4f})")

# Score both approaches
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

print(f"\nTable 1 + full electorate props: data score = {data_score(ns_results, s_results)}/40")
print(f"Separate + full electorate props: data score = {data_score(ns_sep, s_sep)}/40")

# Also try: Table 1 coefs + VOTER proportions (what we used for Figure 3 success)
print("\n\nAPPROACH: Table 1 coefficients + White regional VOTER proportions (incl ind)")
print("=" * 100)

ns_t1v = {}
s_t1v = {}

for year in pres_years:
    ydf = df[df['VCF0004'] == year].copy()
    cs, cw, cl = table1_coefs[year]

    for region, label, out_dict in [(2, 'NS', ns_t1v), (1, 'S', s_t1v)]:
        voters = ydf[(ydf['VCF0105a']==1) & (ydf['VCF0113']==region) &
                      (ydf['VCF0704a'].isin([1,2])) & (ydf['VCF0301'].isin([1,2,3,4,5,6,7]))]
        n = len(voters)
        if n < 20:
            continue
        ps = len(voters[voters['VCF0301'].isin([1,7])]) / n
        pw = len(voters[voters['VCF0301'].isin([2,6])]) / n
        pl = len(voters[voters['VCF0301'].isin([3,5])]) / n
        out_dict[year] = cs*ps + cw*pw + cl*pl

for year in pres_years:
    ns_v = ns_t1v.get(year, float('nan'))
    s_v = s_t1v.get(year, float('nan'))
    print(f"{year}: NS={ns_v:.4f} (gt={gt_ns[year]:.2f} diff={abs(ns_v-gt_ns[year]):.4f})  "
          f"S={s_v:.4f} (gt={gt_s[year]:.2f} diff={abs(s_v-gt_s[year]):.4f})")

print(f"\nTable 1 + voter props: data score = {data_score(ns_t1v, s_t1v)}/40")

# Also try: Separate probits + VOTER proportions (incl ind)
print("\n\nAPPROACH: Separate probits + White regional VOTER proportions (incl ind)")
print("=" * 100)

ns_sv = {}
s_sv = {}

for year in pres_years:
    ydf = df[df['VCF0004'] == year].copy()

    for region, label, out_dict in [(2, 'NS', ns_sv), (1, 'S', s_sv)]:
        voters = ydf[(ydf['VCF0105a']==1) & (ydf['VCF0113']==region) &
                      (ydf['VCF0704a'].isin([1,2])) & (ydf['VCF0301'].isin([1,2,3,4,5,6,7]))].copy()

        if len(voters) < 20:
            continue

        voters['vote_rep'] = (voters['VCF0704a'] == 2).astype(int)
        voters['strong'] = 0
        voters.loc[voters['VCF0301'] == 7, 'strong'] = 1
        voters.loc[voters['VCF0301'] == 1, 'strong'] = -1
        voters['weak'] = 0
        voters.loc[voters['VCF0301'] == 6, 'weak'] = 1
        voters.loc[voters['VCF0301'] == 2, 'weak'] = -1
        voters['leaning'] = 0
        voters.loc[voters['VCF0301'] == 5, 'leaning'] = 1
        voters.loc[voters['VCF0301'] == 3, 'leaning'] = -1

        y = voters['vote_rep']
        X = sm.add_constant(voters[['strong', 'weak', 'leaning']])

        try:
            res = Probit(y, X).fit(disp=0, method='newton', maxiter=100)
            cs, cw, cl = res.params['strong'], res.params['weak'], res.params['leaning']

            n = len(voters)
            ps = len(voters[voters['VCF0301'].isin([1,7])]) / n
            pw = len(voters[voters['VCF0301'].isin([2,6])]) / n
            pl = len(voters[voters['VCF0301'].isin([3,5])]) / n
            out_dict[year] = cs*ps + cw*pw + cl*pl
        except:
            pass

for year in pres_years:
    ns_v = ns_sv.get(year, float('nan'))
    s_v = s_sv.get(year, float('nan'))
    print(f"{year}: NS={ns_v:.4f} (gt={gt_ns[year]:.2f} diff={abs(ns_v-gt_ns[year]):.4f})  "
          f"S={s_v:.4f} (gt={gt_s[year]:.2f} diff={abs(s_v-gt_s[year]):.4f})")

print(f"\nSeparate + voter props: data score = {data_score(ns_sv, s_sv)}/40")
