import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('anes_cumulative.csv', low_memory=False)

pres_years = [1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996]

def compute_avg(v, prop_base):
    """Compute avg coefficient with probit on v, proportions from prop_base."""
    v = v.copy()
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

    n = len(prop_base)
    ps = len(prop_base[prop_base['VCF0301'].isin([1, 7])]) / n
    pw = len(prop_base[prop_base['VCF0301'].isin([2, 6])]) / n
    pl = len(prop_base[prop_base['VCF0301'].isin([3, 5])]) / n

    y = v['vote_rep']
    X = v[['strong', 'weak', 'leaning']]
    X = sm.add_constant(X)
    try:
        model = Probit(y, X)
        result = model.fit(disp=0, method='newton', maxiter=100)
        cs = result.params['strong']
        cw = result.params['weak']
        cl = result.params['leaning']
        return cs * ps + cw * pw + cl * pl
    except:
        return None

print("White Non-South - different proportion methods:")
print(f"{'Year':<6} {'Voter':>10} {'V_ExInd':>10} {'AllPID':>10} {'P_ExInd':>10}")
print("-" * 50)

for year in pres_years:
    year_df = df[df['VCF0004'] == year].copy()
    voters = year_df[year_df['VCF0704a'].isin([1, 2])].copy()
    voters = voters[voters['VCF0301'].isin([1, 2, 3, 4, 5, 6, 7])].copy()
    white_voters = voters[voters['VCF0105a'] == 1].copy()
    ns = white_voters[white_voters['VCF0113'] == 2].copy()

    if len(ns) < 20:
        print(f"{year:<6} -- too few --")
        continue

    # All people with valid PID in this subgroup
    all_pid = year_df[year_df['VCF0301'].isin([1, 2, 3, 4, 5, 6, 7])]
    all_pid_w = all_pid[all_pid['VCF0105a'] == 1]
    all_pid_ns = all_pid_w[all_pid_w['VCF0113'] == 2].copy()

    # Voters only
    avg_v = compute_avg(ns, ns)
    # Voters excluding pure ind
    ns_noind = ns[ns['VCF0301'] != 4].copy()
    avg_ve = compute_avg(ns, ns_noind) if len(ns_noind) >= 20 else None
    # All PID
    avg_a = compute_avg(ns, all_pid_ns) if len(all_pid_ns) >= 20 else None
    # All PID excl pure ind
    all_pid_ns_noind = all_pid_ns[all_pid_ns['VCF0301'] != 4].copy()
    avg_ae = compute_avg(ns, all_pid_ns_noind) if len(all_pid_ns_noind) >= 20 else None

    def fmt(x):
        return f"{x:>10.4f}" if x is not None else f"{'N/A':>10}"

    print(f"{year:<6} {fmt(avg_v)} {fmt(avg_ve)} {fmt(avg_a)} {fmt(avg_ae)}")

print("\n\nWhite South - different proportion methods:")
print(f"{'Year':<6} {'Voter':>10} {'V_ExInd':>10} {'AllPID':>10} {'P_ExInd':>10}")
print("-" * 50)

for year in pres_years:
    year_df = df[df['VCF0004'] == year].copy()
    voters = year_df[year_df['VCF0704a'].isin([1, 2])].copy()
    voters = voters[voters['VCF0301'].isin([1, 2, 3, 4, 5, 6, 7])].copy()
    white_voters = voters[voters['VCF0105a'] == 1].copy()
    s = white_voters[white_voters['VCF0113'] == 1].copy()

    if len(s) < 20:
        print(f"{year:<6} -- too few --")
        continue

    all_pid = year_df[year_df['VCF0301'].isin([1, 2, 3, 4, 5, 6, 7])]
    all_pid_w = all_pid[all_pid['VCF0105a'] == 1]
    all_pid_s = all_pid_w[all_pid_w['VCF0113'] == 1].copy()

    avg_v = compute_avg(s, s)
    s_noind = s[s['VCF0301'] != 4].copy()
    avg_ve = compute_avg(s, s_noind) if len(s_noind) >= 20 else None
    avg_a = compute_avg(s, all_pid_s) if len(all_pid_s) >= 20 else None
    all_pid_s_noind = all_pid_s[all_pid_s['VCF0301'] != 4].copy()
    avg_ae = compute_avg(s, all_pid_s_noind) if len(all_pid_s_noind) >= 20 else None

    def fmt(x):
        return f"{x:>10.4f}" if x is not None else f"{'N/A':>10}"

    print(f"{year:<6} {fmt(avg_v)} {fmt(avg_ve)} {fmt(avg_a)} {fmt(avg_ae)}")
