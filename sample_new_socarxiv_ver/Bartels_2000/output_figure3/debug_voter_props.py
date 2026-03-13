import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('anes_cumulative.csv', low_memory=False)

pres_years = [1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996]

print("Method comparison for average probit coefficient:")
print(f"{'Year':<6} {'All PID':>10} {'Voters':>10} {'Excl Ind':>10} {'Voter ExInd':>10}")
print("-" * 50)

for year in pres_years:
    year_df = df[df['VCF0004'] == year].copy()

    # ALL respondents with valid PID
    all_valid = year_df[year_df['VCF0301'].isin([1,2,3,4,5,6,7])].copy()
    n_all = len(all_valid)
    ps_all = len(all_valid[all_valid['VCF0301'].isin([1,7])]) / n_all
    pw_all = len(all_valid[all_valid['VCF0301'].isin([2,6])]) / n_all
    pl_all = len(all_valid[all_valid['VCF0301'].isin([3,5])]) / n_all

    # VOTERS only with valid PID
    voters = year_df[year_df['VCF0704a'].isin([1, 2])].copy()
    voters = voters[voters['VCF0301'].isin([1, 2, 3, 4, 5, 6, 7])].copy()
    n_voters = len(voters)
    ps_voters = len(voters[voters['VCF0301'].isin([1,7])]) / n_voters
    pw_voters = len(voters[voters['VCF0301'].isin([2,6])]) / n_voters
    pl_voters = len(voters[voters['VCF0301'].isin([3,5])]) / n_voters

    # Excl pure ind (all respondents)
    partisans_all = all_valid[all_valid['VCF0301'].isin([1,2,3,5,6,7])].copy()
    n_part_all = len(partisans_all)
    ps_part = len(partisans_all[partisans_all['VCF0301'].isin([1,7])]) / n_part_all
    pw_part = len(partisans_all[partisans_all['VCF0301'].isin([2,6])]) / n_part_all
    pl_part = len(partisans_all[partisans_all['VCF0301'].isin([3,5])]) / n_part_all

    # Voters excl pure ind
    voters_part = voters[voters['VCF0301'].isin([1,2,3,5,6,7])].copy()
    n_vp = len(voters_part)
    ps_vp = len(voters_part[voters_part['VCF0301'].isin([1,7])]) / n_vp
    pw_vp = len(voters_part[voters_part['VCF0301'].isin([2,6])]) / n_vp
    pl_vp = len(voters_part[voters_part['VCF0301'].isin([3,5])]) / n_vp

    # Run probit
    voters2 = voters.copy()
    voters2['vote_rep'] = (voters2['VCF0704a'] == 2).astype(int)
    voters2['strong'] = 0
    voters2.loc[voters2['VCF0301'] == 7, 'strong'] = 1
    voters2.loc[voters2['VCF0301'] == 1, 'strong'] = -1
    voters2['weak'] = 0
    voters2.loc[voters2['VCF0301'] == 6, 'weak'] = 1
    voters2.loc[voters2['VCF0301'] == 2, 'weak'] = -1
    voters2['leaning'] = 0
    voters2.loc[voters2['VCF0301'] == 5, 'leaning'] = 1
    voters2.loc[voters2['VCF0301'] == 3, 'leaning'] = -1

    y = voters2['vote_rep']
    X = voters2[['strong', 'weak', 'leaning']]
    X = sm.add_constant(X)
    model = Probit(y, X)
    result = model.fit(disp=0, method='newton', maxiter=100)

    cs = result.params['strong']
    cw = result.params['weak']
    cl = result.params['leaning']

    avg_all = cs * ps_all + cw * pw_all + cl * pl_all
    avg_voters = cs * ps_voters + cw * pw_voters + cl * pl_voters
    avg_part = cs * ps_part + cw * pw_part + cl * pl_part
    avg_vp = cs * ps_vp + cw * pw_vp + cl * pl_vp

    print(f"{year:<6} {avg_all:>10.4f} {avg_voters:>10.4f} {avg_part:>10.4f} {avg_vp:>10.4f}")
