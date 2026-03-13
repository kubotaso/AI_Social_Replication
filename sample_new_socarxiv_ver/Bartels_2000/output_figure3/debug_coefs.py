import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('anes_cumulative.csv', low_memory=False)

pres_years = [1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996]

# Paper's Table 1 ground truth
paper_coefs = {
    1952: {'strong': 1.600, 'weak': 0.928, 'leaning': 0.902},
    1956: {'strong': 1.713, 'weak': 0.941, 'leaning': 1.017},
    1960: {'strong': 1.650, 'weak': 0.822, 'leaning': 1.189},
    1964: {'strong': 1.470, 'weak': 0.548, 'leaning': 0.981},
    1968: {'strong': 1.770, 'weak': 0.881, 'leaning': 0.935},
    1972: {'strong': 1.221, 'weak': 0.603, 'leaning': 0.727},
    1976: {'strong': 1.565, 'weak': 0.745, 'leaning': 0.877},
    1980: {'strong': 1.602, 'weak': 0.929, 'leaning': 0.699},
    1984: {'strong': 1.596, 'weak': 0.975, 'leaning': 1.174},
    1988: {'strong': 1.770, 'weak': 0.771, 'leaning': 1.095},
    1992: {'strong': 1.851, 'weak': 0.912, 'leaning': 1.215},
    1996: {'strong': 1.946, 'weak': 1.022, 'leaning': 0.942},
}

print(f"{'Year':<6} {'strong':>8} {'weak':>8} {'lean':>8}  {'p_s':>6} {'p_w':>6} {'p_l':>6}  {'avg_mine':>10} {'avg_paper':>10}")
print("-" * 80)

for year in pres_years:
    year_df = df[df['VCF0004'] == year].copy()

    # Proportions over all valid PID
    valid_pid = year_df[year_df['VCF0301'].isin([1,2,3,4,5,6,7])].copy()
    n_valid = len(valid_pid)

    n_strong = len(valid_pid[valid_pid['VCF0301'].isin([1,7])])
    n_weak = len(valid_pid[valid_pid['VCF0301'].isin([2,6])])
    n_lean = len(valid_pid[valid_pid['VCF0301'].isin([3,5])])

    ps = n_strong / n_valid
    pw = n_weak / n_valid
    pl = n_lean / n_valid

    # My probit
    voters = year_df[year_df['VCF0704a'].isin([1, 2])].copy()
    voters = voters[voters['VCF0301'].isin([1, 2, 3, 4, 5, 6, 7])].copy()
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
    X = voters[['strong', 'weak', 'leaning']]
    X = sm.add_constant(X)
    model = Probit(y, X)
    result = model.fit(disp=0, method='newton', maxiter=100)

    cs = result.params['strong']
    cw = result.params['weak']
    cl = result.params['leaning']

    avg_mine = cs * ps + cw * pw + cl * pl

    # Using paper's coefficients with my proportions
    pc = paper_coefs[year]
    avg_paper = pc['strong'] * ps + pc['weak'] * pw + pc['leaning'] * pl

    print(f"{year:<6} {cs:>8.3f} {cw:>8.3f} {cl:>8.3f}  {ps:>6.3f} {pw:>6.3f} {pl:>6.3f}  {avg_mine:>10.4f} {avg_paper:>10.4f}")
