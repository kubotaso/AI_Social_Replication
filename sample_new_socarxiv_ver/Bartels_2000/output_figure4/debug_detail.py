import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('anes_cumulative.csv', low_memory=False)

pres_years = [1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996]

print("White Non-South:")
print(f"{'Year':<6} {'N':>5} {'strong':>8} {'weak':>8} {'lean':>8} {'p_s':>6} {'p_w':>6} {'p_l':>6} {'avg':>8}")
print("-" * 65)

for year in pres_years:
    year_df = df[df['VCF0004'] == year].copy()
    voters = year_df[year_df['VCF0704a'].isin([1, 2])].copy()
    voters = voters[voters['VCF0301'].isin([1, 2, 3, 4, 5, 6, 7])].copy()
    white_voters = voters[voters['VCF0105a'] == 1].copy()
    group = white_voters[white_voters['VCF0113'] == 2].copy()

    if len(group) < 20:
        print(f"{year:<6} {len(group):>5} -- too few")
        continue

    v = group.copy()
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

    n = len(v)
    ps = len(v[v['VCF0301'].isin([1, 7])]) / n
    pw = len(v[v['VCF0301'].isin([2, 6])]) / n
    pl = len(v[v['VCF0301'].isin([3, 5])]) / n

    y = v['vote_rep']
    X = v[['strong', 'weak', 'leaning']]
    X = sm.add_constant(X)
    try:
        model = Probit(y, X)
        result = model.fit(disp=0, method='newton', maxiter=100)
        cs = result.params['strong']
        cw = result.params['weak']
        cl = result.params['leaning']
        avg = cs * ps + cw * pw + cl * pl
        print(f"{year:<6} {n:>5} {cs:>8.3f} {cw:>8.3f} {cl:>8.3f} {ps:>6.3f} {pw:>6.3f} {pl:>6.3f} {avg:>8.4f}")
    except Exception as e:
        print(f"{year:<6} {n:>5} ERROR: {e}")

print("\n\nWhite South:")
print(f"{'Year':<6} {'N':>5} {'strong':>8} {'weak':>8} {'lean':>8} {'p_s':>6} {'p_w':>6} {'p_l':>6} {'avg':>8}")
print("-" * 65)

for year in pres_years:
    year_df = df[df['VCF0004'] == year].copy()
    voters = year_df[year_df['VCF0704a'].isin([1, 2])].copy()
    voters = voters[voters['VCF0301'].isin([1, 2, 3, 4, 5, 6, 7])].copy()
    white_voters = voters[voters['VCF0105a'] == 1].copy()
    group = white_voters[white_voters['VCF0113'] == 1].copy()

    if len(group) < 20:
        print(f"{year:<6} {len(group):>5} -- too few")
        continue

    v = group.copy()
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

    n = len(v)
    ps = len(v[v['VCF0301'].isin([1, 7])]) / n
    pw = len(v[v['VCF0301'].isin([2, 6])]) / n
    pl = len(v[v['VCF0301'].isin([3, 5])]) / n

    y = v['vote_rep']
    X = v[['strong', 'weak', 'leaning']]
    X = sm.add_constant(X)
    try:
        model = Probit(y, X)
        result = model.fit(disp=0, method='newton', maxiter=100)
        cs = result.params['strong']
        cw = result.params['weak']
        cl = result.params['leaning']
        avg = cs * ps + cw * pw + cl * pl
        print(f"{year:<6} {n:>5} {cs:>8.3f} {cw:>8.3f} {cl:>8.3f} {ps:>6.3f} {pw:>6.3f} {pl:>6.3f} {avg:>8.4f}")
    except Exception as e:
        print(f"{year:<6} {n:>5} ERROR: {e}")

# Also check VCF0105b
print("\n\nVCF0105a value counts (1952):")
year_df = df[df['VCF0004'] == 1952]
print(year_df['VCF0105a'].value_counts().sort_index())
print("\nVCF0105b value counts (1952):")
print(year_df['VCF0105b'].value_counts().sort_index())
