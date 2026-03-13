#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_factor_analysis import compute_nation_level_factor_scores
import pandas as pd
import numpy as np

scores, loadings, means = compute_nation_level_factor_scores()

sd = scores['trad_secrat'].std()
scores['trad_secrat_norm'] = scores['trad_secrat'] / sd
print(f'SD of trad_secrat: {sd:.3f}')
print(f'After normalization: range {scores["trad_secrat_norm"].min():.3f} to {scores["trad_secrat_norm"].max():.3f}')
print(f'  mean={scores["trad_secrat_norm"].mean():.3f}, sd={scores["trad_secrat_norm"].std():.3f}')
print()

for c in ['SWE','JPN','KOR','CHN','TWN','DEU','USA','NGA','PRI','POL','IRL','BRA','MEX','ARG',
          'BGR','CZE','EST','HUN','LVA','LTU','RUS','SVK','SVN','FRA','BEL','ITA','ESP','AUT']:
    row = scores[scores['COUNTRY_ALPHA']==c]
    if len(row) > 0:
        print(f'  {c}: raw={row["trad_secrat"].values[0]:+.3f}, norm={row["trad_secrat_norm"].values[0]:+.3f}')

print()
print('Paper Figure 3 approximate values vs computed (normalized):')
fig3_vals = {'SWE':1.30, 'JPN':1.50, 'KOR':0.95, 'CHN':0.85, 'TWN':0.65,
             'USA':-0.95, 'NGA':-2.00, 'PRI':-2.10, 'POL':-0.35, 'IRL':-1.20, 'BRA':-1.55,
             'BGR':0.80, 'CZE':1.00, 'EST':1.30, 'RUS':0.80, 'ARG':-0.70,
             'FRA':0.05, 'ESP':-0.55, 'ITA':-0.15, 'AUT':-0.05}
for c, fig3 in sorted(fig3_vals.items()):
    row = scores[scores['COUNTRY_ALPHA']==c]
    if len(row) > 0:
        norm = row['trad_secrat_norm'].values[0]
        print(f'  {c}: fig3={fig3:+.2f}, computed={norm:+.2f}, diff={norm-fig3:+.2f}')
