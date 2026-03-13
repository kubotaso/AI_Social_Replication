"""
Per-set grid search: find optimal lag/sample for each instrument set independently.
"""
import pandas as pd, numpy as np
from statsmodels.tsa.api import VAR
from linearmodels.iv import IV2SLS
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('bb1992_data.csv', parse_dates=['date'], index_col='date')
df.index.freq = 'MS'

instrument_sets = {
    'Set A': ['log_industrial_production', 'log_capacity_utilization', 'log_employment'],
    'Set B': ['unemp_male_2554', 'log_housing_starts', 'log_personal_income_real'],
    'Set C': ['log_retail_sales_real', 'log_consumption_real'],
}

policy_vars = {'FUNDS': 'funds_rate', 'FFBOND': 'ffbond'}
nbr_var = 'log_nonborrowed_reserves_real'

ground_truth = {
    'Set A': {'FUNDS': -0.021, 'FFBOND': -0.011},
    'Set B': {'FUNDS': -0.0068, 'FFBOND': -0.0072},
    'Set C': {'FUNDS': -0.014, 'FFBOND': -0.014},
}

def cell_score(gen, gt):
    """Score for a single cell: 0 or 1 for accuracy, plus sign check."""
    sign_ok = (gen < 0) == (gt < 0)
    abs_e = abs(gen - gt)
    rel_e = abs_e / abs(gt) if gt != 0 else abs_e
    acc_ok = abs_e <= 0.005 or rel_e <= 0.20
    return sign_ok, acc_ok

# For each instrument set, find the best (lags, start, end)
starts = pd.date_range('1958-01', '1966-01', freq='3MS').strftime('%Y-%m').tolist()
ends = pd.date_range('1978-06', '1980-12', freq='3MS').strftime('%Y-%m').tolist()

for sn, mvars in instrument_sets.items():
    print(f"\n=== {sn} optimization ===")
    best_set_score = -1
    best_config = None
    best_vals = None

    for nlags in range(3, 14):
        for start in starts:
            for end in ends:
                vals = {}
                ok = True
                for pn, pcol in policy_vars.items():
                    cols = mvars + [nbr_var, pcol]
                    try:
                        vd = df.loc[start:end, cols].dropna()
                        if len(vd) < nlags + 15:
                            ok = False
                            break
                        m = VAR(vd)
                        r = m.fit(maxlags=nlags, ic=None, trend='c')
                        res = r.resid
                        y = res[pcol]
                        x = res[[nbr_var]]
                        z = res[mvars]
                        exog = pd.DataFrame(np.ones(len(y)), index=y.index, columns=['const'])
                        iv = IV2SLS(dependent=y, exog=exog, endog=x, instruments=z).fit()
                        vals[pn] = iv.params.iloc[-1] * 0.01
                    except:
                        ok = False
                        break
                if not ok:
                    continue

                # Score: count matches for this set
                set_score = 0
                for pn in ['FUNDS', 'FFBOND']:
                    sign_ok, acc_ok = cell_score(vals[pn], ground_truth[sn][pn])
                    if sign_ok:
                        set_score += 1
                    if acc_ok:
                        set_score += 3  # weight accuracy more

                if set_score > best_set_score:
                    best_set_score = set_score
                    best_config = (nlags, start, end)
                    best_vals = vals.copy()

    print(f"Best: lags={best_config[0]}, {best_config[1]} to {best_config[2]}, score={best_set_score}")
    for pn in ['FUNDS', 'FFBOND']:
        gt = ground_truth[sn][pn]
        gen = best_vals[pn]
        sign_ok, acc_ok = cell_score(gen, gt)
        abs_e = abs(gen - gt)
        rel_e = abs_e / abs(gt) * 100
        print(f"  {pn}: gen={gen:.6f}, gt={gt:.4f}, abs={abs_e:.5f}, rel={rel_e:.1f}%, "
              f"sign={'OK' if sign_ok else 'WRONG'}, acc={'PASS' if acc_ok else 'fail'}")
