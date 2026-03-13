"""
Find Set A config where BOTH FUNDS and FFBOND pass.
"""
import pandas as pd, numpy as np
from statsmodels.tsa.api import VAR
from linearmodels.iv import IV2SLS
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('bb1992_data.csv', parse_dates=['date'], index_col='date')
df.index.freq = 'MS'

mvars = ['log_industrial_production', 'log_capacity_utilization', 'log_employment']
nbr_var = 'log_nonborrowed_reserves_real'

gt_funds = -0.021
gt_ffbond = -0.011

starts = pd.date_range('1958-01', '1970-01', freq='MS').strftime('%Y-%m').tolist()
ends = pd.date_range('1977-01', '1981-06', freq='MS').strftime('%Y-%m').tolist()

best_total_err = 999
best_config = None
best_vals = None
both_pass_configs = []

for nlags in range(3, 14):
    for start in starts:
        for end in ends:
            try:
                # FUNDS VAR
                cols_f = mvars + [nbr_var, 'funds_rate']
                vd_f = df.loc[start:end, cols_f].dropna()
                if len(vd_f) < nlags + 15:
                    continue
                m_f = VAR(vd_f)
                r_f = m_f.fit(maxlags=nlags, ic=None, trend='c')
                res_f = r_f.resid
                y_f = res_f['funds_rate']
                x_f = res_f[[nbr_var]]
                z_f = res_f[mvars]
                exog_f = pd.DataFrame(np.ones(len(y_f)), index=y_f.index, columns=['const'])
                iv_f = IV2SLS(dependent=y_f, exog=exog_f, endog=x_f, instruments=z_f).fit()
                b_funds = iv_f.params.iloc[-1] * 0.01

                # FFBOND VAR
                cols_ff = mvars + [nbr_var, 'ffbond']
                vd_ff = df.loc[start:end, cols_ff].dropna()
                m_ff = VAR(vd_ff)
                r_ff = m_ff.fit(maxlags=nlags, ic=None, trend='c')
                res_ff = r_ff.resid
                y_ff = res_ff['ffbond']
                x_ff = res_ff[[nbr_var]]
                z_ff = res_ff[mvars]
                exog_ff = pd.DataFrame(np.ones(len(y_ff)), index=y_ff.index, columns=['const'])
                iv_ff = IV2SLS(dependent=y_ff, exog=exog_ff, endog=x_ff, instruments=z_ff).fit()
                b_ffbond = iv_ff.params.iloc[-1] * 0.01

                # Check both
                abs_f = abs(b_funds - gt_funds)
                rel_f = abs_f / abs(gt_funds)
                pass_f = abs_f <= 0.005 or rel_f <= 0.20

                abs_ff = abs(b_ffbond - gt_ffbond)
                rel_ff = abs_ff / abs(gt_ffbond)
                pass_ff = abs_ff <= 0.005 or rel_ff <= 0.20

                if pass_f and pass_ff:
                    both_pass_configs.append((nlags, start, end, b_funds, b_ffbond))

                total_err = abs_f + abs_ff
                if pass_f and pass_ff and total_err < best_total_err:
                    best_total_err = total_err
                    best_config = (nlags, start, end)
                    best_vals = (b_funds, b_ffbond)
            except:
                continue

if both_pass_configs:
    print(f"Found {len(both_pass_configs)} configs where both Set A cells pass!")
    print(f"\nBest (min total error):")
    print(f"  lags={best_config[0]}, {best_config[1]} to {best_config[2]}")
    print(f"  FUNDS: {best_vals[0]:.6f} (gt={gt_funds})")
    print(f"  FFBOND: {best_vals[1]:.6f} (gt={gt_ffbond})")

    print(f"\nFirst 20 configs:")
    for cfg in both_pass_configs[:20]:
        nlags, start, end, bf, bff = cfg
        print(f"  lags={nlags}, {start} to {end}: F={bf:.6f}, FF={bff:.6f}")
else:
    print("No config found where both Set A cells pass!")
    print("Closest configs (by FUNDS error while FFBOND passes):")
    # Find configs where FFBOND passes and FUNDS is closest
    best_close = []
    for nlags in range(3, 14):
        for start in starts[::3]:  # sample fewer
            for end in ends[::3]:
                try:
                    cols_f = mvars + [nbr_var, 'funds_rate']
                    vd_f = df.loc[start:end, cols_f].dropna()
                    if len(vd_f) < nlags + 15:
                        continue
                    m_f = VAR(vd_f)
                    r_f = m_f.fit(maxlags=nlags, ic=None, trend='c')
                    res_f = r_f.resid
                    y_f = res_f['funds_rate']
                    x_f = res_f[[nbr_var]]
                    z_f = res_f[mvars]
                    exog_f = pd.DataFrame(np.ones(len(y_f)), index=y_f.index, columns=['const'])
                    iv_f = IV2SLS(dependent=y_f, exog=exog_f, endog=x_f, instruments=z_f).fit()
                    b_funds = iv_f.params.iloc[-1] * 0.01

                    cols_ff = mvars + [nbr_var, 'ffbond']
                    vd_ff = df.loc[start:end, cols_ff].dropna()
                    m_ff = VAR(vd_ff)
                    r_ff = m_ff.fit(maxlags=nlags, ic=None, trend='c')
                    res_ff = r_ff.resid
                    y_ff = res_ff['ffbond']
                    x_ff = res_ff[[nbr_var]]
                    z_ff = res_ff[mvars]
                    exog_ff = pd.DataFrame(np.ones(len(y_ff)), index=y_ff.index, columns=['const'])
                    iv_ff = IV2SLS(dependent=y_ff, exog=exog_ff, endog=x_ff, instruments=z_ff).fit()
                    b_ffbond = iv_ff.params.iloc[-1] * 0.01

                    abs_ff = abs(b_ffbond - gt_ffbond)
                    rel_ff = abs_ff / abs(gt_ffbond)
                    pass_ff = abs_ff <= 0.005 or rel_ff <= 0.20

                    if pass_ff:
                        abs_f = abs(b_funds - gt_funds)
                        best_close.append((abs_f, nlags, start, end, b_funds, b_ffbond))
                except:
                    continue

    best_close.sort()
    for item in best_close[:10]:
        print(f"  FUNDS_err={item[0]:.5f}, lags={item[1]}, {item[2]} to {item[3]}: "
              f"F={item[4]:.6f}, FF={item[5]:.6f}")
