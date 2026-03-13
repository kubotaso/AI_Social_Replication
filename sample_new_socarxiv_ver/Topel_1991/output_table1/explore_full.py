import pandas as pd
import numpy as np

df = pd.read_csv('data/cps_dws.csv')

deflator = {1978:72.2, 1979:78.6, 1980:85.7, 1981:94.0,
            1982:100.0, 1983:103.9, 1984:107.7, 1985:110.9, 1986:113.8}

# DWREAS 1-6 base (best match for pc and wu)
mask = (df['SEX']==1) & (df['AGE']>=20) & (df['AGE']<=60) & \
       df['DWREAS'].isin([1,2,3,4,5,6]) & df['EMPSTAT'].isin([10,12]) & (df['DWYEARS']<99)
sample = df[mask].copy()
sample['tenure_bin'] = pd.cut(sample['DWYEARS'], bins=[-0.1,5,10,20,100], labels=['0-5','6-10','11-20','21+'])
sample['plant_closing'] = (sample['DWREAS']==1).astype(int)
sample['disp_year'] = sample['YEAR'] - sample['DWLASTWRK']

# Wage sample
vw = (sample['DWWEEKL']>0) & (sample['DWWEEKL']<9000) & \
     (sample['DWWEEKC']>0) & (sample['DWWEEKC']<9000) & (sample['DWLASTWRK']<99)
ws = sample[vw].copy()
ws['def_cur'] = ws['YEAR'].map(deflator)
ws['def_pri'] = ws['disp_year'].map(deflator)
ws = ws.dropna(subset=['def_cur','def_pri'])
ws['real_lwc'] = np.log(ws['DWWEEKC']/ws['def_cur']) - np.log(ws['DWWEEKL']/ws['def_pri'])

us = sample[sample['DWWKSUN']<999]

bins_list = ['0-5','6-10','11-20','21+','Total']

print("UNWEIGHTED RESULTS:")
print(f"{'':30s} {'0-5':>10} {'6-10':>10} {'11-20':>10} {'21+':>10} {'Total':>10}")
for row in ['lwc', 'pc', 'wu']:
    means = []
    ses = []
    for b in bins_list:
        if b == 'Total':
            if row == 'lwc': d = ws
            elif row == 'pc': d = sample
            else: d = us
        else:
            if row == 'lwc': d = ws[ws['tenure_bin']==b]
            elif row == 'pc': d = sample[sample['tenure_bin']==b]
            else: d = us[us['tenure_bin']==b]

        if row == 'lwc':
            m = d['real_lwc'].mean()
            se = d['real_lwc'].std() / np.sqrt(len(d))
        elif row == 'pc':
            m = d['plant_closing'].mean()
            se = np.sqrt(m*(1-m)/len(d))
        else:
            m = d['DWWKSUN'].mean()
            se = d['DWWKSUN'].std() / np.sqrt(len(d))
        means.append(f"{m:.3f}")
        ses.append(f"({se:.3f})")

    labels = {'lwc': 'Avg chg log wkly wage', 'pc': 'Pct plant closing', 'wu': 'Weeks unemp'}
    print(f"{labels[row]:30s} " + " ".join(f"{v:>10}" for v in means))
    print(f"{'':30s} " + " ".join(f"{v:>10}" for v in ses))

# Also weighted
print("\n\nWEIGHTED RESULTS:")
print(f"{'':30s} {'0-5':>10} {'6-10':>10} {'11-20':>10} {'21+':>10} {'Total':>10}")
for row in ['lwc', 'pc', 'wu']:
    means = []
    ses = []
    for b in bins_list:
        if b == 'Total':
            if row == 'lwc': d = ws
            elif row == 'pc': d = sample
            else: d = us
        else:
            if row == 'lwc': d = ws[ws['tenure_bin']==b]
            elif row == 'pc': d = sample[sample['tenure_bin']==b]
            else: d = us[us['tenure_bin']==b]

        w = d['DWSUPPWT']

        if row == 'lwc':
            m = np.average(d['real_lwc'], weights=w)
            v = np.average((d['real_lwc'] - m)**2, weights=w)
            n_eff = w.sum()**2 / (w**2).sum()
            se = np.sqrt(v / n_eff)
        elif row == 'pc':
            m = np.average(d['plant_closing'], weights=w)
            n_eff = w.sum()**2 / (w**2).sum()
            se = np.sqrt(m*(1-m) / n_eff)
        else:
            m = np.average(d['DWWKSUN'], weights=w)
            v = np.average((d['DWWKSUN'] - m)**2, weights=w)
            n_eff = w.sum()**2 / (w**2).sum()
            se = np.sqrt(v / n_eff)

        means.append(f"{m:.3f}")
        ses.append(f"({se:.3f})")

    labels = {'lwc': 'Avg chg log wkly wage', 'pc': 'Pct plant closing', 'wu': 'Weeks unemp'}
    print(f"{labels[row]:30s} " + " ".join(f"{v:>10}" for v in means))
    print(f"{'':30s} " + " ".join(f"{v:>10}" for v in ses))

print("\n\nPAPER VALUES:")
print(f"{'':30s} {'0-5':>10} {'6-10':>10} {'11-20':>10} {'21+':>10} {'Total':>10}")
print(f"{'Avg chg log wkly wage':30s}     -0.095     -0.223     -0.282     -0.439     -0.135")
print(f"{'':30s}     (0.010)     (0.021)     (0.026)     (0.071)     (0.009)")
print(f"{'Pct plant closing':30s}      0.352      0.463      0.528      0.750      0.390")
print(f"{'':30s}     (0.008)     (0.021)     (0.026)     (0.043)     (0.007)")
print(f"{'Weeks unemp':30s}      18.69      24.54      26.66      31.79      20.41")
print(f"{'':30s}     (0.413)     (1.202)     (1.536)     (3.288)     (0.385)")

print(f"\nN (sample): {len(sample)}, N (wages): {len(ws)}, N (unemp): {len(us)}")
print(f"Tenure dist (sample): {sample['tenure_bin'].value_counts().sort_index().to_dict()}")
