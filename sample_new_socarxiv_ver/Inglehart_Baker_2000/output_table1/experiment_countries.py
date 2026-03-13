#!/usr/bin/env python3
"""Test effect of correcting country list to match the paper's 65 societies."""
import pandas as pd, csv, numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_factor_analysis import clean_missing, varimax

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(BASE, 'data/WVS_Time_Series_1981-2022_csv_v5_0.csv'), 'r') as f:
    header = [h.strip('"') for h in next(csv.reader(f))]
cols = ['S002VS', 'COUNTRY_ALPHA', 'S020', 'A006', 'A008', 'A029', 'A030', 'A032', 'A034', 'A042', 'A165', 'E018', 'E025', 'F063', 'F118', 'F120', 'G006', 'Y002']
avail = [c for c in cols if c in header]
wvs = pd.read_csv(os.path.join(BASE, 'data/WVS_Time_Series_1981-2022_csv_v5_0.csv'), usecols=avail, low_memory=False)
wvs = wvs[wvs['S002VS'].isin([2, 3])]
wvs['_src'] = 'wvs'
evs = pd.read_csv(os.path.join(BASE, 'data/EVS_1990_wvs_format.csv'))
evs['_src'] = 'evs'
df = pd.concat([wvs, evs], ignore_index=True, sort=False)

all_v = ['A006', 'F063', 'A029', 'A030', 'A032', 'A034', 'A042', 'F120', 'G006', 'E018', 'Y002', 'A008', 'E025', 'F118', 'A165']
df = clean_missing(df, [c for c in all_v if c in df.columns])
for c in ['A029', 'A030', 'A032', 'A034', 'A042']:
    if c in df.columns:
        df.loc[df[c] == 2, c] = 0
df['god_important'] = np.nan
df.loc[df['_src'] == 'wvs', 'god_important'] = df.loc[df['_src'] == 'wvs', 'F063']
df.loc[df['_src'] == 'evs', 'god_important'] = df.loc[df['_src'] == 'evs', 'A006']
df['F120'] = 11 - df['F120']
df['G006'] = 5 - df['G006']
df['E018'] = 4 - df['E018']
df['Y002'] = 4 - df['Y002']
df['F118'] = 11 - df['F118']

# 5-item autonomy
has5 = df[['A042','A034','A029','A032','A030']].notna().all(axis=1)
has3 = df[['A042','A034','A029']].notna().all(axis=1) & ~has5
df['autonomy_idx'] = np.nan
df.loc[has5,'autonomy_idx'] = df.loc[has5,'A042']+df.loc[has5,'A034']-df.loc[has5,'A029']-df.loc[has5,'A032']-df.loc[has5,'A030']
auto3 = df.loc[has3,'A042']+df.loc[has3,'A034']-df.loc[has3,'A029']
df.loc[has3,'autonomy_idx'] = (auto3+1)/3*5-3

ITEMS = ['god_important', 'autonomy_idx', 'F120', 'G006', 'E018',
         'Y002', 'A008', 'E025', 'F118', 'A165']

paper_nation = {'god_important':0.91,'autonomy_idx':0.89,'F120':0.82,'G006':0.82,'E018':0.72,
                'Y002':0.86,'A008':0.81,'E025':0.80,'F118':0.78,'A165':0.56}

def test_exclusion(df, exclude_list, label):
    df2 = df[~df['COUNTRY_ALPHA'].isin(exclude_list)]
    cm = df2.groupby('COUNTRY_ALPHA')[ITEMS].mean()
    cm = cm.dropna(thresh=7)
    for c in ITEMS: cm[c] = cm[c].fillna(cm[c].mean())

    corr = cm.corr().values
    evals, evecs = np.linalg.eigh(corr)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]; evecs = evecs[:, idx]
    loadings = evecs[:, :2] * np.sqrt(evals[:2])
    loadings, _ = varimax(loadings)
    var_exp = (loadings ** 2).sum(axis=0) / len(ITEMS) * 100

    ti = [ITEMS.index(x) for x in ['god_important', 'autonomy_idx', 'F120', 'G006', 'E018']]
    si = [ITEMS.index(x) for x in ['Y002', 'A008', 'E025', 'F118', 'A165']]
    f1t = sum(abs(loadings[i, 0]) for i in ti)
    f2t = sum(abs(loadings[i, 1]) for i in ti)
    tc = 0 if f1t > f2t else 1; sc = 1 - tc
    if np.mean([loadings[i, tc] for i in ti]) < 0: loadings[:, tc] *= -1
    if np.mean([loadings[i, sc] for i in si]) < 0: loadings[:, sc] *= -1

    # Score
    ls = 0
    for i, item in enumerate(ITEMS):
        dim = tc if i < 5 else sc
        gen = loadings[i, dim]
        gv = paper_nation[item]
        d = abs(gv - gen)
        if d <= 0.03: ls += 2
        elif d <= 0.06: ls += 1

    # Ordering
    tv = [loadings[ITEMS.index(it), tc] for it in ITEMS[:5]]
    sv = [loadings[ITEMS.index(it), sc] for it in ITEMS[5:]]
    tok = all(tv[j] >= tv[j+1] for j in range(4))
    sok = all(sv[j] >= sv[j+1] for j in range(4))
    ord_pts = (2.5 if tok else 0) + (2.5 if sok else 0)

    auto_l = loadings[ITEMS.index('autonomy_idx'), tc]

    print(f"{label:40s} N={len(cm):2d} auto={auto_l:.3f} load={ls}/20 ord={ord_pts:.0f}/5 var={var_exp[tc]:.1f}/{var_exp[sc]:.1f}")
    print(f"  trad order: {[f'{v:.2f}' for v in tv]} {'OK' if tok else 'FAIL'}")
    print(f"  surv order: {[f'{v:.2f}' for v in sv]} {'OK' if sok else 'FAIL'}")

# Test different exclusion lists
test_exclusion(df, ['MNE'], "Current (exclude MNE only)")
test_exclusion(df, ['MNE', 'ALB', 'SLV', 'MLT'], "Paper-correct (excl MNE+ALB+SLV+MLT)")
test_exclusion(df, ['MNE', 'ALB', 'SLV'], "Excl MNE+ALB+SLV")
test_exclusion(df, ['MNE', 'MLT'], "Excl MNE+MLT")
test_exclusion(df, ['MNE', 'ALB'], "Excl MNE+ALB")
test_exclusion(df, ['MNE', 'SLV'], "Excl MNE+SLV")

# Also try with latest per country
from shared_factor_analysis import get_latest_per_country
print("\n--- With latest per country ---")
df_latest = get_latest_per_country(df)
test_exclusion(df_latest, ['MNE', 'ALB', 'SLV', 'MLT'], "Latest, paper-correct excl")
test_exclusion(df_latest, ['MNE'], "Latest, MNE only")
