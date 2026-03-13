#!/usr/bin/env python3
"""
Detailed exploration of NLD EVS 1990 subgroup positions.
Tests both F034=1 and F034=2 as Protestant.
"""
import pandas as pd
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from shared_factor_analysis import (
    load_combined_data, FACTOR_ITEMS, recode_factor_items, varimax
)

# Step 1: Build nation-level PCA (same as Figure 1)
combined = load_combined_data(waves_wvs=[2, 3], include_evs=True)

# Read WVS for Germany split + Ghana
import csv
with open(os.path.join(BASE_DIR, 'data/WVS_Time_Series_1981-2022_csv_v5_0.csv'), 'r') as f:
    reader = csv.reader(f)
    header = [h.strip('"') for h in next(reader)]

needed_extra = ['S002VS', 'COUNTRY_ALPHA', 'S020', 'X048WVS',
                'A008', 'A029', 'A034', 'A042',
                'A165', 'E018', 'E025', 'F025', 'F063', 'F118', 'F120',
                'G006', 'Y002']
available_extra = [c for c in needed_extra if c in header]
wvs_extra = pd.read_csv(os.path.join(BASE_DIR, 'data/WVS_Time_Series_1981-2022_csv_v5_0.csv'),
                         usecols=available_extra, low_memory=False)

# Ghana wave 5
gha = wvs_extra[(wvs_extra['COUNTRY_ALPHA'] == 'GHA') & (wvs_extra['S002VS'] == 5)].copy()
if 'F063' in gha.columns:
    gha['GOD_IMP'] = gha['F063']
for v in ['A042', 'A034', 'A029']:
    if v in gha.columns:
        gha[v] = pd.to_numeric(gha[v], errors='coerce').where(lambda x: x >= 0, np.nan)
if all(v in gha.columns for v in ['A042', 'A034', 'A029']):
    gha['AUTONOMY'] = gha['A042'] + gha['A034'] - gha['A029']
combined = pd.concat([combined, gha], ignore_index=True, sort=False)

# Germany split
deu3 = wvs_extra[(wvs_extra['COUNTRY_ALPHA'] == 'DEU') & (wvs_extra['S002VS'] == 3)].copy()
if 'F063' in deu3.columns:
    deu3['GOD_IMP'] = deu3['F063']
for v in ['A042', 'A034', 'A029']:
    if v in deu3.columns:
        deu3[v] = pd.to_numeric(deu3[v], errors='coerce').where(lambda x: x >= 0, np.nan)
if all(v in deu3.columns for v in ['A042', 'A034', 'A029']):
    deu3['AUTONOMY'] = deu3['A042'] + deu3['A034'] - deu3['A029']
deu_e = deu3[deu3['X048WVS'] >= 276012].copy()
deu_w = deu3[deu3['X048WVS'] < 276012].copy()
deu_e['COUNTRY_ALPHA'] = 'DEU_E'
deu_w['COUNTRY_ALPHA'] = 'DEU_W'
combined = combined[combined['COUNTRY_ALPHA'] != 'DEU']
combined = pd.concat([combined, deu_e, deu_w], ignore_index=True, sort=False)

if 'S020' in combined.columns:
    latest = combined.groupby('COUNTRY_ALPHA')['S020'].max().reset_index()
    latest.columns = ['COUNTRY_ALPHA', 'latest_year']
    combined = combined.merge(latest, on='COUNTRY_ALPHA')
    combined = combined[combined['S020'] == combined['latest_year']].drop('latest_year', axis=1)

combined_recoded = recode_factor_items(combined)
country_means = combined_recoded.groupby('COUNTRY_ALPHA')[FACTOR_ITEMS].mean()
country_means = country_means.dropna(thresh=5)
for col in FACTOR_ITEMS:
    if col in country_means.columns:
        country_means[col] = country_means[col].fillna(country_means[col].mean())

col_means = country_means.mean()
col_stds = country_means.std()
scaled = (country_means - col_means) / col_stds

U, S, Vt = np.linalg.svd(scaled.values, full_matrices=False)
loadings_raw = Vt[:2, :].T * S[:2] / np.sqrt(len(scaled) - 1)
scores_raw = U[:, :2] * S[:2]
loadings_rot, R = varimax(loadings_raw)
scores_rot = scores_raw @ R

loadings_df = pd.DataFrame(loadings_rot, index=FACTOR_ITEMS, columns=['F1', 'F2'])
scores_df = pd.DataFrame(scores_rot, index=country_means.index, columns=['F1', 'F2'])

trad_items = [i for i in ['AUTONOMY', 'F120', 'G006', 'E018'] if i in loadings_df.index]
f1_trad = sum(abs(loadings_df.loc[i, 'F1']) for i in trad_items)
f2_trad = sum(abs(loadings_df.loc[i, 'F2']) for i in trad_items)
trad_col = 'F1' if f1_trad > f2_trad else 'F2'
surv_col = 'F2' if f1_trad > f2_trad else 'F1'

nation_scores = pd.DataFrame({
    'COUNTRY_ALPHA': scores_df.index,
    'trad_secrat': scores_df[trad_col].values,
    'surv_selfexp': scores_df[surv_col].values
}).reset_index(drop=True)

if 'SWE' in nation_scores['COUNTRY_ALPHA'].values:
    swe = nation_scores[nation_scores['COUNTRY_ALPHA'] == 'SWE']
    if swe['trad_secrat'].values[0] < 0:
        nation_scores['trad_secrat'] = -nation_scores['trad_secrat']
    swe = nation_scores[nation_scores['COUNTRY_ALPHA'] == 'SWE']
    if swe['surv_selfexp'].values[0] < 0:
        nation_scores['surv_selfexp'] = -nation_scores['surv_selfexp']

# Scale factors
PAPER_POSITIONS = {
    'NLD': (1.2, 0.5), 'DEU_W': (0.7, 1.3), 'CHE': (1.0, 0.6),
    'SWE': (1.8, 1.3), 'USA': (1.5, -0.7), 'IND': (-0.5, -0.8), 'NGA': (-0.3, -1.8)
}
raw_s, raw_t, pap_s, pap_t = [], [], [], []
for _, row in nation_scores.iterrows():
    code = row['COUNTRY_ALPHA']
    if code in PAPER_POSITIONS:
        raw_s.append(row['surv_selfexp'])
        raw_t.append(row['trad_secrat'])
        pap_s.append(PAPER_POSITIONS[code][0])
        pap_t.append(PAPER_POSITIONS[code][1])
scale_surv = np.std(pap_s) / np.std(raw_s)
scale_trad = np.std(pap_t) / np.std(raw_t)
print(f'Scale factors: surv={scale_surv:.4f}, trad={scale_trad:.4f}')

# Print NLD national info
nld_ns = nation_scores[nation_scores['COUNTRY_ALPHA'] == 'NLD']
if len(nld_ns) > 0:
    nld_raw_s = nld_ns.iloc[0]['surv_selfexp']
    nld_raw_t = nld_ns.iloc[0]['trad_secrat']
    print(f'NLD national: raw=({nld_raw_s:.3f},{nld_raw_t:.3f}) paper=(1.2,0.5)')
else:
    print('NLD not in nation_scores! NLD is from EVS 1990 only')
    nld_raw_s = np.nan
    nld_raw_t = np.nan

# Check where NLD appears in combined data
nld_combined = combined_recoded[combined_recoded['COUNTRY_ALPHA'] == 'NLD']
print(f'NLD in combined: {len(nld_combined)} rows, source years: {nld_combined["S020"].value_counts().to_dict() if "S020" in nld_combined.columns else "unknown"}')

# Step 2: Load EVS 1990 NLD and compute subgroup positions
evs = pd.read_csv(os.path.join(BASE_DIR, 'data/EVS_1990_wvs_format.csv'))
nld_evs = evs[evs['COUNTRY_ALPHA'] == 'NLD'].copy()

# Build GOD_IMP
if 'A006' in nld_evs.columns:
    nld_evs['GOD_IMP'] = nld_evs['A006']
elif 'F063' in nld_evs.columns:
    nld_evs['GOD_IMP'] = nld_evs['F063']

# Build AUTONOMY
for v in ['A042', 'A034', 'A029']:
    if v in nld_evs.columns:
        nld_evs[v] = pd.to_numeric(nld_evs[v], errors='coerce')
        nld_evs[v] = nld_evs[v].where(nld_evs[v] >= 0, np.nan)
if all(v in nld_evs.columns for v in ['A042', 'A034', 'A029']):
    nld_evs['AUTONOMY'] = nld_evs['A042'] + nld_evs['A034'] - nld_evs['A029']

nld_rec = recode_factor_items(nld_evs)
nld_rec['F034'] = pd.to_numeric(nld_evs['F034'], errors='coerce')

print('\nEVS NLD factor item means by group:')
nld_overall = nld_rec[FACTOR_ITEMS].mean()
print(f'Overall NLD: {nld_overall.round(3).to_dict()}')

for f034_val, label in [(1, 'F034=1'), (2, 'F034=2')]:
    g = nld_rec[nld_rec['F034'] == f034_val]
    if len(g) > 0:
        gmean = g[FACTOR_ITEMS].mean()
        print(f'{label} (n={len(g)}): {gmean.round(3).to_dict()}')

# Compute raw PCA scores for each group
def get_raw_score(group_mean):
    gs = (group_mean - col_means) / col_stds
    gs = gs.fillna(0)
    score_2d = gs.values @ Vt[:2, :].T
    score_rot = score_2d @ R
    if f1_trad > f2_trad:
        trad_val = score_rot[0]
        surv_val = score_rot[1]
    else:
        trad_val = score_rot[1]
        surv_val = score_rot[0]
    # Apply sign corrections
    swe_ns = nation_scores[nation_scores['COUNTRY_ALPHA'] == 'SWE']
    # Already applied sign corrections to nation_scores, need to track
    return surv_val, trad_val

print('\nRaw PCA projections:')
for f034_val, label in [(1, 'F034=1'), (2, 'F034=2'), (None, 'Overall')]:
    if f034_val is not None:
        g = nld_rec[nld_rec['F034'] == f034_val]
    else:
        g = nld_rec
    if len(g) > 0:
        gmean = g[FACTOR_ITEMS].mean()
        surv_raw, trad_raw = get_raw_score(gmean)
        print(f'{label} (n={len(g)}): raw surv={surv_raw:.3f}, trad={trad_raw:.3f}')
        if not np.isnan(nld_raw_s):
            dev_s = surv_raw - nld_raw_s
            dev_t = trad_raw - nld_raw_t
            anchored_s = 1.2 + dev_s * scale_surv
            anchored_t = 0.5 + dev_t * scale_trad
            print(f'  dev=({dev_s:.3f},{dev_t:.3f}), anchored=({anchored_s:.3f},{anchored_t:.3f})')
            print(f'  Target: NLD Protestant=(1.3,0.5), NLD Catholic=(1.0,0.4)')
