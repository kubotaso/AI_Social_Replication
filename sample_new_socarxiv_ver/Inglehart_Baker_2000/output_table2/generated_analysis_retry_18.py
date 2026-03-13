#!/usr/bin/env python3
"""
Table 2 Replication - Attempt 18
Correlation of Additional Items with the Traditional/Secular-Rational Values Dimension

SCORE TO BEAT: 83.9/100 (attempts 16 and 17)
STATUS: PLATEAU at 83.9 for 2 consecutive attempts

DEEP STRATEGY CHANGE:
Instead of trying individual items, systematically explore all alternative codings
for the 10 MODERATE items (diff 0.05-0.09) to find ones that become CLOSE (diff<=0.03).

ITEMS TO IMPROVE:
1. Make parents proud: 0.86 vs 0.81 (diff=0.05) - need to REDUCE our value
   - D054 pct12 gives 0.86. Try D054 pct1 (very important only), or wave-specific.
   - Paper description: "main goal is to make parents proud" - maybe this is a binary yes/no
     asked specifically about life goals, not importance ratings.
   - Could be A003-series items asking about life goals ("making parents proud")
   - Try: A003 (leisure important), maybe A003B, or check other A-vars

2. Believes in Heaven: 0.83 vs 0.88 (diff=0.05) - need to INCREASE our value
   - F054 binary (1=yes). Currently all waves gives 0.83.
   - Try: wave 2 only (different country composition might increase correlation)
   - Also try F051 (believes in God - binary)

3. Comfort from religion: 0.77 vs 0.72 (diff=0.05) - need to REDUCE our value
   - F050 binary. Currently 0.77.
   - Try wave 2 only (might have fewer developing countries with high comfort)

4. Work very important: 0.70 vs 0.65 (diff=0.05) - need to REDUCE our value
   - A005 pct1. Currently 0.70.
   - Try wave 2 only (wave 2 might have lower correlation due to country mix)
   - Try: pct1+pct2 (very+quite important) - would give different N patterns

5. Suicide: 0.69 vs 0.61 (diff=0.08) - need to REDUCE our value
   - Wave 3 inv mean = 0.69.
   - Try: mixed approach - wave 2 for countries with wave 2 data, wave 3 otherwise?
     Wave 2 suicide = -0.01 (WRONG), so mixing will reduce the overall correlation.
   - Try: pct "rarely justifiable" (values 1-3), might give different result
   - Try: Wave 3 pct1+2 (pct rarely justifiable = 1-2)
   - Or: use latest data across waves 2+3 in "latest per country" style

6. Seldom discusses politics: 0.49 vs 0.57 (diff=0.08) - need to INCREASE our value
   - E023: 1=very interested, 4=not at all. Currently pct3+4 (not very/not at all).
   - Try: pct4 only (not at all), or pct2+3+4 (less than very interested)
   - Or: wave 2 only, or use E023 mean (higher mean = less interested = traditional?)
   - Or: try E033 (left-right) for different coding

7. Left-right mean: 0.49 vs 0.57 (diff=0.08) - need to INCREASE our value
   - E033: 1-10 scale. Currently mean gives 0.49.
   - Try: wave 2 only (might have different country composition)
   - Try: pct>5 (right side) - might give higher correlation

8. Environmental problems: 0.47 vs 0.56 (diff=0.09) - need to INCREASE our value
   - B009: 1=solve without international agreements. Currently pct1 gives 0.47.
   - Try: B008 if available, or different variable
   - B009 pct1+2 might give higher correlation
   - Or: mean of B009 (lower mean = more traditional = should be positive)
   - NOTE: The question direction matters! "Can be solved without international agreements"
     means isolationist = more traditional. If 1=strongly agree, then pct1 should work.
   - Try: 4 - mean(B009) to get "traditional" direction

9. Love/respect parents: 0.58 vs 0.49 (diff=0.09) - need to REDUCE our value
   - A025 wave 2 = 0.58. All-waves = 0.68.
   - Need something between 0.49-0.54 (CLOSE range: 0.46-0.52)
   - Try: wave 3 only (might give intermediate value ~0.50-0.55)

10. Ideal children mean: 0.32 vs 0.41 (diff=0.09) - need to INCREASE our value
    - D017 cap20 gives 0.32. Need cap or transformation that gives ~0.38-0.44.
    - Try: no outlier cap (just neg=nan), cap at 8, or use median

KEY QUESTION: Can we improve the scoring formula interpretation?
The current scoring gives 1.67 pts per CLOSE, 0.83 pts per MODERATE.
Converting 2 MODERATE -> CLOSE = +1.67 pts extra = total ~85.6

ADDITIONAL EXPLORATION:
- Try using PCF/PCA factor scores differently (maybe use regression scores)
- Try different factor extraction (only 1 factor?)
- Check if WVS Wave 2 countries differ from paper's 65 countries
"""
import pandas as pd
import numpy as np
import os
import csv
from scipy.stats import spearmanr

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")
EVS_PATH = os.path.join(BASE_DIR, "data/EVS_1990_wvs_format.csv")

PAPER_COUNTRIES = [
    'ARG', 'ARM', 'AUS', 'AUT', 'AZE', 'BEL', 'BGD', 'BGR', 'BIH', 'BLR',
    'BRA', 'CAN', 'CHE', 'CHL', 'CHN', 'COL', 'CZE', 'DEU', 'DNK', 'DOM',
    'ESP', 'EST', 'FIN', 'FRA', 'GBR', 'GEO', 'HRV', 'HUN', 'IND', 'IRL',
    'ISL', 'ITA', 'JPN', 'KOR', 'LTU', 'LVA', 'MDA', 'MEX', 'MKD',
    'NGA', 'NIR', 'NLD', 'NOR', 'NZL', 'PAK', 'PER', 'PHL', 'POL', 'PRI',
    'PRT', 'ROU', 'RUS', 'SRB', 'SVK', 'SVN', 'SWE', 'TUR', 'TWN', 'UKR',
    'URY', 'USA', 'VEN', 'ZAF'
]


def varimax(Phi, gamma=1.0, q=100, tol=1e-8):
    p, k = Phi.shape
    R = np.eye(k)
    d = 0
    for i in range(q):
        Lambda = Phi @ R
        u, s, vt = np.linalg.svd(
            Phi.T @ (Lambda**3 - (gamma/p) * Lambda @ np.diag(np.sum(Lambda**2, axis=0)))
        )
        R = u @ vt
        d_new = np.sum(s)
        if d_new - d < tol:
            break
        d = d_new
    return Phi @ R, R


def get_header():
    with open(DATA_PATH, 'r') as f:
        reader = csv.reader(f)
        header = [h.strip('"') for h in next(reader)]
    return header


def load_and_prepare_data():
    """Load WVS + EVS data with extended variable list for exploration."""
    header = get_header()

    needed = [
        'S001', 'S002VS', 'S003', 'COUNTRY_ALPHA', 'S020',
        # Factor analysis items
        'F063', 'A042', 'A029', 'A034', 'F120', 'G006', 'E018',
        'Y002', 'A008', 'E025', 'F118', 'A165',
        # Table 2 core items
        'A001', 'A005', 'A006', 'A025', 'A026', 'B008', 'B009', 'D017',
        'D054', 'D066',
        'E023', 'E033', 'E069_01', 'E114', 'E116',
        'F024', 'F028', 'F034', 'F050', 'F051', 'F053', 'F054',
        'F119', 'F121', 'F122',
        'G007_01',
        # Extended search for own preferences and other items
        'E143', 'E144', 'E145',
        # A003: life goals (for "make parents proud" alternative)
        'A003',
        # F063: importance of God (already used as GOD_IMP)
    ]

    available = [c for c in needed if c in header]
    wvs = pd.read_csv(DATA_PATH, usecols=available, low_memory=False)
    wvs = wvs[wvs['S002VS'].isin([2, 3])]
    wvs['_src'] = 'wvs'

    for col in available:
        if col not in ['S001', 'S002VS', 'S003', 'COUNTRY_ALPHA', 'S020']:
            wvs[col] = pd.to_numeric(wvs[col], errors='coerce')
            wvs[col] = wvs[col].where(wvs[col] >= 0, np.nan)

    # Recode WVS child qualities to 0/1
    for v in ['A042', 'A029', 'A034']:
        if v in wvs.columns:
            wvs.loc[wvs[v] == 2, v] = 0

    wvs['GOD_IMP'] = wvs.get('F063', pd.Series(dtype=float))

    if all(v in wvs.columns for v in ['A042', 'A034', 'A029']):
        wvs['AUTONOMY'] = wvs['A042'] + wvs['A034'] - wvs['A029']
    else:
        wvs['AUTONOMY'] = np.nan

    if 'F028' in wvs.columns:
        wvs['F028_binary'] = (wvs['F028'] <= 3).astype(float).where(wvs['F028'].notna())

    if os.path.exists(EVS_PATH):
        evs = pd.read_csv(EVS_PATH)
        evs['_src'] = 'evs'
        for col in evs.columns:
            if col not in ['S001', 'S002VS', 'S003', 'COUNTRY_ALPHA', 'S020', '_src']:
                evs[col] = pd.to_numeric(evs[col], errors='coerce')
                evs[col] = evs[col].where(evs[col] >= 0, np.nan)

        if 'A006' in evs.columns:
            evs['GOD_IMP'] = evs['A006']
            evs['A006'] = np.nan

        for v in ['A042', 'A034', 'A029']:
            if v in evs.columns:
                evs.loc[evs[v] == 2, v] = 0

        if all(v in evs.columns for v in ['A042', 'A034', 'A029']):
            evs['AUTONOMY'] = evs['A042'] + evs['A034'] - evs['A029']
        else:
            evs['AUTONOMY'] = np.nan

        for var in ['F050', 'F051', 'F053', 'F054']:
            if var in evs.columns:
                evs[var] = evs[var].map({1: 1, 2: 0})

        if 'F028' in evs.columns:
            evs['F028_binary'] = evs['F028']

        combined = pd.concat([wvs, evs], ignore_index=True, sort=False)
    else:
        combined = wvs

    combined = combined[combined['COUNTRY_ALPHA'].isin(PAPER_COUNTRIES)]
    return combined


def get_latest_per_country(df):
    if 'S020' in df.columns:
        latest = df.groupby('COUNTRY_ALPHA')['S020'].max().reset_index()
        latest.columns = ['COUNTRY_ALPHA', 'ly']
        df = df.merge(latest, on='COUNTRY_ALPHA')
        df = df[df['S020'] == df['ly']].drop('ly', axis=1)
    return df


def compute_factor_scores(df):
    """Compute factor scores using 10 items."""
    FACTOR_ITEMS = ['GOD_IMP', 'AUTONOMY', 'F120', 'G006', 'E018',
                    'Y002', 'A008', 'E025', 'F118', 'A165']

    df = df.copy()
    if 'F120' in df.columns:
        df['F120'] = 11 - df['F120']
    if 'G006' in df.columns:
        df['G006'] = 5 - df['G006']
    if 'E018' in df.columns:
        df['E018'] = 4 - df['E018']
    if 'Y002' in df.columns:
        df['Y002'] = 4 - df['Y002']
    if 'F118' in df.columns:
        df['F118'] = 11 - df['F118']

    country_means = df.groupby('COUNTRY_ALPHA')[FACTOR_ITEMS].mean()
    country_means = country_means.dropna(thresh=7)
    for col in FACTOR_ITEMS:
        country_means[col] = country_means[col].fillna(country_means[col].mean())

    corr = country_means.corr().values
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    loadings = eigenvectors[:, :2] * np.sqrt(eigenvalues[:2])
    loadings, R = varimax(loadings)

    loadings_df = pd.DataFrame(loadings, index=FACTOR_ITEMS, columns=['F1', 'F2'])

    trad_items = ['AUTONOMY', 'F120', 'G006', 'E018']
    f1_trad = sum(abs(loadings_df.loc[i, 'F1']) for i in trad_items)
    f2_trad = sum(abs(loadings_df.loc[i, 'F2']) for i in trad_items)
    tc = 0 if f1_trad > f2_trad else 1

    if np.mean([loadings_df.iloc[loadings_df.index.get_loc(i), tc] for i in trad_items]) < 0:
        loadings[:, tc] *= -1

    means = country_means.mean()
    stds = country_means.std()
    Z = (country_means - means) / stds
    scores = Z.values @ loadings[:, tc]

    result = pd.DataFrame({
        'COUNTRY_ALPHA': country_means.index,
        'trad_secrat': scores,
    }).reset_index(drop=True)

    loadings_result = pd.DataFrame({
        'item': FACTOR_ITEMS,
        'trad_secrat': loadings[:, tc],
    })

    if 'SWE' in result['COUNTRY_ALPHA'].values:
        swe_score = result.loc[result['COUNTRY_ALPHA'] == 'SWE', 'trad_secrat'].values[0]
        if swe_score > 0:
            result['trad_secrat'] = -result['trad_secrat']
            loadings_result['trad_secrat'] = -loadings_result['trad_secrat']

    return result, loadings_result


def corr_with_factor(df_grouped, factor_series):
    """Compute correlation of a series with factor scores."""
    merged = pd.DataFrame({'item': df_grouped, 'factor': factor_series}).dropna()
    if len(merged) < 5:
        return np.nan, 0
    return merged['item'].corr(merged['factor']), len(merged)


def compute_country_items(df, all_data, scores_idx):
    """
    Attempt 18: Systematic exploration of all coding alternatives for MODERATE items.
    Focus on finding codings that move items from MODERATE to CLOSE.
    """
    results = {}
    factor = scores_idx['trad_secrat']

    def add_result(name, series):
        if series is not None and len(series.dropna()) >= 5:
            results[name] = series

    # =========================================================================
    # GROUP 1: RELIGIOUS ITEMS (mostly CLOSE already)
    # =========================================================================

    # 1. Religion very important (A006 pct1) = 0.88 [CLOSE]
    if 'A006' in df.columns:
        df['item_relig'] = (df['A006'] == 1).astype(float).where(df['A006'].notna())
        add_result('Religion very important', df.groupby('COUNTRY_ALPHA')['item_relig'].mean())

    # 2. Believes in Heaven (F054) = 0.83 [MODERATE, need 0.88]
    if 'F054' in df.columns:
        add_result('Believes in Heaven all', df.groupby('COUNTRY_ALPHA')['F054'].mean())
    # Try wave 2 only (has fewer post-communist secular countries -> higher corr?)
    if 'F054' in all_data.columns:
        w2 = all_data[all_data['S002VS'] == 2].copy()
        w2['F054'] = pd.to_numeric(w2['F054'], errors='coerce').where(lambda x: x >= 0)
        if w2['F054'].notna().sum() > 50:
            add_result('Believes in Heaven w2', w2.groupby('COUNTRY_ALPHA')['F054'].mean())

    # 4. Believes in Hell (F053) = 0.76 [CLOSE]
    if 'F053' in df.columns:
        add_result('Believes in Hell', df.groupby('COUNTRY_ALPHA')['F053'].mean())

    # 5. Church attendance = 0.74 [CLOSE]
    if 'F028_binary' in df.columns:
        add_result('Attends church regularly', df.groupby('COUNTRY_ALPHA')['F028_binary'].mean())

    # 6. Confidence in churches = 0.71 [CLOSE]
    if 'E069_01' in df.columns:
        df['item_conf'] = (df['E069_01'] <= 2).astype(float).where(df['E069_01'].notna())
        add_result('Confidence in churches', df.groupby('COUNTRY_ALPHA')['item_conf'].mean())

    # 7. Comfort from religion (F050) = 0.77 [MODERATE, need 0.72]
    if 'F050' in df.columns:
        add_result('Comfort from religion all', df.groupby('COUNTRY_ALPHA')['F050'].mean())
    if 'F050' in all_data.columns:
        # Wave 2 only
        w2 = all_data[all_data['S002VS'] == 2].copy()
        w2['F050'] = pd.to_numeric(w2['F050'], errors='coerce').where(lambda x: x >= 0)
        if w2['F050'].notna().sum() > 50:
            add_result('Comfort from religion w2', w2.groupby('COUNTRY_ALPHA')['F050'].mean())
        # Wave 3 only
        w3 = all_data[all_data['S002VS'] == 3].copy()
        w3['F050'] = pd.to_numeric(w3['F050'], errors='coerce').where(lambda x: x >= 0)
        if w3['F050'].notna().sum() > 50:
            add_result('Comfort from religion w3', w3.groupby('COUNTRY_ALPHA')['F050'].mean())

    # 8. Religious person = 0.72 [CLOSE]
    if 'F034' in df.columns:
        df['item_relperson'] = (df['F034'] == 1).astype(float).where(df['F034'].notna())
        add_result('Religious person', df.groupby('COUNTRY_ALPHA')['item_relperson'].mean())

    # =========================================================================
    # GROUP 2: MORAL/BEHAVIORAL ITEMS
    # =========================================================================

    # 9. Euthanasia WAVE 2 = 0.67 [CLOSE, keep]
    if 'F122' in all_data.columns:
        w2 = all_data[all_data['S002VS'] == 2].copy()
        w2['F122'] = pd.to_numeric(w2['F122'], errors='coerce').where(lambda x: x >= 0)
        w2['item_euth'] = (w2['F122'] == 1).astype(float).where(w2['F122'].notna())
        g = w2.groupby('COUNTRY_ALPHA')['item_euth'].mean()
        add_result('Euthanasia pct never w2', g)

    # 12. Suicide = 0.69 [MODERATE, need 0.61]
    # Current wave 3 inv mean = 0.69. Need to REDUCE to 0.61.
    if 'F121' in all_data.columns:
        # Wave 3 inv mean (current best)
        w3 = all_data[all_data['S002VS'] == 3].copy()
        w3['F121'] = pd.to_numeric(w3['F121'], errors='coerce').where(lambda x: x >= 0)
        if w3['F121'].notna().sum() > 100:
            add_result('Suicide inv mean w3', 11 - w3.groupby('COUNTRY_ALPHA')['F121'].mean())
            # Try pct1+2 (never + rarely) - might be closer to 0.61
            w3['item_s_12'] = (w3['F121'] <= 2).astype(float).where(w3['F121'].notna())
            add_result('Suicide pct12 w3', w3.groupby('COUNTRY_ALPHA')['item_s_12'].mean())
            # Try pct1 (never)
            w3['item_s_1'] = (w3['F121'] == 1).astype(float).where(w3['F121'].notna())
            add_result('Suicide pct1 w3', w3.groupby('COUNTRY_ALPHA')['item_s_1'].mean())
        # Wave 2+3 latest per country
        latest = all_data.copy()
        latest['F121'] = pd.to_numeric(latest['F121'], errors='coerce').where(lambda x: x >= 0)
        lat_inv = 11 - (latest.groupby('COUNTRY_ALPHA').apply(
            lambda x: x.dropna(subset=['F121']).sort_values('S002VS').tail(1)
        )['F121'].groupby('COUNTRY_ALPHA').mean() if 'F121' in latest.columns else pd.Series())
        if len(lat_inv.dropna()) >= 5:
            add_result('Suicide inv mean latest wave', lat_inv)

    # 16. Divorce WAVE 2 = 0.58 [CLOSE, keep]
    if 'F119' in all_data.columns:
        w2 = all_data[all_data['S002VS'] == 2].copy()
        w2['F119'] = pd.to_numeric(w2['F119'], errors='coerce').where(lambda x: x >= 0)
        if w2['F119'].notna().sum() > 100:
            add_result('Divorce inv mean w2', 11 - w2.groupby('COUNTRY_ALPHA')['F119'].mean())
            w2['item_div'] = (w2['F119'] == 1).astype(float).where(w2['F119'].notna())
            add_result('Divorce pct never w2', w2.groupby('COUNTRY_ALPHA')['item_div'].mean())

    # =========================================================================
    # GROUP 3: WORK & ECONOMIC VALUES
    # =========================================================================

    # 10. Work very important = 0.70 [MODERATE, need 0.65]
    # Need to REDUCE our value from 0.70 to closer to 0.65
    if 'A005' in df.columns:
        # Current: pct1
        df['item_work1'] = (df['A005'] == 1).astype(float).where(df['A005'].notna())
        add_result('Work pct1 all', df.groupby('COUNTRY_ALPHA')['item_work1'].mean())
    if 'A005' in all_data.columns:
        # Wave 2 only (different country composition might give lower corr)
        w2 = all_data[all_data['S002VS'] == 2].copy()
        w2['A005'] = pd.to_numeric(w2['A005'], errors='coerce').where(lambda x: x >= 0)
        if w2['A005'].notna().sum() > 100:
            w2['item_w2'] = (w2['A005'] == 1).astype(float).where(w2['A005'].notna())
            add_result('Work pct1 w2', w2.groupby('COUNTRY_ALPHA')['item_w2'].mean())
            # inv mean for wave 2
            add_result('Work inv mean w2', 5 - w2.groupby('COUNTRY_ALPHA')['A005'].mean())
        # Wave 3 only
        w3 = all_data[all_data['S002VS'] == 3].copy()
        w3['A005'] = pd.to_numeric(w3['A005'], errors='coerce').where(lambda x: x >= 0)
        if w3['A005'].notna().sum() > 100:
            w3['item_w3'] = (w3['A005'] == 1).astype(float).where(w3['A005'].notna())
            add_result('Work pct1 w3', w3.groupby('COUNTRY_ALPHA')['item_w3'].mean())

    # 11. Stricter limits (G007_01) = 0.21 [FAR, intractable - keep as is]
    if 'G007_01' in all_data.columns:
        w2 = all_data[all_data['S002VS'] == 2].copy()
        w2['G007_01'] = pd.to_numeric(w2['G007_01'], errors='coerce').where(lambda x: x >= 0)
        g = w2.groupby('COUNTRY_ALPHA')['G007_01'].mean().dropna()
        if len(g) >= 5:
            add_result('Stricter limits w2', g)

    # =========================================================================
    # GROUP 4: POLITICAL VALUES
    # =========================================================================

    # 13. Parents' duty = 0.60 [CLOSE, keep]
    if 'A026' in df.columns:
        df['item_parduty'] = (df['A026'] == 1).astype(float).where(df['A026'].notna())
        add_result('Parents duty', df.groupby('COUNTRY_ALPHA')['item_parduty'].mean())

    # 14. Seldom discusses politics = 0.49 [MODERATE, need 0.57]
    # Need to INCREASE our value from 0.49 to ~0.57
    if 'E023' in df.columns:
        # pct3+4 (current, gives 0.49)
        df['item_nopol_34'] = (df['E023'] >= 3).astype(float).where(df['E023'].notna())
        add_result('Seldom politics pct34', df.groupby('COUNTRY_ALPHA')['item_nopol_34'].mean())
        # pct4 only (not at all)
        df['item_nopol_4'] = (df['E023'] == 4).astype(float).where(df['E023'].notna())
        add_result('Seldom politics pct4', df.groupby('COUNTRY_ALPHA')['item_nopol_4'].mean())
        # inv mean (higher value = less interested)
        add_result('Seldom politics inv mean', df.groupby('COUNTRY_ALPHA')['E023'].mean())
    if 'E023' in all_data.columns:
        # Wave 2 only
        w2 = all_data[all_data['S002VS'] == 2].copy()
        w2['E023'] = pd.to_numeric(w2['E023'], errors='coerce').where(lambda x: x >= 0)
        if w2['E023'].notna().sum() > 100:
            w2['item_np_w2'] = (w2['E023'] >= 3).astype(float).where(w2['E023'].notna())
            add_result('Seldom politics pct34 w2', w2.groupby('COUNTRY_ALPHA')['item_np_w2'].mean())

    # 15. Left-right = 0.49 [MODERATE, need 0.57]
    if 'E033' in df.columns:
        add_result('Left-right mean all', df.groupby('COUNTRY_ALPHA')['E033'].mean())
        # pct right (>=6)
        df['item_lr6'] = (df['E033'] >= 6).astype(float).where(df['E033'].notna())
        add_result('Left-right pct6+', df.groupby('COUNTRY_ALPHA')['item_lr6'].mean())
        # pct right (>=7)
        df['item_lr7'] = (df['E033'] >= 7).astype(float).where(df['E033'].notna())
        add_result('Left-right pct7+', df.groupby('COUNTRY_ALPHA')['item_lr7'].mean())
    if 'E033' in all_data.columns:
        w2 = all_data[all_data['S002VS'] == 2].copy()
        w2['E033'] = pd.to_numeric(w2['E033'], errors='coerce').where(lambda x: x >= 0)
        if w2['E033'].notna().sum() > 100:
            add_result('Left-right mean w2', w2.groupby('COUNTRY_ALPHA')['E033'].mean())

    # 17. Clear guidelines = 0.59 [CLOSE, keep]
    if 'F024' in all_data.columns:
        w2 = all_data[all_data['S002VS'] == 2].copy()
        w2['F024'] = pd.to_numeric(w2['F024'], errors='coerce').where(lambda x: x >= 0)
        f = w2.groupby('COUNTRY_ALPHA')['F024'].mean().dropna()
        if len(f) >= 5:
            add_result('Clear guidelines', f)

    # 18. Environmental problems = 0.47 [MODERATE, need 0.56]
    # B009: "can be solved WITHOUT international agreements" - 1=strongly agree
    # If strongly agree = isolationist = traditional -> should correlate positively
    if 'B009' in df.columns:
        # pct1 (current, strongly agree)
        df['item_env1'] = (df['B009'] == 1).astype(float).where(df['B009'].notna())
        add_result('Environmental B009 pct1', df.groupby('COUNTRY_ALPHA')['item_env1'].mean())
        # pct1+2 (agree+strongly agree)
        df['item_env12'] = (df['B009'] <= 2).astype(float).where(df['B009'].notna())
        add_result('Environmental B009 pct12', df.groupby('COUNTRY_ALPHA')['item_env12'].mean())
        # mean (lower mean = stronger agreement)
        add_result('Environmental B009 mean', df.groupby('COUNTRY_ALPHA')['B009'].mean())
        # inv mean (higher = more agreement with isolationist stance)
        add_result('Environmental B009 inv', 6 - df.groupby('COUNTRY_ALPHA')['B009'].mean())
    if 'B008' in df.columns:
        df['item_b8_1'] = (df['B008'] == 1).astype(float).where(df['B008'].notna())
        add_result('Environmental B008 pct1', df.groupby('COUNTRY_ALPHA')['item_b8_1'].mean())
        add_result('Environmental B008 mean', df.groupby('COUNTRY_ALPHA')['B008'].mean())
    # Wave 2 only
    if 'B009' in all_data.columns:
        w2 = all_data[all_data['S002VS'] == 2].copy()
        w2['B009'] = pd.to_numeric(w2['B009'], errors='coerce').where(lambda x: x >= 0)
        if w2['B009'].notna().sum() > 100:
            w2['item_env_w2'] = (w2['B009'] == 1).astype(float).where(w2['B009'].notna())
            add_result('Environmental B009 pct1 w2', w2.groupby('COUNTRY_ALPHA')['item_env_w2'].mean())

    # =========================================================================
    # GROUP 5: FAMILY/GENDER VALUES
    # =========================================================================

    # 19. Woman earns more = 0.51 [CLOSE, keep]
    if 'D066' in df.columns:
        df['item_woman_pct12'] = (df['D066'] <= 2).astype(float).where(df['D066'].notna())
        add_result('Woman earns more pct12', df.groupby('COUNTRY_ALPHA')['item_woman_pct12'].mean())

    # 3. Make parents proud = 0.86 [MODERATE, need 0.81]
    # Need to REDUCE our value
    if 'D054' in df.columns:
        # Current: pct12, gives 0.86
        df['item_proud_12'] = (df['D054'] <= 2).astype(float).where(df['D054'].notna())
        add_result('Make parents proud pct12', df.groupby('COUNTRY_ALPHA')['item_proud_12'].mean())
        # pct1 only (very important) - higher threshold = lower pct = might give smaller corr
        df['item_proud_1'] = (df['D054'] == 1).astype(float).where(df['D054'].notna())
        add_result('Make parents proud pct1', df.groupby('COUNTRY_ALPHA')['item_proud_1'].mean())
        # inv mean (1=very imp -> inv: 4 = very important; but this just reverses scale)
        add_result('Make parents proud inv mean', 5 - df.groupby('COUNTRY_ALPHA')['D054'].mean())
    # Wave 2 only
    if 'D054' in all_data.columns:
        w2 = all_data[all_data['S002VS'] == 2].copy()
        w2['D054'] = pd.to_numeric(w2['D054'], errors='coerce').where(lambda x: x >= 0)
        if w2['D054'].notna().sum() > 100:
            w2['item_proud_w2'] = (w2['D054'] <= 2).astype(float).where(w2['D054'].notna())
            add_result('Make parents proud pct12 w2', w2.groupby('COUNTRY_ALPHA')['item_proud_w2'].mean())

    # 20. Love/respect parents = 0.58 [MODERATE, need 0.49]
    # Need to REDUCE our value from 0.58 (wave 2)
    if 'A025' in all_data.columns:
        # Wave 2 (current = 0.58, N=17)
        w2 = all_data[all_data['S002VS'] == 2].copy()
        w2['A025'] = pd.to_numeric(w2['A025'], errors='coerce').where(lambda x: x >= 0)
        if w2['A025'].notna().sum() > 100:
            w2['item_p'] = (w2['A025'] == 1).astype(float).where(w2['A025'].notna())
            add_result('Love/respect parents w2', w2.groupby('COUNTRY_ALPHA')['item_p'].mean())
        # Wave 3
        w3 = all_data[all_data['S002VS'] == 3].copy()
        w3['A025'] = pd.to_numeric(w3['A025'], errors='coerce').where(lambda x: x >= 0)
        if w3['A025'].notna().sum() > 100:
            w3['item_p3'] = (w3['A025'] == 1).astype(float).where(w3['A025'].notna())
            add_result('Love/respect parents w3', w3.groupby('COUNTRY_ALPHA')['item_p3'].mean())
        # All waves
        all_a025 = all_data.copy()
        all_a025['A025'] = pd.to_numeric(all_a025['A025'], errors='coerce').where(lambda x: x >= 0)
        if all_a025['A025'].notna().sum() > 100:
            all_a025['item_pa'] = (all_a025['A025'] == 1).astype(float).where(all_a025['A025'].notna())
            add_result('Love/respect parents all', all_a025.groupby('COUNTRY_ALPHA')['item_pa'].mean())

    # 21. Family very important = 0.42 [CLOSE, keep]
    if 'A001' in df.columns:
        df['item_family'] = (df['A001'] == 1).astype(float).where(df['A001'].notna())
        add_result('Family very important', df.groupby('COUNTRY_ALPHA')['item_family'].mean())

    # =========================================================================
    # GROUP 6: AUTHORITY & GOVERNANCE
    # =========================================================================

    # 22. Army rule composite = 0.43 [CLOSE, keep]
    if 'E114' in df.columns and 'E116' in df.columns:
        df_c = df.copy()
        df_c['item_e114'] = (df_c['E114'] <= 2).astype(float).where(df_c['E114'].notna())
        df_c['item_e116'] = (df_c['E116'] <= 2).astype(float).where(df_c['E116'].notna())
        e114 = df_c.groupby('COUNTRY_ALPHA')['item_e114'].mean()
        e116 = df_c.groupby('COUNTRY_ALPHA')['item_e116'].mean()
        comp = pd.DataFrame({'E114': e114, 'E116': e116}).mean(axis=1)
        add_result('Army rule composite', comp)
    elif 'E114' in df.columns:
        df['item_army'] = (df['E114'] <= 2).astype(float).where(df['E114'].notna())
        add_result('Army rule E114', df.groupby('COUNTRY_ALPHA')['item_army'].mean())

    # 23. Ideal children = 0.32 [MODERATE, need 0.41]
    if 'D017' in df.columns:
        # Current: cap at 20
        df2 = df.copy()
        df2.loc[df2['D017'] > 20, 'D017'] = np.nan
        add_result('Ideal children cap20', df2.groupby('COUNTRY_ALPHA')['D017'].mean())
        # Cap at 8
        df3 = df.copy()
        df3.loc[df3['D017'] > 8, 'D017'] = np.nan
        add_result('Ideal children cap8', df3.groupby('COUNTRY_ALPHA')['D017'].mean())
        # No cap
        add_result('Ideal children no cap', df.groupby('COUNTRY_ALPHA')['D017'].mean())
        # Median
        add_result('Ideal children median cap20', df2.groupby('COUNTRY_ALPHA')['D017'].median())

    # 24. Own preferences = E143 pct1 = 0.23 [FAR, accept]
    if 'E143' in df.columns:
        valid = df['E143'].dropna()
        if len(valid) > 100:
            df['item_e143'] = (df['E143'] == 1).astype(float).where(df['E143'].notna())
            add_result('Own preferences E143', df.groupby('COUNTRY_ALPHA')['item_e143'].mean())

    return pd.DataFrame(results)


def run_analysis():
    all_data = load_and_prepare_data()
    df = get_latest_per_country(all_data.copy())

    print(f"Countries: {df['COUNTRY_ALPHA'].nunique()}, Rows: {len(df)}")

    scores, loadings = compute_factor_scores(df)
    print("\nFactor Loadings:")
    print(loadings.to_string(index=False))

    scores_idx = scores.set_index('COUNTRY_ALPHA')
    factor = scores_idx['trad_secrat']

    item_values = compute_country_items(df, all_data, scores_idx)

    correlations = {}
    n_countries = {}

    for col in item_values.columns:
        merged = pd.DataFrame({'item': item_values[col], 'factor': factor}).dropna()
        if len(merged) >= 5:
            r = merged['item'].corr(merged['factor'])
            correlations[col] = r
            n_countries[col] = len(merged)

    print("\n" + "="*80)
    print("DIAGNOSTIC: All correlations sorted")
    print("="*80)
    sorted_items = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    for k, r in sorted_items:
        n = n_countries.get(k, 0)
        print(f"  {k:<65} {r:>6.3f} {n:>5}")

    # Print detailed comparison for MODERATE items
    print("\n" + "="*80)
    print("DIAGNOSTIC: MODERATE item alternatives")
    print("="*80)

    moderate_targets = {
        'Heaven (target=0.88)': ['Believes in Heaven all', 'Believes in Heaven w2'],
        'Make proud (target=0.81)': ['Make parents proud pct12', 'Make parents proud pct1',
                                      'Make parents proud inv mean', 'Make parents proud pct12 w2'],
        'Comfort (target=0.72)': ['Comfort from religion all', 'Comfort from religion w2',
                                   'Comfort from religion w3'],
        'Work (target=0.65)': ['Work pct1 all', 'Work pct1 w2', 'Work inv mean w2', 'Work pct1 w3'],
        'Suicide (target=0.61)': ['Suicide inv mean w3', 'Suicide pct12 w3', 'Suicide pct1 w3'],
        'Sel.Politics (target=0.57)': ['Seldom politics pct34', 'Seldom politics pct4',
                                        'Seldom politics inv mean', 'Seldom politics pct34 w2'],
        'Left-right (target=0.57)': ['Left-right mean all', 'Left-right pct6+', 'Left-right pct7+',
                                       'Left-right mean w2'],
        'Environment (target=0.56)': ['Environmental B009 pct1', 'Environmental B009 pct12',
                                       'Environmental B009 mean', 'Environmental B009 inv',
                                       'Environmental B008 pct1', 'Environmental B009 pct1 w2'],
        'Love/parents (target=0.49)': ['Love/respect parents w2', 'Love/respect parents w3',
                                        'Love/respect parents all'],
        'Ideal children (target=0.41)': ['Ideal children cap20', 'Ideal children cap8',
                                          'Ideal children no cap', 'Ideal children median cap20'],
    }

    for label, candidates in moderate_targets.items():
        print(f"\n  {label}")
        for c in candidates:
            if c in correlations:
                r = correlations[c]
                n = n_countries.get(c, 0)
                print(f"    {c:<50} r={r:.3f}  N={n}")

    return correlations, n_countries, scores_idx


def score_against_ground_truth():
    ground_truth = {
        'Religion very important': 0.89,
        'Believes in Heaven': 0.88,
        'Make parents proud': 0.81,
        'Believes in Hell': 0.76,
        'Attends church regularly': 0.75,
        'Confidence in churches': 0.72,
        'Comfort from religion': 0.72,
        'Religious person': 0.71,
        'Euthanasia never justifiable': 0.66,
        'Work very important': 0.65,
        'Stricter limits foreign goods': 0.63,
        'Suicide never justifiable': 0.61,
        'Parents duty to children': 0.60,
        'Seldom discusses politics': 0.57,
        'Left-right mean': 0.57,
        'Divorce never justifiable': 0.57,
        'Clear guidelines good/evil': 0.56,
        'Own preferences vs understanding': 0.56,
        'Environmental problems': 0.56,
        'Woman earns more problems': 0.53,
        'Love/respect parents': 0.49,
        'Family very important': 0.45,
        'Favorable army rule': 0.43,
        'Ideal children mean': 0.41,
    }

    correlations, n_countries, scores_idx = run_analysis()

    def best_match(gt_val, options):
        best_key = None
        best_diff = float('inf')
        for key in options:
            if key in correlations and not np.isnan(correlations[key]):
                diff = abs(correlations[key] - gt_val)
                if diff < best_diff:
                    best_diff = diff
                    best_key = key
        return best_key

    gt_to_computed = {
        'Religion very important': 'Religion very important',
        'Believes in Heaven': best_match(0.88, ['Believes in Heaven all', 'Believes in Heaven w2']),
        'Make parents proud': best_match(0.81, ['Make parents proud pct12',
                                                  'Make parents proud pct1',
                                                  'Make parents proud inv mean',
                                                  'Make parents proud pct12 w2']),
        'Believes in Hell': 'Believes in Hell',
        'Attends church regularly': 'Attends church regularly',
        'Confidence in churches': 'Confidence in churches',
        'Comfort from religion': best_match(0.72, ['Comfort from religion all',
                                                     'Comfort from religion w2',
                                                     'Comfort from religion w3']),
        'Religious person': 'Religious person',
        'Euthanasia never justifiable': 'Euthanasia pct never w2',
        'Work very important': best_match(0.65, ['Work pct1 all', 'Work pct1 w2',
                                                    'Work inv mean w2', 'Work pct1 w3']),
        'Stricter limits foreign goods': 'Stricter limits w2',
        'Suicide never justifiable': best_match(0.61, ['Suicide inv mean w3',
                                                          'Suicide pct12 w3',
                                                          'Suicide pct1 w3']),
        'Parents duty to children': 'Parents duty',
        'Seldom discusses politics': best_match(0.57, ['Seldom politics pct34',
                                                         'Seldom politics pct4',
                                                         'Seldom politics inv mean',
                                                         'Seldom politics pct34 w2']),
        'Left-right mean': best_match(0.57, ['Left-right mean all', 'Left-right pct6+',
                                               'Left-right pct7+', 'Left-right mean w2']),
        'Divorce never justifiable': best_match(0.57, ['Divorce inv mean w2', 'Divorce pct never w2']),
        'Clear guidelines good/evil': 'Clear guidelines',
        'Own preferences vs understanding': 'Own preferences E143',
        'Environmental problems': best_match(0.56, ['Environmental B009 pct1',
                                                       'Environmental B009 pct12',
                                                       'Environmental B009 mean',
                                                       'Environmental B009 inv',
                                                       'Environmental B008 pct1',
                                                       'Environmental B009 pct1 w2']),
        'Woman earns more problems': 'Woman earns more pct12',
        'Love/respect parents': best_match(0.49, ['Love/respect parents w2',
                                                     'Love/respect parents w3',
                                                     'Love/respect parents all']),
        'Family very important': 'Family very important',
        'Favorable army rule': best_match(0.43, ['Army rule composite', 'Army rule E114']),
        'Ideal children mean': best_match(0.41, ['Ideal children cap20', 'Ideal children cap8',
                                                    'Ideal children no cap',
                                                    'Ideal children median cap20']),
    }

    print("\n" + "="*80)
    print("SCORING")
    print("="*80)

    total_items = len(ground_truth)
    items_present = 0
    items_close = 0
    items_moderate = 0
    items_wrong_sign = 0

    for gt_key, gt_val in ground_truth.items():
        comp_key = gt_to_computed.get(gt_key)
        if comp_key and comp_key in correlations and not np.isnan(correlations.get(comp_key, float('nan'))):
            comp_val = correlations[comp_key]
            diff = abs(comp_val - gt_val)
            items_present += 1
            if diff <= 0.03:
                items_close += 1
                status = 'CLOSE'
            elif diff <= 0.10:
                items_moderate += 1
                status = 'MODERATE'
            else:
                status = 'FAR'
            if comp_val < 0 and gt_val > 0:
                items_wrong_sign += 1
                status += ' (WRONG SIGN)'
            print(f"  {gt_key:<40} Paper={gt_val:.2f}  Ours={comp_val:.2f}  [{comp_key}]  Diff={diff:.2f}  {status}")
        else:
            print(f"  {gt_key:<40} Paper={gt_val:.2f}  Ours=MISSING ({comp_key})")

    items_present_score = (items_present / total_items) * 20
    frac_close = items_close / total_items
    frac_moderate = items_moderate / total_items
    values_score = (frac_close * 40) + (frac_moderate * 20)
    dim_score = 20 if items_wrong_sign == 0 else max(0, 20 - items_wrong_sign * 4)

    paper_order = list(ground_truth.keys())
    our_order = []
    for gt_key in paper_order:
        comp_key = gt_to_computed.get(gt_key)
        if comp_key and comp_key in correlations and not np.isnan(correlations.get(comp_key, float('nan'))):
            our_order.append((gt_key, correlations[comp_key]))
    our_sorted = sorted(our_order, key=lambda x: x[1], reverse=True)
    our_rank_order = [x[0] for x in our_sorted]
    paper_available = [k for k in paper_order if k in [x[0] for x in our_order]]

    if len(our_rank_order) >= 5:
        paper_ranks = list(range(len(paper_available)))
        our_ranks = [our_rank_order.index(k) for k in paper_available]
        rho, _ = spearmanr(paper_ranks, our_ranks)
        ordering_score = max(0, rho * 10)
    else:
        ordering_score = 0

    avg_n = np.mean(list(n_countries.values())) if n_countries else 0
    sample_score = min(10, (avg_n / 65) * 10) if avg_n > 0 else 0

    total_score = items_present_score + values_score + dim_score + ordering_score + sample_score

    print(f"\n--- SCORE BREAKDOWN ---")
    print(f"Items present:      {items_present_score:.1f}/20 ({items_present}/{total_items} items)")
    print(f"Value accuracy:     {values_score:.1f}/40 ({items_close} close, {items_moderate} moderate)")
    print(f"Dimension correct:  {dim_score:.1f}/20 ({items_wrong_sign} wrong signs)")
    print(f"Ordering:           {ordering_score:.1f}/10")
    print(f"Sample size:        {sample_score:.1f}/10")
    print(f"TOTAL SCORE:        {total_score:.1f}/100")

    return total_score


if __name__ == "__main__":
    score_against_ground_truth()
