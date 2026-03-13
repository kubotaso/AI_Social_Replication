#!/usr/bin/env python3
"""
Table 5a Replication: Unstandardized Coefficients from Regression of
Traditional/Secular-Rational Values on Modernization and Cultural Heritage.

Inglehart & Baker (2000), Table 5a.
6 OLS models with nation-level DV (factor scores).
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from shared_factor_analysis import compute_nation_level_factor_scores, COUNTRY_NAMES

# Ground truth from Table 5a
GROUND_TRUTH = {
    'Model 1': {
        'GDP': {'coef': 0.066, 'se': 0.031, 'sig': '*'},
        'Industrial': {'coef': 0.052, 'se': 0.012, 'sig': '***'},
        'Adj_R2': 0.42,
        'N': 49,
    },
    'Model 2': {
        'GDP': {'coef': 0.086, 'se': 0.043, 'sig': '*'},
        'Industrial': {'coef': 0.051, 'se': 0.014, 'sig': '***'},
        'Education_enrollment': {'coef': -0.01, 'se': 0.01, 'sig': ''},
        'Service': {'coef': -0.054, 'se': 0.039, 'sig': ''},
        'Education_enrollment_2': {'coef': -0.005, 'se': 0.012, 'sig': ''},
        'Adj_R2': 0.37,
        'N': 46,
    },
    'Model 3': {
        'GDP': {'coef': 0.131, 'se': 0.036, 'sig': '**'},
        'Industrial': {'coef': 0.023, 'se': 0.015, 'sig': ''},
        'Ex_Communist': {'coef': 1.05, 'se': 0.351, 'sig': '**'},
        'Adj_R2': 0.50,
        'N': 49,
    },
    'Model 4': {
        'GDP': {'coef': 0.042, 'se': 0.029, 'sig': ''},
        'Industrial': {'coef': 0.061, 'se': 0.011, 'sig': '***'},
        'Catholic': {'coef': -0.767, 'se': 0.216, 'sig': '**'},
        'Adj_R2': 0.53,
        'N': 49,
    },
    'Model 5': {
        'GDP': {'coef': 0.080, 'se': 0.027, 'sig': '**'},
        'Industrial': {'coef': 0.052, 'se': 0.011, 'sig': '***'},
        'Confucian': {'coef': 1.57, 'se': 0.370, 'sig': '***'},
        'Adj_R2': 0.57,
        'N': 49,
    },
    'Model 6': {
        'GDP': {'coef': 0.122, 'se': 0.030, 'sig': '***'},
        'Industrial': {'coef': 0.030, 'se': 0.012, 'sig': '*'},
        'Ex_Communist': {'coef': 0.952, 'se': 0.282, 'sig': '***'},
        'Catholic': {'coef': -0.409, 'se': 0.188, 'sig': '*'},
        'Confucian': {'coef': 1.39, 'se': 0.329, 'sig': '***'},
        'Adj_R2': 0.70,
        'N': 49,
    },
}


def get_significance(pvalue):
    """Return significance stars."""
    if pvalue < 0.001:
        return '***'
    elif pvalue < 0.01:
        return '**'
    elif pvalue < 0.05:
        return '*'
    return ''


def run_analysis():
    """Run Table 5a replication."""

    # Step 1: Compute factor scores
    scores, loadings, means = compute_nation_level_factor_scores()
    print(f"Factor scores computed for {len(scores)} countries")

    # Step 2: Load World Bank data
    wb = pd.read_csv(os.path.join(BASE_DIR, 'data/world_bank_indicators.csv'))

    # Get GDP per capita (PPP, constant intl $) - use earliest available year
    # Paper says "1980" but our data starts at 1990
    gdp = wb[wb['indicator'] == 'NY.GDP.PCAP.PP.KD'][['economy', 'YR1990']].copy()
    gdp.columns = ['economy', 'gdp']
    gdp = gdp.dropna()
    gdp['gdp'] = gdp['gdp'] / 1000  # Convert to $1000s

    # Get industry employment
    ind = wb[wb['indicator'] == 'SL.IND.EMPL.ZS'][['economy', 'YR1991']].copy()
    ind.columns = ['economy', 'industrial']
    ind = ind.dropna()

    # Get service sector employment
    svc = wb[wb['indicator'] == 'SL.SRV.EMPL.ZS'][['economy', 'YR1991']].copy()
    svc.columns = ['economy', 'service']
    svc = svc.dropna()

    # Get education enrollment rates - try to combine available data
    # Primary + Secondary + Tertiary enrollment
    edu_vars = {}
    for ind_code, name in [('SE.PRM.ENRR', 'primary'), ('SE.SEC.ENRR', 'secondary'), ('SE.TER.ENRR', 'tertiary')]:
        sub = wb[wb['indicator'] == ind_code][['economy']].copy()
        # Try each year from 1990 back to find data
        for yr in ['YR1990', 'YR1991', 'YR1992', 'YR1993', 'YR1994', 'YR1995',
                    'YR1989', 'YR1988', 'YR1987', 'YR1986', 'YR1985']:
            if yr in wb.columns:
                col_data = wb[wb['indicator'] == ind_code][['economy', yr]].dropna()
                if len(col_data) > 0:
                    col_data.columns = ['economy', name]
                    edu_vars[name] = col_data
                    break

    # Merge all data
    data = scores.rename(columns={'COUNTRY_ALPHA': 'economy'})
    data = data.merge(gdp, on='economy', how='inner')
    data = data.merge(ind, on='economy', how='inner')

    print(f"After merging GDP + Industry: {len(data)} countries")
    print(f"Countries: {sorted(data['economy'].tolist())}")

    # Create cultural dummies
    # Ex-Communist countries
    ex_communist = ['ARM', 'AZE', 'BLR', 'BIH', 'BGR', 'CHN', 'HRV', 'CZE',
                    'EST', 'GEO', 'HUN', 'LVA', 'LTU', 'MKD', 'MDA', 'POL',
                    'ROU', 'RUS', 'SVK', 'SVN', 'UKR', 'SRB']

    # Historically Catholic - from instruction_summary
    hist_catholic = ['FRA', 'BEL', 'ITA', 'ESP', 'PRT', 'AUT', 'IRL', 'POL',
                     'HRV', 'SVN', 'HUN', 'CZE', 'SVK',
                     'ARG', 'BRA', 'CHL', 'COL', 'DOM', 'MEX', 'PER', 'PRI',
                     'URY', 'VEN', 'PHL']

    # Historically Confucian
    hist_confucian = ['JPN', 'KOR', 'TWN', 'CHN']

    data['ex_communist'] = data['economy'].isin(ex_communist).astype(int)
    data['catholic'] = data['economy'].isin(hist_catholic).astype(int)
    data['confucian'] = data['economy'].isin(hist_confucian).astype(int)

    # The paper uses N=49. We need to identify which 49 countries they use.
    # They likely exclude some countries with missing WB data.
    # Let's see what we have
    print(f"\nTotal countries with factor scores + GDP + Industry: {len(data)}")

    # Add service sector
    data_svc = data.merge(svc, on='economy', how='left')

    # Add education if available
    for name, edf in edu_vars.items():
        data_svc = data_svc.merge(edf, on='economy', how='left')

    # DV is trad_secrat (Traditional vs Secular-Rational)
    dv = 'trad_secrat'

    results = {}

    # Model 1: GDP + Industrial (N=49)
    m1_data = data.dropna(subset=[dv, 'gdp', 'industrial'])
    X1 = sm.add_constant(m1_data[['gdp', 'industrial']])
    y1 = m1_data[dv]
    model1 = sm.OLS(y1, X1).fit()
    results['Model 1'] = model1

    # Model 2: GDP + Industrial + Education + Service + Education (N=46)
    # The paper has two "% enrolled in education" rows (possibly primary vs. higher education)
    m2_data = data_svc.dropna(subset=[dv, 'gdp', 'industrial'])
    if 'service' in m2_data.columns:
        m2_data = m2_data.dropna(subset=['service'])
    # Add education vars if available
    edu_cols = [c for c in ['primary', 'secondary', 'tertiary'] if c in m2_data.columns]
    if edu_cols:
        m2_data = m2_data.dropna(subset=edu_cols)

    m2_ivs = ['gdp', 'industrial']
    if edu_cols:
        m2_ivs.extend(edu_cols[:1])  # First education variable
    if 'service' in m2_data.columns:
        m2_ivs.append('service')
    if len(edu_cols) > 1:
        m2_ivs.append(edu_cols[1])  # Second education variable

    if len(m2_data) > len(m2_ivs) + 1:
        X2 = sm.add_constant(m2_data[m2_ivs])
        y2 = m2_data[dv]
        model2 = sm.OLS(y2, X2).fit()
        results['Model 2'] = model2
    else:
        print("WARNING: Not enough data for Model 2")
        results['Model 2'] = None

    # Model 3: GDP + Industrial + Ex-Communist
    m3_data = data.dropna(subset=[dv, 'gdp', 'industrial'])
    X3 = sm.add_constant(m3_data[['gdp', 'industrial', 'ex_communist']])
    y3 = m3_data[dv]
    model3 = sm.OLS(y3, X3).fit()
    results['Model 3'] = model3

    # Model 4: GDP + Industrial + Catholic
    m4_data = data.dropna(subset=[dv, 'gdp', 'industrial'])
    X4 = sm.add_constant(m4_data[['gdp', 'industrial', 'catholic']])
    y4 = m4_data[dv]
    model4 = sm.OLS(y4, X4).fit()
    results['Model 4'] = model4

    # Model 5: GDP + Industrial + Confucian
    m5_data = data.dropna(subset=[dv, 'gdp', 'industrial'])
    X5 = sm.add_constant(m5_data[['gdp', 'industrial', 'confucian']])
    y5 = m5_data[dv]
    model5 = sm.OLS(y5, X5).fit()
    results['Model 5'] = model5

    # Model 6: GDP + Industrial + Ex-Communist + Catholic + Confucian
    m6_data = data.dropna(subset=[dv, 'gdp', 'industrial'])
    X6 = sm.add_constant(m6_data[['gdp', 'industrial', 'ex_communist', 'catholic', 'confucian']])
    y6 = m6_data[dv]
    model6 = sm.OLS(y6, X6).fit()
    results['Model 6'] = model6

    # Print results
    output_lines = []
    output_lines.append("=" * 90)
    output_lines.append("Table 5a: Unstandardized Coefficients - Traditional/Secular-Rational Values")
    output_lines.append("=" * 90)

    header = f"{'Variable':<40} {'M1':>8} {'M2':>8} {'M3':>8} {'M4':>8} {'M5':>8} {'M6':>8}"
    output_lines.append(header)
    output_lines.append("-" * 90)

    var_rows = [
        ('GDP per capita ($1000s)', 'gdp'),
        ('% Industrial sector', 'industrial'),
    ]

    # Additional M2 vars
    m2_extra = []
    if results.get('Model 2') is not None:
        for v in m2_ivs:
            if v not in ['gdp', 'industrial']:
                m2_extra.append(v)

    all_vars = var_rows + [(v, v) for v in m2_extra] + [
        ('Ex-Communist', 'ex_communist'),
        ('Catholic', 'catholic'),
        ('Confucian', 'confucian'),
    ]

    for var_label, var_name in all_vars:
        coef_row = f"{var_label:<40}"
        se_row = f"{'':<40}"

        for mname in ['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5', 'Model 6']:
            model = results.get(mname)
            if model is not None and var_name in model.params.index:
                coef = model.params[var_name]
                se = model.bse[var_name]
                pval = model.pvalues[var_name]
                sig = get_significance(pval)
                coef_row += f" {coef:7.3f}{sig:<3}"
                se_row += f" ({se:6.3f})  "
            else:
                coef_row += f" {'---':>10}"
                se_row += f" {'':>10}"

        output_lines.append(coef_row)
        output_lines.append(se_row)

    output_lines.append("-" * 90)

    # Adjusted R-squared
    r2_row = f"{'Adjusted R-squared':<40}"
    for mname in ['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5', 'Model 6']:
        model = results.get(mname)
        if model is not None:
            r2_row += f" {model.rsquared_adj:8.2f}"
        else:
            r2_row += f" {'---':>8}"
    output_lines.append(r2_row)

    # N
    n_row = f"{'N':<40}"
    for mname in ['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5', 'Model 6']:
        model = results.get(mname)
        if model is not None:
            n_row += f" {int(model.nobs):>8}"
        else:
            n_row += f" {'---':>8}"
    output_lines.append(n_row)

    output_lines.append("=" * 90)

    output = '\n'.join(output_lines)
    print(output)

    # Detailed results for scoring
    print("\n\n=== DETAILED RESULTS FOR SCORING ===")
    for mname in ['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5', 'Model 6']:
        model = results.get(mname)
        if model is not None:
            print(f"\n{mname}:")
            print(model.summary())

    return results


def score_against_ground_truth():
    """Compute alignment score against paper values."""
    results = run_analysis()

    total_points = 0
    max_points = 0
    details = []

    for mname, gt in GROUND_TRUTH.items():
        model = results.get(mname)
        if model is None:
            details.append(f"{mname}: Model not computed")
            max_points += 100
            continue

        # Check each coefficient
        var_map = {
            'GDP': 'gdp',
            'Industrial': 'industrial',
            'Ex_Communist': 'ex_communist',
            'Catholic': 'catholic',
            'Confucian': 'confucian',
            'Service': 'service',
            'Education_enrollment': None,
            'Education_enrollment_2': None,
        }

        for var_key, var_info in gt.items():
            if var_key in ['Adj_R2', 'N']:
                continue

            col_name = var_map.get(var_key)
            if col_name is None:
                # Education variables - try to match
                if 'Education' in var_key:
                    edu_cols = [c for c in model.params.index if c not in ['const', 'gdp', 'industrial', 'service', 'ex_communist', 'catholic', 'confucian']]
                    if edu_cols:
                        col_name = edu_cols[0] if 'enrollment' in var_key.lower() and '_2' not in var_key else (edu_cols[1] if len(edu_cols) > 1 else None)

            if col_name is not None and col_name in model.params.index:
                gen_coef = model.params[col_name]
                gen_se = model.bse[col_name]
                gen_sig = get_significance(model.pvalues[col_name])

                # Coefficient match (within 0.05)
                max_points += 5
                coef_diff = abs(gen_coef - var_info['coef'])
                if coef_diff <= 0.05:
                    total_points += 5
                    details.append(f"{mname} {var_key} coef: {gen_coef:.3f} vs {var_info['coef']:.3f} MATCH")
                elif coef_diff <= 0.1:
                    total_points += 3
                    details.append(f"{mname} {var_key} coef: {gen_coef:.3f} vs {var_info['coef']:.3f} PARTIAL")
                else:
                    details.append(f"{mname} {var_key} coef: {gen_coef:.3f} vs {var_info['coef']:.3f} MISS (diff={coef_diff:.3f})")

                # SE match (within 0.02)
                max_points += 3
                se_diff = abs(gen_se - var_info['se'])
                if se_diff <= 0.02:
                    total_points += 3
                    details.append(f"  SE: {gen_se:.3f} vs {var_info['se']:.3f} MATCH")
                elif se_diff <= 0.05:
                    total_points += 1.5
                    details.append(f"  SE: {gen_se:.3f} vs {var_info['se']:.3f} PARTIAL")
                else:
                    details.append(f"  SE: {gen_se:.3f} vs {var_info['se']:.3f} MISS")

                # Significance match
                max_points += 5
                if gen_sig == var_info['sig']:
                    total_points += 5
                    details.append(f"  Sig: {gen_sig} vs {var_info['sig']} MATCH")
                else:
                    details.append(f"  Sig: {gen_sig} vs {var_info['sig']} MISS")
            else:
                max_points += 13  # coef + se + sig
                if col_name is None:
                    details.append(f"{mname} {var_key}: Variable not mapped")
                else:
                    details.append(f"{mname} {var_key}: Variable {col_name} not in model")

        # Adj R2
        max_points += 5
        gen_r2 = model.rsquared_adj
        paper_r2 = gt['Adj_R2']
        r2_diff = abs(gen_r2 - paper_r2)
        if r2_diff <= 0.02:
            total_points += 5
            details.append(f"{mname} Adj R2: {gen_r2:.2f} vs {paper_r2:.2f} MATCH")
        elif r2_diff <= 0.05:
            total_points += 3
            details.append(f"{mname} Adj R2: {gen_r2:.2f} vs {paper_r2:.2f} PARTIAL")
        else:
            details.append(f"{mname} Adj R2: {gen_r2:.2f} vs {paper_r2:.2f} MISS (diff={r2_diff:.2f})")

        # N
        max_points += 5
        gen_n = int(model.nobs)
        paper_n = gt['N']
        n_pct_diff = abs(gen_n - paper_n) / paper_n
        if n_pct_diff <= 0.05:
            total_points += 5
            details.append(f"{mname} N: {gen_n} vs {paper_n} MATCH")
        elif n_pct_diff <= 0.15:
            total_points += 3
            details.append(f"{mname} N: {gen_n} vs {paper_n} PARTIAL")
        else:
            details.append(f"{mname} N: {gen_n} vs {paper_n} MISS")

    score = int(100 * total_points / max_points) if max_points > 0 else 0

    print("\n\n=== SCORING ===")
    for d in details:
        print(d)
    print(f"\nTotal: {total_points}/{max_points} = {score}/100")

    return score


if __name__ == "__main__":
    score = score_against_ground_truth()
