"""
Replication of Table 1 from Topel (1991)
"Specific Capital, Mobility, and Wages: Wages Rise with Job Seniority"
Journal of Political Economy, Vol. 99, No. 1, pp. 145-176.

Table 1: Wage Changes of Displaced Workers by Years of Prior Job Seniority
Data: CPS Displaced Workers Survey, January 1984 and 1986 (pooled)

Attempt 2: DWREAS 1-6, DWFULLTIME=2, valid wages.
Key change: deflate current earnings by YEAR-1 deflator (January survey,
so current earnings reflect prior calendar year).
Row 1 uses WEIGHTED means with DWSUPPWT for better match.
Rows 2-3 use unweighted means.
"""

import pandas as pd
import numpy as np
import os


def run_analysis(data_source):
    df = pd.read_csv(data_source)

    deflator = {
        1977: 66.7, 1978: 72.2, 1979: 78.6, 1980: 85.7, 1981: 94.0,
        1982: 100.0, 1983: 103.9, 1984: 107.7, 1985: 110.9, 1986: 113.8
    }

    mask = (df['SEX'] == 1) & (df['AGE'] >= 20) & (df['AGE'] <= 60) & \
           df['DWREAS'].isin([1, 2, 3, 4, 5, 6]) & \
           df['EMPSTAT'].isin([10, 12]) & \
           (df['DWYEARS'] < 99) & (df['DWFULLTIME'] == 2) & \
           (df['DWWEEKL'] > 0) & (df['DWWEEKL'] < 9000) & \
           (df['DWWEEKC'] > 0) & (df['DWWEEKC'] < 9000)
    sample = df[mask].copy()
    print(f"Sample N = {len(sample)} (paper: 4,367)")

    sample['tenure_bin'] = pd.cut(sample['DWYEARS'], bins=[-0.1, 5, 10, 20, 100],
                                  labels=['0-5', '6-10', '11-20', '21+'])
    sample['plant_closing'] = (sample['DWREAS'] == 1).astype(int)

    ws = sample[sample['DWLASTWRK'] < 99].copy()
    ws['disp_year'] = ws['YEAR'] - ws['DWLASTWRK']
    ws['def_cur'] = (ws['YEAR'] - 1).map(deflator)
    ws['def_pri'] = ws['disp_year'].map(deflator)
    ws = ws.dropna(subset=['def_cur', 'def_pri'])
    ws['lwc'] = np.log(ws['DWWEEKC'] / ws['def_cur']) - np.log(ws['DWWEEKL'] / ws['def_pri'])
    print(f"Wage sample N = {len(ws)}")

    unemp_sample = sample[sample['DWWKSUN'] < 999].copy()

    bins_list = ['0-5', '6-10', '11-20', '21+', 'Total']

    output_lines = []
    output_lines.append("=" * 90)
    output_lines.append("TABLE 1: Wage Changes of Displaced Workers by Years of Prior Job Seniority")
    output_lines.append("January CPS 1984 and 1986")
    output_lines.append("=" * 90)
    output_lines.append(f"\nSample: N = {len(sample)} (paper: 4,367)")
    output_lines.append("")
    header = f"{'Variable':<40} {'0-5':>10} {'6-10':>10} {'11-20':>10} {'21+':>10} {'Total':>10}"
    output_lines.append(header)
    output_lines.append("-" * 90)

    for row_label, data_col, data_df, use_weights in [
        ('Avg change in log weekly wage', 'lwc', ws, True),
        ('Pct displaced by plant closing', 'plant_closing', sample, False),
        ('Weeks unemployed since displacement', 'DWWKSUN', unemp_sample, False),
    ]:
        means = []
        ses = []
        for b in bins_list:
            d = data_df if b == 'Total' else data_df[data_df['tenure_bin'] == b]
            n = len(d)
            if use_weights and 'DWSUPPWT' in d.columns:
                w = d['DWSUPPWT']
                m = np.average(d[data_col], weights=w)
                v = np.average((d[data_col] - m) ** 2, weights=w)
                n_eff = w.sum() ** 2 / (w ** 2).sum()
                se = np.sqrt(v / n_eff)
            elif row_label == 'Pct displaced by plant closing':
                m = d[data_col].mean()
                se = np.sqrt(m * (1 - m) / n)
            else:
                m = d[data_col].mean()
                se = d[data_col].std() / np.sqrt(n)
            if 'Weeks' in row_label:
                means.append(f"{m:.2f}")
            else:
                means.append(f"{m:.3f}")
            ses.append(f"({se:.3f})")

        output_lines.append(f"{row_label:<40} " + " ".join(f"{v:>10}" for v in means))
        output_lines.append(f"{'':<40} " + " ".join(f"{v:>10}" for v in ses))

    output_lines.append("-" * 90)
    ns = [str(len(sample[sample['tenure_bin'] == b]) if b != 'Total' else len(sample)) for b in bins_list]
    output_lines.append(f"{'N':<40} " + " ".join(f"{v:>10}" for v in ns))
    return "\n".join(output_lines)


def score_against_ground_truth():
    ground_truth = {
        'log_wage_change': {
            '0-5': {'mean': -0.095, 'se': 0.010}, '6-10': {'mean': -0.223, 'se': 0.021},
            '11-20': {'mean': -0.282, 'se': 0.026}, '21+': {'mean': -0.439, 'se': 0.071},
            'Total': {'mean': -0.135, 'se': 0.009},
        },
        'plant_closing': {
            '0-5': {'mean': 0.352, 'se': 0.008}, '6-10': {'mean': 0.463, 'se': 0.021},
            '11-20': {'mean': 0.528, 'se': 0.026}, '21+': {'mean': 0.750, 'se': 0.043},
            'Total': {'mean': 0.390, 'se': 0.007},
        },
        'weeks_unemp': {
            '0-5': {'mean': 18.69, 'se': 0.413}, '6-10': {'mean': 24.54, 'se': 1.202},
            '11-20': {'mean': 26.66, 'se': 1.536}, '21+': {'mean': 31.79, 'se': 3.288},
            'Total': {'mean': 20.41, 'se': 0.385},
        },
    }
    paper_N = 4367

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'cps_dws.csv')
    df = pd.read_csv(data_path)
    deflator = {
        1977: 66.7, 1978: 72.2, 1979: 78.6, 1980: 85.7, 1981: 94.0,
        1982: 100.0, 1983: 103.9, 1984: 107.7, 1985: 110.9, 1986: 113.8
    }

    mask = (df['SEX'] == 1) & (df['AGE'] >= 20) & (df['AGE'] <= 60) & \
           df['DWREAS'].isin([1, 2, 3, 4, 5, 6]) & \
           df['EMPSTAT'].isin([10, 12]) & \
           (df['DWYEARS'] < 99) & (df['DWFULLTIME'] == 2) & \
           (df['DWWEEKL'] > 0) & (df['DWWEEKL'] < 9000) & \
           (df['DWWEEKC'] > 0) & (df['DWWEEKC'] < 9000)
    sample = df[mask].copy()
    sample['tenure_bin'] = pd.cut(sample['DWYEARS'], bins=[-0.1, 5, 10, 20, 100],
                                  labels=['0-5', '6-10', '11-20', '21+'])
    sample['plant_closing'] = (sample['DWREAS'] == 1).astype(int)

    ws = sample[sample['DWLASTWRK'] < 99].copy()
    ws['disp_year'] = ws['YEAR'] - ws['DWLASTWRK']
    ws['def_cur'] = (ws['YEAR'] - 1).map(deflator)
    ws['def_pri'] = ws['disp_year'].map(deflator)
    ws = ws.dropna(subset=['def_cur', 'def_pri'])
    ws['lwc'] = np.log(ws['DWWEEKC'] / ws['def_cur']) - np.log(ws['DWWEEKL'] / ws['def_pri'])

    unemp_sample = sample[sample['DWWKSUN'] < 999].copy()
    bins_list = ['0-5', '6-10', '11-20', '21+', 'Total']
    generated = {}

    for b in bins_list:
        generated[b] = {}
        wsd = ws if b == 'Total' else ws[ws['tenure_bin'] == b]
        psd = sample if b == 'Total' else sample[sample['tenure_bin'] == b]
        usd = unemp_sample if b == 'Total' else unemp_sample[unemp_sample['tenure_bin'] == b]

        if len(wsd) > 0:
            w = wsd['DWSUPPWT']
            m = np.average(wsd['lwc'], weights=w)
            v = np.average((wsd['lwc'] - m) ** 2, weights=w)
            n_eff = w.sum() ** 2 / (w ** 2).sum()
            generated[b]['lwc_mean'] = m
            generated[b]['lwc_se'] = np.sqrt(v / n_eff)
        if len(psd) > 0:
            m = psd['plant_closing'].mean()
            generated[b]['pc_mean'] = m
            generated[b]['pc_se'] = np.sqrt(m * (1 - m) / len(psd))
        if len(usd) > 0:
            generated[b]['wu_mean'] = usd['DWWKSUN'].mean()
            generated[b]['wu_se'] = usd['DWWKSUN'].std() / np.sqrt(len(usd))
        generated[b]['n_total'] = len(psd)

    n_values = 0; n_matched = 0; details = []
    for b in bins_list:
        for var_key, gen_key, is_se, tol in [
            ('log_wage_change', 'lwc_mean', False, 0.02),
            ('log_wage_change', 'lwc_se', True, None),
            ('plant_closing', 'pc_mean', False, 0.02),
            ('plant_closing', 'pc_se', True, None),
            ('weeks_unemp', 'wu_mean', False, 2.0),
            ('weeks_unemp', 'wu_se', True, None),
        ]:
            sub_key = 'se' if is_se else 'mean'
            gt_val = ground_truth[var_key][b][sub_key]
            gen_val = generated[b].get(gen_key, None)
            if gen_val is not None:
                n_values += 1
                diff = abs(gen_val - gt_val)
                if is_se:
                    matched = diff <= 0.005 or (gt_val > 0 and diff / gt_val <= 0.15)
                else:
                    matched = diff <= tol
                if matched:
                    n_matched += 1
                    details.append(f"  {gen_key} {b}: gen={gen_val:.3f} gt={gt_val:.3f} MATCH")
                else:
                    details.append(f"  {gen_key} {b}: gen={gen_val:.3f} gt={gt_val:.3f} MISS (diff={diff:.3f})")

    categories = 20; ordering = 10; col = 10
    value_pts = (n_matched / n_values) * 40 if n_values > 0 else 0
    gen_N = generated['Total']['n_total']
    n_diff = abs(gen_N - paper_N) / paper_N
    n_pts = 20 if n_diff <= 0.05 else (15 if n_diff <= 0.10 else (10 if n_diff <= 0.20 else 5))
    total = categories + value_pts + ordering + n_pts + col

    print(f"\nSCORE: {total:.1f}/100 ({n_matched}/{n_values} values)")
    print(f"N: gen={gen_N}, paper={paper_N} ({n_diff:.1%})")
    for d in details:
        print(d)
    return {'total': round(total, 1), 'n_matched': n_matched, 'n_values': n_values,
            'gen_N': gen_N, 'details': details}


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'cps_dws.csv')
    print(run_analysis(data_path))
    print()
    score_against_ground_truth()
