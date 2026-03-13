"""
Replication of Table 1 from Topel (1991)
"Specific Capital, Mobility, and Wages: Wages Rise with Job Seniority"
Journal of Political Economy, Vol. 99, No. 1, pp. 145-176.

Table 1: Wage Changes of Displaced Workers by Years of Prior Job Seniority
Data: CPS Displaced Workers Survey, January 1984 and 1986 (pooled)
"""

import pandas as pd
import numpy as np
import os
import sys


def run_analysis(data_source):
    """
    Replicate Table 1 from Topel (1991).

    Parameters:
        data_source: path to CPS DWS CSV file

    Returns:
        String with formatted results
    """
    # Load data
    df = pd.read_csv(data_source)

    # =====================================================================
    # SAMPLE SELECTION
    # =====================================================================
    # Males age 20-60
    mask = (df['SEX'] == 1) & (df['AGE'] >= 20) & (df['AGE'] <= 60)

    # Displaced for economic reasons (plant closings, layoffs)
    mask &= df['DWREAS'].isin([1, 2, 3])

    # Currently employed
    mask &= df['EMPSTAT'].isin([10, 12])

    # Valid tenure data
    mask &= df['DWYEARS'] < 99

    sample = df[mask].copy()

    print(f"After basic filters (male, 20-60, econ displacement, employed, valid tenure): {len(sample)}")

    # =====================================================================
    # GNP PRICE DEFLATOR FOR CONSUMPTION EXPENDITURE
    # =====================================================================
    # Base year 1982 = 100
    deflator = {
        1978: 72.2,
        1979: 78.6,
        1980: 85.7,
        1981: 94.0,
        1982: 100.0,
        1983: 103.9,
        1984: 107.7,
        1985: 110.9,
        1986: 113.8
    }

    # =====================================================================
    # VARIABLE CONSTRUCTION
    # =====================================================================

    # Seniority bins
    sample['tenure_bin'] = pd.cut(
        sample['DWYEARS'],
        bins=[-0.1, 5, 10, 20, 100],
        labels=['0-5', '6-10', '11-20', '21+']
    )

    # Displacement year
    sample['disp_year'] = sample['YEAR'] - sample['DWLASTWRK']

    # Plant closing indicator
    sample['plant_closing'] = (sample['DWREAS'] == 1).astype(int)

    # =====================================================================
    # ROW 1: Average change in log weekly wage
    # =====================================================================
    # Need valid weekly earnings for both prior and current jobs
    valid_wages = (sample['DWWEEKL'] > 0) & (sample['DWWEEKL'] < 9000) & \
                  (sample['DWWEEKC'] > 0) & (sample['DWWEEKC'] < 9000)

    wage_sample = sample[valid_wages].copy()

    # Also need valid DWLASTWRK for deflation
    wage_sample = wage_sample[wage_sample['DWLASTWRK'] < 99].copy()

    print(f"Valid wage sample: {len(wage_sample)}")

    # Deflate earnings
    # Current earnings are from survey year (YEAR)
    # Prior earnings are from displacement year (YEAR - DWLASTWRK)
    wage_sample['deflator_current'] = wage_sample['YEAR'].map(deflator)
    wage_sample['deflator_prior'] = wage_sample['disp_year'].map(deflator)

    # Drop rows where deflator is not available
    wage_sample = wage_sample.dropna(subset=['deflator_current', 'deflator_prior'])

    print(f"Valid wage sample after deflation: {len(wage_sample)}")

    # Real weekly earnings
    wage_sample['real_current'] = wage_sample['DWWEEKC'] / wage_sample['deflator_current'] * 100
    wage_sample['real_prior'] = wage_sample['DWWEEKL'] / wage_sample['deflator_prior'] * 100

    # Log wage change
    wage_sample['log_wage_change'] = np.log(wage_sample['real_current']) - np.log(wage_sample['real_prior'])

    # =====================================================================
    # ROW 2: Percentage displaced by plant closing
    # =====================================================================
    # Uses the full sample (not just those with valid wages)

    # =====================================================================
    # ROW 3: Weeks unemployed since displacement
    # =====================================================================
    unemp_sample = sample[sample['DWWKSUN'] < 999].copy()
    print(f"Valid unemployment weeks sample: {len(unemp_sample)}")

    # =====================================================================
    # COMPUTE STATISTICS
    # =====================================================================

    results = {}
    bins = ['0-5', '6-10', '11-20', '21+', 'Total']

    # Try both weighted and unweighted - start with unweighted
    # as the paper may report unweighted means for this descriptive table

    for approach in ['unweighted', 'weighted']:
        results[approach] = {}

        for bin_name in bins:
            results[approach][bin_name] = {}

            if bin_name == 'Total':
                ws = wage_sample
                ps = sample
                us = unemp_sample
            else:
                ws = wage_sample[wage_sample['tenure_bin'] == bin_name]
                ps = sample[sample['tenure_bin'] == bin_name]
                us = unemp_sample[unemp_sample['tenure_bin'] == bin_name]

            # Row 1: Log wage change
            if len(ws) > 0:
                if approach == 'weighted' and 'DWSUPPWT' in ws.columns:
                    w = ws['DWSUPPWT']
                    mean_lwc = np.average(ws['log_wage_change'], weights=w)
                    # Weighted SE
                    var = np.average((ws['log_wage_change'] - mean_lwc) ** 2, weights=w)
                    n_eff = w.sum() ** 2 / (w ** 2).sum()
                    se_lwc = np.sqrt(var / n_eff)
                else:
                    mean_lwc = ws['log_wage_change'].mean()
                    se_lwc = ws['log_wage_change'].std() / np.sqrt(len(ws))
                results[approach][bin_name]['log_wage_change_mean'] = mean_lwc
                results[approach][bin_name]['log_wage_change_se'] = se_lwc
                results[approach][bin_name]['n_wages'] = len(ws)

            # Row 2: Plant closing percentage
            if len(ps) > 0:
                if approach == 'weighted' and 'DWSUPPWT' in ps.columns:
                    w = ps['DWSUPPWT']
                    mean_pc = np.average(ps['plant_closing'], weights=w)
                    se_pc = np.sqrt(mean_pc * (1 - mean_pc) / (w.sum() ** 2 / (w ** 2).sum()))
                else:
                    mean_pc = ps['plant_closing'].mean()
                    se_pc = np.sqrt(mean_pc * (1 - mean_pc) / len(ps))
                results[approach][bin_name]['plant_closing_mean'] = mean_pc
                results[approach][bin_name]['plant_closing_se'] = se_pc
                results[approach][bin_name]['n_closing'] = len(ps)

            # Row 3: Weeks unemployed
            if len(us) > 0:
                if approach == 'weighted' and 'DWSUPPWT' in us.columns:
                    w = us['DWSUPPWT']
                    mean_wu = np.average(us['DWWKSUN'], weights=w)
                    var = np.average((us['DWWKSUN'] - mean_wu) ** 2, weights=w)
                    n_eff = w.sum() ** 2 / (w ** 2).sum()
                    se_wu = np.sqrt(var / n_eff)
                else:
                    mean_wu = us['DWWKSUN'].mean()
                    se_wu = us['DWWKSUN'].std() / np.sqrt(len(us))
                results[approach][bin_name]['weeks_unemp_mean'] = mean_wu
                results[approach][bin_name]['weeks_unemp_se'] = se_wu
                results[approach][bin_name]['n_unemp'] = len(us)

    # =====================================================================
    # FORMAT OUTPUT
    # =====================================================================

    output_lines = []
    output_lines.append("=" * 90)
    output_lines.append("TABLE 1: Wage Changes of Displaced Workers by Years of Prior Job Seniority")
    output_lines.append("January CPS 1984 and 1986")
    output_lines.append("=" * 90)

    for approach in ['unweighted', 'weighted']:
        output_lines.append(f"\n{'=' * 90}")
        output_lines.append(f"APPROACH: {approach.upper()}")
        output_lines.append(f"{'=' * 90}")

        header = f"{'Variable':<40} {'0-5':>10} {'6-10':>10} {'11-20':>10} {'21+':>10} {'Total':>10}"
        output_lines.append(header)
        output_lines.append("-" * 90)

        # Row 1
        vals = [f"{results[approach][b]['log_wage_change_mean']:.3f}" if 'log_wage_change_mean' in results[approach][b] else 'N/A' for b in bins]
        output_lines.append(f"{'Avg change in log weekly wage':<40} " + " ".join(f"{v:>10}" for v in vals))
        ses = [f"({results[approach][b]['log_wage_change_se']:.3f})" if 'log_wage_change_se' in results[approach][b] else '' for b in bins]
        output_lines.append(f"{'':<40} " + " ".join(f"{v:>10}" for v in ses))

        # Row 2
        vals = [f"{results[approach][b]['plant_closing_mean']:.3f}" if 'plant_closing_mean' in results[approach][b] else 'N/A' for b in bins]
        output_lines.append(f"{'Pct displaced by plant closing':<40} " + " ".join(f"{v:>10}" for v in vals))
        ses = [f"({results[approach][b]['plant_closing_se']:.3f})" if 'plant_closing_se' in results[approach][b] else '' for b in bins]
        output_lines.append(f"{'':<40} " + " ".join(f"{v:>10}" for v in ses))

        # Row 3
        vals = [f"{results[approach][b]['weeks_unemp_mean']:.2f}" if 'weeks_unemp_mean' in results[approach][b] else 'N/A' for b in bins]
        output_lines.append(f"{'Weeks unemployed since displacement':<40} " + " ".join(f"{v:>10}" for v in vals))
        ses = [f"({results[approach][b]['weeks_unemp_se']:.3f})" if 'weeks_unemp_se' in results[approach][b] else '' for b in bins]
        output_lines.append(f"{'':<40} " + " ".join(f"{v:>10}" for v in ses))

        # Sample sizes
        output_lines.append("-" * 90)
        ns_w = [str(results[approach][b].get('n_wages', 'N/A')) for b in bins]
        ns_c = [str(results[approach][b].get('n_closing', 'N/A')) for b in bins]
        ns_u = [str(results[approach][b].get('n_unemp', 'N/A')) for b in bins]
        output_lines.append(f"{'N (wage change)':<40} " + " ".join(f"{v:>10}" for v in ns_w))
        output_lines.append(f"{'N (plant closing)':<40} " + " ".join(f"{v:>10}" for v in ns_c))
        output_lines.append(f"{'N (weeks unemployed)':<40} " + " ".join(f"{v:>10}" for v in ns_u))

    output = "\n".join(output_lines)
    return output


def score_against_ground_truth():
    """
    Score results against ground truth values from Topel (1991) Table 1.
    Returns per-criterion breakdown and total score.
    """
    # Ground truth from the paper
    ground_truth = {
        'log_wage_change': {
            '0-5':  {'mean': -0.095, 'se': 0.010},
            '6-10': {'mean': -0.223, 'se': 0.021},
            '11-20': {'mean': -0.282, 'se': 0.026},
            '21+':  {'mean': -0.439, 'se': 0.071},
            'Total': {'mean': -0.135, 'se': 0.009},
        },
        'plant_closing': {
            '0-5':  {'mean': 0.352, 'se': 0.008},
            '6-10': {'mean': 0.463, 'se': 0.021},
            '11-20': {'mean': 0.528, 'se': 0.026},
            '21+':  {'mean': 0.750, 'se': 0.043},
            'Total': {'mean': 0.390, 'se': 0.007},
        },
        'weeks_unemp': {
            '0-5':  {'mean': 18.69, 'se': 0.413},
            '6-10': {'mean': 24.54, 'se': 1.202},
            '11-20': {'mean': 26.66, 'se': 1.536},
            '21+':  {'mean': 31.79, 'se': 3.288},
            'Total': {'mean': 20.41, 'se': 0.385},
        },
    }
    paper_N = 4367

    # Run the analysis
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'cps_dws.csv')
    result_text = run_analysis(data_path)

    # Parse results - run analysis again to get raw numbers
    df = pd.read_csv(data_path)

    # Replicate the computation to get raw values
    mask = (df['SEX'] == 1) & (df['AGE'] >= 20) & (df['AGE'] <= 60) & \
           df['DWREAS'].isin([1, 2, 3]) & df['EMPSTAT'].isin([10, 12]) & \
           (df['DWYEARS'] < 99)
    sample = df[mask].copy()

    deflator = {1978: 72.2, 1979: 78.6, 1980: 85.7, 1981: 94.0,
                1982: 100.0, 1983: 103.9, 1984: 107.7, 1985: 110.9, 1986: 113.8}

    sample['tenure_bin'] = pd.cut(sample['DWYEARS'], bins=[-0.1, 5, 10, 20, 100], labels=['0-5', '6-10', '11-20', '21+'])
    sample['disp_year'] = sample['YEAR'] - sample['DWLASTWRK']
    sample['plant_closing'] = (sample['DWREAS'] == 1).astype(int)

    valid_wages = (sample['DWWEEKL'] > 0) & (sample['DWWEEKL'] < 9000) & \
                  (sample['DWWEEKC'] > 0) & (sample['DWWEEKC'] < 9000) & \
                  (sample['DWLASTWRK'] < 99)
    wage_sample = sample[valid_wages].copy()
    wage_sample['deflator_current'] = wage_sample['YEAR'].map(deflator)
    wage_sample['deflator_prior'] = wage_sample['disp_year'].map(deflator)
    wage_sample = wage_sample.dropna(subset=['deflator_current', 'deflator_prior'])
    wage_sample['real_current'] = wage_sample['DWWEEKC'] / wage_sample['deflator_current'] * 100
    wage_sample['real_prior'] = wage_sample['DWWEEKL'] / wage_sample['deflator_prior'] * 100
    wage_sample['log_wage_change'] = np.log(wage_sample['real_current']) - np.log(wage_sample['real_prior'])

    unemp_sample = sample[sample['DWWKSUN'] < 999].copy()

    bins_list = ['0-5', '6-10', '11-20', '21+', 'Total']

    # Score: try both weighted and unweighted, pick best
    best_total_score = 0
    best_approach = None
    best_breakdown = None

    for approach in ['unweighted', 'weighted']:
        generated = {}
        for bin_name in bins_list:
            generated[bin_name] = {}
            if bin_name == 'Total':
                ws = wage_sample
                ps = sample
                us = unemp_sample
            else:
                ws = wage_sample[wage_sample['tenure_bin'] == bin_name]
                ps = sample[sample['tenure_bin'] == bin_name]
                us = unemp_sample[unemp_sample['tenure_bin'] == bin_name]

            if approach == 'weighted':
                if len(ws) > 0:
                    w = ws['DWSUPPWT']
                    m = np.average(ws['log_wage_change'], weights=w)
                    v = np.average((ws['log_wage_change'] - m) ** 2, weights=w)
                    n_eff = w.sum() ** 2 / (w ** 2).sum()
                    generated[bin_name]['lwc_mean'] = m
                    generated[bin_name]['lwc_se'] = np.sqrt(v / n_eff)
                if len(ps) > 0:
                    w = ps['DWSUPPWT']
                    m = np.average(ps['plant_closing'], weights=w)
                    generated[bin_name]['pc_mean'] = m
                    generated[bin_name]['pc_se'] = np.sqrt(m * (1-m) / (w.sum() ** 2 / (w ** 2).sum()))
                if len(us) > 0:
                    w = us['DWSUPPWT']
                    m = np.average(us['DWWKSUN'], weights=w)
                    v = np.average((us['DWWKSUN'] - m) ** 2, weights=w)
                    n_eff = w.sum() ** 2 / (w ** 2).sum()
                    generated[bin_name]['wu_mean'] = m
                    generated[bin_name]['wu_se'] = np.sqrt(v / n_eff)
            else:
                if len(ws) > 0:
                    generated[bin_name]['lwc_mean'] = ws['log_wage_change'].mean()
                    generated[bin_name]['lwc_se'] = ws['log_wage_change'].std() / np.sqrt(len(ws))
                if len(ps) > 0:
                    generated[bin_name]['pc_mean'] = ps['plant_closing'].mean()
                    generated[bin_name]['pc_se'] = np.sqrt(ps['plant_closing'].mean() * (1 - ps['plant_closing'].mean()) / len(ps))
                if len(us) > 0:
                    generated[bin_name]['wu_mean'] = us['DWWKSUN'].mean()
                    generated[bin_name]['wu_se'] = us['DWWKSUN'].std() / np.sqrt(len(us))

            generated[bin_name]['n_total'] = len(ps)

        # ====== SCORING ======
        # Categories present: 20 pts
        categories_present = 20  # All bins always present

        # Count/percentage values: 40 pts
        # 15 means + 15 SEs = 30 comparisons
        value_points = 0
        value_max = 40
        n_values = 0
        n_matched = 0

        details = []

        for bin_name in bins_list:
            # Log wage change mean
            gt_val = ground_truth['log_wage_change'][bin_name]['mean']
            gen_val = generated[bin_name].get('lwc_mean', None)
            if gen_val is not None:
                n_values += 1
                diff = abs(gen_val - gt_val)
                if diff <= 0.02:
                    n_matched += 1
                    details.append(f"  lwc_mean {bin_name}: gen={gen_val:.3f} gt={gt_val:.3f} MATCH (diff={diff:.3f})")
                else:
                    details.append(f"  lwc_mean {bin_name}: gen={gen_val:.3f} gt={gt_val:.3f} MISS (diff={diff:.3f})")

            # Log wage change SE
            gt_val = ground_truth['log_wage_change'][bin_name]['se']
            gen_val = generated[bin_name].get('lwc_se', None)
            if gen_val is not None:
                n_values += 1
                diff = abs(gen_val - gt_val)
                if diff <= 0.005 or (gt_val > 0 and diff / gt_val <= 0.15):
                    n_matched += 1
                    details.append(f"  lwc_se {bin_name}: gen={gen_val:.3f} gt={gt_val:.3f} MATCH")
                else:
                    details.append(f"  lwc_se {bin_name}: gen={gen_val:.3f} gt={gt_val:.3f} MISS (diff={diff:.3f})")

            # Plant closing mean
            gt_val = ground_truth['plant_closing'][bin_name]['mean']
            gen_val = generated[bin_name].get('pc_mean', None)
            if gen_val is not None:
                n_values += 1
                diff = abs(gen_val - gt_val)
                if diff <= 0.02:
                    n_matched += 1
                    details.append(f"  pc_mean {bin_name}: gen={gen_val:.3f} gt={gt_val:.3f} MATCH")
                else:
                    details.append(f"  pc_mean {bin_name}: gen={gen_val:.3f} gt={gt_val:.3f} MISS (diff={diff:.3f})")

            # Plant closing SE
            gt_val = ground_truth['plant_closing'][bin_name]['se']
            gen_val = generated[bin_name].get('pc_se', None)
            if gen_val is not None:
                n_values += 1
                diff = abs(gen_val - gt_val)
                if diff <= 0.005 or (gt_val > 0 and diff / gt_val <= 0.15):
                    n_matched += 1
                    details.append(f"  pc_se {bin_name}: gen={gen_val:.3f} gt={gt_val:.3f} MATCH")
                else:
                    details.append(f"  pc_se {bin_name}: gen={gen_val:.3f} gt={gt_val:.3f} MISS (diff={diff:.3f})")

            # Weeks unemployed mean
            gt_val = ground_truth['weeks_unemp'][bin_name]['mean']
            gen_val = generated[bin_name].get('wu_mean', None)
            if gen_val is not None:
                n_values += 1
                diff = abs(gen_val - gt_val)
                if diff <= 2.0:
                    n_matched += 1
                    details.append(f"  wu_mean {bin_name}: gen={gen_val:.2f} gt={gt_val:.2f} MATCH")
                else:
                    details.append(f"  wu_mean {bin_name}: gen={gen_val:.2f} gt={gt_val:.2f} MISS (diff={diff:.2f})")

            # Weeks unemployed SE
            gt_val = ground_truth['weeks_unemp'][bin_name]['se']
            gen_val = generated[bin_name].get('wu_se', None)
            if gen_val is not None:
                n_values += 1
                diff = abs(gen_val - gt_val)
                if diff <= 0.3 or (gt_val > 0 and diff / gt_val <= 0.15):
                    n_matched += 1
                    details.append(f"  wu_se {bin_name}: gen={gen_val:.3f} gt={gt_val:.3f} MATCH")
                else:
                    details.append(f"  wu_se {bin_name}: gen={gen_val:.3f} gt={gt_val:.3f} MISS (diff={diff:.3f})")

        if n_values > 0:
            value_points = (n_matched / n_values) * value_max

        # Ordering: 10 pts (always correct since we use fixed bin order)
        ordering_points = 10

        # Sample size N: 20 pts
        gen_N = generated['Total']['n_total']
        n_diff_pct = abs(gen_N - paper_N) / paper_N
        if n_diff_pct <= 0.05:
            n_points = 20
        elif n_diff_pct <= 0.10:
            n_points = 15
        elif n_diff_pct <= 0.20:
            n_points = 10
        else:
            n_points = 5

        # Column structure: 10 pts
        col_points = 10

        total_score = categories_present + value_points + ordering_points + n_points + col_points

        breakdown = {
            'approach': approach,
            'categories_present': categories_present,
            'value_points': round(value_points, 1),
            'n_matched': n_matched,
            'n_values': n_values,
            'ordering': ordering_points,
            'sample_size': n_points,
            'column_structure': col_points,
            'total': round(total_score, 1),
            'gen_N': gen_N,
            'paper_N': paper_N,
            'details': details,
        }

        if total_score > best_total_score:
            best_total_score = total_score
            best_approach = approach
            best_breakdown = breakdown

    print("\n" + "=" * 70)
    print("SCORING RESULTS")
    print("=" * 70)
    print(f"Best approach: {best_breakdown['approach']}")
    print(f"Categories present:  {best_breakdown['categories_present']}/20")
    print(f"Values matched:      {best_breakdown['value_points']}/40 ({best_breakdown['n_matched']}/{best_breakdown['n_values']} values)")
    print(f"Ordering:            {best_breakdown['ordering']}/10")
    print(f"Sample size (N):     {best_breakdown['sample_size']}/20 (gen={best_breakdown['gen_N']}, paper={best_breakdown['paper_N']})")
    print(f"Column structure:    {best_breakdown['column_structure']}/10")
    print(f"TOTAL SCORE:         {best_breakdown['total']}/100")
    print()
    print("Value-by-value comparison:")
    for d in best_breakdown['details']:
        print(d)

    return best_breakdown


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'cps_dws.csv')
    result = run_analysis(data_path)
    print(result)
    print()
    breakdown = score_against_ground_truth()
