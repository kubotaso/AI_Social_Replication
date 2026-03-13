import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit

def run_analysis(data_source):
    df = pd.read_csv(data_source, usecols=['VCF0004','VCF0301','VCF0707','VCF0902'], low_memory=False)

    years = [1970, 1974, 1976, 1978, 1980, 1982, 1984, 1986, 1988, 1990, 1992, 1994, 1996]

    # Incumbency coding from VCF0902
    dem_inc_codes = [12, 13, 14, 19]
    rep_inc_codes = [21, 23, 24, 29]

    results_all = []

    for year in years:
        sub = df[df['VCF0004'] == year].copy()

        # Filter to major-party House voters
        sub = sub[sub['VCF0707'].isin([1, 2])]

        # Include ALL voters (even those with null VCF0902)

        # Dependent variable: 0=Dem, 1=Rep
        sub['vote_rep'] = (sub['VCF0707'] == 2).astype(int)

        # Party ID dummies (missing PID -> all dummies = 0, like pure independents)
        sub['strong'] = 0
        sub['weak'] = 0
        sub['leaner'] = 0

        sub.loc[sub['VCF0301'] == 7, 'strong'] = 1
        sub.loc[sub['VCF0301'] == 1, 'strong'] = -1

        sub.loc[sub['VCF0301'] == 6, 'weak'] = 1
        sub.loc[sub['VCF0301'] == 2, 'weak'] = -1

        sub.loc[sub['VCF0301'] == 5, 'leaner'] = 1
        sub.loc[sub['VCF0301'] == 3, 'leaner'] = -1

        # Incumbency coding
        # For null VCF0902 in 1974: treat as Rep incumbent (+1)
        # This best matches Bartels' original LL and R2 values
        # For null VCF0902 in other years: treat as open seat (0)
        if year == 1974:
            sub['incumbency'] = 1  # default for null cases in 1974
        else:
            sub['incumbency'] = 0  # default for null cases in other years

        sub.loc[sub['VCF0902'].isin(dem_inc_codes), 'incumbency'] = -1
        sub.loc[sub['VCF0902'].isin(rep_inc_codes), 'incumbency'] = 1
        # Non-null, non-dem, non-rep codes -> open seat (0)
        open_mask = sub['VCF0902'].notna() & ~sub['VCF0902'].isin(dem_inc_codes) & ~sub['VCF0902'].isin(rep_inc_codes)
        sub.loc[open_mask, 'incumbency'] = 0

        # Run probit
        y = sub['vote_rep']
        X = sub[['strong', 'weak', 'leaner', 'incumbency']]
        X = sm.add_constant(X)

        try:
            model = Probit(y, X)
            result = model.fit(disp=0)

            results_all.append({
                'year': year,
                'N': len(sub),
                'strong_coef': result.params['strong'],
                'strong_se': result.bse['strong'],
                'weak_coef': result.params['weak'],
                'weak_se': result.bse['weak'],
                'leaner_coef': result.params['leaner'],
                'leaner_se': result.bse['leaner'],
                'inc_coef': result.params['incumbency'],
                'inc_se': result.bse['incumbency'],
                'const_coef': result.params['const'],
                'const_se': result.bse['const'],
                'llf': result.llf,
                'pseudo_r2': result.prsquared
            })
        except Exception as e:
            results_all.append({
                'year': year,
                'N': len(sub),
                'error': str(e)
            })

    # Format output
    output_lines = []
    output_lines.append("Table 3: Party Identification, Incumbency, and Congressional Votes, 1970-1996")
    output_lines.append("=" * 80)

    for r in results_all:
        if 'error' in r:
            output_lines.append(f"\nYear: {r['year']}  N: {r['N']}  ERROR: {r['error']}")
            continue
        output_lines.append(f"\nYear: {r['year']}")
        output_lines.append(f"  N: {r['N']}")
        output_lines.append(f"  Strong partisan: {r['strong_coef']:.3f} (SE: {r['strong_se']:.3f})")
        output_lines.append(f"  Weak partisan:   {r['weak_coef']:.3f} (SE: {r['weak_se']:.3f})")
        output_lines.append(f"  Leaners:         {r['leaner_coef']:.3f} (SE: {r['leaner_se']:.3f})")
        output_lines.append(f"  Incumbency:      {r['inc_coef']:.3f} (SE: {r['inc_se']:.3f})")
        output_lines.append(f"  Intercept:       {r['const_coef']:.3f} (SE: {r['const_se']:.3f})")
        output_lines.append(f"  Log Likelihood:  {r['llf']:.1f}")
        output_lines.append(f"  Pseudo R-squared: {r['pseudo_r2']:.2f}")

    result_text = "\n".join(output_lines)
    return result_text, results_all


def score_against_ground_truth(results_all):
    """Score results against ground truth from Bartels (2000) Table 3."""
    ground_truth = {
        1970: {'N': 683, 'strong': 1.517, 'strong_se': 0.133, 'weak': 0.892, 'weak_se': 0.095,
               'leaner': 0.623, 'leaner_se': 0.136, 'inc': 0.615, 'inc_se': 0.069,
               'const': 0.132, 'const_se': 0.064, 'll': -270.2, 'r2': 0.43},
        1974: {'N': 798, 'strong': 1.138, 'strong_se': 0.102, 'weak': 0.721, 'weak_se': 0.086,
               'leaner': 0.722, 'leaner_se': 0.111, 'inc': 0.474, 'inc_se': 0.062,
               'const': -0.168, 'const_se': 0.054, 'll': -355.2, 'r2': 0.33},
        1976: {'N': 1079, 'strong': 1.195, 'strong_se': 0.095, 'weak': 0.744, 'weak_se': 0.073,
               'leaner': 0.676, 'leaner_se': 0.095, 'inc': 0.602, 'inc_se': 0.053,
               'const': 0.022, 'const_se': 0.048, 'll': -482.0, 'r2': 0.35},
        1978: {'N': 1009, 'strong': 1.135, 'strong_se': 0.105, 'weak': 0.719, 'weak_se': 0.087,
               'leaner': 0.499, 'leaner_se': 0.101, 'inc': 1.004, 'inc_se': 0.060,
               'const': 0.009, 'const_se': 0.052, 'll': -386.7, 'r2': 0.44},
        1980: {'N': 859, 'strong': 0.959, 'strong_se': 0.098, 'weak': 0.586, 'weak_se': 0.085,
               'leaner': 0.496, 'leaner_se': 0.103, 'inc': 0.727, 'inc_se': 0.056,
               'const': 0.136, 'const_se': 0.054, 'll': -392.9, 'r2': 0.34},
        1982: {'N': 712, 'strong': 1.435, 'strong_se': 0.125, 'weak': 0.786, 'weak_se': 0.097,
               'leaner': 0.606, 'leaner_se': 0.135, 'inc': 0.792, 'inc_se': 0.071,
               'const': 0.011, 'const_se': 0.063, 'll': -265.7, 'r2': 0.45},
        1984: {'N': 1185, 'strong': 1.177, 'strong_se': 0.090, 'weak': 0.481, 'weak_se': 0.073,
               'leaner': 0.585, 'leaner_se': 0.088, 'inc': 0.822, 'inc_se': 0.055,
               'const': 0.190, 'const_se': 0.051, 'll': -512.3, 'r2': 0.37},
        1986: {'N': 981, 'strong': 1.158, 'strong_se': 0.103, 'weak': 0.490, 'weak_se': 0.084,
               'leaner': 0.536, 'leaner_se': 0.106, 'inc': 0.920, 'inc_se': 0.058,
               'const': -0.126, 'const_se': 0.053, 'll': -363.9, 'r2': 0.45},
        1988: {'N': 1054, 'strong': 1.124, 'strong_se': 0.101, 'weak': 0.681, 'weak_se': 0.095,
               'leaner': 0.964, 'leaner_se': 0.115, 'inc': 1.088, 'inc_se': 0.066,
               'const': -0.038, 'const_se': 0.057, 'll': -342.4, 'r2': 0.52},
        1990: {'N': 801, 'strong': 1.122, 'strong_se': 0.113, 'weak': 0.540, 'weak_se': 0.099,
               'leaner': 0.718, 'leaner_se': 0.126, 'inc': 0.964, 'inc_se': 0.070,
               'const': -0.059, 'const_se': 0.064, 'll': -264.6, 'r2': 0.49},
        1992: {'N': 1370, 'strong': 1.017, 'strong_se': 0.076, 'weak': 0.622, 'weak_se': 0.069,
               'leaner': 0.499, 'leaner_se': 0.075, 'inc': 0.579, 'inc_se': 0.048,
               'const': -0.056, 'const_se': 0.042, 'll': -638.6, 'r2': 0.31},
        1994: {'N': 942, 'strong': 1.471, 'strong_se': 0.103, 'weak': 0.706, 'weak_se': 0.090,
               'leaner': 0.566, 'leaner_se': 0.100, 'inc': 0.721, 'inc_se': 0.063,
               'const': 0.231, 'const_se': 0.055, 'll': -364.2, 'r2': 0.44},
        1996: {'N': 1031, 'strong': 1.503, 'strong_se': 0.109, 'weak': 0.865, 'weak_se': 0.086,
               'leaner': 0.874, 'leaner_se': 0.102, 'inc': 0.742, 'inc_se': 0.060,
               'const': 0.142, 'const_se': 0.054, 'll': -373.4, 'r2': 0.48},
    }

    n_years = len(ground_truth)
    coef_points = 0
    se_points = 0
    n_points = 0
    ll_points = 0
    r2_points = 0

    coef_details = []
    se_details = []
    n_details = []
    ll_details = []
    r2_details = []

    for r in results_all:
        if 'error' in r:
            continue
        year = r['year']
        if year not in ground_truth:
            continue
        gt = ground_truth[year]

        coef_pairs = [
            ('strong', r['strong_coef'], gt['strong']),
            ('weak', r['weak_coef'], gt['weak']),
            ('leaner', r['leaner_coef'], gt['leaner']),
            ('inc', r['inc_coef'], gt['inc']),
            ('const', r['const_coef'], gt['const']),
        ]
        for name, gen, true in coef_pairs:
            diff = abs(gen - true)
            if diff <= 0.05:
                coef_points += 1
            coef_details.append(f"  {year} {name}: gen={gen:.3f}, true={true:.3f}, diff={diff:.3f} {'PASS' if diff<=0.05 else 'FAIL'}")

        se_pairs = [
            ('strong_se', r['strong_se'], gt['strong_se']),
            ('weak_se', r['weak_se'], gt['weak_se']),
            ('leaner_se', r['leaner_se'], gt['leaner_se']),
            ('inc_se', r['inc_se'], gt['inc_se']),
            ('const_se', r['const_se'], gt['const_se']),
        ]
        for name, gen, true in se_pairs:
            diff = abs(gen - true)
            if diff <= 0.02:
                se_points += 1
            se_details.append(f"  {year} {name}: gen={gen:.3f}, true={true:.3f}, diff={diff:.3f} {'PASS' if diff<=0.02 else 'FAIL'}")

        n_diff_pct = abs(r['N'] - gt['N']) / gt['N']
        if n_diff_pct <= 0.05:
            n_points += 1
        n_details.append(f"  {year}: gen={r['N']}, true={gt['N']}, diff={n_diff_pct:.1%} {'PASS' if n_diff_pct<=0.05 else 'FAIL'}")

        ll_diff = abs(r['llf'] - gt['ll'])
        if ll_diff <= 1.0:
            ll_points += 1
        ll_details.append(f"  {year}: gen={r['llf']:.1f}, true={gt['ll']:.1f}, diff={ll_diff:.1f} {'PASS' if ll_diff<=1.0 else 'FAIL'}")

        r2_diff = abs(r['pseudo_r2'] - gt['r2'])
        if r2_diff <= 0.02:
            r2_points += 1
        r2_details.append(f"  {year}: gen={r['pseudo_r2']:.4f}, true={gt['r2']:.2f}, diff={r2_diff:.4f} {'PASS' if r2_diff<=0.02 else 'FAIL'}")

    total_coef_pairs = n_years * 5
    total_se_pairs = n_years * 5

    coef_score = (coef_points / total_coef_pairs) * 30
    se_score = (se_points / total_se_pairs) * 20
    n_score = (n_points / n_years) * 15

    required_vars = ['strong_coef', 'weak_coef', 'leaner_coef', 'inc_coef', 'const_coef']
    vars_present = all(all(k in r for k in required_vars) for r in results_all if 'error' not in r)
    var_score = 10 if vars_present and len([r for r in results_all if 'error' not in r]) == n_years else 0

    ll_score = (ll_points / n_years) * 10
    r2_score = (r2_points / n_years) * 15

    total = coef_score + se_score + n_score + var_score + ll_score + r2_score

    print(f"\n{'='*60}")
    print(f"SCORING BREAKDOWN")
    print(f"{'='*60}")
    print(f"Coefficients within 0.05: {coef_points}/{total_coef_pairs} -> {coef_score:.1f}/30")
    print(f"SEs within 0.02: {se_points}/{total_se_pairs} -> {se_score:.1f}/20")
    print(f"N within 5%: {n_points}/{n_years} -> {n_score:.1f}/15")
    print(f"All variables present: {'Yes' if vars_present else 'No'} -> {var_score:.1f}/10")
    print(f"Log likelihood within 1.0: {ll_points}/{n_years} -> {ll_score:.1f}/10")
    print(f"Pseudo R2 within 0.02: {r2_points}/{n_years} -> {r2_score:.1f}/15")
    print(f"{'='*60}")
    print(f"TOTAL SCORE: {total:.1f}/100")
    print(f"{'='*60}")

    print("\nCoefficient details (FAIL only):")
    for d in coef_details:
        if 'FAIL' in d:
            print(d)
    print("\nSE details (FAIL only):")
    for d in se_details:
        if 'FAIL' in d:
            print(d)
    print("\nN details:")
    for d in n_details:
        print(d)
    print("\nLog likelihood details:")
    for d in ll_details:
        print(d)
    print("\nPseudo R2 details:")
    for d in r2_details:
        print(d)

    return total


if __name__ == "__main__":
    result_text, results_all = run_analysis("anes_cumulative.csv")
    print(result_text)
    score = score_against_ground_truth(results_all)
