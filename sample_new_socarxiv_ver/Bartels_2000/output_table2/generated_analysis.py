import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
import warnings
warnings.filterwarnings('ignore')


def run_analysis(data_source):
    df = pd.read_csv(data_source, low_memory=False)

    # 22 election years for House vote (1954 excluded - no VCF0707 data)
    house_years = [1952, 1956, 1958, 1960, 1962, 1964, 1966, 1968, 1970, 1972,
                   1974, 1976, 1978, 1980, 1982, 1984, 1986, 1988, 1990, 1992,
                   1994, 1996]
    df = df[df['VCF0004'].isin(house_years)].copy()

    # Dependent variable: 2-party House vote
    # VCF0707: 1=Democrat, 2=Republican
    # Convert to numeric, coerce errors to NaN
    df['VCF0707'] = pd.to_numeric(df['VCF0707'], errors='coerce')
    df = df[df['VCF0707'].isin([1, 2])].copy()
    df['vote_rep'] = (df['VCF0707'] == 2).astype(int)

    # Independent variables from VCF0301 (7-point party ID)
    df['VCF0301'] = pd.to_numeric(df['VCF0301'], errors='coerce')
    df = df[df['VCF0301'].isin([1, 2, 3, 4, 5, 6, 7])].copy()

    # Construct symmetric party attachment variables
    # Strong: +1 for Strong Rep (7), -1 for Strong Dem (1), 0 otherwise
    df['strong'] = 0
    df.loc[df['VCF0301'] == 7, 'strong'] = 1
    df.loc[df['VCF0301'] == 1, 'strong'] = -1

    # Weak: +1 for Weak Rep (6), -1 for Weak Dem (2), 0 otherwise
    df['weak'] = 0
    df.loc[df['VCF0301'] == 6, 'weak'] = 1
    df.loc[df['VCF0301'] == 2, 'weak'] = -1

    # Leaning: +1 for Ind-Rep (5), -1 for Ind-Dem (3), 0 otherwise
    df['leaning'] = 0
    df.loc[df['VCF0301'] == 5, 'leaning'] = 1
    df.loc[df['VCF0301'] == 3, 'leaning'] = -1

    results_text = "Table 2: Party Identification and Congressional Votes, 1952-1996\n"
    results_text += "=" * 110 + "\n\n"
    results_text += f"{'Year':<8} {'N':<7} {'Strong':<16} {'Weak':<16} {'Leaners':<16} {'Intercept':<16} {'LogLik':<12} {'Pseudo-R2':<10}\n"
    results_text += "-" * 110 + "\n"

    results_dict = {}

    for year in house_years:
        year_df = df[df['VCF0004'] == year].copy()
        n = len(year_df)

        if n == 0:
            results_text += f"{year:<8} -- NO DATA --\n"
            continue

        y = year_df['vote_rep']
        X = year_df[['strong', 'weak', 'leaning']]
        X = sm.add_constant(X)

        try:
            model = Probit(y, X)
            result = model.fit(disp=0, method='newton', maxiter=100)

            coef_strong = result.params['strong']
            coef_weak = result.params['weak']
            coef_leaning = result.params['leaning']
            intercept = result.params['const']

            se_strong = result.bse['strong']
            se_weak = result.bse['weak']
            se_leaning = result.bse['leaning']
            se_intercept = result.bse['const']

            ll = result.llf
            pseudo_r2 = result.prsquared

            results_dict[year] = {
                'N': n,
                'strong': coef_strong, 'se_strong': se_strong,
                'weak': coef_weak, 'se_weak': se_weak,
                'leaning': coef_leaning, 'se_leaning': se_leaning,
                'intercept': intercept, 'se_intercept': se_intercept,
                'loglik': ll, 'pseudo_r2': pseudo_r2
            }

            results_text += (
                f"{year:<8} {n:<7} "
                f"{coef_strong:>6.3f} ({se_strong:.3f})  "
                f"{coef_weak:>6.3f} ({se_weak:.3f})  "
                f"{coef_leaning:>6.3f} ({se_leaning:.3f})  "
                f"{intercept:>6.3f} ({se_intercept:.3f})  "
                f"{ll:>9.1f}  {pseudo_r2:.2f}\n"
            )

        except Exception as e:
            results_text += f"{year:<8} {n:<7} ERROR: {e}\n"

    results_text += "\nNote: Probit coefficients with standard errors in parentheses; major-party House voters only.\n"

    # Score against ground truth
    score = score_against_ground_truth(results_dict)
    results_text += f"\n{'=' * 110}\nAutomated Score: {score}/100\n"

    # Print detailed scoring breakdown
    results_text += "\n" + detailed_scoring(results_dict) + "\n"

    return results_text


def score_against_ground_truth(results):
    """Score results against paper's ground truth values for Table 2."""
    ground_truth = get_ground_truth()

    total_points = 0

    # Coefficient signs and magnitudes (30 points)
    coef_score = 0
    coef_max = 0
    for year in ground_truth:
        if year not in results:
            continue
        for var in ['strong', 'weak', 'leaning', 'intercept']:
            coef_max += 1
            gt_val = ground_truth[year][var]
            gen_val = results[year][var]
            if abs(gt_val - gen_val) < 0.05:
                coef_score += 1
            elif abs(gt_val - gen_val) < 0.10:
                coef_score += 0.5
    if coef_max > 0:
        total_points += 30 * (coef_score / coef_max)

    # Standard errors (20 points)
    se_score = 0
    se_max = 0
    for year in ground_truth:
        if year not in results:
            continue
        for var in ['se_strong', 'se_weak', 'se_leaning', 'se_intercept']:
            se_max += 1
            gt_val = ground_truth[year][var]
            gen_val = results[year][var]
            if abs(gt_val - gen_val) < 0.02:
                se_score += 1
            elif abs(gt_val - gen_val) < 0.05:
                se_score += 0.5
    if se_max > 0:
        total_points += 20 * (se_score / se_max)

    # Sample size N (15 points)
    n_score = 0
    n_max = 0
    for year in ground_truth:
        if year not in results:
            continue
        n_max += 1
        gt_n = ground_truth[year]['N']
        gen_n = results[year]['N']
        if abs(gt_n - gen_n) / gt_n < 0.05:
            n_score += 1
        elif abs(gt_n - gen_n) / gt_n < 0.10:
            n_score += 0.5
    if n_max > 0:
        total_points += 15 * (n_score / n_max)

    # All variables present (10 points)
    vars_present = all(year in results for year in ground_truth)
    if vars_present:
        total_points += 10

    # Log likelihood (10 points)
    ll_score = 0
    ll_max = 0
    for year in ground_truth:
        if year not in results:
            continue
        ll_max += 1
        gt_ll = ground_truth[year]['loglik']
        gen_ll = results[year]['loglik']
        if abs(gt_ll - gen_ll) < 1.0:
            ll_score += 1
        elif abs(gt_ll - gen_ll) < 5.0:
            ll_score += 0.5
    if ll_max > 0:
        total_points += 10 * (ll_score / ll_max)

    # Pseudo-R2 (15 points)
    r2_score = 0
    r2_max = 0
    for year in ground_truth:
        if year not in results:
            continue
        r2_max += 1
        gt_r2 = ground_truth[year]['pseudo_r2']
        gen_r2 = results[year]['pseudo_r2']
        if abs(gt_r2 - gen_r2) < 0.02:
            r2_score += 1
        elif abs(gt_r2 - gen_r2) < 0.05:
            r2_score += 0.5
    if r2_max > 0:
        total_points += 15 * (r2_score / r2_max)

    return round(total_points, 1)


def detailed_scoring(results):
    """Print detailed scoring breakdown per year."""
    ground_truth = get_ground_truth()
    text = "DETAILED SCORING BREAKDOWN\n" + "=" * 80 + "\n\n"

    for year in sorted(ground_truth.keys()):
        if year not in results:
            text += f"{year}: MISSING\n"
            continue

        gt = ground_truth[year]
        gen = results[year]

        text += f"\n{year}: N={gen['N']} (paper: {gt['N']}, diff={gen['N']-gt['N']})\n"
        for var, se_var in [('strong', 'se_strong'), ('weak', 'se_weak'),
                            ('leaning', 'se_leaning'), ('intercept', 'se_intercept')]:
            coef_diff = gen[var] - gt[var]
            se_diff = gen[se_var] - gt[se_var]
            coef_ok = "OK" if abs(coef_diff) < 0.05 else ("PARTIAL" if abs(coef_diff) < 0.10 else "MISS")
            se_ok = "OK" if abs(se_diff) < 0.02 else ("PARTIAL" if abs(se_diff) < 0.05 else "MISS")
            text += (f"  {var:<10}: coef={gen[var]:>7.3f} (paper={gt[var]:>7.3f}, diff={coef_diff:>+7.4f}) [{coef_ok}] "
                     f"  SE={gen[se_var]:>6.3f} (paper={gt[se_var]:>6.3f}, diff={se_diff:>+7.4f}) [{se_ok}]\n")

        ll_diff = gen['loglik'] - gt['loglik']
        ll_ok = "OK" if abs(ll_diff) < 1.0 else ("PARTIAL" if abs(ll_diff) < 5.0 else "MISS")
        text += f"  LogLik:     {gen['loglik']:>9.1f} (paper={gt['loglik']:>9.1f}, diff={ll_diff:>+7.1f}) [{ll_ok}]\n"

        r2_diff = gen['pseudo_r2'] - gt['pseudo_r2']
        r2_ok = "OK" if abs(r2_diff) < 0.02 else ("PARTIAL" if abs(r2_diff) < 0.05 else "MISS")
        text += f"  Pseudo-R2:  {gen['pseudo_r2']:>7.4f} (paper={gt['pseudo_r2']:>5.2f}, diff={r2_diff:>+7.4f}) [{r2_ok}]\n"

    return text


def get_ground_truth():
    """Ground truth values from Table 2 of Bartels (2000), page 43."""
    return {
        1952: {'N': 975, 'strong': 1.495, 'se_strong': 0.098, 'weak': 1.011, 'se_weak': 0.081, 'leaning': 0.619, 'se_leaning': 0.102, 'intercept': 0.258, 'se_intercept': 0.053, 'loglik': -402.8, 'pseudo_r2': 0.40},
        1956: {'N': 1157, 'strong': 1.621, 'se_strong': 0.096, 'weak': 1.148, 'se_weak': 0.079, 'leaning': 0.959, 'se_leaning': 0.115, 'intercept': 0.069, 'se_intercept': 0.050, 'loglik': -406.0, 'pseudo_r2': 0.49},
        1958: {'N': 817, 'strong': 1.654, 'se_strong': 0.111, 'weak': 0.991, 'se_weak': 0.087, 'leaning': 0.653, 'se_leaning': 0.145, 'intercept': -0.122, 'se_intercept': 0.059, 'loglik': -282.4, 'pseudo_r2': 0.48},
        1960: {'N': 759, 'strong': 1.426, 'se_strong': 0.107, 'weak': 1.059, 'se_weak': 0.092, 'leaning': 0.857, 'se_leaning': 0.144, 'intercept': -0.065, 'se_intercept': 0.058, 'loglik': -295.4, 'pseudo_r2': 0.43},
        1962: {'N': 698, 'strong': 1.695, 'se_strong': 0.129, 'weak': 0.999, 'se_weak': 0.092, 'leaning': 0.646, 'se_leaning': 0.147, 'intercept': -0.080, 'se_intercept': 0.063, 'loglik': -249.8, 'pseudo_r2': 0.47},
        1964: {'N': 957, 'strong': 1.423, 'se_strong': 0.096, 'weak': 0.680, 'se_weak': 0.073, 'leaning': 0.689, 'se_leaning': 0.118, 'intercept': -0.230, 'se_intercept': 0.050, 'loglik': -402.4, 'pseudo_r2': 0.35},
        1966: {'N': 677, 'strong': 1.294, 'se_strong': 0.112, 'weak': 0.840, 'se_weak': 0.086, 'leaning': 0.362, 'se_leaning': 0.130, 'intercept': -0.066, 'se_intercept': 0.057, 'loglik': -319.7, 'pseudo_r2': 0.31},
        1968: {'N': 871, 'strong': 1.293, 'se_strong': 0.099, 'weak': 0.705, 'se_weak': 0.075, 'leaning': 0.604, 'se_leaning': 0.104, 'intercept': 0.131, 'se_intercept': 0.050, 'loglik': -431.2, 'pseudo_r2': 0.28},
        1970: {'N': 683, 'strong': 1.384, 'se_strong': 0.116, 'weak': 0.830, 'se_weak': 0.087, 'leaning': 0.553, 'se_leaning': 0.126, 'intercept': 0.048, 'se_intercept': 0.058, 'loglik': -315.3, 'pseudo_r2': 0.33},
        1972: {'N': 1337, 'strong': 1.225, 'se_strong': 0.084, 'weak': 0.772, 'se_weak': 0.061, 'leaning': 0.716, 'se_leaning': 0.082, 'intercept': -0.124, 'se_intercept': 0.040, 'loglik': -652.0, 'pseudo_r2': 0.29},
        1974: {'N': 798, 'strong': 1.148, 'se_strong': 0.099, 'weak': 0.693, 'se_weak': 0.082, 'leaning': 0.704, 'se_leaning': 0.107, 'intercept': -0.222, 'se_intercept': 0.052, 'loglik': -385.9, 'pseudo_r2': 0.27},
        1976: {'N': 1079, 'strong': 1.150, 'se_strong': 0.088, 'weak': 0.677, 'se_weak': 0.068, 'leaning': 0.616, 'se_leaning': 0.090, 'intercept': -0.120, 'se_intercept': 0.043, 'loglik': -553.4, 'pseudo_r2': 0.25},
        1978: {'N': 1009, 'strong': 0.974, 'se_strong': 0.086, 'weak': 0.641, 'se_weak': 0.072, 'leaning': 0.312, 'se_leaning': 0.083, 'intercept': -0.123, 'se_intercept': 0.043, 'loglik': -562.8, 'pseudo_r2': 0.18},
        1980: {'N': 859, 'strong': 0.924, 'se_strong': 0.088, 'weak': 0.561, 'se_weak': 0.076, 'leaning': 0.495, 'se_leaning': 0.094, 'intercept': -0.037, 'se_intercept': 0.047, 'loglik': -486.9, 'pseudo_r2': 0.18},
        1982: {'N': 712, 'strong': 1.265, 'se_strong': 0.104, 'weak': 0.726, 'se_weak': 0.085, 'leaning': 0.636, 'se_leaning': 0.122, 'intercept': -0.008, 'se_intercept': 0.056, 'loglik': -339.7, 'pseudo_r2': 0.30},
        1984: {'N': 1185, 'strong': 1.119, 'se_strong': 0.090, 'weak': 0.462, 'se_weak': 0.067, 'leaning': 0.496, 'se_leaning': 0.080, 'intercept': -0.128, 'se_intercept': 0.041, 'loglik': -642.7, 'pseudo_r2': 0.21},
        1986: {'N': 981, 'strong': 1.111, 'se_strong': 0.085, 'weak': 0.521, 'se_weak': 0.072, 'leaning': 0.490, 'se_leaning': 0.090, 'intercept': -0.196, 'se_intercept': 0.045, 'loglik': -512.4, 'pseudo_r2': 0.22},
        1988: {'N': 1054, 'strong': 0.979, 'se_strong': 0.075, 'weak': 0.714, 'se_weak': 0.077, 'leaning': 0.717, 'se_leaning': 0.089, 'intercept': -0.272, 'se_intercept': 0.044, 'loglik': -534.1, 'pseudo_r2': 0.25},
        1990: {'N': 801, 'strong': 1.179, 'se_strong': 0.094, 'weak': 0.567, 'se_weak': 0.084, 'leaning': 0.673, 'se_leaning': 0.107, 'intercept': -0.286, 'se_intercept': 0.052, 'loglik': -378.6, 'pseudo_r2': 0.28},
        1992: {'N': 1370, 'strong': 1.043, 'se_strong': 0.072, 'weak': 0.650, 'se_weak': 0.065, 'leaning': 0.547, 'se_leaning': 0.071, 'intercept': -0.211, 'se_intercept': 0.038, 'loglik': -716.4, 'pseudo_r2': 0.23},
        1994: {'N': 942, 'strong': 1.353, 'se_strong': 0.091, 'weak': 0.702, 'se_weak': 0.082, 'leaning': 0.561, 'se_leaning': 0.090, 'intercept': 0.088, 'se_intercept': 0.048, 'loglik': -440.4, 'pseudo_r2': 0.32},
        1996: {'N': 1031, 'strong': 1.427, 'se_strong': 0.092, 'weak': 0.749, 'se_weak': 0.075, 'leaning': 0.664, 'se_leaning': 0.091, 'intercept': 0.210, 'se_intercept': 0.048, 'loglik': -464.2, 'pseudo_r2': 0.35},
    }


if __name__ == "__main__":
    result = run_analysis("anes_cumulative.csv")
    print(result)
