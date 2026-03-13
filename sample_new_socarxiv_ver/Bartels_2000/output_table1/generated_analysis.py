import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
import warnings
warnings.filterwarnings('ignore')


def run_analysis(data_source):
    df = pd.read_csv(data_source, low_memory=False)

    # Filter to presidential election years
    pres_years = [1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996]
    df = df[df['VCF0004'].isin(pres_years)].copy()

    # Dependent variable: 2-party presidential vote
    # VCF0704a: 1=Democrat, 2=Republican
    df = df[df['VCF0704a'].isin([1, 2])].copy()
    df['vote_rep'] = (df['VCF0704a'] == 2).astype(int)

    # Independent variables from VCF0301 (7-point party ID)
    # Exclude missing (NaN or 0)
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

    results_text = "Table 1: Party Identification and Presidential Votes, 1952-1996\n"
    results_text += "=" * 80 + "\n\n"
    results_text += f"{'Year':<8} {'N':<7} {'Strong':<16} {'Weak':<16} {'Leaners':<16} {'Intercept':<16} {'LogLik':<12} {'Pseudo-R2':<10}\n"
    results_text += "-" * 105 + "\n"

    results_dict = {}

    for year in pres_years:
        year_df = df[df['VCF0004'] == year].copy()
        n = len(year_df)

        if n == 0:
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

    results_text += "\nNote: Probit coefficients with standard errors in parentheses; major-party voters only.\n"

    # Score against ground truth
    score = score_against_ground_truth(results_dict)
    results_text += f"\n{'=' * 80}\nAutomated Score: {score}/100\n"

    return results_text


def score_against_ground_truth(results):
    """Score results against paper's ground truth values."""
    ground_truth = {
        1952: {'N': 1181, 'strong': 1.600, 'se_strong': 0.096, 'weak': 0.928, 'se_weak': 0.077, 'leaning': 0.902, 'se_leaning': 0.106, 'intercept': 0.633, 'se_intercept': 0.057, 'loglik': -490.1, 'pseudo_r2': 0.39},
        1956: {'N': 1266, 'strong': 1.713, 'se_strong': 0.097, 'weak': 0.941, 'se_weak': 0.075, 'leaning': 1.017, 'se_leaning': 0.118, 'intercept': 0.644, 'se_intercept': 0.055, 'loglik': -489.8, 'pseudo_r2': 0.43},
        1960: {'N': 885, 'strong': 1.650, 'se_strong': 0.114, 'weak': 0.822, 'se_weak': 0.079, 'leaning': 1.189, 'se_leaning': 0.153, 'intercept': 0.208, 'se_intercept': 0.057, 'loglik': -345.9, 'pseudo_r2': 0.44},
        1964: {'N': 1111, 'strong': 1.470, 'se_strong': 0.094, 'weak': 0.548, 'se_weak': 0.067, 'leaning': 0.981, 'se_leaning': 0.122, 'intercept': -0.339, 'se_intercept': 0.048, 'loglik': -448.9, 'pseudo_r2': 0.36},
        1968: {'N': 911, 'strong': 1.770, 'se_strong': 0.121, 'weak': 0.881, 'se_weak': 0.080, 'leaning': 0.935, 'se_leaning': 0.120, 'intercept': 0.442, 'se_intercept': 0.059, 'loglik': -363.9, 'pseudo_r2': 0.42},
        1972: {'N': 1587, 'strong': 1.221, 'se_strong': 0.079, 'weak': 0.603, 'se_weak': 0.058, 'leaning': 0.727, 'se_leaning': 0.078, 'intercept': 0.589, 'se_intercept': 0.040, 'loglik': -789.0, 'pseudo_r2': 0.24},
        1976: {'N': 1322, 'strong': 1.565, 'se_strong': 0.102, 'weak': 0.745, 'se_weak': 0.062, 'leaning': 0.877, 'se_leaning': 0.088, 'intercept': 0.122, 'se_intercept': 0.042, 'loglik': -604.3, 'pseudo_r2': 0.34},
        1980: {'N': 877, 'strong': 1.602, 'se_strong': 0.113, 'weak': 0.929, 'se_weak': 0.086, 'leaning': 0.699, 'se_leaning': 0.107, 'intercept': 0.487, 'se_intercept': 0.058, 'loglik': -376.3, 'pseudo_r2': 0.37},
        1984: {'N': 1376, 'strong': 1.596, 'se_strong': 0.092, 'weak': 0.975, 'se_weak': 0.072, 'leaning': 1.174, 'se_leaning': 0.096, 'intercept': 0.451, 'se_intercept': 0.048, 'loglik': -514.1, 'pseudo_r2': 0.45},
        1988: {'N': 1195, 'strong': 1.770, 'se_strong': 0.107, 'weak': 0.771, 'se_weak': 0.073, 'leaning': 1.095, 'se_leaning': 0.094, 'intercept': 0.162, 'se_intercept': 0.048, 'loglik': -440.3, 'pseudo_r2': 0.47},
        1992: {'N': 1357, 'strong': 1.851, 'se_strong': 0.109, 'weak': 0.912, 'se_weak': 0.072, 'leaning': 1.215, 'se_leaning': 0.092, 'intercept': -0.113, 'se_intercept': 0.047, 'loglik': -443.4, 'pseudo_r2': 0.52},
        1996: {'N': 1034, 'strong': 1.946, 'se_strong': 0.129, 'weak': 1.022, 'se_weak': 0.083, 'leaning': 0.942, 'se_leaning': 0.101, 'intercept': -0.233, 'se_intercept': 0.056, 'loglik': -327.3, 'pseudo_r2': 0.54},
    }

    total_points = 0
    max_points = 100

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


if __name__ == "__main__":
    result = run_analysis("anes_cumulative.csv")
    print(result)
