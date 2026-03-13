"""
Table 1 Attempt 4: Try to recover missing party IDs by using VCF0303 (3-category)
as a fallback for respondents where VCF0301 is missing but VCF0303 is valid.
If VCF0303 indicates Democrat or Republican, we know they have SOME party attachment
but not the specific level. We can't use them properly, but let's see if VCF0303
adds any valid observations.
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
import warnings
warnings.filterwarnings('ignore')


def run_analysis(data_source):
    df = pd.read_csv(data_source, low_memory=False)

    pres_years = [1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996]
    df = df[df['VCF0004'].isin(pres_years)].copy()

    # Filter to 2-party presidential voters
    df = df[df['VCF0704a'].isin([1, 2])].copy()
    df['vote_rep'] = (df['VCF0704a'] == 2).astype(int)

    # Check if VCF0303 provides any additional valid observations
    # VCF0303: 1=Democrats (incl leaners), 2=Independents, 3=Republicans (incl leaners)
    has_pid7 = df['VCF0301'].isin([1, 2, 3, 4, 5, 6, 7])
    has_pid3 = df['VCF0303'].isin([1, 2, 3])
    extra = has_pid3 & ~has_pid7

    print(f"Respondents with VCF0303 but not VCF0301: {extra.sum()}")
    for yr in pres_years:
        yr_mask = df['VCF0004'] == yr
        print(f"  {yr}: {(extra & yr_mask).sum()} extra obs (VCF0303 only)")

    # Since VCF0303 doesn't give us the 7-point scale, we can't use these
    # observations for the probit with strong/weak/leaning variables.
    # Fall back to standard approach.

    df = df[has_pid7].copy()

    df['strong'] = 0
    df.loc[df['VCF0301'] == 7, 'strong'] = 1
    df.loc[df['VCF0301'] == 1, 'strong'] = -1

    df['weak'] = 0
    df.loc[df['VCF0301'] == 6, 'weak'] = 1
    df.loc[df['VCF0301'] == 2, 'weak'] = -1

    df['leaning'] = 0
    df.loc[df['VCF0301'] == 5, 'leaning'] = 1
    df.loc[df['VCF0301'] == 3, 'leaning'] = -1

    results_text = "Table 1: Party Identification and Presidential Votes, 1952-1996\n"
    results_text += "(Attempt 4: Checked VCF0303 as fallback - no additional valid obs)\n"
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

        model = Probit(y, X)
        result = model.fit(disp=0, method='newton', maxiter=100)

        results_dict[year] = {
            'N': n,
            'strong': result.params['strong'], 'se_strong': result.bse['strong'],
            'weak': result.params['weak'], 'se_weak': result.bse['weak'],
            'leaning': result.params['leaning'], 'se_leaning': result.bse['leaning'],
            'intercept': result.params['const'], 'se_intercept': result.bse['const'],
            'loglik': result.llf, 'pseudo_r2': result.prsquared
        }

        r = results_dict[year]
        results_text += (
            f"{year:<8} {n:<7} "
            f"{r['strong']:>6.3f} ({r['se_strong']:.3f})  "
            f"{r['weak']:>6.3f} ({r['se_weak']:.3f})  "
            f"{r['leaning']:>6.3f} ({r['se_leaning']:.3f})  "
            f"{r['intercept']:>6.3f} ({r['se_intercept']:.3f})  "
            f"{r['loglik']:>9.1f}  {r['pseudo_r2']:.2f}\n"
        )

    score = score_against_ground_truth(results_dict)
    results_text += f"\n{'=' * 80}\nAutomated Score: {score}/100\n"
    return results_text


def score_against_ground_truth(results):
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
    coef_score = 0
    coef_max = 0
    for year in ground_truth:
        if year not in results: continue
        for var in ['strong', 'weak', 'leaning', 'intercept']:
            coef_max += 1
            diff = abs(ground_truth[year][var] - results[year][var])
            if diff < 0.05: coef_score += 1
            elif diff < 0.10: coef_score += 0.5
    if coef_max > 0: total_points += 30 * (coef_score / coef_max)

    se_score = 0
    se_max = 0
    for year in ground_truth:
        if year not in results: continue
        for var in ['se_strong', 'se_weak', 'se_leaning', 'se_intercept']:
            se_max += 1
            diff = abs(ground_truth[year][var] - results[year][var])
            if diff < 0.02: se_score += 1
            elif diff < 0.05: se_score += 0.5
    if se_max > 0: total_points += 20 * (se_score / se_max)

    n_score = 0
    n_max = 0
    for year in ground_truth:
        if year not in results: continue
        n_max += 1
        diff = abs(ground_truth[year]['N'] - results[year]['N']) / ground_truth[year]['N']
        if diff < 0.05: n_score += 1
        elif diff < 0.10: n_score += 0.5
    if n_max > 0: total_points += 15 * (n_score / n_max)

    if all(year in results for year in ground_truth): total_points += 10

    ll_score = 0
    ll_max = 0
    for year in ground_truth:
        if year not in results: continue
        ll_max += 1
        diff = abs(ground_truth[year]['loglik'] - results[year]['loglik'])
        if diff < 1.0: ll_score += 1
        elif diff < 5.0: ll_score += 0.5
    if ll_max > 0: total_points += 10 * (ll_score / ll_max)

    r2_score = 0
    r2_max = 0
    for year in ground_truth:
        if year not in results: continue
        r2_max += 1
        diff = abs(ground_truth[year]['pseudo_r2'] - results[year]['pseudo_r2'])
        if diff < 0.02: r2_score += 1
        elif diff < 0.05: r2_score += 0.5
    if r2_max > 0: total_points += 15 * (r2_score / r2_max)

    return round(total_points, 1)


if __name__ == "__main__":
    result = run_analysis("anes_cumulative.csv")
    print(result)
