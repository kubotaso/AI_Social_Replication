import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit


def construct_pid_vars(df, pid_col, suffix):
    """Construct directional party ID variables from 7-point scale."""
    df[f'strong_{suffix}'] = np.where(df[pid_col] == 7, 1, np.where(df[pid_col] == 1, -1, 0))
    df[f'weak_{suffix}'] = np.where(df[pid_col] == 6, 1, np.where(df[pid_col] == 2, -1, 0))
    df[f'lean_{suffix}'] = np.where(df[pid_col] == 5, 1, np.where(df[pid_col] == 3, -1, 0))
    return df


def run_probit(df, dep_var, indep_vars):
    """Run probit regression and return results."""
    X = sm.add_constant(df[indep_vars])
    model = Probit(df[dep_var], X).fit(disp=0)
    return model


def run_iv_probit(df, dep_var, endog_vars, instrument_vars):
    """Run IV probit (2-stage: OLS first stage, Probit second stage).

    First stage: OLS of each endogenous variable on all instruments
    Second stage: Probit of dep_var on predicted values from first stage

    Returns the second-stage probit model and log-likelihood/pseudo-R2
    from the lagged probit (which should match per the paper).
    """
    # First stage: OLS of each current PID var on all lagged PID vars
    predicted = pd.DataFrame(index=df.index)
    for var in endog_vars:
        X_first = sm.add_constant(df[instrument_vars])
        ols_model = sm.OLS(df[var], X_first).fit()
        predicted[var] = ols_model.predict(X_first)

    # Second stage: Probit with predicted values
    X_second = sm.add_constant(predicted[endog_vars])
    iv_model = Probit(df[dep_var], X_second).fit(disp=0)

    return iv_model


def prepare_1960_data():
    """Prepare 1960 panel data."""
    df = pd.read_csv('panel_1960.csv')

    # Filter to valid House voters with valid PID in both years
    mask = (
        df['VCF0707'].isin([1.0, 2.0]) &
        df['VCF0301'].isin([1, 2, 3, 4, 5, 6, 7]) &
        df['VCF0301_lagged'].isin([1, 2, 3, 4, 5, 6, 7])
    )
    df = df[mask].copy()

    # Dependent: Republican House vote (0=Dem, 1=Rep)
    df['house_rep'] = (df['VCF0707'] == 2.0).astype(int)

    # Construct PID variables
    df = construct_pid_vars(df, 'VCF0301', 'curr')
    df = construct_pid_vars(df, 'VCF0301_lagged', 'lag')

    return df


def prepare_1976_data():
    """Prepare 1976 panel data."""
    df = pd.read_csv('panel_1976.csv')

    mask = (
        df['VCF0707'].isin([1.0, 2.0]) &
        df['VCF0301'].isin([1, 2, 3, 4, 5, 6, 7]) &
        df['VCF0301_lagged'].isin([1, 2, 3, 4, 5, 6, 7])
    )
    df = df[mask].copy()

    df['house_rep'] = (df['VCF0707'] == 2.0).astype(int)
    df = construct_pid_vars(df, 'VCF0301', 'curr')
    df = construct_pid_vars(df, 'VCF0301_lagged', 'lag')

    return df


def prepare_1992_data():
    """Prepare 1992 panel data using CDF for proper House vote coding.

    The panel_1992.csv has V925701 which codes ballot position, not party.
    Instead, use the CDF which has VCF0707 properly coded (1=Dem, 2=Rep).
    Match 1992 panel respondents (VCF0006a in 1990xxxx range) to their
    1990 CDF entries for lagged PID.
    """
    cdf = pd.read_csv('anes_cumulative.csv', low_memory=False)

    # Get 1992 panel respondents from CDF (those with 1990-era IDs)
    cdf92 = cdf[cdf['VCF0004'] == 1992].copy()
    cdf92_panel = cdf92[cdf92['VCF0006a'] < 19920000].copy()

    # Get 1990 wave for lagged PID
    cdf90 = cdf[cdf['VCF0004'] == 1990].copy()

    # Merge to get lagged PID
    merged = cdf92_panel.merge(
        cdf90[['VCF0006a', 'VCF0301']],
        on='VCF0006a',
        suffixes=('', '_lag')
    )

    # Filter to valid House voters with valid PID in both years
    mask = (
        merged['VCF0707'].isin([1.0, 2.0]) &
        merged['VCF0301'].isin([1, 2, 3, 4, 5, 6, 7]) &
        merged['VCF0301_lag'].isin([1, 2, 3, 4, 5, 6, 7])
    )
    df = merged[mask].copy()

    df['house_rep'] = (df['VCF0707'] == 2.0).astype(int)
    df = construct_pid_vars(df, 'VCF0301', 'curr')
    df = construct_pid_vars(df, 'VCF0301_lag', 'lag')

    return df


def format_results(year, n, model_curr, model_lag, model_iv):
    """Format results for one panel year."""
    lines = []
    lines.append(f"\n--- {year} Panel (N={n}) ---\n")

    for label, model in [("Current party ID", model_curr),
                          ("Lagged party ID", model_lag),
                          ("IV estimates", model_iv)]:
        lines.append(f"{label}:")

        # Get variable names - handle different naming for current vs lagged vs IV
        params = model.params
        bse = model.bse

        # Map to standard names
        coef_map = {}
        for name in params.index:
            if 'strong' in name:
                coef_map['Strong partisan'] = name
            elif 'weak' in name:
                coef_map['Weak partisan'] = name
            elif 'lean' in name:
                coef_map['Leaning partisan'] = name
            elif name == 'const':
                coef_map['Intercept'] = name

        for display_name in ['Strong partisan', 'Weak partisan', 'Leaning partisan', 'Intercept']:
            var_name = coef_map[display_name]
            lines.append(f"  {display_name:20s}: {params[var_name]:7.3f} ({bse[var_name]:.3f})")

        lines.append(f"  Log-likelihood:       {model.llf:.1f}")
        lines.append(f"  Pseudo-R2:            {model.prsquared:.2f}")
        lines.append("")

    return "\n".join(lines)


def run_analysis(data_source=None):
    """Replicate Table 5 from Bartels (2000)."""

    results = []
    results.append("Table 5: Current versus Lagged Party Identification and Congressional Votes")
    results.append("=" * 80)

    curr_vars = ['strong_curr', 'weak_curr', 'lean_curr']
    lag_vars = ['strong_lag', 'weak_lag', 'lean_lag']

    # === 1960 Panel ===
    df60 = prepare_1960_data()
    model_curr_60 = run_probit(df60, 'house_rep', curr_vars)
    model_lag_60 = run_probit(df60, 'house_rep', lag_vars)
    model_iv_60 = run_iv_probit(df60, 'house_rep', curr_vars, lag_vars)
    results.append(format_results(1960, len(df60), model_curr_60, model_lag_60, model_iv_60))

    # === 1976 Panel ===
    df76 = prepare_1976_data()
    model_curr_76 = run_probit(df76, 'house_rep', curr_vars)
    model_lag_76 = run_probit(df76, 'house_rep', lag_vars)
    model_iv_76 = run_iv_probit(df76, 'house_rep', curr_vars, lag_vars)
    results.append(format_results(1976, len(df76), model_curr_76, model_lag_76, model_iv_76))

    # === 1992 Panel ===
    df92 = prepare_1992_data()
    model_curr_92 = run_probit(df92, 'house_rep', curr_vars)
    model_lag_92 = run_probit(df92, 'house_rep', lag_vars)
    model_iv_92 = run_iv_probit(df92, 'house_rep', curr_vars, lag_vars)
    results.append(format_results(1992, len(df92), model_curr_92, model_lag_92, model_iv_92))

    output = "\n".join(results)
    print(output)

    # Run automated scoring
    score = score_against_ground_truth(
        model_curr_60, model_lag_60, model_iv_60, len(df60),
        model_curr_76, model_lag_76, model_iv_76, len(df76),
        model_curr_92, model_lag_92, model_iv_92, len(df92)
    )

    return output


def score_against_ground_truth(
    mc60, ml60, mi60, n60,
    mc76, ml76, mi76, n76,
    mc92, ml92, mi92, n92
):
    """Score results against ground truth from the paper."""

    ground_truth = {
        '1960': {
            'N': 911,
            'current': {
                'strong': (1.358, 0.094), 'weak': (1.028, 0.083),
                'lean': (0.855, 0.131), 'intercept': (0.035, 0.053),
                'llf': -372.7, 'r2': 0.41
            },
            'lagged': {
                'strong': (1.363, 0.092), 'weak': (0.842, 0.078),
                'lean': (0.564, 0.125), 'intercept': (0.068, 0.051),
                'llf': -403.9, 'r2': 0.36
            },
            'iv': {
                'strong': (1.715, 0.173), 'weak': (0.728, 0.239),
                'lean': (1.081, 0.696), 'intercept': (0.032, 0.057),
                'llf': -403.9, 'r2': 0.36
            }
        },
        '1976': {
            'N': 682,
            'current': {
                'strong': (1.087, 0.105), 'weak': (0.624, 0.086),
                'lean': (0.622, 0.110), 'intercept': (-0.123, 0.054),
                'llf': -358.2, 'r2': 0.24
            },
            'lagged': {
                'strong': (0.966, 0.104), 'weak': (0.738, 0.089),
                'lean': (0.486, 0.109), 'intercept': (-0.063, 0.053),
                'llf': -371.3, 'r2': 0.21
            },
            'iv': {
                'strong': (1.123, 0.178), 'weak': (0.745, 0.251),
                'lean': (0.725, 0.438), 'intercept': (-0.102, 0.055),
                'llf': -371.3, 'r2': 0.21
            }
        },
        '1992': {
            'N': 760,
            'current': {
                'strong': (0.975, 0.094), 'weak': (0.627, 0.084),
                'lean': (0.472, 0.098), 'intercept': (-0.211, 0.051),
                'llf': -408.2, 'r2': 0.20
            },
            'lagged': {
                'strong': (1.061, 0.100), 'weak': (0.404, 0.077),
                'lean': (0.519, 0.101), 'intercept': (-0.168, 0.051),
                'llf': -416.2, 'r2': 0.19
            },
            'iv': {
                'strong': (1.516, 0.180), 'weak': (-0.225, 0.268),
                'lean': (1.824, 0.513), 'intercept': (-0.125, 0.053),
                'llf': -416.2, 'r2': 0.19
            }
        }
    }

    models = {
        '1960': {'current': mc60, 'lagged': ml60, 'iv': mi60, 'N': n60},
        '1976': {'current': mc76, 'lagged': ml76, 'iv': mi76, 'N': n76},
        '1992': {'current': mc92, 'lagged': ml92, 'iv': mi92, 'N': n92}
    }

    total_coef_score = 0
    total_se_score = 0
    total_n_score = 0
    total_var_score = 0
    total_llf_score = 0
    total_r2_score = 0

    n_coefs = 0
    n_ses = 0
    n_panels = 0
    n_models = 0
    n_r2 = 0

    print("\n" + "=" * 80)
    print("SCORING")
    print("=" * 80)

    for year in ['1960', '1976', '1992']:
        gt = ground_truth[year]
        m = models[year]
        n_panels += 1

        # Score N
        n_diff_pct = abs(m['N'] - gt['N']) / gt['N']
        if n_diff_pct <= 0.05:
            n_score = 1.0
        elif n_diff_pct <= 0.10:
            n_score = 0.7
        elif n_diff_pct <= 0.20:
            n_score = 0.4
        else:
            n_score = 0.1
        total_n_score += n_score
        print(f"\n{year}: N={m['N']} (target {gt['N']}), diff={n_diff_pct:.1%}, score={n_score:.1f}")

        for model_type in ['current', 'lagged', 'iv']:
            gt_model = gt[model_type]
            gen_model = m[model_type]
            n_models += 1

            params = gen_model.params
            bse = gen_model.bse

            # Map variable names
            var_map = {}
            for name in params.index:
                if 'strong' in name:
                    var_map['strong'] = name
                elif 'weak' in name:
                    var_map['weak'] = name
                elif 'lean' in name:
                    var_map['lean'] = name
                elif name == 'const':
                    var_map['intercept'] = name

            # Check all variables present
            all_present = len(var_map) == 4
            total_var_score += 1.0 if all_present else 0.0

            # Score coefficients and SEs
            for var_key in ['strong', 'weak', 'lean', 'intercept']:
                gt_coef, gt_se = gt_model[var_key]
                gen_coef = params[var_map[var_key]]
                gen_se = bse[var_map[var_key]]

                n_coefs += 1
                n_ses += 1

                coef_diff = abs(gen_coef - gt_coef)
                se_diff = abs(gen_se - gt_se)

                coef_score = max(0, 1.0 - coef_diff / 0.05) if coef_diff <= 0.15 else 0.0
                se_score = max(0, 1.0 - se_diff / 0.02) if se_diff <= 0.06 else 0.0

                total_coef_score += coef_score
                total_se_score += se_score

                if coef_diff > 0.05 or se_diff > 0.02:
                    print(f"  {year} {model_type} {var_key}: coef={gen_coef:.3f} (target {gt_coef:.3f}, diff={coef_diff:.3f}), "
                          f"se={gen_se:.3f} (target {gt_se:.3f}, diff={se_diff:.3f})")

            # Score log-likelihood
            llf_diff = abs(gen_model.llf - gt_model['llf'])
            llf_score = max(0, 1.0 - llf_diff / 1.0) if llf_diff <= 3.0 else 0.0
            total_llf_score += llf_score

            # Score pseudo-R2
            n_r2 += 1
            r2_diff = abs(gen_model.prsquared - gt_model['r2'])
            r2_score = max(0, 1.0 - r2_diff / 0.02) if r2_diff <= 0.06 else 0.0
            total_r2_score += r2_score

            if llf_diff > 1.0 or r2_diff > 0.02:
                print(f"  {year} {model_type}: LL={gen_model.llf:.1f} (target {gt_model['llf']:.1f}, diff={llf_diff:.1f}), "
                      f"R2={gen_model.prsquared:.4f} (target {gt_model['r2']:.2f}, diff={r2_diff:.4f})")

    # Compute weighted score (out of 100)
    # Coefficients: 30 points
    coef_pct = total_coef_score / n_coefs if n_coefs > 0 else 0
    coef_points = 30 * coef_pct

    # Standard errors: 20 points
    se_pct = total_se_score / n_ses if n_ses > 0 else 0
    se_points = 20 * se_pct

    # Sample size: 15 points
    n_pct = total_n_score / n_panels if n_panels > 0 else 0
    n_points = 15 * n_pct

    # Variables present: 10 points
    var_pct = total_var_score / n_models if n_models > 0 else 0
    var_points = 10 * var_pct

    # Log-likelihood: 10 points
    llf_pct = total_llf_score / n_models if n_models > 0 else 0
    llf_points = 10 * llf_pct

    # Pseudo-R2: 15 points
    r2_pct = total_r2_score / n_r2 if n_r2 > 0 else 0
    r2_points = 15 * r2_pct

    total = coef_points + se_points + n_points + var_points + llf_points + r2_points

    print(f"\n{'='*40}")
    print(f"SCORE BREAKDOWN:")
    print(f"  Coefficients:  {coef_points:5.1f}/30 ({coef_pct:.1%})")
    print(f"  Std errors:    {se_points:5.1f}/20 ({se_pct:.1%})")
    print(f"  Sample size:   {n_points:5.1f}/15 ({n_pct:.1%})")
    print(f"  Variables:     {var_points:5.1f}/10 ({var_pct:.1%})")
    print(f"  Log-likelihood:{llf_points:5.1f}/10 ({llf_pct:.1%})")
    print(f"  Pseudo-R2:     {r2_points:5.1f}/15 ({r2_pct:.1%})")
    print(f"  TOTAL:         {total:5.1f}/100")
    print(f"{'='*40}")

    return total


if __name__ == "__main__":
    result = run_analysis()
