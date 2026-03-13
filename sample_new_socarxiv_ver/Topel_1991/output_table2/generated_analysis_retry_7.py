#!/usr/bin/env python3
"""
Table 2 Replication: Models of Annual Within-Job Wage Growth
Topel (1991)

Attempt 7 (counting as attempt 5):
CRITICAL FIX: Construct experience to INCREMENT BY 1 EACH YEAR,
not from age - education - 6 each year. This is what Topel does:
"experience and tenure progress at the same rate."

Use person's first observation to set initial experience:
  initial_experience = age_first - education_fixed - 6
Then: experience_t = initial_experience + (year_t - year_first)
This GUARANTEES d_exp = 1 for all consecutive years.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import os
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_FILE = os.path.join(PROJECT_DIR, 'data', 'psid_panel.csv')

CPS_WAGE_INDEX = {
    1968: 1.000, 1969: 1.032, 1970: 1.091, 1971: 1.115, 1972: 1.113,
    1973: 1.151, 1974: 1.167, 1975: 1.188, 1976: 1.117, 1977: 1.121,
    1978: 1.133, 1979: 1.128, 1980: 1.128, 1981: 1.109, 1982: 1.103,
    1983: 1.089
}

GNP_DEFLATOR = {
    1967: 100.00, 1968: 104.28, 1969: 109.13, 1970: 113.94, 1971: 118.92,
    1972: 123.16, 1973: 130.27, 1974: 143.08, 1975: 155.56, 1976: 163.42,
    1977: 173.43, 1978: 186.18, 1979: 201.33, 1980: 220.39, 1981: 241.02,
    1982: 255.09, 1983: 264.00
}

EDUC_MAP = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}


def run_analysis(data_source=DATA_FILE):
    """Run Table 2 replication."""

    print("="*70)
    print("TABLE 2 REPLICATION - Attempt 7 (Constructed Experience)")
    print("="*70)

    df = pd.read_csv(data_source)
    print(f"\n  Raw: {len(df)} obs, {df['person_id'].nunique()} persons")

    # =========================================================================
    # Education: fix per person
    # =========================================================================
    df['educ_raw'] = df['education_clean'].copy()
    cat_mask = ~df['year'].isin([1975, 1976])
    df.loc[cat_mask, 'educ_raw'] = df.loc[cat_mask, 'education_clean'].map(
        {**EDUC_MAP, 9: np.nan}
    )

    def get_fixed_educ(group):
        good = group[group['year'].isin([1975, 1976])]['educ_raw'].dropna()
        if len(good) > 0:
            return good.iloc[0]
        mapped = group['educ_raw'].dropna()
        if len(mapped) > 0:
            modes = mapped.mode()
            return modes.iloc[0] if len(modes) > 0 else mapped.median()
        return np.nan

    person_educ = df.groupby('person_id').apply(get_fixed_educ)
    df['education_fixed'] = df['person_id'].map(person_educ)
    df = df[df['education_fixed'].notna()].copy()

    # =========================================================================
    # Experience: CONSTRUCTED to increment by 1 each year
    # =========================================================================
    # For each person, compute initial experience from first observation:
    #   initial_exp = age_first - education_fixed - 6
    # Then: experience_t = initial_exp + (year_t - year_first)
    df = df.sort_values(['person_id', 'year']).reset_index(drop=True)

    first_obs = df.groupby('person_id').first()[['age', 'year']].reset_index()
    first_obs.columns = ['person_id', 'age_first', 'year_first']
    df = df.merge(first_obs, on='person_id')

    df['initial_exp'] = df['age_first'] - df['education_fixed'] - 6
    df['initial_exp'] = df['initial_exp'].clip(lower=0)
    df['experience'] = df['initial_exp'] + (df['year'] - df['year_first'])
    df['experience'] = df['experience'].clip(lower=0)

    # Drop persons with initial experience < 1
    person_min_exp = df.groupby('person_id')['experience'].min()
    valid_persons = person_min_exp[person_min_exp >= 1].index
    df = df[df['person_id'].isin(valid_persons)].copy()

    # Tenure (paper starts at 0)
    df['tenure'] = df['tenure_topel'] - 1

    # Real wage for reporting
    df['gnp_defl'] = df['year'].map(GNP_DEFLATOR)
    df['log_real_wage'] = df['log_hourly_wage'] - np.log(df['gnp_defl'] / 100.0)

    print(f"  Mean education: {df['education_fixed'].mean():.2f} (paper: 12.645)")
    print(f"  Mean experience: {df['experience'].mean():.1f} (paper: 20.021)")
    print(f"  Persons: {df['person_id'].nunique()}")

    # =========================================================================
    # Within-job first differences
    # =========================================================================
    df = df.sort_values(['person_id', 'job_id', 'year']).reset_index(drop=True)
    grp = df.groupby(['person_id', 'job_id'])
    df['prev_year'] = grp['year'].shift(1)
    df['prev_log_wage'] = grp['log_hourly_wage'].shift(1)
    df['prev_log_real_wage'] = grp['log_real_wage'].shift(1)
    df['prev_tenure'] = grp['tenure'].shift(1)
    df['prev_experience'] = grp['experience'].shift(1)

    within = df[
        (df['prev_year'].notna()) &
        (df['year'] - df['prev_year'] == 1)
    ].copy()

    within['d_log_wage'] = within['log_hourly_wage'] - within['prev_log_wage']
    within['d_log_real_wage'] = within['log_real_wage'] - within['prev_log_real_wage']
    within['d_exp'] = within['experience'] - within['prev_experience']

    # ALL within-job obs should now have d_exp == 1
    n_bad = (within['d_exp'] != 1).sum()
    print(f"\n  Obs with d_exp != 1: {n_bad} (should be 0)")

    # Drop outliers
    n0 = len(within)
    within = within[within['d_log_wage'].between(-2, 2)].copy()
    print(f"  Dropped {n0 - len(within)} outliers")

    within = within[within['experience'] >= 1].copy()

    N = len(within)
    mean_real = within['d_log_real_wage'].mean()
    n_persons = within['person_id'].nunique()

    print(f"\n  Final sample:")
    print(f"    N = {N} (paper: 8,683)")
    print(f"    Persons = {n_persons} (paper: 1,540)")
    print(f"    Mean d_log_real_wage = {mean_real:.3f} (paper: .026)")
    print(f"    SD d_log_wage = {within['d_log_wage'].std():.4f}")

    # =========================================================================
    # Construct regressors
    # =========================================================================
    t = within['tenure'].values.astype(float)
    pt = within['prev_tenure'].values.astype(float)
    e = within['experience'].values.astype(float)
    pe = within['prev_experience'].values.astype(float)

    within['d_tenure'] = t - pt
    within['d_tenure_sq'] = t**2 - pt**2
    within['d_tenure_cu'] = t**3 - pt**3
    within['d_tenure_qu'] = t**4 - pt**4
    within['d_exp_sq'] = e**2 - pe**2
    within['d_exp_cu'] = e**3 - pe**3
    within['d_exp_qu'] = e**4 - pe**4

    year_dummies = pd.get_dummies(within['year'], prefix='yr', dtype=float)
    yr_cols = sorted(year_dummies.columns.tolist())[1:]

    y = within['d_log_wage'].values

    # =========================================================================
    # Run models
    # =========================================================================
    def run_ols(y_vals, var_list):
        X_main = within[var_list].copy()
        X = pd.concat([X_main.reset_index(drop=True),
                       year_dummies[yr_cols].reset_index(drop=True)], axis=1)
        valid = np.isfinite(X.values).all(axis=1) & np.isfinite(y_vals)
        model = sm.OLS(y_vals[valid], X.loc[valid].values, hasconst=True).fit()
        return model, var_list + yr_cols

    def gc(m, n, v):
        if v in n: return m.params[n.index(v)], m.bse[n.index(v)]
        return None, None

    m1, n1 = run_ols(y, ['d_tenure', 'd_exp_sq', 'd_exp_cu', 'd_exp_qu'])
    m2, n2 = run_ols(y, ['d_tenure', 'd_tenure_sq', 'd_exp_sq', 'd_exp_cu', 'd_exp_qu'])
    m3, n3 = run_ols(y, ['d_tenure', 'd_tenure_sq', 'd_tenure_cu', 'd_tenure_qu',
                          'd_exp_sq', 'd_exp_cu', 'd_exp_qu'])

    # =========================================================================
    # Format results
    # =========================================================================
    results = []
    results.append("=" * 80)
    results.append("TABLE 2: Models of Annual Within-Job Wage Growth")
    results.append("PSID White Males, 1968-83")
    results.append(f"(Dependent Variable Is Change in Log Real Wage; Mean = {mean_real:.3f})")
    results.append("=" * 80)
    results.append("")

    def fmt(c, s, scale=1):
        if c is None: return f"{'...':>10s}        "
        cv, sv = c*scale, s*scale
        p = 2*(1-stats.norm.cdf(abs(cv/sv))) if sv > 0 else 1
        st = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
        return f"{cv:>10.4f} ({sv:.4f}){st}"

    header = f"{'':30s} {'(1)':>22s} {'(2)':>22s} {'(3)':>22s}"
    results.append(header)
    results.append("-" * 96)

    for label, var, scale in [
        ('Delta Tenure', 'd_tenure', 1),
        ('Delta Tenure^2 (x10^2)', 'd_tenure_sq', 100),
        ('Delta Tenure^3 (x10^3)', 'd_tenure_cu', 1000),
        ('Delta Tenure^4 (x10^4)', 'd_tenure_qu', 10000),
        ('Delta Experience^2 (x10^2)', 'd_exp_sq', 100),
        ('Delta Experience^3 (x10^3)', 'd_exp_cu', 1000),
        ('Delta Experience^4 (x10^4)', 'd_exp_qu', 10000),
    ]:
        c1, s1 = gc(m1, n1, var)
        c2, s2 = gc(m2, n2, var)
        c3, s3 = gc(m3, n3, var)
        results.append(f"{label:30s} {fmt(c1,s1,scale):>22s} {fmt(c2,s2,scale):>22s} {fmt(c3,s3,scale):>22s}")

    results.append(f"{'R^2':30s} {m1.rsquared:>22.3f} {m2.rsquared:>22.3f} {m3.rsquared:>22.3f}")
    se1, se2, se3 = np.sqrt(m1.mse_resid), np.sqrt(m2.mse_resid), np.sqrt(m3.mse_resid)
    results.append(f"{'Standard error':30s} {se1:>22.3f} {se2:>22.3f} {se3:>22.3f}")
    results.append(f"{'N':30s} {int(m1.nobs):>22d} {int(m2.nobs):>22d} {int(m3.nobs):>22d}")

    # Predicted wage growth
    results.append("")
    results.append("PREDICTED WITHIN-JOB WAGE GROWTH BY YEARS OF JOB TENURE")
    results.append("(Workers with 10 Years of Labor Market Experience)")
    results.append("")

    bt = gc(m3, n3, 'd_tenure')[0] or 0
    bt2 = gc(m3, n3, 'd_tenure_sq')[0] or 0
    bt3 = gc(m3, n3, 'd_tenure_cu')[0] or 0
    bt4 = gc(m3, n3, 'd_tenure_qu')[0] or 0
    bx2 = gc(m3, n3, 'd_exp_sq')[0] or 0
    bx3 = gc(m3, n3, 'd_exp_cu')[0] or 0
    bx4 = gc(m3, n3, 'd_exp_qu')[0] or 0

    preds = []
    for T in range(1, 11):
        t, tp = T, T-1
        x, xp = 10+T, 10+T-1
        p = (bt + bt2*(t**2-tp**2) + bt3*(t**3-tp**3) + bt4*(t**4-tp**4) +
             bx2*(x**2-xp**2) + bx3*(x**3-xp**3) + bx4*(x**4-xp**4))
        preds.append(p)

    results.append("Tenure:    " + "  ".join([f"{t:>5d}" for t in range(1,11)]))
    results.append("Growth:    " + "  ".join([f"{p:>5.3f}" for p in preds]))
    results.append("Paper:     " + "  ".join([f"{v:>5.3f}" for v in [.068,.060,.052,.046,.041,.037,.033,.030,.028,.026]]))

    # Detailed comparison
    results.append("")
    results.append("DETAILED COEFFICIENT COMPARISON")
    gt_items = [
        (1, 'd_tenure', 1, 0.1242, 0.0161),
        (1, 'd_exp_sq', 100, -0.6051, 0.1430),
        (1, 'd_exp_cu', 1000, 0.1460, 0.0482),
        (1, 'd_exp_qu', 10000, 0.0131, 0.0054),
        (2, 'd_tenure', 1, 0.1265, 0.0162),
        (2, 'd_tenure_sq', 100, -0.0518, 0.0178),
        (2, 'd_exp_sq', 100, -0.6144, 0.1430),
        (2, 'd_exp_cu', 1000, 0.1620, 0.0485),
        (2, 'd_exp_qu', 10000, 0.0151, 0.0055),
        (3, 'd_tenure', 1, 0.1258, 0.0162),
        (3, 'd_tenure_sq', 100, -0.4592, 0.1080),
        (3, 'd_tenure_cu', 1000, 0.1846, 0.0526),
        (3, 'd_tenure_qu', 10000, -0.0245, 0.0079),
        (3, 'd_exp_sq', 100, -0.4067, 0.1546),
        (3, 'd_exp_cu', 1000, 0.0989, 0.0517),
        (3, 'd_exp_qu', 10000, 0.0089, 0.0058),
    ]
    models = {1: (m1, n1), 2: (m2, n2), 3: (m3, n3)}
    for mod, var, scale, gt_c, gt_s in gt_items:
        m, n = models[mod]
        c, s = gc(m, n, var)
        if c is not None:
            gen_c = c * scale
            diff = gen_c - gt_c
            match = "YES" if abs(diff) <= 0.05 else "no"
            results.append(f"  M{mod} {var:>15s}: {gen_c:>10.4f} (paper: {gt_c:>10.4f}, diff: {diff:>8.4f}) {match}")

    output = "\n".join(results)
    print("\n" + output)

    score = compute_score(m1, n1, m2, n2, m3, n3, int(m1.nobs), preds)
    print(f"\nSCORE: {score['total']:.0f}/100")
    for k, v in score['breakdown'].items():
        print(f"  {k}: {v['earned']:.1f}/{v['possible']}")

    return output, score


def compute_score(m1, n1, m2, n2, m3, n3, N, preds):
    """Score against ground truth."""

    def gc(m, n, v):
        if v in n: return m.params[n.index(v)], m.bse[n.index(v)]
        return None, None

    gt = [
        (1, 'd_tenure', 1, 0.1242, 0.0161),
        (1, 'd_exp_sq', 100, -0.6051, 0.1430),
        (1, 'd_exp_cu', 1000, 0.1460, 0.0482),
        (1, 'd_exp_qu', 10000, 0.0131, 0.0054),
        (2, 'd_tenure', 1, 0.1265, 0.0162),
        (2, 'd_tenure_sq', 100, -0.0518, 0.0178),
        (2, 'd_exp_sq', 100, -0.6144, 0.1430),
        (2, 'd_exp_cu', 1000, 0.1620, 0.0485),
        (2, 'd_exp_qu', 10000, 0.0151, 0.0055),
        (3, 'd_tenure', 1, 0.1258, 0.0162),
        (3, 'd_tenure_sq', 100, -0.4592, 0.1080),
        (3, 'd_tenure_cu', 1000, 0.1846, 0.0526),
        (3, 'd_tenure_qu', 10000, -0.0245, 0.0079),
        (3, 'd_exp_sq', 100, -0.4067, 0.1546),
        (3, 'd_exp_cu', 1000, 0.0989, 0.0517),
        (3, 'd_exp_qu', 10000, 0.0089, 0.0058),
    ]

    models = {1: (m1, n1), 2: (m2, n2), 3: (m3, n3)}
    bd = {}

    cm = sum(1 for mod,var,sc,gtc,_ in gt if gc(models[mod][0],models[mod][1],var)[0] is not None and abs(gc(models[mod][0],models[mod][1],var)[0]*sc - gtc) <= 0.05)
    bd['coef_magnitudes'] = {'earned': 25*cm/len(gt), 'possible': 25}

    sm_ = sum(1 for mod,var,sc,_,gts in gt if gc(models[mod][0],models[mod][1],var)[1] is not None and abs(gc(models[mod][0],models[mod][1],var)[1]*sc - gts) <= 0.02)
    bd['standard_errors'] = {'earned': 15*sm_/len(gt), 'possible': 15}

    r = abs(N-8683)/8683
    bd['sample_size'] = {'earned': 15 if r<=0.05 else 10 if r<=0.10 else 5 if r<=0.20 else 0, 'possible': 15}

    sig_m = 0
    for mod,var,sc,gtc,gts in gt:
        c,s = gc(models[mod][0],models[mod][1],var)
        if c is not None and s is not None:
            gp = 2*(1-stats.norm.cdf(abs(gtc/gts)))
            ep = 2*(1-stats.norm.cdf(abs(c*sc/(s*sc))))
            gs = '***' if gp<0.01 else '**' if gp<0.05 else '*' if gp<0.1 else ''
            es = '***' if ep<0.01 else '**' if ep<0.05 else '*' if ep<0.1 else ''
            if gs == es: sig_m += 1
    bd['significance'] = {'earned': 25*sig_m/len(gt), 'possible': 25}

    bd['variables_present'] = {'earned': 10, 'possible': 10}

    r2c = [abs(m1.rsquared-0.022)<=0.02, abs(m2.rsquared-0.023)<=0.02, abs(m3.rsquared-0.025)<=0.02]
    bd['r_squared'] = {'earned': 10*sum(r2c)/3, 'possible': 10}

    return {'total': sum(v['earned'] for v in bd.values()), 'breakdown': bd}


if __name__ == '__main__':
    output, score = run_analysis()
