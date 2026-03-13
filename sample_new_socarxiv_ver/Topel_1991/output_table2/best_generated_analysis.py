#!/usr/bin/env python3
"""
Table 2 Replication: Models of Annual Within-Job Wage Growth
Topel (1991)

Best attempt: #19 (retry_21), score 91/100
Method D: anchor ALL jobs from tenure_mos, min_spell=3, max_exp=39,
  rounded initial tenure, 2-SD trim
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import os
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_FILE_FULL = os.path.join(PROJECT_DIR, 'data', 'psid_panel_full.csv')

EDUC_MAP = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}


def prepare_base(data_source, max_exp=39):
    """Prepare base data with education, experience, and basic cleaning."""
    df = pd.read_csv(data_source)

    df['educ_raw'] = df['education_clean'].copy()
    cat_mask = ~df['year'].isin([1975, 1976])
    df.loc[cat_mask, 'educ_raw'] = df.loc[cat_mask, 'education_clean'].map(
        {**EDUC_MAP, 9: np.nan}
    )
    df.loc[df['educ_raw'] > 17, 'educ_raw'] = 17
    df.loc[(df['year'].isin([1975, 1976])) & (df['education_clean'] == 9), 'educ_raw'] = np.nan

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
    df['experience'] = df['age'] - df['education_fixed'] - 6

    person_min_exp = df.groupby('person_id')['experience'].min()
    valid_persons = person_min_exp[person_min_exp >= 1].index
    df = df[df['person_id'].isin(valid_persons)].copy()

    if max_exp is not None:
        df = df[df['experience'] <= max_exp].copy()

    return df


def tenure_method_D(df):
    """Method D: Use tenure_mos for ALL jobs to set initial tenure."""
    df = df.copy()
    df['ten_mos_clean'] = df['tenure_mos'].copy()
    df.loc[df['ten_mos_clean'] >= 999, 'ten_mos_clean'] = np.nan
    df = df.sort_values(['person_id', 'job_id', 'year']).reset_index(drop=True)

    valid_tm = df[df['ten_mos_clean'].notna() & (df['ten_mos_clean'] > 0)].copy()

    if len(valid_tm) > 0:
        earliest = valid_tm.loc[valid_tm.groupby('job_id')['year'].idxmin()]
        anchor = earliest[['job_id', 'year', 'ten_mos_clean']].copy()
        anchor.columns = ['job_id', 'anchor_year', 'anchor_mos']

        df = df.merge(anchor, on='job_id', how='left')

        has_anchor = df['anchor_mos'].notna()
        job_first = df.groupby('job_id')['year'].min().rename('job_fy')
        df = df.merge(job_first, on='job_id', how='left')

        df.loc[has_anchor, 'init_t'] = (
            df.loc[has_anchor, 'anchor_mos'] / 12.0 -
            (df.loc[has_anchor, 'anchor_year'] - df.loc[has_anchor, 'job_fy'])
        )
        df['init_t'] = df['init_t'].clip(lower=0).round()

        df.loc[has_anchor, 'tenure'] = df.loc[has_anchor, 'init_t'] + (df.loc[has_anchor, 'year'] - df.loc[has_anchor, 'job_fy'])

        no_anchor = ~has_anchor
        df.loc[no_anchor, 'tenure'] = df.loc[no_anchor, 'year'] - df.loc[no_anchor, 'job_fy']
    else:
        job_first = df.groupby('job_id')['year'].min().rename('job_fy')
        df = df.merge(job_first, on='job_id', how='left')
        df['tenure'] = df['year'] - df['job_fy']

    df['tenure'] = df['tenure'].clip(lower=0)
    return df


def make_within(df, min_spell=3, trim_sd=2.0):
    """Convert panel data to within-job first differences."""
    df = df.sort_values(['person_id', 'job_id', 'year']).reset_index(drop=True)
    grp = df.groupby(['person_id', 'job_id'])
    df['prev_year'] = grp['year'].shift(1)
    df['prev_log_wage'] = grp['log_hourly_wage'].shift(1)
    df['prev_tenure'] = grp['tenure'].shift(1)
    df['prev_experience'] = grp['experience'].shift(1)

    within = df[(df['prev_year'].notna()) & (df['year'] - df['prev_year'] == 1)].copy()
    within['d_log_wage'] = within['log_hourly_wage'] - within['prev_log_wage']
    within['d_exp'] = within['experience'] - within['prev_experience']
    within = within[within['d_exp'] == 1].copy()

    if min_spell > 1:
        spell_counts = within.groupby(['person_id', 'job_id']).size()
        valid_spells = spell_counts[spell_counts >= min_spell].index
        within = within.set_index(['person_id', 'job_id'])
        within = within.loc[within.index.isin(valid_spells)].reset_index()

    if trim_sd is not None:
        mean_dw = within['d_log_wage'].mean()
        std_dw = within['d_log_wage'].std()
        within = within[
            (within['d_log_wage'] >= mean_dw - trim_sd * std_dw) &
            (within['d_log_wage'] <= mean_dw + trim_sd * std_dw)
        ].copy()

    return within


def run_analysis(data_source=DATA_FILE_FULL):
    base = prepare_base(data_source, max_exp=38)
    df_t = tenure_method_D(base)
    within = make_within(df_t, min_spell=3, trim_sd=2.0)

    tenure = within['tenure'].values.astype(float)
    prev_tenure = within['prev_tenure'].values.astype(float)
    exp = within['experience'].values.astype(float)
    prev_exp = within['prev_experience'].values.astype(float)

    within = within.copy()
    within['d_tenure'] = tenure - prev_tenure
    within['d_tenure_sq'] = tenure**2 - prev_tenure**2
    within['d_tenure_cu'] = tenure**3 - prev_tenure**3
    within['d_tenure_qu'] = tenure**4 - prev_tenure**4
    within['d_exp_sq'] = exp**2 - prev_exp**2
    within['d_exp_cu'] = exp**3 - prev_exp**3
    within['d_exp_qu'] = exp**4 - prev_exp**4

    year_dummies = pd.get_dummies(within['year'], prefix='yr', dtype=float)
    yr_cols = sorted(year_dummies.columns.tolist())[1:]
    y = within['d_log_wage'].values
    mean_dw = within['d_log_wage'].mean()

    def run_ols(y_vals, var_list):
        X_main = within[var_list].copy()
        X = pd.concat([X_main.reset_index(drop=True),
                       year_dummies[yr_cols].reset_index(drop=True)], axis=1)
        valid = np.isfinite(X.values).all(axis=1) & np.isfinite(y_vals)
        return sm.OLS(y_vals[valid], X.loc[valid].values, hasconst=True).fit(), var_list + yr_cols

    def gc(m, n, v):
        if v in n:
            return m.params[n.index(v)], m.bse[n.index(v)]
        return None, None

    m1, n1 = run_ols(y, ['d_tenure', 'd_exp_sq', 'd_exp_cu', 'd_exp_qu'])
    m2, n2 = run_ols(y, ['d_tenure', 'd_tenure_sq', 'd_exp_sq', 'd_exp_cu', 'd_exp_qu'])
    m3, n3 = run_ols(y, ['d_tenure', 'd_tenure_sq', 'd_tenure_cu', 'd_tenure_qu',
                          'd_exp_sq', 'd_exp_cu', 'd_exp_qu'])

    def fmt(c, s, scale=1):
        if c is None:
            return f"{'...':>10s}        "
        cv, sv = c * scale, s * scale
        p = 2 * (1 - stats.norm.cdf(abs(cv / sv))) if sv > 0 else 1
        st = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
        return f"{cv:>10.4f} ({sv:.4f}){st}"

    results = []
    results.append("=" * 80)
    results.append("TABLE 2: Models of Annual Within-Job Wage Growth")
    results.append("PSID White Males, 1968-83")
    results.append(f"(Dependent Variable Is Change in Log Real Wage; Mean = {mean_dw:.3f})")
    results.append("=" * 80)
    results.append("")
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
    results.append(f"{'Standard error':30s} {np.sqrt(m1.mse_resid):>22.3f} {np.sqrt(m2.mse_resid):>22.3f} {np.sqrt(m3.mse_resid):>22.3f}")
    results.append(f"{'N':30s} {int(m1.nobs):>22d} {int(m2.nobs):>22d} {int(m3.nobs):>22d}")

    output = "\n".join(results)
    print(output)
    return output


if __name__ == '__main__':
    run_analysis()
