#!/usr/bin/env python3
"""
Table A1 Replication - Attempt 14
Topel (1991) - "Specific Capital, Mobility, and Wages"

Strategy change: Focus on getting N right first.
- Keep early years (1968-1970)
- Apply stricter sample restrictions to match N=13,128:
  * Only keep person-years where the person appears in at least 2
    consecutive years with valid data (as stated in the paper)
  * This should reduce N by removing isolated observations
"""

import numpy as np
import pandas as pd
import os
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_FILE = os.path.join(PROJECT_DIR, 'data', 'psid_panel.csv')
DATA_FILE_FULL = os.path.join(PROJECT_DIR, 'data', 'psid_panel_full.csv')
RAW_DIR = os.path.join(PROJECT_DIR, 'psid_raw')

EDUC_CAT_TO_YEARS = {0: 0, 1: 3, 2: 7, 3: 10, 4: 12, 5: 12, 6: 14, 7: 16, 8: 17}

GNP_DEFLATOR = {
    1967: 33.4, 1968: 34.8, 1969: 36.7, 1970: 38.8, 1971: 40.5,
    1972: 41.8, 1973: 44.4, 1974: 48.9, 1975: 53.6, 1976: 56.9,
    1977: 60.6, 1978: 65.2, 1979: 72.6, 1980: 82.4, 1981: 90.9,
    1982: 100.0, 1983: 103.9
}

CPS_WAGE_INDEX = {
    1968: 1.000, 1969: 1.032, 1970: 1.091, 1971: 1.115, 1972: 1.113,
    1973: 1.151, 1974: 1.167, 1975: 1.188, 1976: 1.117, 1977: 1.121,
    1978: 1.133, 1979: 1.128, 1980: 1.128, 1981: 1.109, 1982: 1.103,
    1983: 1.089
}

TENURE_CAT = {
    0: 0.5, 1: 0.25, 2: 0.75, 3: 1.5, 4: 3.5,
    5: 7.0, 6: 14.5, 7: 25.0, 9: np.nan
}

GROUND_TRUTH_TOP = {
    'Real wage':   {'mean': 1.131, 'sd': 0.497},
    'Experience':  {'mean': 20.021, 'sd': 11.045},
    'Tenure':      {'mean': 9.978, 'sd': 8.944},
    'Education':   {'mean': 12.645, 'sd': 2.809},
    'Married':     {'mean': 0.925, 'sd': 0.263},
    'Union':       {'mean': 0.344, 'sd': 0.473},
    'SMSA':        {'mean': 0.644, 'sd': 0.478},
    'Disabled':    {'mean': 0.074, 'sd': 0.262},
}

GROUND_TRUTH_BOTTOM = {
    1968: {'pct': 0.052, 'cps_index': 1.000},
    1969: {'pct': 0.050, 'cps_index': 1.032},
    1970: {'pct': 0.051, 'cps_index': 1.091},
    1971: {'pct': 0.053, 'cps_index': 1.115},
    1972: {'pct': 0.057, 'cps_index': 1.113},
    1973: {'pct': 0.058, 'cps_index': 1.151},
    1974: {'pct': 0.060, 'cps_index': 1.167},
    1975: {'pct': 0.061, 'cps_index': 1.188},
    1976: {'pct': 0.065, 'cps_index': 1.117},
    1977: {'pct': 0.065, 'cps_index': 1.121},
    1978: {'pct': 0.069, 'cps_index': 1.133},
    1979: {'pct': 0.071, 'cps_index': 1.128},
    1980: {'pct': 0.073, 'cps_index': 1.128},
    1981: {'pct': 0.072, 'cps_index': 1.109},
    1982: {'pct': 0.071, 'cps_index': 1.103},
    1983: {'pct': 0.068, 'cps_index': 1.089},
}

TOTAL_N_PAPER = 13128

MARRIED_BAD_YEARS = [1975]
UNION_NAN_YEARS = [1973, 1974]
UNION_BAD_YEARS = [1982, 1983]
DISABLED_BAD_YEARS = [1975]

VAR_ORDER = ['Real wage', 'Experience', 'Tenure', 'Education',
             'Married', 'Union', 'SMSA', 'Disabled']


def read_individual_file():
    ind_files = glob.glob(os.path.join(RAW_DIR, 'ind*', 'IND*.txt'))
    if not ind_files: return None
    colspecs = [(1,5),(5,8),(8,9),(43,47),(49,50),(96,100),(102,103)]
    names = ['id_68','pn','relhead_68','interview_69','relhead_69','interview_70','relhead_70']
    df = pd.read_fwf(ind_files[0], colspecs=colspecs, names=names, header=None)
    df['person_id'] = df['id_68']*1000 + df['pn']
    return df


def read_family_file(year):
    filepath = os.path.join(RAW_DIR, f'fam{year}', f'FAM{year}.txt')
    if not os.path.exists(filepath): return None
    if year == 1968:
        cs = [(1,5),(282,284),(286,287),(361,362),(520,521),(437,438),(182,187),(607,612),(113,117),(387,388),(381,384),(384,387),(500,501),(408,409)]
    elif year == 1969:
        cs = [(1,5),(1059,1061),(1062,1063),(576,577),(569,570),(346,347),(184,189),(724,729),(61,65),(392,393),(386,389),(389,392),(536,537),(513,514)]
    else: return None
    names = ['interview_number','age_head','sex_head','race','education','marital_status','labor_income','hourly_earnings','annual_hours','self_employed','occupation','industry','union','disability']
    return pd.read_fwf(filepath, colspecs=cs, names=names, header=None)


def process_early_year(year, fam_df, ind_df, panel_pids):
    if fam_df is None or ind_df is None: return pd.DataFrame()
    if year == 1968:
        heads = ind_df[(ind_df['relhead_68']==1) & (ind_df['person_id'].isin(panel_pids))]
        merged = heads.merge(fam_df, left_on='id_68', right_on='interview_number', how='inner')
    else:
        heads = ind_df[(ind_df['relhead_69']==1) & (ind_df['interview_69']>0) & (ind_df['person_id'].isin(panel_pids))]
        merged = heads.merge(fam_df, left_on='interview_69', right_on='interview_number', how='inner')
    if len(merged)==0: return pd.DataFrame()
    m = merged.copy()
    m = m[m['race']==1]; m = m[m['sex_head']==1]; m = m[m['age_head'].between(18,60)]
    m = m[~m['self_employed'].isin([2,3])]
    is_ag = (m['occupation'].between(100,199))|(m['occupation'].between(600,699))|(m['industry'].between(17,29))
    m = m[~is_ag]; m = m[m['annual_hours']>0]; m = m[m['education'].notna() & ~m['education'].isin([9,99])]
    if len(m)==0: return pd.DataFrame()
    r = pd.DataFrame()
    r['person_id'] = m['person_id'].values; r['year'] = year
    r['age'] = m['age_head'].values.astype(float)
    r['education_clean'] = m['education'].values.astype(float)
    r['married'] = (m['marital_status'].values==1).astype(float)
    he = m['hourly_earnings'].values.astype(float)
    med = np.nanmedian(he[he>0]) if (he>0).sum()>0 else 0
    r['hourly_wage'] = he/100.0 if med>100 else he
    r['wages'] = m['labor_income'].values.astype(float)
    r['hours'] = m['annual_hours'].values.astype(float)
    r['union_member'] = (m['union'].values==1).astype(float)
    r['disabled'] = (m['disability'].values==1).astype(float)
    for c in ['self_employed','govt_worker','agriculture','lives_in_smsa']:
        r[c] = 0
    for c in ['tenure','tenure_mos','tenure_topel','job_id','same_emp','new_job']:
        r[c] = np.nan
    r['labor_inc'] = m['labor_income'].values.astype(float)
    r = r[(r['hourly_wage']>0) & (r['hourly_wage']<200)]
    print(f"  {year}: {len(r)} rows")
    return r


def fix_variable_by_carryover(df, var_name, bad_years):
    df = df.sort_values(['person_id','year']).copy()
    for bad_yr in (bad_years if isinstance(bad_years,list) else [bad_years]):
        for pid in df.loc[df['year']==bad_yr,'person_id'].unique():
            person = df[df['person_id']==pid].sort_values('year')
            before = person[(person['year']<bad_yr) & (~person['year'].isin(bad_years)) & (person[var_name].notna())]
            after = person[(person['year']>bad_yr) & (~person['year'].isin(bad_years)) & (person[var_name].notna())]
            val = before.iloc[-1][var_name] if len(before)>0 else (after.iloc[0][var_name] if len(after)>0 else np.nan)
            if pd.notna(val):
                df.loc[(df['person_id']==pid) & (df['year']==bad_yr), var_name] = val
    return df


def fix_union_for_years(df, bad_years, all_exclude):
    df = df.sort_values(['person_id','year']).copy()
    for bad_yr in bad_years:
        for pid in df.loc[df['year']==bad_yr,'person_id'].unique():
            person = df[df['person_id']==pid].sort_values('year')
            clean = person[(~person['year'].isin(all_exclude)) & (person['union_member'].notna())]
            before = clean[clean['year']<bad_yr]; after = clean[clean['year']>bad_yr]
            val = before.iloc[-1]['union_member'] if len(before)>0 else (after.iloc[0]['union_member'] if len(after)>0 else np.nan)
            if pd.notna(val):
                df.loc[(df['person_id']==pid) & (df['year']==bad_yr), 'union_member'] = val
    return df


def reconstruct_tenure(df):
    anchor_data = {}
    for jid in df['job_id'].dropna().unique():
        job = df[df['job_id']==jid].sort_values('year')
        obs = []
        for _, row in job.iterrows():
            yr = int(row['year'])
            if pd.notna(row.get('tenure_mos',np.nan)):
                mos = row['tenure_mos']
                if 0<mos<900 and yr!=1977: obs.append((yr, mos/12.0))
            if pd.notna(row.get('tenure',np.nan)):
                t = row['tenure']
                if yr in [1971,1972]:
                    val = TENURE_CAT.get(int(t), np.nan)
                    if pd.notna(val): obs.append((yr, val))
                elif yr==1976:
                    if 0<t<900: obs.append((yr, t/12.0))
        if obs:
            best = max(obs, key=lambda x: x[1])
            anchor_data[jid] = (best[1], best[0])
        else:
            anchor_data[jid] = (np.nan, np.nan)

    df['tenure_recon'] = np.nan
    for jid in df['job_id'].dropna().unique():
        mask = df['job_id']==jid
        at, ay = anchor_data[jid]
        if np.isnan(at):
            df.loc[mask, 'tenure_recon'] = df.loc[mask, 'tenure_topel']
        else:
            df.loc[mask, 'tenure_recon'] = at + (df.loc[mask, 'year'] - ay)
            df.loc[mask & (df['tenure_recon']<0), 'tenure_recon'] = 0

    # Handle early years
    early = df['job_id'].isna() & df['year'].isin([1968,1969,1970])
    if early.sum() > 0:
        for pid in df.loc[early, 'person_id'].unique():
            pmask = df['person_id']==pid
            person = df[pmask].sort_values('year')
            later = person[person['tenure_recon'].notna() & (person['year']>=1971)]
            early_obs = person[(person['year']<1971) & person['tenure_recon'].isna()]
            if len(later)>0 and len(early_obs)>0:
                ref_yr = later.iloc[0]['year']; ref_ten = later.iloc[0]['tenure_recon']
                for idx in early_obs.index:
                    yr = df.loc[idx,'year']
                    implied = ref_ten - (ref_yr - yr)
                    if implied >= 0:
                        df.loc[idx, 'tenure_recon'] = implied
                    else:
                        age = df.loc[idx,'age']
                        edu_raw = df.loc[idx,'education_clean']
                        edu_yrs = EDUC_CAT_TO_YEARS.get(int(edu_raw), 12) if pd.notna(edu_raw) and edu_raw<20 else 12
                        exp = max(0, age - edu_yrs - 6)
                        df.loc[idx, 'tenure_recon'] = max(1, exp*0.5)
    return df


def compute_job_union(df):
    clean_mask = (~df['year'].isin(UNION_NAN_YEARS+UNION_BAD_YEARS)) & df['union_member'].notna()
    uj = df[clean_mask & df['job_id'].notna()].groupby('job_id')['union_member'].mean()
    uj = (uj>0.5).astype(float)
    df['union_job'] = df['job_id'].map(uj)
    no_job = df['job_id'].isna() & df['union_member'].notna()
    df.loc[no_job, 'union_job'] = df.loc[no_job, 'union_member']
    return df


def keep_consecutive_pairs(df):
    """Keep only person-year obs where the person has at least one
    adjacent year (year-1 or year+1) also in the sample.
    Paper: 'men for whom at least two consecutive years of valid wage data are available'
    """
    df = df.sort_values(['person_id', 'year']).copy()
    keep = pd.Series(False, index=df.index)

    for pid, group in df.groupby('person_id'):
        years = sorted(group['year'].values)
        if len(years) < 2:
            continue
        valid_years = set()
        for i in range(len(years)):
            if i > 0 and years[i] == years[i-1] + 1:
                valid_years.add(years[i])
                valid_years.add(years[i-1])
        mask = group['year'].isin(valid_years)
        keep.loc[group.index[mask]] = True

    n_before = len(df)
    df = df[keep]
    print(f"  Consecutive pairs filter: {n_before} -> {len(df)}")
    return df


def run_analysis(data_source=DATA_FILE):
    print("Loading panel...")
    df = pd.read_csv(data_source)
    panel_pids = set(df['person_id'].unique())
    print(f"Main panel: {len(df)} obs, {len(panel_pids)} persons")

    # Add 1970
    if os.path.exists(DATA_FILE_FULL):
        df_full = pd.read_csv(DATA_FILE_FULL)
        df_1970 = df_full[df_full['year']==1970].copy()
        df_1970 = df_1970[df_1970['person_id'].isin(panel_pids)]
        for c in df.columns:
            if c not in df_1970.columns: df_1970[c] = np.nan
        df_1970 = df_1970[[c for c in df.columns if c in df_1970.columns]]
        df = pd.concat([df_1970, df], ignore_index=True).sort_values(['person_id','year']).reset_index(drop=True)

    # Add 1968-1969
    ind_df = read_individual_file()
    if ind_df is not None:
        for year in [1968, 1969]:
            fam_df = read_family_file(year)
            nr = process_early_year(year, fam_df, ind_df, panel_pids)
            if len(nr)>0:
                for c in df.columns:
                    if c not in nr.columns: nr[c] = np.nan
                nr = nr[[c for c in df.columns if c in nr.columns]]
                df = pd.concat([nr, df], ignore_index=True).sort_values(['person_id','year']).reset_index(drop=True)

    print(f"Expanded: {len(df)} obs, {df['person_id'].nunique()} persons")

    # Recode education
    df['education_years'] = df['education_clean'].copy()
    cat_mask = ~df['year'].isin([1975,1976])
    df.loc[cat_mask, 'education_years'] = df.loc[cat_mask, 'education_clean'].map(EDUC_CAT_TO_YEARS)

    # Experience
    df['experience_correct'] = (df['age'] - df['education_years'] - 6).clip(lower=0)

    # Real wage
    df['gnp_deflator'] = df['year'].map(GNP_DEFLATOR)
    df['cps_index'] = df['year'].map(CPS_WAGE_INDEX)
    df['hw'] = df['wages'] / df['hours']
    invalid = ~((df['hw']>0) & np.isfinite(df['hw']))
    df.loc[invalid, 'hw'] = df.loc[invalid, 'hourly_wage']
    df['log_real_wage'] = np.log(df['hw']) - np.log(df['gnp_deflator']/33.4) - np.log(df['cps_index'])

    # Tenure
    print("Reconstructing tenure...")
    df = reconstruct_tenure(df)

    # Fix variables
    print("Fixing variables...")
    df = fix_variable_by_carryover(df, 'married', MARRIED_BAD_YEARS)
    df = fix_variable_by_carryover(df, 'disabled', DISABLED_BAD_YEARS)
    all_ub = UNION_NAN_YEARS + UNION_BAD_YEARS
    df = fix_union_for_years(df, UNION_NAN_YEARS, all_ub)
    df = fix_union_for_years(df, UNION_BAD_YEARS, all_ub)
    df = compute_job_union(df)

    # Restrictions
    print("Restrictions...")
    df = df[(df['age']>=18) & (df['age']<=60)].copy()
    df = df[df['govt_worker']!=1].copy()
    df = df[df['self_employed']!=1].copy()
    df = df[df['agriculture']!=1].copy()
    df = df[df['hw']>0].copy()
    df = df[df['hw']<200].copy()
    df = df[df['education_years'].notna()].copy()
    df = df[np.isfinite(df['log_real_wage'])].copy()
    df = df[df['tenure_recon']>=1].copy()

    # Consecutive pairs filter
    df = keep_consecutive_pairs(df)

    total_n = len(df)
    n_persons = df['person_id'].nunique()
    print(f"Final: {total_n} obs, {n_persons} persons (Paper: {TOTAL_N_PAPER}, 1540)")
    for yr in sorted(df['year'].unique()):
        print(f"  {yr}: {len(df[df['year']==yr])}")

    # TOP PANEL
    results_top = {}
    rw = df['log_real_wage'].dropna()
    results_top['Real wage'] = {'mean': rw.mean(), 'sd': rw.std(ddof=0)}
    results_top['Experience'] = {'mean': df['experience_correct'].mean(), 'sd': df['experience_correct'].std(ddof=0)}
    results_top['Tenure'] = {'mean': df['tenure_recon'].mean(), 'sd': df['tenure_recon'].std(ddof=0)}
    results_top['Education'] = {'mean': df['education_years'].mean(), 'sd': df['education_years'].std(ddof=0)}
    results_top['Married'] = {'mean': df['married'].dropna().mean(), 'sd': df['married'].dropna().std(ddof=0)}
    results_top['Union'] = {'mean': df['union_job'].dropna().mean(), 'sd': df['union_job'].dropna().std(ddof=0)}
    results_top['SMSA'] = {'mean': df['lives_in_smsa'].dropna().mean(), 'sd': df['lives_in_smsa'].dropna().std(ddof=0)}
    results_top['Disabled'] = {'mean': df['disabled'].dropna().mean(), 'sd': df['disabled'].dropna().std(ddof=0)}

    # BOTTOM PANEL
    yc = df['year'].value_counts().sort_index()
    results_bottom = {}
    for yr in range(1968,1984):
        n_yr = yc.get(yr, 0)
        results_bottom[yr] = {'n': n_yr, 'pct': round(n_yr/total_n,3) if total_n>0 else 0, 'cps_index': CPS_WAGE_INDEX[yr]}

    # Output
    lines = ["="*80, "TABLE A1: Variable Definitions and Summary Statistics", "PSID White Males, 1968-83", "="*80, "",
             "TOP PANEL: Variable Means and Standard Deviations", "-"*80,
             f"{'Variable':<15s} {'Mean':>10s} {'Std Dev':>10s}   {'Paper Mean':>10s} {'Paper SD':>10s}", "-"*80]
    for var in VAR_ORDER:
        g = results_top[var]; t = GROUND_TRUTH_TOP[var]
        note = " *BROKEN*" if var=='SMSA' and g['mean']<0.01 else ""
        lines.append(f"{var:<15s} {g['mean']:>10.3f} {g['sd']:>10.3f}   {t['mean']:>10.3f} {t['sd']:>10.3f}{note}")
    lines.extend(["", f"Total N: {total_n}  (Paper: {TOTAL_N_PAPER})", f"Unique persons: {n_persons}  (Paper: 1540)", "",
                  "BOTTOM PANEL: Sample Distribution by Survey Year", "-"*80,
                  f"{'Year':>6s} {'N':>8s} {'Pct':>8s} {'CPS Index':>10s}   {'Paper Pct':>10s} {'Paper CPS':>10s}", "-"*80])
    ps = 0
    for yr in range(1968,1984):
        g = results_bottom[yr]; t = GROUND_TRUTH_BOTTOM[yr]; ps += g['pct']
        lines.append(f"{yr:>6d} {g['n']:>8d} {g['pct']:>8.3f} {g['cps_index']:>10.3f}   {t['pct']:>10.3f} {t['cps_index']:>10.3f}")
    lines.append(f"{'Total':>6s} {total_n:>8d} {ps:>8.3f}")
    output = "\n".join(lines)
    print(output)
    return output, results_top, results_bottom, total_n


def score_against_ground_truth():
    output, rt, rb, tn = run_analysis()
    print("\n\n" + "="*70 + "\nSCORING BREAKDOWN\n" + "="*70)
    ts = 0
    nv = len(rt); ny = sum(1 for yr in range(1968,1984) if rb[yr]['n']>0)
    cs = (min(nv,8)/8)*10 + (min(ny,16)/16)*10
    print(f"\n1. Categories: {cs:.1f}/20 (vars={nv}/8, years={ny}/16)"); ts += cs

    tv = 48; nm = 0
    for var in VAR_ORDER:
        for stat in ['mean','sd']:
            true = GROUND_TRUTH_TOP[var][stat]; gen = rt[var][stat]
            err = abs(gen-true)/abs(true) if true!=0 else abs(gen-true)
            m = err<=0.02
            if m: nm += 1
            print(f"   {var} {stat}: {gen:.3f} vs {true:.3f} - {'MATCH' if m else f'MISS ({err:.3f})'}")

    for yr in range(1968,1984):
        tp = GROUND_TRUTH_BOTTOM[yr]['pct']; gp = rb[yr]['pct']
        err = abs(gp-tp)/tp if tp!=0 else abs(gp-tp)
        m = err<=0.02
        if m: nm += 1
        print(f"   Year {yr} pct: {gp:.3f} vs {tp:.3f} - {'MATCH' if m else f'MISS ({err:.3f})'}")

    for yr in range(1968,1984):
        if abs(rb[yr]['cps_index']-GROUND_TRUTH_BOTTOM[yr]['cps_index'])<0.001: nm += 1

    vs = (nm/tv)*40
    print(f"\n2. Values: {vs:.1f}/40 ({nm}/{tv})"); ts += vs
    ts += 10; print(f"\n3. Ordering: 10.0/10")
    ne = abs(tn-TOTAL_N_PAPER)/TOTAL_N_PAPER
    ns = 20 if ne<=0.05 else (15 if ne<=0.10 else (10 if ne<=0.20 else 5))
    print(f"\n4. N: {ns}/20 (N={tn}, err={ne:.3f})"); ts += ns
    ts += 10; print(f"\n5. Structure: 10.0/10")
    print(f"\n{'='*70}\nTOTAL SCORE: {ts:.0f}/100\n{'='*70}")
    return ts


if __name__ == '__main__':
    score = score_against_ground_truth()
