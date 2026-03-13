"""
Investigate specific discrepancies in Table 8 replication:
1. India 1990-91: our 34 vs paper 28 - big miss
2. Rounding issues for partials
3. Weight variations
"""
import pandas as pd
import numpy as np
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
wvs_path = os.path.join(base, 'data', 'WVS_Time_Series_1981-2022_csv_v5_0.csv')
evs_path = os.path.join(base, 'data', 'ZA4460_v3-0-0.dta')

wvs = pd.read_csv(wvs_path, usecols=['S002VS', 'S003', 'COUNTRY_ALPHA', 'F001', 'S020', 'S017', 'X048WVS'])

def pct_weighted(f001, weight):
    mask = f001 > 0
    f = f001[mask]
    w = weight[mask].fillna(1)
    if len(f) == 0: return None, 0
    pct = ((f == 1) * w).sum() / w.sum() * 100
    return pct, len(f)

def pct_unweighted(f001):
    valid = f001[f001 > 0]
    if len(valid) == 0: return None, 0
    return (valid == 1).mean() * 100, len(valid)

# ========== INDIA INVESTIGATION ==========
print("=" * 70)
print("INDIA Wave 2 (1990-1991) Investigation")
print("=" * 70)
india_w2 = wvs[(wvs['COUNTRY_ALPHA'] == 'IND') & (wvs['S002VS'] == 2)]
print(f"India W2 total rows: {len(india_w2)}")
print(f"F001 distribution:")
print(india_w2['F001'].value_counts().sort_index())
print()

# Try different approaches
pct_w, n_w = pct_weighted(india_w2['F001'], india_w2['S017'])
pct_u, n_u = pct_unweighted(india_w2['F001'])
print(f"Weighted (S017): {pct_w:.2f}% (n={n_w})")
print(f"Unweighted: {pct_u:.2f}% (n={n_u})")

# Check S017 distribution for India
print(f"\nS017 stats for India W2:")
print(india_w2['S017'].describe())
print(f"S017 unique values: {sorted(india_w2['S017'].dropna().unique())[:20]}")

# Maybe India has sub-samples? Check X048WVS
print(f"\nX048WVS (region) for India W2:")
print(india_w2['X048WVS'].value_counts().sort_index().head(20))

# Check if there's a subsample issue - perhaps the paper used only urban/literate sample
# India 1990 WVS was conducted in urban areas only in some versions
# Try with only certain regions
for region_code in india_w2['X048WVS'].unique()[:5]:
    sub = india_w2[india_w2['X048WVS'] == region_code]
    if len(sub) > 50:
        pct, n = pct_unweighted(sub['F001'])
        if pct is not None:
            print(f"  Region {region_code}: {pct:.1f}% (n={n})")

# ========== ROUNDING INVESTIGATION ==========
print()
print("=" * 70)
print("ROUNDING INVESTIGATION - All partial matches")
print("=" * 70)

# For each country/period with a partial match, show the exact floating-point percentage
# to see if different rounding could help

evs = pd.read_stata(evs_path, convert_categoricals=False)

partials = {
    ('Japan', 'W3', '1995-1998'): ('wvs', 'JPN', 3, 26),
    ('Spain', 'W2-EVS', '1990-1991'): ('evs', 'ES', None, 27),
    ('China', 'W2', '1990-1991'): ('wvs', 'CHN', 2, 30),
    ('Lithuania', 'W3', '1995-1998'): ('wvs', 'LTU', 3, 42),
    ('Brazil', 'W2', '1990-1991'): ('wvs', 'BRA', 2, 44),
    ('Chile', 'W2', '1990-1991'): ('wvs', 'CHL', 2, 54),
    ('India', 'W3', '1995-1998'): ('wvs', 'IND', 3, 23),
    ('Nigeria', 'W2', '1990-1991'): ('wvs', 'NGA', 2, 60),
    ('Nigeria', 'W3', '1995-1998'): ('wvs', 'NGA', 3, 50),
    ('South Africa', 'W1', '1981'): ('wvs', 'ZAF', 1, 38),
    ('Turkey', 'W3', '1995-1998'): ('wvs', 'TUR', 3, 50),
    ('West Germany', 'net', 'net_change'): ('derived', None, None, 12),
}

print(f"\n{'Country':<22} {'Period':<12} {'Paper':>6} {'Weighted':>10} {'Unweighted':>12} {'Weight_needed':>14}")
print("-" * 80)

for (country, source, period), (dtype, code, wave, paper_val) in partials.items():
    if dtype == 'wvs' and wave is not None:
        sub = wvs[(wvs['COUNTRY_ALPHA'] == code) & (wvs['S002VS'] == wave)]
        pw, nw = pct_weighted(sub['F001'], sub['S017'])
        pu, nu = pct_unweighted(sub['F001'])
        pw_str = f"{pw:.2f}" if pw else "N/A"
        pu_str = f"{pu:.2f}" if pu else "N/A"
        print(f"{country:<22} {period:<12} {paper_val:>6} {pw_str:>10} {pu_str:>12}")
    elif dtype == 'evs':
        sub = evs[evs['c_abrv'] == code]
        pu, nu = pct_unweighted(sub['q322'])
        # Also try with weight_g
        if 'weight_g' in evs.columns:
            mask = sub['q322'] > 0
            q = sub.loc[mask, 'q322']
            w = sub.loc[mask, 'weight_g'].fillna(1)
            pw = ((q == 1) * w).sum() / w.sum() * 100
        else:
            pw = None
        pw_str = f"{pw:.2f}" if pw else "N/A"
        pu_str = f"{pu:.2f}" if pu else "N/A"
        print(f"{country:<22} {period:<12} {paper_val:>6} {pw_str:>10} {pu_str:>12}")

# ========== West Germany W3 investigation ==========
print()
print("=" * 70)
print("WEST GERMANY W3 Investigation")
print("=" * 70)
deu_w3 = wvs[(wvs['COUNTRY_ALPHA'] == 'DEU') & (wvs['S002VS'] == 3)]
print(f"Total Germany W3: {len(deu_w3)}")
print(f"X048WVS values: {sorted(deu_w3['X048WVS'].unique())}")

# Try different West Germany definitions
west_standard = [276001,276002,276003,276004,276005,276006,276007,276008,276009,276010,276019]
west_no_berlin = [276001,276002,276003,276004,276005,276006,276007,276008,276009,276010]
west_with_all_berlin = [276001,276002,276003,276004,276005,276006,276007,276008,276009,276010,276019,276020]
east_standard = [276012,276013,276014,276015,276016]
east_with_berlin = [276012,276013,276014,276015,276016,276020]

for name, codes in [('West (std)', west_standard), ('West (no Berlin)', west_no_berlin),
                     ('West (all Berlin)', west_with_all_berlin),
                     ('East (std)', east_standard), ('East (with Berlin)', east_with_berlin)]:
    sub = deu_w3[deu_w3['X048WVS'].isin(codes)]
    pw, nw = pct_weighted(sub['F001'], sub['S017'])
    pu, nu = pct_unweighted(sub['F001'])
    pw_str = f"{pw:.2f}" if pw else "N/A"
    pu_str = f"{pu:.2f}" if pu else "N/A"
    print(f"  {name:<25}: weighted={pw_str}, unweighted={pu_str} (n={nu})")

# For West Germany, paper says 41. We get 41 unweighted, 41 weighted. Let's check net change
# W3 West = 41 (matches), EVS 1990 West = 30 (matches)
# So net change should be 41-30 = 11. Paper says 12 with 1981=29.
# Net change is 41-29=12 but we only have 30 and 41, giving 11.
print("  => West Germany net_change: 41 - 30 = 11. Paper: 41 - 29 = 12.")

# ========== East Germany W3 investigation ==========
print()
print("=" * 70)
print("EAST GERMANY W3 Investigation")
print("=" * 70)
# Paper says 47
e_sub = deu_w3[deu_w3['X048WVS'].isin(east_standard)]
pw, nw = pct_weighted(e_sub['F001'], e_sub['S017'])
pu, nu = pct_unweighted(e_sub['F001'])
print(f"  East (std): weighted={pw:.2f}, unweighted={pu:.2f} (n={nu})")

e_sub2 = deu_w3[deu_w3['X048WVS'].isin(east_with_berlin)]
pw2, nw2 = pct_weighted(e_sub2['F001'], e_sub2['S017'])
pu2, nu2 = pct_unweighted(e_sub2['F001'])
print(f"  East (with Berlin-E): weighted={pw2:.2f}, unweighted={pu2:.2f} (n={nu2})")

# Also check if using S017 vs unweighted changes things
# We want 47 for East Germany
print(f"  Paper target: 47")
