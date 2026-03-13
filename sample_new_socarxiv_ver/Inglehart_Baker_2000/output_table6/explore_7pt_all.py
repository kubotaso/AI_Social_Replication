import pandas as pd
import numpy as np

evs_long = pd.read_stata('data/ZA4460_v3-0-0.dta', convert_categoricals=False)

# Check: for which countries does excluding value 8 improve the match?
paper_vals = {
    'BE': ('Belgium', 35), 'BG': ('Bulgaria', 9), 'CA': ('Canada', 40),
    'FI': ('Finland', 13), 'FR': ('France', 17),
    'GB-GBN': ('Great Britain', 25), 'GB-NIR': ('Northern Ireland', 69),
    'HU': ('Hungary', 34), 'IE': ('Ireland', 88), 'IS': ('Iceland', 9),
    'IT': ('Italy', 47), 'LV': ('Latvia', 9), 'NL': ('Netherlands', 31),
    'NO': ('Norway', 13), 'PL': ('Poland', 85), 'SE': ('Sweden', 10),
    'SI': ('Slovenia', 35), 'US': ('United States', 59),
}

print(f"{'Country':<22} {'Paper':>5} {'8pt':>5} {'7pt(-8)':>7} {'8pt_diff':>8} {'7pt_diff':>8} {'Better':>7}")
print("-" * 70)

for alpha, (name, paper_val) in sorted(paper_vals.items(), key=lambda x: x[1][0]):
    sub = evs_long[evs_long['c_abrv'] == alpha]
    if len(sub) == 0:
        continue

    valid_8 = sub[sub['q336'].isin([1,2,3,4,5,6,7,8])]
    valid_7 = sub[sub['q336'].isin([1,2,3,4,5,6,7])]
    monthly = sub[sub['q336'].isin([1,2,3])]

    pct_8 = round(len(monthly)/len(valid_8)*100) if len(valid_8) > 0 else None
    pct_7 = round(len(monthly)/len(valid_7)*100) if len(valid_7) > 0 else None

    d8 = abs(pct_8 - paper_val) if pct_8 is not None else 99
    d7 = abs(pct_7 - paper_val) if pct_7 is not None else 99

    better = "7pt" if d7 < d8 else ("same" if d7 == d8 else "8pt")

    print(f"  {name:<20} {paper_val:>5} {pct_8:>5} {pct_7:>7} {d8:>8} {d7:>8} {better:>7}")

# Also check Germany split
de = evs_long[evs_long['country'] == 276]
for region, c1 in [('West', 900), ('East', 901)]:
    sub = de[de['country1'] == c1]
    valid_8 = sub[sub['q336'].isin([1,2,3,4,5,6,7,8])]
    valid_7 = sub[sub['q336'].isin([1,2,3,4,5,6,7])]
    monthly = sub[sub['q336'].isin([1,2,3])]

    pct_8 = round(len(monthly)/len(valid_8)*100) if len(valid_8) > 0 else None
    pct_7 = round(len(monthly)/len(valid_7)*100) if len(valid_7) > 0 else None

    paper = {'West': 33, 'East': 20}[region]
    d8 = abs(pct_8 - paper) if pct_8 is not None else 99
    d7 = abs(pct_7 - paper) if pct_7 is not None else 99

    better = "7pt" if d7 < d8 else ("same" if d7 == d8 else "8pt")
    print(f"  {region+' Germany':<20} {paper:>5} {pct_8:>5} {pct_7:>7} {d8:>8} {d7:>8} {better:>7}")
