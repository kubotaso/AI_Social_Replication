import pandas as pd
import numpy as np

evs_long = pd.read_stata('data/ZA4460_v3-0-0.dta', convert_categoricals=False)
de = evs_long[evs_long['country'] == 276]

# Split East/West using c_abrv1
de_west = de[de['c_abrv1'] == 'DE-W']
de_east = de[de['c_abrv1'] == 'DE-E']

print(f"West Germany N: {len(de_west)}")
print(f"East Germany N: {len(de_east)}")

# Church attendance using q336
for label, subset in [('West', de_west), ('East', de_east), ('All', de)]:
    print(f"\n{label} Germany q336:")
    print(subset['q336'].value_counts().sort_index())
    valid = subset[subset['q336'].isin([1,2,3,4,5,6,7,8])]
    monthly = subset[subset['q336'].isin([1,2,3])]
    if len(valid) > 0:
        pct = len(monthly)/len(valid)*100
        print(f"  % monthly: {pct:.1f}% (N={len(valid)})")
        print(f"  % monthly (rounded): {round(pct)}")

# Also check: what if we use the EVS formatted data with the c_abrv1 split
# The EVS_1990_wvs_format.csv has F063 and COUNTRY_ALPHA
evs = pd.read_csv('data/EVS_1990_wvs_format.csv', low_memory=False)
# Does it have a region variable?
print(f"\nEVS formatted columns: {list(evs.columns)}")
de_evs = evs[evs['COUNTRY_ALPHA'] == 'DEU']
print(f"EVS Germany N: {len(de_evs)}")
print(f"F063 for Germany:")
print(de_evs['F063'].value_counts().sort_index())
valid = de_evs[de_evs['F063'].isin([1,2,3,4,5,6,7,8])]
monthly = de_evs[de_evs['F063'].isin([1,2,3])]
print(f"Overall: {len(monthly)/len(valid)*100:.1f}%")

# Check if the EVS data has separate East/West entries
print(f"\nAll unique COUNTRY_ALPHA values:")
print(sorted(evs['COUNTRY_ALPHA'].unique()))
