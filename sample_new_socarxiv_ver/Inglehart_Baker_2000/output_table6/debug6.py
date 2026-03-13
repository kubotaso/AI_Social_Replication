"""Debug6: Find church attendance in EVS Stata file - check q320-q340 range."""
import pandas as pd
import numpy as np

evs_orig = pd.read_stata('data/ZA4460_v3-0-0.dta', convert_categoricals=False)

# Church attendance is likely in the religion section
# Check variables around q320-q340 which seem to be in the religion block
# q320 is probably the church attendance question

print("=== q320 ===")
print(evs_orig['q320'].value_counts().sort_index())
print()

print("=== q322 ===")
print(evs_orig['q322'].value_counts().sort_index())
print()

print("=== q323 ===")
print(evs_orig['q323'].value_counts().sort_index())
print()

# Check Hungary for q320
hun = evs_orig[evs_orig['c_abrv'] == 'HU']
if len(hun) == 0:
    # Try different country codes
    for code_col in ['country', 'c_abrv', 'c_abrv1']:
        print(f"\n{code_col} unique values:")
        print(sorted(evs_orig[code_col].dropna().unique())[:30])

print("\n=== HUNGARY q320 ===")
hun = evs_orig[evs_orig['c_abrv'] == 'HU']
if len(hun) == 0:
    hun = evs_orig[evs_orig['country'] == 348]
    if len(hun) == 0:
        # Check country values
        print("Country values:", sorted(evs_orig['country'].dropna().unique()))
print(f"Hungary n={len(hun)}")
if len(hun) > 0:
    print(hun['q320'].value_counts().sort_index())

# Also look at variable labels if available
try:
    evs_labels = pd.read_stata('data/ZA4460_v3-0-0.dta', convert_categoricals=True, iterator=True)
    chunk = evs_labels.get_chunk(1)
    # Try to get variable labels
except:
    pass

# Check with Stata variable labels
try:
    reader = pd.io.stata.StataReader('data/ZA4460_v3-0-0.dta')
    vl = reader.variable_labels()
    # Print labels for religious variables
    for v, label in vl.items():
        if any(kw in label.lower() for kw in ['church', 'attend', 'service', 'religious']):
            print(f"  {v}: {label}")
    reader.close()
except Exception as e:
    print(f"Error reading labels: {e}")
