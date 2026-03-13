"""Debug4: Check original EVS Stata file for Hungary 1990 and Germany East/West."""
import pandas as pd
import numpy as np

# Load the original EVS Stata file
evs_orig = pd.read_stata('data/ZA4460_v3-0-0.dta', convert_categoricals=False)
print("EVS Stata shape:", evs_orig.shape)
print("EVS Stata columns count:", len(evs_orig.columns))

# Check for church attendance variable
church_cols = [c for c in evs_orig.columns if 'church' in c.lower() or 'attend' in c.lower() or 'relig' in c.lower()[:10]]
print("\nChurch/attend/relig columns:", church_cols[:20])

# Check for variables starting with F
f_cols = [c for c in evs_orig.columns if c.startswith('F') or c.startswith('f')]
print("\nF-prefix columns:", sorted(f_cols)[:30])

# Check for known variable names
for v in ['F028', 'f028', 'V147', 'v147', 'F063', 'f063', 'V148', 'v148',
          'V24', 'v24', 'Q1', 'Q2', 'V22', 'v22']:
    if v in evs_orig.columns:
        print(f"  Found: {v}")

# Check country variable
country_cols = [c for c in evs_orig.columns if 'country' in c.lower() or 'nation' in c.lower() or c.upper() == 'S003']
print("\nCountry columns:", country_cols[:10])

# Try to find the church attendance variable by looking at all column names
print("\nAll columns (first 100):")
print(list(evs_orig.columns)[:100])
