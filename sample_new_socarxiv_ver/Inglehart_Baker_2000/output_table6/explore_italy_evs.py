"""
Deep dive into Italy Wave 2 EVS data
"""
import pandas as pd
import numpy as np
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
evs_path = os.path.join(base, "data", "ZA4460_v3-0-0.dta")

evs = pd.read_stata(evs_path, convert_categoricals=False)
print("EVS columns sample:", list(evs.columns[:30]))
print()

# Find country column
country_cols = [c for c in evs.columns if 'country' in c.lower() or c.startswith('c_') or c == 'country1']
print("Country cols:", country_cols)

# Find the S003-equivalent
s003_cols = [c for c in evs.columns if 's003' in c.lower() or 'S003' in c]
print("S003 cols:", s003_cols)

# Show all column names to find correct country identifier
print("\nAll EVS columns:")
for i, c in enumerate(evs.columns):
    print(f"  {i}: {c}")
