import pandas as pd
import numpy as np

evs_long = pd.read_stata('data/ZA4460_v3-0-0.dta', convert_categoricals=False)
print(f"ZA4460 shape: {evs_long.shape}")
print(f"Year range: {evs_long['year'].min()} to {evs_long['year'].max()}")
print(f"Unique years: {sorted(evs_long['year'].unique())}")
print(f"studyno unique: {sorted(evs_long['studyno'].unique())}")

# Check studyno distribution
print("\nstudyno value counts:")
print(evs_long['studyno'].value_counts().sort_index())

# Check if version column provides info
print(f"\nversion unique: {sorted(evs_long['version'].unique())}")

# This file is ZA4460 which is EVS 1990 (2nd wave)
# The 1st wave is ZA4438
# But our file only contains 1990-1993 data

# Let me check if there's any indication of wave 1 data
print(f"\nAll years and countries:")
for yr in sorted(evs_long['year'].unique()):
    countries = sorted(evs_long[evs_long['year']==yr]['c_abrv'].unique())
    print(f"  Year {yr}: {countries}")
