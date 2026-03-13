"""Debug5: Find church attendance variable in original EVS Stata file."""
import pandas as pd
import numpy as np

evs_orig = pd.read_stata('data/ZA4460_v3-0-0.dta', convert_categoricals=False)

# The variable we need is about church attendance frequency
# In WVS it's F028 (or F063 in some EVS mappings)
# Let's check variables around q320-q400 range for religious questions
# Also check if there's a codebook-referenced variable
print("All columns sorted:")
all_cols = sorted(evs_orig.columns)
print(all_cols)
