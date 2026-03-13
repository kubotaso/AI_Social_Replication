"""
Check what columns exist in the WVS Time Series CSV
"""
import pandas as pd
import numpy as np
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
wvs_path = os.path.join(base, "data", "WVS_Time_Series_1981-2022_csv_v5_0.csv")

# Read just the header
wvs_header = pd.read_csv(wvs_path, nrows=0)
cols = list(wvs_header.columns)
print(f"Total columns: {len(cols)}")
print()
print("All columns:")
for i, c in enumerate(cols):
    print(f"  {i+1}: {c}")
