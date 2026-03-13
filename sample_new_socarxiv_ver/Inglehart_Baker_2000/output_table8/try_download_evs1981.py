"""Try to download EVS 1981 data or find it in existing datasets"""
import os
import subprocess

# Check if we can find EVS data through the WVS/EVS Joint dataset
# The WVS Time Series v5 might include EVS data for Wave 1 in some versions
# Let's check what's in the WVS data more carefully

import pandas as pd

# Strategy 1: Check if there's a WVS_EVS joint file we haven't noticed
data_dir = 'data'
for f in os.listdir(data_dir):
    print(f"Data file: {f} ({os.path.getsize(os.path.join(data_dir,f))/(1024*1024):.1f} MB)")

# Strategy 2: Check the ZA4460 codebook for references to 1981 data
print("\nChecking ZA4460 codebook for 1981 references...")

# Strategy 3: Look at the WVS Variable Report for clues about EVS 1981
print("\nChecking WVS variable report...")

# Strategy 4: Try to use the EVS trend file (ZA4804) which integrates waves 1-4
# Or try ZA7500 (EVS Trend File 1981-2017)
print("\nNote: EVS 1981 data is available as ZA4438 or part of ZA7500 (EVS Trend File)")
print("These require GESIS registration to download.")
print("Without this data, we cannot compute 1981 values for European countries.")

# Strategy 5: Check if the existing WVS Time Series has hidden EVS data
# Some versions integrate EVS data. Let's check for country codes that would
# indicate EVS 1981 participants
wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                   usecols=['S002VS','S003','COUNTRY_ALPHA','F001','S020'])

# Check for European countries in any wave with 1981 year
europe_1981 = wvs[(wvs['S020']==1981) | (wvs['S020']==1982)]
print(f"\nWVS records from 1981-1982:")
print(europe_1981.groupby(['COUNTRY_ALPHA','S002VS','S020']).size().reset_index(name='n'))

# Check for Belgium, France, Germany, GB, etc. in any wave
target_countries = ['BEL','CAN','FRA','DEU','GBR','ISL','IRL','ITA','NLD','NOR','ESP','SWE','USA']
for cc in target_countries:
    sub = wvs[(wvs['COUNTRY_ALPHA']==cc) & (wvs['S002VS'].isin([1,2]))]
    if len(sub) > 0:
        f_valid = (sub['F001'] > 0).sum()
        print(f"{cc} waves 1-2: n={len(sub)}, F001 valid={f_valid}, years={sorted(sub['S020'].unique())}")

print("\n\nConclusion: EVS 1981 data is NOT available in our current datasets.")
print("The paper used EVS 1981 data for European countries in the 1981 column.")
print("We need ZA4438 or ZA7500 from GESIS to get this data.")
