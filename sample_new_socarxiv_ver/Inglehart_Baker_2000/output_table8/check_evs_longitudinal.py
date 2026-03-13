"""Check EVS Longitudinal Data File for 1981 wave data"""
import pandas as pd
import os

# Check the small combined file first
path1 = '/Users/kubotaso/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/OldFiles/Replication_Claude_IB_v2/data/wvs_evs_1981_2000_4wave.dta'
print("=" * 70)
print("Checking wvs_evs_1981_2000_4wave.dta")
print("=" * 70)

try:
    df = pd.read_stata(path1, convert_categoricals=False)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    # Check for F001 or meaning of life variable
    for col in df.columns:
        if 'f001' in col.lower() or 'meaning' in col.lower() or 'q322' in col.lower():
            print(f"  Found relevant col: {col}")
    # Check years/waves
    for col in df.columns:
        if 'wave' in col.lower() or 'year' in col.lower() or 's002' in col.lower() or 's020' in col.lower():
            print(f"  {col}: {sorted(df[col].unique())[:20]}")
    # Check country variable
    for col in df.columns:
        if 'country' in col.lower() or 'alpha' in col.lower() or 's003' in col.lower() or 'c_abrv' in col.lower():
            print(f"  {col}: {sorted(df[col].unique())[:30]}")
except Exception as e:
    print(f"Error: {e}")

# Check the full EVS Longitudinal file
path2 = '/Users/kubotaso/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/OldFiles/Replication_Claude_IB/data/ZA4804_v3-1-0.dta/ZA4804_v3-1-0.dta'
print()
print("=" * 70)
print("Checking ZA4804_v3-1-0.dta (EVS Longitudinal)")
print("=" * 70)

try:
    # Read just column names first
    cols = pd.read_stata(path2, convert_categoricals=False, iterator=True)
    columns = cols.variable_labels()
    print(f"Total variables: {len(columns)}")

    # Find F001 or meaning of life variable
    for var, label in columns.items():
        if any(x in var.lower() for x in ['f001', 'v107', 'e032']) or \
           any(x in label.lower() for x in ['meaning', 'purpose of life', 'think about']):
            print(f"  {var}: {label}")

    # Find wave/year variable
    for var, label in columns.items():
        if any(x in var.lower() for x in ['wave', 'studyno', 's002']):
            print(f"  {var}: {label}")

    # Find country variable
    for var, label in columns.items():
        if any(x in var.lower() for x in ['country', 'c_abrv', 's003']):
            print(f"  {var}: {label}")

    # Now read actual data with just the key columns
    # Read a small sample first
    df = pd.read_stata(path2, convert_categoricals=False,
                       columns=['studynoc', 'country', 'c_abrv'] if 'c_abrv' in columns else None)
    print(f"\nShape: {df.shape}")

    # Check studynoc to identify waves
    if 'studynoc' in df.columns:
        print(f"studynoc values: {sorted(df['studynoc'].unique())}")
    if 'wave' in df.columns:
        print(f"wave values: {sorted(df['wave'].unique())}")

except Exception as e:
    print(f"Error reading full file: {e}")
    # Try reading just first 1000 rows
    try:
        df = pd.read_stata(path2, convert_categoricals=False,
                           chunksize=1000)
        chunk = next(df)
        print(f"Chunk shape: {chunk.shape}")
        print(f"Columns (first 30): {list(chunk.columns[:30])}")
        for col in chunk.columns:
            if any(x in col.lower() for x in ['f001', 'v107', 'meaning', 'wave', 'study', 'country', 'c_abrv', 'year']):
                vals = chunk[col].unique()
                print(f"  {col}: {vals[:10]}")
    except Exception as e2:
        print(f"Error reading chunk: {e2}")

# Also check the extracted CSVs
path3 = '/Users/kubotaso/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/OldFiles/Replication_Claude_IB_v3/data/evs_extracted_v2.csv'
print()
print("=" * 70)
print("Checking evs_extracted_v2.csv")
print("=" * 70)

try:
    df = pd.read_csv(path3, nrows=5)
    print(f"Columns: {list(df.columns)}")
    df_full = pd.read_csv(path3)
    print(f"Shape: {df_full.shape}")
    for col in df_full.columns:
        if any(x in col.lower() for x in ['f001', 'meaning', 'wave', 'year', 'country', 's002', 's020']):
            vals = sorted(df_full[col].dropna().unique())
            print(f"  {col}: {vals[:20]}")
except Exception as e:
    print(f"Error: {e}")
