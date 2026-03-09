"""
Download all required data for Bernanke and Blinder (1992) replication from FRED.
Uses pandas-datareader (no API key needed for FRED web reader).
"""

import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
import warnings
warnings.filterwarnings('ignore')

START = '1947-01-01'  # Get extra early data for safety
END = '1990-12-31'

# FRED series mapping
FRED_SERIES = {
    # Real activity variables
    'INDPRO': 'industrial_production',      # Industrial Production Index
    'CUMFNS': 'capacity_utilization',       # Capacity Utilization: Manufacturing
    'PAYEMS': 'employment',                 # All Employees: Total Nonfarm
    'HOUST': 'housing_starts',              # Housing Starts: Total
    'RSAFS': 'retail_sales_nominal',        # Advance Retail Sales: Retail and Food Services
    'PI': 'personal_income_nominal',        # Personal Income
    'DGORDER': 'durable_goods_orders_nominal',  # Manufacturers New Orders: Durable Goods
    'PCE': 'consumption_nominal',           # Personal Consumption Expenditures

    # Price level
    'CPIAUCSL': 'cpi',                      # Consumer Price Index for All Urban Consumers

    # Monetary aggregates
    'M1SL': 'm1',                           # M1 Money Stock
    'M2SL': 'm2',                           # M2 Money Stock

    # Interest rates
    'FEDFUNDS': 'funds_rate',               # Effective Federal Funds Rate
    'TB3MS': 'tbill_3m',                    # 3-Month Treasury Bill
    'TB6MS': 'tbill_6m',                    # 6-Month Treasury Bill
    'GS1': 'treasury_1y',                   # 1-Year Treasury Constant Maturity
    'GS10': 'treasury_10y',                 # 10-Year Treasury Constant Maturity

    # Unemployment
    'LNS14000061': 'unemp_male_2554',       # Unemployment Rate - Men 25-54
    'UNRATE': 'unemp_rate',                 # Civilian Unemployment Rate (backup)

    # Commercial paper rate
    'CP6M': 'cpaper_6m',                    # 6-Month AA Financial Commercial Paper Rate

    # Bank balance sheet (try multiple series)
    'LOANS': 'bank_loans',                  # Loans and Leases in Bank Credit
    'INVEST': 'bank_investments',           # Investments in Bank Credit (securities)
    'TCDSL': 'bank_deposits_check',         # Total Checkable Deposits

    # Reserves
    'BOGNONBR': 'nonborrowed_reserves',     # Non-Borrowed Reserves
    'TOTRESNS': 'total_reserves',           # Total Reserves
    'REQRESNS': 'required_reserves',        # Required Reserves
}

def download_fred_data():
    """Download all series from FRED."""
    results = {}
    failed = []

    for fred_code, name in FRED_SERIES.items():
        try:
            df = pdr.DataReader(fred_code, 'fred', START, END)
            results[name] = df[fred_code]
            print(f"  OK: {fred_code} -> {name} ({len(df)} obs, {df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')})")
        except Exception as e:
            failed.append((fred_code, name, str(e)))
            print(f"  FAIL: {fred_code} -> {name}: {e}")

    return results, failed


def try_alternative_series(failed_names, results):
    """Try alternative FRED codes for failed series."""
    alternatives = {
        'cpaper_6m': [
            ('WCP6M', 'fred'),       # 6-month commercial paper (weekly -> monthly)
            ('DCPN6M', 'fred'),      # 6-month nonfinancial CP
            ('CPN6M', 'fred'),       # Another CP code
        ],
        'bank_loans': [
            ('TOTLL', 'fred'),       # Total Loans and Leases
            ('TOTLLNSA', 'fred'),
            ('H8B1023NCBCMG', 'fred'),  # H.8 Loans
        ],
        'bank_investments': [
            ('USGSEC', 'fred'),      # U.S. Government Securities at All Commercial Banks
            ('H8B1058NCBCMG', 'fred'),
        ],
        'bank_deposits_check': [
            ('DPSACBM027SBOG', 'fred'),  # Deposits at All Commercial Banks
            ('TCDSL', 'fred'),
            ('DEMDEPSL', 'fred'),        # Demand Deposits
        ],
        'nonborrowed_reserves': [
            ('NONBORRES', 'fred'),
            ('BOGNONBR', 'fred'),
        ],
    }

    for fred_code, name, error in failed_names:
        if name in alternatives and name not in results:
            for alt_code, src in alternatives[name]:
                try:
                    df = pdr.DataReader(alt_code, src, START, END)
                    results[name] = df.iloc[:, 0]
                    print(f"  ALT OK: {alt_code} -> {name} ({len(df)} obs)")
                    break
                except Exception as e:
                    print(f"  ALT FAIL: {alt_code} -> {name}: {e}")


def construct_derived_variables(df):
    """Construct derived variables needed for the paper."""
    # FFBOND = funds rate - 10-year bond rate
    if 'funds_rate' in df.columns and 'treasury_10y' in df.columns:
        df['ffbond'] = df['funds_rate'] - df['treasury_10y']

    # CPBILL = 6-month commercial paper rate - 6-month T-bill rate
    if 'cpaper_6m' in df.columns and 'tbill_6m' in df.columns:
        df['cpbill'] = df['cpaper_6m'] - df['tbill_6m']
    elif 'cpaper_6m' in df.columns and 'tbill_3m' in df.columns:
        # Approximate with 3-month T-bill if 6-month unavailable
        df['cpbill'] = df['cpaper_6m'] - df['tbill_3m']
        print("  NOTE: CPBILL approximated using 3-month T-bill rate")

    # TERM = 10-year - 1-year Treasury rate
    if 'treasury_10y' in df.columns and 'treasury_1y' in df.columns:
        df['term'] = df['treasury_10y'] - df['treasury_1y']

    # Real variables (deflate by CPI, base period doesn't matter for logs)
    if 'cpi' in df.columns:
        cpi = df['cpi']
        cpi_base = cpi.loc['1982-01-01':'1984-12-01'].mean() if '1982-01-01' in cpi.index else cpi.mean()
        deflator = cpi / cpi_base

        nominal_to_real = {
            'retail_sales_nominal': 'retail_sales_real',
            'personal_income_nominal': 'personal_income_real',
            'durable_goods_orders_nominal': 'durable_goods_real',
            'consumption_nominal': 'consumption_real',
        }

        for nom, real in nominal_to_real.items():
            if nom in df.columns:
                df[real] = df[nom] / deflator

    # Log transformations for real activity variables and monetary aggregates
    log_vars = [
        'industrial_production', 'capacity_utilization', 'employment',
        'housing_starts', 'retail_sales_real', 'personal_income_real',
        'durable_goods_real', 'consumption_real', 'cpi', 'm1', 'm2',
        'bank_loans', 'bank_investments', 'bank_deposits_check',
    ]
    for var in log_vars:
        if var in df.columns:
            df[f'log_{var}'] = np.log(df[var])

    return df


def main():
    print("=" * 60)
    print("Downloading FRED data for Bernanke & Blinder (1992)")
    print("=" * 60)

    # Download
    print("\nPhase 1: Primary FRED downloads...")
    results, failed = download_fred_data()

    # Try alternatives for failed series
    if failed:
        print(f"\nPhase 2: Trying alternatives for {len(failed)} failed series...")
        try_alternative_series(failed, results)

    # Combine into DataFrame
    print("\nPhase 3: Combining data...")
    df = pd.DataFrame(results)
    df.index.name = 'date'

    # Ensure monthly frequency
    df = df.resample('MS').first()

    # Construct derived variables
    print("\nPhase 4: Constructing derived variables...")
    df = construct_derived_variables(df)

    # Save
    output_file = 'bb1992_data.csv'
    df.to_csv(output_file)
    print(f"\nSaved to {output_file}")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}")
    print(f"\nColumns ({len(df.columns)}):")
    for col in sorted(df.columns):
        non_null = df[col].notna().sum()
        first_valid = df[col].first_valid_index()
        last_valid = df[col].last_valid_index()
        if first_valid is not None:
            print(f"  {col}: {non_null} obs, {first_valid.strftime('%Y-%m')} to {last_valid.strftime('%Y-%m')}")
        else:
            print(f"  {col}: NO DATA")

    # Check coverage for key sample periods
    print("\n" + "=" * 60)
    print("Sample period coverage check")
    print("=" * 60)

    key_vars_table1 = ['log_industrial_production', 'log_cpi', 'log_m1', 'log_m2', 'funds_rate', 'tbill_3m', 'treasury_10y']
    for var in key_vars_table1:
        if var in df.columns:
            subset = df.loc['1959-07':'1989-12', var].dropna()
            print(f"  {var}: {len(subset)} obs in 1959:7-1989:12")
        else:
            print(f"  {var}: MISSING")

    print("\nDone!")
    return df


if __name__ == '__main__':
    df = main()
