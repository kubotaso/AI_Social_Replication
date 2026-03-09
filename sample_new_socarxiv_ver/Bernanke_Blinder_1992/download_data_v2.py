"""
Download and build complete dataset for BB1992 replication.
V2: Uses confirmed working FRED codes, adds missing series.
"""

import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
import warnings
warnings.filterwarnings('ignore')

START = '1947-01-01'
END = '1990-12-31'

# All confirmed working FRED series
FRED_SERIES = {
    # Real activity
    'INDPRO': 'industrial_production',
    'CUMFNS': 'capacity_utilization',
    'PAYEMS': 'employment',
    'HOUST': 'housing_starts',
    'RSALES': 'retail_sales_nominal',        # Retail sales (nominal)
    'PI': 'personal_income_nominal',
    'PCE': 'consumption_nominal',
    # Price
    'CPIAUCSL': 'cpi',
    # Monetary aggregates
    'M1SL': 'm1',
    'M2SL': 'm2',
    # Interest rates
    'FEDFUNDS': 'funds_rate',
    'TB3MS': 'tbill_3m',
    'TB6MS': 'tbill_6m',
    'GS1': 'treasury_1y',
    'GS10': 'treasury_10y',
    'CP6M': 'cpaper_6m',                    # Only from 1970
    'MPRIME': 'prime_rate',                  # Bank prime rate (from 1949)
    # Unemployment
    'LNS14000061': 'unemp_male_2554',
    'UNRATE': 'unemp_rate',
    # Bank balance sheet
    'LOANS': 'bank_loans',
    'INVEST': 'bank_investments',
    'TCDSL': 'bank_deposits_check',
    'H8B1001NCBCMG': 'bank_credit_total',   # Total bank credit (from 1947)
    'DPSACBM027SBOG': 'bank_deposits_total', # Total deposits (from 1973)
    # Reserves
    'BOGNONBR': 'nonborrowed_reserves',
    'TOTRESNS': 'total_reserves',
    'REQRESNS': 'required_reserves',
}

# Additional attempts for durable goods orders
DURABLE_ATTEMPTS = [
    'DGORDER',      # Modern series
    'A33DNO',       # New Orders Nondefense Capital Goods
    'AMDMNO',       # Mfrs New Orders Durable Goods
    'AMDMUO',       # Mfrs Unfilled Orders Durable Goods
    'AMTUNO',       # Total Mfrs Unfilled Orders
    'UMDMNO',       # Value of Mfrs New Orders Durable
    'ANDENO',       # New Orders for Nondefense Capital Goods excl Aircraft
    'ACOGNO',       # New Orders for Consumer Goods
    'AMNMNO',       # Mfrs New Orders Nondurable
    'NAPM',         # ISM Manufacturing PMI (proxy)
    'NAPMNOI',      # ISM New Orders Index
]

def download_all():
    results = {}
    print("Phase 1: Downloading main series...")
    for code, name in FRED_SERIES.items():
        try:
            df = pdr.DataReader(code, 'fred', START, END)
            results[name] = df.iloc[:, 0]
            n = df.iloc[:, 0].dropna().shape[0]
            print(f"  OK: {code} -> {name} ({n} obs)")
        except Exception as e:
            print(f"  FAIL: {code} -> {name}: {e}")

    print("\nPhase 2: Trying durable goods orders alternatives...")
    for code in DURABLE_ATTEMPTS:
        try:
            df = pdr.DataReader(code, 'fred', START, END)
            n = df.iloc[:, 0].dropna().shape[0]
            first = df.first_valid_index()
            print(f"  OK: {code} -> {n} obs from {first.strftime('%Y-%m')}")
            if first.year <= 1959:
                results['durable_goods_orders_nominal'] = df.iloc[:, 0]
                print(f"  ** Using {code} for durable goods orders **")
                break
            else:
                print(f"  (starts too late, need pre-1959)")
        except Exception as e:
            print(f"  FAIL: {code}: {str(e)[:60]}")

    # Build DataFrame
    print("\nPhase 3: Building DataFrame...")
    df = pd.DataFrame(results)
    df.index.name = 'date'
    df = df.resample('MS').first()

    # Construct derived variables
    print("\nPhase 4: Constructing derived variables...")

    # CPI deflation
    cpi = df['cpi']
    cpi_82 = cpi.loc['1982-01-01':'1984-12-01'].mean()
    deflator = cpi / cpi_82

    # Real variables
    for nom, real in [
        ('retail_sales_nominal', 'retail_sales_real'),
        ('personal_income_nominal', 'personal_income_real'),
        ('consumption_nominal', 'consumption_real'),
    ]:
        if nom in df.columns:
            df[real] = df[nom] / deflator

    if 'durable_goods_orders_nominal' in df.columns:
        df['durable_goods_real'] = df['durable_goods_orders_nominal'] / deflator

    # Spreads
    df['ffbond'] = df['funds_rate'] - df['treasury_10y']
    if 'cpaper_6m' in df.columns and 'tbill_6m' in df.columns:
        df['cpbill'] = df['cpaper_6m'] - df['tbill_6m']
    if 'treasury_10y' in df.columns and 'treasury_1y' in df.columns:
        df['term'] = df['treasury_10y'] - df['treasury_1y']

    # Log transformations
    log_vars = [
        'industrial_production', 'capacity_utilization', 'employment',
        'housing_starts', 'retail_sales_real', 'personal_income_real',
        'durable_goods_real', 'consumption_real', 'cpi', 'm1', 'm2',
        'bank_loans', 'bank_investments', 'bank_deposits_check',
        'bank_credit_total', 'bank_deposits_total',
        'nonborrowed_reserves', 'total_reserves', 'required_reserves',
    ]
    for var in log_vars:
        if var in df.columns:
            valid = df[var] > 0
            df.loc[valid, f'log_{var}'] = np.log(df.loc[valid, var])

    # Bank balance sheet derived: securities = total loans and investments - loans
    # In the paper: deposits = total deposits, securities = total loans and investments - loans
    # Using LOANS and INVEST directly (INVEST = investment securities at commercial banks)
    # The paper defines: securities = total loans and investments - total loans
    # bank_credit_total is total bank credit (loans + investments)
    # So securities = bank_credit_total - bank_loans
    if 'bank_credit_total' in df.columns and 'bank_loans' in df.columns:
        df['bank_securities'] = df['bank_credit_total'] - df['bank_loans']
        valid = df['bank_securities'] > 0
        df.loc[valid, 'log_bank_securities'] = np.log(df.loc[valid, 'bank_securities'])

    # Real bank variables (deflated by CPI)
    for bvar in ['bank_loans', 'bank_investments', 'bank_deposits_check', 'bank_securities', 'bank_deposits_total']:
        if bvar in df.columns:
            df[f'{bvar}_real'] = df[bvar] / deflator
            valid = df[f'{bvar}_real'] > 0
            df.loc[valid, f'log_{bvar}_real'] = np.log(df.loc[valid, f'{bvar}_real'])

    # Real nonborrowed reserves
    if 'nonborrowed_reserves' in df.columns:
        df['nonborrowed_reserves_real'] = df['nonborrowed_reserves'] / deflator
        valid = df['nonborrowed_reserves_real'] > 0
        df.loc[valid, 'log_nonborrowed_reserves_real'] = np.log(df.loc[valid, 'nonborrowed_reserves_real'])

    # Save
    outfile = 'bb1992_data.csv'
    df.to_csv(outfile)
    print(f"\nSaved: {outfile}")
    print(f"Shape: {df.shape}")
    print(f"Range: {df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}")

    # Coverage report
    print("\n" + "="*60)
    print("COVERAGE REPORT")
    print("="*60)

    key_periods = {
        'Tables 1-2 (1959:7-1989:12)': ('1959-07', '1989-12'),
        'Tables 3-4 (1961:7-1989:12)': ('1961-07', '1989-12'),
        'Table 5-6 (1959:8-1979:9)': ('1959-08', '1979-09'),
        'Figure 4 (1959:1-1978:12)': ('1959-01', '1978-12'),
    }

    key_vars = {
        'Tables 1-2': ['log_industrial_production', 'log_capacity_utilization',
                       'log_employment', 'unemp_male_2554', 'log_housing_starts',
                       'log_personal_income_real', 'log_retail_sales_real',
                       'log_consumption_real', 'log_cpi', 'log_m1', 'log_m2',
                       'funds_rate', 'tbill_3m', 'treasury_10y'],
        'Tables 3-4': ['cpbill', 'term', 'funds_rate'],
        'Figure 4': ['funds_rate', 'unemp_male_2554', 'log_cpi',
                     'log_bank_loans_real', 'log_bank_investments_real', 'log_bank_deposits_check_real'],
    }

    for label, (start, end) in key_periods.items():
        print(f"\n{label}:")
        for var_group, vars_list in key_vars.items():
            if label.startswith(var_group.split()[0]):
                for var in vars_list:
                    if var in df.columns:
                        n = df.loc[start:end, var].dropna().shape[0]
                        expected = len(pd.date_range(start, end, freq='MS'))
                        status = "OK" if n >= expected * 0.95 else f"SHORT ({n}/{expected})"
                        print(f"  {var}: {n} obs - {status}")
                    else:
                        print(f"  {var}: MISSING")

    return df


if __name__ == '__main__':
    df = download_all()
