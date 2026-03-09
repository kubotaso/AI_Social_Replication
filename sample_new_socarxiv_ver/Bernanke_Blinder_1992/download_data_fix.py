"""Fix missing/short series for BB1992 replication."""

import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
import warnings
warnings.filterwarnings('ignore')

START = '1947-01-01'
END = '1990-12-31'

# Try alternative codes for missing series
attempts = {
    'retail_sales': [
        'RETAILSMNSA',    # Retail Sales, not SA
        'RETAILSMSA',     # Retail Sales, SA
        'MRTSSM44000USS', # Monthly Retail Trade
        'RETAILMPCSMSA',  # Retail trade
        'RSALES',
        'CMRMTSPL',       # Real Mfg and Trade Industries Sales
        'RRSFS',          # Real Retail and Food Services Sales
    ],
    'durable_goods': [
        'AMDMNO',         # Manufacturers' New Orders: Durable Goods
        'ADXTNO',
        'AMDMNOX',
        'NEWORDER',
        'AMTMNO',         # Manufacturers New Orders: Total Manufacturing
    ],
    'cpaper_historical': [
        'DCPF3M',        # 3-Month AA Financial Commercial Paper Rate (longer)
        'DCPN3M',        # 3-Month Nonfinancial CP
        'DCPF6M',        # 6-Month Financial CP
        'RIFSPPFAAD90NB', # Financial CP 90-day
        'WCP3M',         # 3-Month CP
        'AAA',           # AAA corporate bond rate (proxy)
        'PRIME',         # Prime rate
        'MPRIME',        # Bank Prime Loan Rate
    ],
    'total_deposits': [
        'DPSACBW027SBOG',   # Deposits at All Commercial Banks (weekly)
        'DPSACBM027SBOG',   # Deposits at All Commercial Banks (monthly)
        'DEMDEPSL',         # Demand Deposits at Commercial Banks
        'SAVDEP',           # Savings Deposits
        'TCDSL',            # Total Checkable Deposits
        'DEPBKALL',         # Total Deposits, All Commercial Banks
        'H8B3053NCBA',      # Total deposits
    ],
    'total_loans_and_inv': [
        'TOTBKCR',          # Bank Credit at All Commercial Banks
        'H8B1001NCBCMG',    # Bank Credit (monthly)
    ],
}

for category, codes in attempts.items():
    print(f"\n--- {category} ---")
    for code in codes:
        try:
            df = pdr.DataReader(code, 'fred', START, END)
            first = df.first_valid_index()
            last = df.last_valid_index()
            n = df.iloc[:,0].dropna().shape[0]
            print(f"  OK: {code} -> {n} obs, {first.strftime('%Y-%m')} to {last.strftime('%Y-%m')}")
        except Exception as e:
            print(f"  FAIL: {code}: {str(e)[:80]}")
