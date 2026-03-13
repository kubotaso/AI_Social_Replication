# Data Acquisition Instructions

## 1. World Values Survey (WVS) Longitudinal Data File

The WVS data requires free registration to download. Please follow these steps:

### Option A: WVS Time-Series Data (Preferred)

1. Go to: https://www.worldvaluessurvey.org/WVSDocumentationWVL.jsp
2. Register for a free account if you don't have one
3. Download the **WVS Time-Series (1981-2022)** data file
4. Choose one of these formats (in order of preference):
   - **CSV** format (easiest to work with)
   - **Stata (.dta)** format
   - **SPSS (.sav)** format
   - **R (.rds)** format
5. Also download the **codebook** (PDF or other documentation)
6. Save the data file to: `data/WVS_TimeSeries.csv` (or `.dta`, `.sav`, `.rds`)
7. Save the codebook to: `data/WVS_codebook.pdf`

### Option B: Individual Wave Files (Alternative)

If the longitudinal file is not available, download individual wave files:
1. Wave 1 (1981-1984): https://www.worldvaluessurvey.org/WVSDocumentationWV1.jsp
2. Wave 2 (1990-1994): https://www.worldvaluessurvey.org/WVSDocumentationWV2.jsp
3. Wave 3 (1995-1998): https://www.worldvaluessurvey.org/WVSDocumentationWV3.jsp
4. Save each to `data/` directory

### Option C: ICPSR (Alternative)

1. Go to: https://www.icpsr.umich.edu
2. Register for a free account
3. Search for "World Values Survey" or study numbers:
   - ICPSR 2790 (WVS 1981-1984)
   - ICPSR 6160 (WVS 1990-1993)
   - ICPSR 2790 (WVS combined file)
4. Download in Stata or CSV format
5. Save to `data/` directory

## 2. World Bank Data (Automated)

This will be downloaded automatically using the `wbgapi` Python package. No manual action needed.

## After Download

Please let me know when the WVS data files are ready in the `data/` directory. I will then:
1. Inspect the data
2. Verify all required variables are present
3. Continue with the replication
