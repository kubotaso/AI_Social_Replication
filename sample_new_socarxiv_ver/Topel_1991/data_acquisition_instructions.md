# Data Acquisition Instructions

Two datasets are needed: PSID panel data (for Tables 2-7, A1) and CPS Displaced Workers Survey (for Table 1).

---

## 1. PSID Data (CRITICAL — needed for 8 of 9 tables)

The PSID requires free registration to download data.

### Option A: PSID Data Center Custom Extract (RECOMMENDED)

1. **Register** at https://simba.isr.umich.edu/u/Register.aspx (free)
2. **Log in** at https://simba.isr.umich.edu
3. Go to **Data Center** → **Previous Cart/Extract** or **Create New Cart**
4. Select the following variables across waves 1968-1983:

**From Family Files (each year 1968-1983):**
- Head's age
- Head's sex
- Head's race
- Head's education (years of schooling)
- Head's marital status
- Head's labor income / wages and salary
- Head's annual work hours
- Head's hourly earnings (if available)
- Head's employment status
- Head's occupation (1-digit or 3-digit)
- Head's industry
- Whether head is self-employed
- Whether head works for government
- Head's union membership status
- Head's tenure on current job
- Whether head changed employer since last interview
- Head's state of residence / region
- Head's SMSA residence
- Head's disability status
- Family interview number
- Survey year
- Whether family is from SRC (random) or SEO (poverty) sample

**From Cross-Year Individual File:**
- Person ID (1968 interview number + person number)
- Sex
- Sequence number
- Relationship to head
- Response status for each year

5. **Download** as CSV or Stata format
6. **Save** all files to the `data/` directory in this project

### Option B: PSID Packaged ZIP Files

1. **Register** at https://simba.isr.umich.edu (free)
2. Go to https://simba.isr.umich.edu/Zips/ZipMain.aspx
3. Download the **Family Files** for each year: 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983
4. Download the **Cross-Year Individual File** (1968-2021 or whichever version is current)
5. Save all ZIP files to the `psid_raw/` directory
6. Each ZIP contains: ASCII data file, SAS/Stata/SPSS setup code, codebook, and documentation

---

## 2. CPS Displaced Workers Survey (needed for Table 1)

### Option A: IPUMS CPS (RECOMMENDED — easiest)

1. **Register** at https://cps.ipums.org/cps/ (free)
2. Select **"Get Data"**
3. Select samples: **January 1984 Displaced Workers** and **January 1986 Displaced Workers**
4. Select variables:
   - AGE, SEX, RACE
   - DUTEFIRST (or prior job tenure)
   - DWREAS (reason for displacement: plant closing, layoff)
   - DWLOST (type of job lost)
   - DWWKSLK (weeks without work since displacement)
   - EARNWK (current weekly earnings)
   - DWERNWK (weekly earnings on prior job)
   - DWCURJOB (currently employed)
   - WTFINL (final weight)
5. Download as CSV
6. Save to `data/cps_dws_ipums.csv`

### Option B: Use Raw NBER Files (already downloaded)

The raw CPS files for January 1984 and 1986 are already downloaded:
- `data/cpsjan84.raw` (96 MB, fixed-width)
- `data/cpsjan86.raw` (84 MB, fixed-width)
- `data/cpsjan84.pdf` (codebook, 81 pages)
- `data/cpsjan86.pdf` (codebook, 175 pages)

However, the displaced workers supplement questions appear to be in a separate supplement file not yet identified. The IPUMS option (A) is strongly preferred.

---

## After downloading:

Please let me know when the files are ready and I will:
1. Inspect the data files
2. Verify the required variables are present
3. Proceed with the replication

**The PSID data is the priority** — it is needed for Tables 2, 3, 4A, 4B, 5, 6, 7, and A1.
