import pandas as pd
import numpy as np

# 1. Load the data
# We skip the first 11 rows of metadata and define our own column names
col_names = ['province', 'naics', 'year', 'exports_value', 'exporters_number']
df = pd.read_csv('1210009801-noSymbol.csv', skiprows=11, names=col_names, low_memory=False)

# 2. Forward fill the Province and NAICS columns
# (StatCan only puts the name in the first row of each group)
df['province'] = df['province'].ffill()
df['naics'] = df['naics'].ffill()

# 3. Remove footer rows (where year is not a number)
df = df.dropna(subset=['year'])

# 4. Clean Numeric Values
# This handles commas (399,747,297), symbols (.., F), and 0s
def clean_statcan_numeric(val):
    if pd.isna(val): return np.nan
    val = str(val).strip()
    if val in ['..', 'F', 'x', '...']: return np.nan
    if val == '0s': return 0.0
    val = val.replace(',', '') # Remove commas
    try:
        return float(val)
    except ValueError:
        return np.nan

df['exports_value'] = df['exports_value'].apply(clean_statcan_numeric)
df['exporters_number'] = df['exporters_number'].apply(clean_statcan_numeric)

# 5. Clean text columns
# Remove footnote numbers from names (e.g., 'Territories 3' -> 'Territories')
df['province'] = df['province'].str.replace(r'\s\d+$', '', regex=True).str.strip()
df['naics'] = df['naics'].str.strip()

# 6. Filter Years (2007 - 2023)
df['year'] = pd.to_numeric(df['year'], errors='coerce').astype(int)
df_final = df[df['year'].between(2007, 2024)].copy()

# Save the result
df_final.to_csv('trade_clean_output.csv', index=False)

print("Data cleaning complete. Sample of output:")
print(df_final.head())
