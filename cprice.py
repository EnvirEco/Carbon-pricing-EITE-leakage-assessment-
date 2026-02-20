import pandas as pd

# ────────────────────────────────────────────────
#  CONFIGURATION - adjust paths as needed
# ────────────────────────────────────────────────
INPUT_PANEL   = "did_panel_province_sector_year.csv"          # your original file
OUTPUT_PANEL  = "did_panel_with_carbon_price.csv"             # where to save merged result

# ────────────────────────────────────────────────
#  Carbon price lookup table (CAD/tCO₂e)
#  Format: {province: {year: price}}
# ────────────────────────────────────────────────
carbon_prices = {
    'Alberta': {
        2007: 15, 2008: 15, 2009: 15, 2010: 15, 2011: 15, 2012: 15,
        2013: 15, 2014: 15, 2015: 15, 2016: 20, 2017: 30, 2018: 30,
        2019: 30, 2020: 30, 2021: 40, 2022: 50, 2023: 65, 2024: 80,
        2025: 95
    },
    'British Columbia': {
        2008: 10, 2009: 15, 2010: 20, 2011: 25, 2012: 30, 2013: 30,
        2014: 30, 2015: 30, 2016: 30, 2017: 30, 2018: 35, 2019: 40,
        2020: 45, 2021: 50, 2022: 50, 2023: 65, 2024: 80, 2025: 95
    },
    'Quebec': {
        2007: 3.5, 2008: 3.5, 2009: 3.5, 2010: 3.5, 2011: 3.5, 2012: 3.5,
        2013: 11.0, 2014: 11.5, 2015: 12.3, 2016: 12.6, 2017: 14.6,
        2018: 14.8, 2019: 16.5, 2020: 17.0, 2021: 20.4, 2022: 29.9,
        2023: 34.8, 2024: 42.3, 2025: 48.0
    },
    'Ontario': {
        2017: 14, 2018: 15,
        2019: 20, 2020: 30, 2021: 40, 2022: 50, 2023: 65, 2024: 80,
        2025: 95
    },
    # Federal backstop provinces (and others post-2018/2019)
    'Saskatchewan': {y: 0 for y in range(2007,2017)} | {
        2017: 0, 2018: 0, 2019: 20, 2020: 30, 2021: 40, 2022: 50,
        2023: 65, 2024: 80, 2025: 0   # paused industrial in 2025
    },
    'Manitoba': {y: 0 for y in range(2007,2019)} | {
        2019: 20, 2020: 30, 2021: 40, 2022: 50, 2023: 65, 2024: 80,
        2025: 95
    },
    'New Brunswick': {y: 0 for y in range(2007,2019)} | {
        2019: 20, 2020: 30, 2021: 40, 2022: 50, 2023: 65, 2024: 80,
        2025: 95
    },
    'Nova Scotia': {y: 0 for y in range(2007,2019)} | {
        2019: 20, 2020: 30, 2021: 40, 2022: 50, 2023: 65, 2024: 80,
        2025: 95   # cap-and-trade, but using federal equiv for simplicity
    },
    'Newfoundland and Labrador': {y: 0 for y in range(2007,2019)} | {
        2019: 20, 2020: 30, 2021: 40, 2022: 50, 2023: 65, 2024: 80,
        2025: 95
    },
    'Prince Edward Island': {y: 0 for y in range(2007,2019)} | {
        2019: 20, 2020: 30, 2021: 40, 2022: 50, 2023: 65, 2024: 80,
        2025: 95
    },
    # Territories usually follow federal backstop
    'Yukon': {y: 0 for y in range(2007,2019)} | {
        2019: 20, 2020: 30, 2021: 40, 2022: 50, 2023: 65, 2024: 80,
        2025: 95
    },
    'Northwest Territories': {y: 0 for y in range(2007,2019)} | {
        2019: 20, 2020: 30, 2021: 40, 2022: 50, 2023: 65, 2024: 80,
        2025: 95
    },
    'Nunavut': {y: 0 for y in range(2007,2019)} | {
        2019: 20, 2020: 30, 2021: 40, 2022: 50, 2023: 65, 2024: 80,
        2025: 95
    },
}

# ────────────────────────────────────────────────
#  MAIN MERGE LOGIC
# ────────────────────────────────────────────────
print("Reading panel file...")
df = pd.read_csv(INPUT_PANEL)

# Ensure year is integer
df['year'] = df['year'].astype(int)

# Create carbon_price column (default 0)
df['carbon_price'] = 0.0

print("Applying carbon prices...")
for prov, year_dict in carbon_prices.items():
    mask = df['province'].str.strip() == prov.strip()
    for y, price in year_dict.items():
        df.loc[mask & (df['year'] == y), 'carbon_price'] = price

# Quick diagnostics
print("\nDiagnostics:")
print("  Unique provinces:", sorted(df['province'].unique()))
print("  Years range:   ", df['year'].min(), "–", df['year'].max())
print("  Rows with price > 0:", (df['carbon_price'] > 0).sum(), f"({(df['carbon_price'] > 0).mean():.1%})")
print("\nPrice distribution:\n", df['carbon_price'].value_counts().sort_index())

print("\nSample rows with price:")
print(df[['province', 'year', 'naics_3digit', 'treatment', 'carbon_price']].head(12))

# Save merged file
df.to_csv(OUTPUT_PANEL, index=False)
print(f"\nSaved merged file to: {OUTPUT_PANEL}")