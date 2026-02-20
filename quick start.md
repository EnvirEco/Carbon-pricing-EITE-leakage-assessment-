# Carbon-pricing-EITE-leakage-assessment-

**Quick Start: Carbon Pricing Impact Model**
This two-stage pipeline constructs a panel dataset from Canadian emissions and trade records and then estimates the impact of carbon pricing on industrial exports using a Difference-in-Differences (DiD) framework.

**1. Prerequisites & Environment**
  Ensure you have Python 3.8+ installed with the following libraries:
  •	pandas, numpy, statsmodels, scipy, openpyxl

**2. Required Input Files**
  Place these files in your working directory:
  •	GHGRP_extended.csv: Facility-level GHG emissions data.
  •	trade_clean_output.csv: Provincial export data by NAICS code.
  •	policy.csv: Historical carbon prices by province and year.
  •	03_DID_REGRESSION_CLEAN.py: The estimation script.
  •	MASTER_BUILD_PANEL_v3.py: The data construction script.

**3. Required Input Files**
  Place these files in your working directory:
  •	GHGRP_extended.csv: Facility-level GHG emissions data.
  •	trade_clean_output.csv: Provincial export data by NAICS code.
  •	policy.csv: Historical carbon prices by province and year.
  •	03_DID_REGRESSION_CLEAN.py: The estimation script.
  •	MASTER_BUILD_PANEL_v3.py: The data construction script.

**4. Execution Steps**
  Step A: Build the Dataset
    Run the panel builder to merge emissions, trade, and policy data while calculating carbon intensities and EITE classifications.
      
      Bash
      python MASTER_BUILD_PANEL_v3.py
      
    •	Output: did_panel_final_clean.csv
    •	Validation: Ensure the script reports ~10 provinces and verifies positive export/emissions matches.
  Step B: Run the Regression Analysis
  Execute the DiD model.
  This script performs multiple specifications and applies Wild Cluster Bootstrapping to account for the small number of provincial clusters (n=10).

    Bash
    python 03_DID_REGRESSION_CLEAN.py

**4. Understanding the Outputs**
  All results are saved to the /outputs folder:
  File	Description
    summary_table.csv	High-level coefficients for all major model specifications.
    event_study_stacked.csv	Lead/lag coefficients to validate parallel trends.
    results_all_models.json	Comprehensive statistics (p-values, R², SE) for every model.
    province_specific.csv	Individual carbon price elasticities for each province.

**5. Key Specifications in the Model**
  •	Spec 1A-1C: Baseline TWFE vs. Sector-Year FE vs. Province Trends.
  •	Spec 2: Interaction between carbon price and continuous carbon intensity.
  •	Spec 4: Heterogeneity tests for EITE (Energy-Intensive, Trade-Exposed) vs. Non-EITE sectors.
  •	Robustness: A Stacked DiD approach to handle staggered policy adoption and prevent "negative weighting" bias.

