"""
DID REGRESSION: CARBON PRICING AND TRADE LEAKAGE (CLEAN VERSION)
==================================================================
Estimate effects of staggered carbon pricing on industrial exports,
with heterogeneity by sectoral carbon intensity.

Key design choices:
1. NO province × year interactions (enables DID identification)
2. Wild cluster bootstrap (appropriate for n=10 provinces)
3. Simplified SPEC 2: carbon_price × carbon_intensity only
4. Log exports scaled to millions (clearer interpretation)
5. Corrected comparison logic (SPEC 6)
"""

import pandas as pd
import numpy as np
import json
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

PANEL_FILE = '/mnt/data/did_panel_final_clean.csv' if Path('/mnt/data/did_panel_final_clean.csv').exists() else 'did_panel_final_clean.csv'
OUTPUT_DIR = './outputs'

# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_data(panel):
    """Prepare panel for regression analysis"""
    panel = panel.copy()
    
    # Rename common variants
    if 'exports_value' in panel.columns and 'export_value' not in panel.columns:
        panel = panel.rename(columns={'exports_value': 'export_value'})
    
    if 'naics_3digit' not in panel.columns and 'naics' in panel.columns:
        panel['naics_3digit'] = (
            panel['naics']
            .astype(str)
            .str.extract(r'(\d{3})', expand=False)
            .astype(float)
        )
    
    # Emissions handling
    if 'emissions' not in panel.columns:
        emissions_cols = ['emission', 'emissions_total', 'ghg_emissions', 'ghg_emissions_total']
        for col in emissions_cols:
            if col in panel.columns:
                panel = panel.rename(columns={col: 'emissions'})
                break
        if 'emissions' not in panel.columns:
            warnings.warn("No emissions column found; defaulting to 0.")
            panel['emissions'] = 0
    
    # Carbon price handling
    if 'carbon_price' not in panel.columns:
        raise KeyError("Expected 'carbon_price' column in panel data.")
    panel['carbon_price'] = panel['carbon_price'].fillna(0)
    
    # ========================================================================
    # Carbon Intensity Measurement (Based on Pre-Treatment Period)
    # ========================================================================
    # Use 2004-2006 as truly pre-treatment baseline (before AB/BC/QC policies)
    baseline_period = panel[(panel['year'] >= 2004) & (panel['year'] <= 2006)].copy()
    
    if baseline_period.empty:
        # Fallback to earliest 3 years in data
        years_sorted = sorted(panel['year'].unique())[:3]
        baseline_period = panel[panel['year'].isin(years_sorted)].copy()
    
    baseline_ci = baseline_period.groupby('naics_3digit').agg({
        'emissions': 'sum',
        'export_value': 'sum'
    }).reset_index()
    
    # Units note:
    # export_value is reported in THOUSANDS of dollars.
    # Therefore:
    # - actual dollars = export_value * 1,000
    # - million dollars (MUSD) = export_value / 1,000
    # Carbon intensity targets:
    # - carbon_intensity: tCO2e per dollar (legacy variable used by quartile bins)
    # - ci_per_musd: tCO2e per $1,000,000 exports (used in interactions)
    baseline_ci['carbon_intensity'] = baseline_ci['emissions'] / ((baseline_ci['export_value'] * 1_000) + 1e-9)
    baseline_ci['exports_musd'] = baseline_ci['export_value'] / 1_000
    baseline_ci['ci_per_musd'] = baseline_ci['emissions'] / (baseline_ci['exports_musd'] + 1e-9)
    
    # Normalize to 0-1 scale
    min_ci = baseline_ci['carbon_intensity'].min()
    max_ci = baseline_ci['carbon_intensity'].max()
    if pd.isna(min_ci) or pd.isna(max_ci) or max_ci == min_ci:
        baseline_ci['carbon_intensity_norm'] = np.nan
    else:
        baseline_ci['carbon_intensity_norm'] = (
            (baseline_ci['carbon_intensity'] - min_ci) / (max_ci - min_ci)
        )
    
    baseline_ci = baseline_ci[['naics_3digit', 'carbon_intensity', 'carbon_intensity_norm', 'ci_per_musd', 'exports_musd']]
    
    # Merge back to full panel.
    # If panel already has carbon-intensity columns, preserve existing non-null
    # values and backfill from baseline values computed above.
    panel = panel.merge(
        baseline_ci,
        on='naics_3digit',
        how='left',
        suffixes=('', '_baseline'),
    )

    for col in ['carbon_intensity', 'carbon_intensity_norm', 'ci_per_musd']:
        baseline_col = f'{col}_baseline'
        if baseline_col in panel.columns:
            if col in panel.columns:
                panel[col] = panel[col].fillna(panel[baseline_col])
            else:
                panel[col] = panel[baseline_col]

    panel = panel.drop(
        columns=[
            c for c in ['carbon_intensity_baseline', 'carbon_intensity_norm_baseline', 'ci_per_musd_baseline']
            if c in panel.columns
        ]
    )

    if 'carbon_intensity' not in panel.columns:
        raise KeyError("Failed to create carbon intensity data after merge.")
    
    # ========================================================================
    # Create Categories (Quartiles Based on Positive CI)
    # ========================================================================
    positive_ci = panel.loc[panel['carbon_intensity'] > 0, 'carbon_intensity']
    
    if positive_ci.empty:
        panel['carbon_intensity_cat'] = 'None'
        panel['high_carbon'] = 0
    else:
        # Quartile categorization
        if len(positive_ci.unique()) >= 4:
            carbon_cats = pd.qcut(positive_ci, q=4, duplicates='drop')
        else:
            carbon_cats = pd.cut(positive_ci, bins=4)
        
        # Map back to panel
        panel['carbon_intensity_cat'] = pd.Series(
            carbon_cats, index=positive_ci.index
        ).reindex(panel.index)
        
        # Rename categories
        num_bins = carbon_cats.cat.categories.size
        label_map = {
            4: ['Low', 'Medium-Low', 'Medium-High', 'High'],
            3: ['Low', 'Medium', 'High'],
            2: ['Low', 'High'],
            1: ['All'],
        }
        labels = label_map.get(num_bins, [f'Bin{i+1}' for i in range(num_bins)])
        
        if panel['carbon_intensity_cat'].dtype.name == 'category':
            panel['carbon_intensity_cat'] = panel['carbon_intensity_cat'].cat.rename_categories(labels)
        
        if panel['carbon_intensity_cat'].dtype.name == 'category':
            if 'None' not in panel['carbon_intensity_cat'].cat.categories:
                panel['carbon_intensity_cat'] = panel['carbon_intensity_cat'].cat.add_categories(['None'])
        panel['carbon_intensity_cat'] = panel['carbon_intensity_cat'].fillna('None')
        panel['high_carbon'] = (panel['carbon_intensity_cat'] == 'High').astype(int)

    # ========================================================================
    # Cohort and Treatment-Timing Indicators
    # ========================================================================
    treatment_starts = {
        'Alberta': 2007,
        'British Columbia': 2008,
        'Quebec': 2007,
        'Ontario': 2017,
        'Saskatchewan': 2019,
    }

    def assign_cohort(row):
        prov = row['province']
        year = row['year']
        start = treatment_starts.get(prov, 2019)
        if year < start:
            return 'Pre-Treatment'
        if start <= 2009:
            return 'Early (pre-2010)'
        if start <= 2018:
            return 'Mid (2010s)'
        return 'Late (2019+)'

    panel['cohort'] = panel.apply(assign_cohort, axis=1)
    panel['eite_dummy'] = panel.get('eite', 0).fillna(0).astype(int)
    panel['oil_sector'] = panel['naics_3digit'].isin([211, 212, 324]).astype(int)

    print(f"  ✓ Added cohorts: {panel['cohort'].value_counts().to_dict()}")
    
    # ========================================================================
    # Create Log Transform (Scaled to Millions for Interpretability)
    # ========================================================================
    # export_value is in $1,000s => divide by 1,000 to obtain $M
    panel['export_millions'] = panel['export_value'] / 1_000
    panel['log_exports'] = np.log(panel['export_millions'] + 1e-6)

    if 'emissions_intensity' not in panel.columns:
        panel['emissions_intensity'] = panel['emissions'] / (panel['export_value'] + 1e-6)
    if 'log_intensity' not in panel.columns:
        panel['log_intensity'] = np.log(panel['emissions_intensity'])
    
    # Note: Coefficient = elasticity (% change per $1/t)
    # Interpretation: coef = 0.0129 means 1.29% change in exports per $1/t
    
    # ========================================================================
    # Centered Variables (for Cleaner Interpretation)
    # ========================================================================
    panel['ci_per_musd_centered'] = panel['ci_per_musd'] - panel['ci_per_musd'].mean()
    panel['carbon_price_centered'] = (
        panel['carbon_price'] - panel['carbon_price'].mean()
    )
    
    # ========================================================================
    # Sector Indicators
    # ========================================================================
    panel['province_trend'] = panel.groupby('province')['year'].transform(lambda s: s - s.min())


    # ========================================================================
    # Province-cycle control: Bartik-style predicted export shock
    # ========================================================================
    # Construct a province-year shock driven by (i) each province's *baseline*
    # export mix across sectors and (ii) national sector export growth in each year.
    #
    # This helps absorb provincial boom/bust dynamics driven by sector composition,
    # without directly controlling for potentially post-treatment variables like GDP.
    #
    # Steps:
    # 1) Baseline shares: s_{p,k} from baseline period exports (2004–2006 or fallback)
    # 2) National sector growth: g_{k,t} = Δlog(X_{k,t}) across all provinces
    # 3) Bartik shock: B_{p,t} = Σ_k s_{p,k} * g_{k,t}
    baseline_ps = baseline_period.groupby(['province', 'naics_3digit'], as_index=False)['export_value'].sum()
    prov_tot = baseline_ps.groupby('province', as_index=False)['export_value'].sum().rename(columns={'export_value': 'prov_baseline_total'})
    baseline_ps = baseline_ps.merge(prov_tot, on='province', how='left')
    baseline_ps['baseline_share'] = baseline_ps['export_value'] / (baseline_ps['prov_baseline_total'] + 1e-9)
    baseline_ps = baseline_ps[['province', 'naics_3digit', 'baseline_share']]

    nat_sector = panel.groupby(['naics_3digit', 'year'], as_index=False)['export_value'].sum()
    nat_sector['log_nat_exports'] = np.log(nat_sector['export_value'] + 1e-9)
    nat_sector = nat_sector.sort_values(['naics_3digit', 'year'])
    nat_sector['nat_dlog_exports'] = nat_sector.groupby('naics_3digit')['log_nat_exports'].diff().fillna(0.0)
    nat_sector = nat_sector[['naics_3digit', 'year', 'nat_dlog_exports']]

    bartik_long = baseline_ps.merge(nat_sector, on='naics_3digit', how='left')
    bartik_long['bartik_component'] = bartik_long['baseline_share'] * bartik_long['nat_dlog_exports']
    bartik = bartik_long.groupby(['province', 'year'], as_index=False)['bartik_component'].sum()
    bartik = bartik.rename(columns={'bartik_component': 'bartik_dlog'})
    panel = panel.merge(bartik, on=['province', 'year'], how='left')
    panel['bartik_dlog'] = panel['bartik_dlog'].fillna(0.0)

    # Optional: cumulative Bartik index (captures sustained booms/busts)
    panel = panel.sort_values(['province', 'year'])
    panel['bartik_cum'] = panel.groupby('province')['bartik_dlog'].cumsum()

    return panel, baseline_ci


def _wild_cluster_bootstrap(model, data, cluster_var='province', n_boot=999, seed=42):
    """
    Wild cluster bootstrap (Rademacher) for small number of clusters.
    
    Appropriate when:
    - Small number of clusters (e.g., 10 provinces)
    - Standard errors from conventional clustering may be overstated
    
    Args:
        model: fitted OLS model
        data: dataframe with cluster variable
        cluster_var: column name for cluster variable
        n_boot: number of bootstrap iterations
        seed: random seed for reproducibility
    
    Returns:
        (se, pvalues, conf_int) where each is indexed by model.params.index
    """
    rng = np.random.default_rng(seed)
    
    # Extract design matrix and residuals
    row_labels = getattr(model.model.data, 'row_labels', data.index)
    if row_labels is None:
        row_labels = data.index
    
    design = model.model.exog
    resid = model.resid.to_numpy()
    groups = data.loc[row_labels, cluster_var].to_numpy()
    uniq_clusters = np.unique(groups)
    
    # Check minimum cluster count
    if uniq_clusters.size < 2:
        warnings.warn(f"Only {uniq_clusters.size} unique cluster(s); bootstrap not feasible.")
        return None
    
    # Bootstrap loop
    boot_betas = np.empty((n_boot, design.shape[1]))
    for b in range(n_boot):
        # Rademacher weights: +1 or -1 with equal probability
        signs = rng.choice([-1.0, 1.0], size=uniq_clusters.size)
        sign_map = dict(zip(uniq_clusters, signs))
        weights = np.array([sign_map[g] for g in groups])
        
        # Resample with cluster weights
        y_star = model.fittedvalues.to_numpy() + resid * weights
        boot_fit = sm.OLS(y_star, design).fit()
        boot_betas[b, :] = boot_fit.params
    
    # Calculate bootstrap SE
    wb_se = pd.Series(boot_betas.std(axis=0, ddof=1), index=model.params.index)
    
    # Calculate p-values using bootstrap t-statistics under the null.
    # We center by estimated params and scale by bootstrap SE so inference lines
    # up with wb_se/wb_conf_int used throughout the reporting.
    wb_se_safe = np.where(wb_se.to_numpy() <= 1e-12, np.nan, wb_se.to_numpy())
    boot_t = (boot_betas - model.params.to_numpy()) / wb_se_safe
    t_obs = model.params.to_numpy() / wb_se_safe
    wb_pvals = pd.Series(
        np.nanmean(np.abs(boot_t) >= np.abs(t_obs), axis=0),
        index=model.params.index,
    ).fillna(1.0)
    
    # Calculate 95% CI
    ci_low = np.percentile(boot_betas, 2.5, axis=0)
    ci_high = np.percentile(boot_betas, 97.5, axis=0)
    wb_ci = pd.DataFrame({0: ci_low, 1: ci_high}, index=model.params.index)
    
    return wb_se, wb_pvals, wb_ci


def run_regression(data, formula, title, cluster_var='province', use_bootstrap=True):
    """
    Run regression with wild cluster bootstrap inference.
    
    Args:
        data: DataFrame with all variables
        formula: statsmodels formula
        title: description of regression
        cluster_var: column name for clustering (usually 'province')
        use_bootstrap: if True, use wild cluster bootstrap; else use standard clustering
    
    Returns:
        model object with .wb_se, .wb_pvalues, .wb_conf_int attributes
    """
    if cluster_var not in data.columns:
        raise KeyError(f"Cluster variable '{cluster_var}' not found.")
    
    data_clean = data.dropna(subset=[cluster_var]).copy()
    if data_clean.empty:
        raise ValueError(f"No data for regression: {title}")
    
    # Fit OLS model
    model = smf.ols(formula, data=data_clean, missing='drop').fit()
    
    # Apply wild cluster bootstrap
    if use_bootstrap:
        wb = _wild_cluster_bootstrap(model, data_clean, cluster_var=cluster_var)
        if wb is not None:
            model.wb_se, model.wb_pvalues, model.wb_conf_int = wb
        else:
            # Fallback to conventional clustering if bootstrap fails
            warnings.warn(f"Bootstrap failed for {title}; using standard errors.")
            model.wb_se = model.bse
            model.wb_pvalues = model.pvalues
            model.wb_conf_int = model.conf_int()
    else:
        model.wb_se = model.bse
        model.wb_pvalues = model.pvalues
        model.wb_conf_int = model.conf_int()
    
    return model


def extract_coefs(model, var_name):
    """Extract coefficient and inference statistics"""
    if var_name not in model.params.index:
        return {'coef': np.nan, 'se': np.nan, 'pval': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan}
    
    return {
        'coef': model.params[var_name],
        'se': model.wb_se[var_name],
        'pval': model.wb_pvalues[var_name],
        'ci_lower': model.wb_conf_int.loc[var_name, 0],
        'ci_upper': model.wb_conf_int.loc[var_name, 1],
    }


def print_results(model, title):
    """Print regression results using wild-bootstrap inference objects."""
    print("\n" + "="*100)
    print(title)
    print("="*100)

    coef_table = pd.DataFrame({
        'Coef.': model.params,
        'Std.Err.': model.wb_se,
        'P>|t|': model.wb_pvalues,
        '[0.025': model.wb_conf_int[0],
        '0.975]': model.wb_conf_int[1],
    })
    print(coef_table.to_string(float_format=lambda x: f"{x:0.6f}"))
    print(f"\nR-squared: {model.rsquared:.4f}")
    print(f"Observations: {model.nobs}")
    print(f"Degrees of Freedom: {model.df_resid}")
    print("Note: Inference columns above use wild cluster bootstrap (Rademacher weights).")
    print("      Clustered by province.")


def append_summary_rows(summary_rows, model, spec_name, extra_data=None):
    """Append coefficient-level summary rows for CSV export."""
    if model is None:
        if extra_data is not None:
            summary_rows.append({
                'spec_name': spec_name,
                'coef_name': 'extra_data',
                'coef': np.nan,
                'wb_se': np.nan,
                'wb_pval': np.nan,
                'ci_low': np.nan,
                'ci_high': np.nan,
                'nobs': np.nan,
                'r2': np.nan,
                'extra_data': extra_data,
            })
        return

    for coef_name in model.params.index:
        row = {
            'spec_name': spec_name,
            'coef_name': coef_name,
            'coef': model.params[coef_name],
            'wb_se': model.wb_se.get(coef_name, np.nan),
            'wb_pval': model.wb_pvalues.get(coef_name, np.nan),
            'ci_low': model.wb_conf_int.loc[coef_name, 0] if coef_name in model.wb_conf_int.index else np.nan,
            'ci_high': model.wb_conf_int.loc[coef_name, 1] if coef_name in model.wb_conf_int.index else np.nan,
            'nobs': model.nobs,
            'r2': model.rsquared,
        }
        if extra_data is not None:
            row['extra_data'] = extra_data
        summary_rows.append(row)


def export_all_model_results(models_dict, output_dir='./outputs', prefix='did_regression_results'):
    """Export fitted model results to flattened CSV and structured JSON files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    flat_rows = []
    nested_results = {}

    for model_name, model in models_dict.items():
        if model is None:
            continue

        params = model.params
        se = model.std_errors if hasattr(model, 'std_errors') else model.bse
        pvalues = model.pvalues

        wb_se = getattr(model, 'wb_se', None)
        wb_pvalues = getattr(model, 'wb_pvalues', None)
        wb_conf_int = getattr(model, 'wb_conf_int', None)

        for var in params.index:
            row = {
                'model': model_name,
                'variable': var,
                'coef': params.get(var, np.nan),
                'se_conventional': se.get(var, np.nan),
                'pval_conventional': pvalues.get(var, np.nan),
            }

            if wb_se is not None and isinstance(wb_se, pd.Series):
                row['se_wild_bootstrap'] = wb_se.get(var, np.nan)
            if wb_pvalues is not None and isinstance(wb_pvalues, pd.Series):
                row['pval_wild_bootstrap'] = wb_pvalues.get(var, np.nan)
            if wb_conf_int is not None and isinstance(wb_conf_int, pd.DataFrame) and var in wb_conf_int.index:
                row['ci_low_wb'] = wb_conf_int.loc[var, 0]
                row['ci_high_wb'] = wb_conf_int.loc[var, 1]

            if hasattr(model, 'rsquared'):
                row['rsquared'] = model.rsquared
            if hasattr(model, 'nobs'):
                row['nobs'] = model.nobs
            if hasattr(model, 'df_resid'):
                row['df_resid'] = model.df_resid

            flat_rows.append(row)

        nested_results[model_name] = {
            'coefficients': params.to_dict(),
            'se_conventional': se.to_dict(),
            'pvalues_conventional': pvalues.to_dict(),
            'nobs': int(model.nobs) if hasattr(model, 'nobs') else None,
            'rsquared': float(model.rsquared) if hasattr(model, 'rsquared') else None,
        }
        if wb_se is not None:
            nested_results[model_name]['se_wild_bootstrap'] = wb_se.to_dict()
        if wb_pvalues is not None:
            nested_results[model_name]['pvalues_wild_bootstrap'] = wb_pvalues.to_dict()
        if wb_conf_int is not None:
            nested_results[model_name]['ci_wild_bootstrap'] = wb_conf_int.to_dict(orient='index')

    if flat_rows:
        results_df = pd.DataFrame(flat_rows)
        col_order = [
            'model', 'variable', 'coef',
            'se_conventional', 'pval_conventional',
            'se_wild_bootstrap', 'pval_wild_bootstrap',
            'ci_low_wb', 'ci_high_wb',
            'rsquared', 'nobs', 'df_resid',
        ]
        col_order = [c for c in col_order if c in results_df.columns]
        results_df = results_df[col_order]

        csv_path = output_dir / f'{prefix}_all_models.csv'
        results_df.to_csv(csv_path, index=False)
        print(f"✓ Saved CSV results: {csv_path}")
    else:
        csv_path = None
        print('No results to save to CSV.')

    json_path = output_dir / f'{prefix}_all_models.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(
            nested_results,
            f,
            indent=2,
            default=lambda x: float(x) if isinstance(x, np.floating) else x,
        )

    print(f"✓ Saved JSON results: {json_path}")

    return csv_path, json_path




def choose_event_study_reference(data, preferred='-1', event_col='event_time_str'):
    """Return a valid event-time reference level for Patsy treatment coding."""
    levels = data[event_col].dropna().astype(str)
    levels = [lvl for lvl in levels.unique().tolist() if lvl != 'Never']

    if not levels:
        return preferred
    if preferred in levels:
        return preferred

    numeric_levels = []
    for lvl in levels:
        try:
            numeric_levels.append((lvl, int(lvl)))
        except ValueError:
            continue

    if numeric_levels:
        # Prefer the latest pre-treatment period if available, otherwise the closest event time to -1.
        pre_periods = sorted((n for _, n in numeric_levels if n < 0))
        if pre_periods:
            return str(pre_periods[-1])

        closest = min(numeric_levels, key=lambda x: (abs(x[1] - (-1)), abs(x[1])))
        return str(closest[1])

    return sorted(levels)[0]


def make_event_study_formula(reference_level):
    return (
        f'log_exports ~ C(event_time_str, Treatment(reference="{reference_level}")) '
        '+ C(stack_cohort) + C(year) + C(province) + C(naics_3digit)'
    )


def make_event_study_term(k, reference_level):
    return f'C(event_time_str, Treatment(reference="{reference_level}"))[T.{k}]'
def _first_treat_year_by_province(panel):
    """Compute first treatment year per province; never-treated => NaN."""
    out = {}
    for prov, g in panel.sort_values('year').groupby('province'):
        treated_years = g.loc[g['treatment'] == 1, 'year']
        out[prov] = int(treated_years.min()) if not treated_years.empty else np.nan
    return out


def build_stacked_event_data(panel, event_window=5):
    """Create cohort-stacked DID event-study sample."""
    first_treat = _first_treat_year_by_province(panel)
    cohorts = sorted({int(y) for y in first_treat.values() if pd.notna(y)})
    never_treated = [p for p, y in first_treat.items() if pd.isna(y)]

    stacks = []
    for cohort in cohorts:
        treated_provs = [p for p, y in first_treat.items() if pd.notna(y) and int(y) == cohort]
        keep_provs = treated_provs + never_treated
        if not treated_provs or not keep_provs:
            continue

        sub = panel[panel['province'].isin(keep_provs)].copy()
        sub['stack_cohort'] = cohort
        sub['ever_treated'] = sub['province'].isin(treated_provs).astype(int)
        sub['event_time'] = np.where(sub['ever_treated'] == 1, sub['year'] - cohort, np.nan)
        treated_mask = sub['ever_treated'] == 1
        sub = sub[(~treated_mask) | (sub['event_time'].between(-event_window, event_window))]
        sub['event_time_str'] = sub['event_time'].round().astype('Int64').astype('string').fillna('Never')
        stacks.append(sub)

    if not stacks:
        return pd.DataFrame()

    return pd.concat(stacks, ignore_index=True)


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

if __name__ == '__main__':

    print("\n" + "="*100)
    print("DID REGRESSION: CARBON PRICING AND TRADE LEAKAGE")
    print("Staggered Adoption Design - Across Canadian Provinces")
    print("="*100)

    print("\n[SETUP] Loading and preparing panel data...")
    panel = pd.read_csv(PANEL_FILE)
    panel, baseline_ci = prepare_data(panel)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print(f"  Observations: {len(panel)}")
    print(f"  Provinces: {panel['province'].nunique()}")
    print(f"  Sectors (NAICS 3-digit): {panel['naics_3digit'].nunique()}")
    print(f"  Years: {int(panel['year'].min())}-{int(panel['year'].max())}")
    print("  Baseline CI scale: tCO2e per $1,000,000 exports (ci_per_musd)")

    # Unit diagnostics (export_value is in CAD thousands)
    ci_min = panel['ci_per_musd'].min()
    ci_p25 = panel['ci_per_musd'].quantile(0.25)
    ci_p50 = panel['ci_per_musd'].quantile(0.50)
    ci_p75 = panel['ci_per_musd'].quantile(0.75)
    ci_max = panel['ci_per_musd'].max()
    print("  CI distribution (tCO2e per $M):")
    print(f"    min={ci_min:,.3f}, p25={ci_p25:,.3f}, p50={ci_p50:,.3f}, p75={ci_p75:,.3f}, max={ci_max:,.3f}")
    print("  Sanity benchmark: at $500M exports, CI of 0.5-2.0 tCO2e/$M implies 250-1,000 tCO2e emissions.")

    summary_rows = []

    print("\n" + "="*100)
    print("SPECIFICATION 1A/1B/1C: BASELINE vs SHOCK-CONTROLLED TWFE")
    print("="*100)

    spec1a_formula = 'log_exports ~ carbon_price + C(province) + C(year)'
    model_spec1a = run_regression(panel, spec1a_formula, 'SPEC 1A baseline', cluster_var='province')
    print_results(model_spec1a, 'SPEC 1A (baseline TWFE): log_exports ~ carbon_price + C(province) + C(year)')
    coef_1a = extract_coefs(model_spec1a, 'carbon_price')
    print("\n>>> RESULT (SPEC 1A)")
    print(f"    Coefficient: {coef_1a['coef']:.6f}")
    print(f"    p-value: {coef_1a['pval']:.6f}")
    append_summary_rows(summary_rows, model_spec1a, 'spec_1a_baseline_twfe')

    spec1b_formula = 'log_exports ~ carbon_price + C(province) + C(year) + C(naics_3digit):C(year)'
    model_spec1b = run_regression(panel, spec1b_formula, 'SPEC 1B shock-controlled', cluster_var='province')
    print_results(model_spec1b, 'SPEC 1B (shock-controlled): + sector×year FE')
    coef_1b = extract_coefs(model_spec1b, 'carbon_price')
    print("\n>>> RESULT (SPEC 1B)")
    print(f"    Coefficient: {coef_1b['coef']:.6f}")
    print(f"    p-value: {coef_1b['pval']:.6f}")
    append_summary_rows(summary_rows, model_spec1b, 'spec_1b_shock_sector_year_fe')

    spec1c_formula = 'log_exports ~ carbon_price + C(province) + C(year) + C(province):year'
    model_spec1c = run_regression(panel, spec1c_formula, 'SPEC 1C shock-controlled', cluster_var='province')
    print_results(model_spec1c, 'SPEC 1C (shock-controlled): + province-specific linear trends')
    coef_1c = extract_coefs(model_spec1c, 'carbon_price')
    print("\n>>> RESULT (SPEC 1C)")
    print(f"    Coefficient: {coef_1c['coef']:.6f}")
    print(f"    p-value: {coef_1c['pval']:.6f}")
    append_summary_rows(summary_rows, model_spec1c, 'spec_1c_shock_province_trends')

    print("\n" + "="*100)
    print("INTENSITY SPECIFICATION 1A/1B/1C")
    print("="*100)

    intensity_1a_formula = 'log_intensity ~ carbon_price + C(province) + C(year)'
    intensity_1a = run_regression(panel, intensity_1a_formula, 'INTENSITY SPEC 1A', cluster_var='province')
    print_results(intensity_1a, 'INTENSITY SPEC 1A: log_intensity ~ carbon_price + C(province) + C(year)')
    append_summary_rows(summary_rows, intensity_1a, 'intensity_spec_1a')

    intensity_1b_formula = 'log_intensity ~ carbon_price + C(province) + C(year) + C(naics_3digit)*C(year)'
    intensity_1b = run_regression(panel, intensity_1b_formula, 'INTENSITY SPEC 1B', cluster_var='province')
    print_results(intensity_1b, 'INTENSITY SPEC 1B: + C(naics_3digit)*C(year)')
    append_summary_rows(summary_rows, intensity_1b, 'intensity_spec_1b')

    intensity_1c_formula = 'log_intensity ~ carbon_price + C(province) + C(year) + province_trend'
    intensity_1c = run_regression(panel, intensity_1c_formula, 'INTENSITY SPEC 1C', cluster_var='province')
    print_results(intensity_1c, 'INTENSITY SPEC 1C: + province_trend')
    append_summary_rows(summary_rows, intensity_1c, 'intensity_spec_1c')

    print("\n" + "="*100)
    print("SPECIFICATION 2: INTERACTION WITH CI IN tCO2e per $1M exports")
    print("="*100)

    spec2_formula = 'log_exports ~ carbon_price * ci_per_musd_centered + C(province) + C(year)'
    model_spec2 = run_regression(panel, spec2_formula, 'SPEC 2', cluster_var='province')
    print_results(model_spec2, 'SPEC 2: log_exports ~ carbon_price × ci_per_musd_centered + FE')

    coef_2_main = extract_coefs(model_spec2, 'carbon_price')
    coef_2_interact = extract_coefs(model_spec2, 'carbon_price:ci_per_musd_centered')
    append_summary_rows(summary_rows, model_spec2, 'spec_2_ci_per_musd_interaction')

    ci_levels = panel['ci_per_musd'].quantile([0.25, 0.50, 0.75]).to_dict()
    ci_mean = panel['ci_per_musd'].mean()
    print("\n>>> INTERPRETATION (SPEC 2)")
    print(f"    Effect of $1/t at CI = {ci_mean:.2f} tCO2e per $M exports: {coef_2_main['coef']*100:.3f}%")
    print(f"    Interaction slope (per +1 tCO2e per $M): {coef_2_interact['coef']*100:.5f} pp")
    for q, ci_val in ci_levels.items():
        marginal = coef_2_main['coef'] + coef_2_interact['coef'] * (ci_val - ci_mean)
        print(f"    Marginal effect at p{int(q*100):02d} CI = {ci_val:.2f}: {marginal*100:.3f}% per $1/t")

    print("\n" + "="*100)
    print("SPECIFICATION 3: HETEROGENEOUS EFFECT BY CI CATEGORY")
    print("="*100)

    categories_present = [c for c in panel['carbon_intensity_cat'].dropna().unique().tolist() if c != 'None']
    if len(categories_present) < 2:
        print('  ⚠ SPEC 3 skipped: fewer than two CI categories available.')
    else:
        preferred_order = ['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High']
        ref_cat = next((c for c in preferred_order if c in categories_present), categories_present[0])
        spec3_formula = (
            f'log_exports ~ carbon_price * C(carbon_intensity_cat, Treatment(reference="{ref_cat}")) '
            '+ C(province) + C(year)'
        )
        model_spec3 = run_regression(panel, spec3_formula, 'SPEC 3', cluster_var='province')
        print_results(model_spec3, f'SPEC 3: carbon_price × CI category (ref={ref_cat}) + FE')
        append_summary_rows(summary_rows, model_spec3, 'spec_3_ci_category_interaction')

        base_effect = extract_coefs(model_spec3, 'carbon_price')['coef']
        print("\n>>> INTERPRETATION (SPEC 3)")
        print(f"    Reference category ({ref_cat}) effect: {base_effect*100:.3f}% per $1/t")
        for cat in sorted(categories_present):
            if cat == ref_cat:
                continue
            term = f'carbon_price:C(carbon_intensity_cat, Treatment(reference="{ref_cat}"))[T.{cat}]'
            if term in model_spec3.params.index:
                delta = model_spec3.params[term]
                total = base_effect + delta
                print(f"    {cat}: delta vs {ref_cat} = {delta*100:.3f} pp, total = {total*100:.3f}% per $1/t")

    print("\n" + "="*100)
    print("SPECIFICATION 4: CARBON PRICE × NON-EITE DUMMY")
    print("Tests whether negative export effects are concentrated in non-EITE sectors")
    print("="*100)

    panel['carbon_price_non_eite'] = panel['carbon_price'] * (1 - panel['eite_dummy'])
    panel['carbon_price_non_oil'] = panel['carbon_price'] * (1 - panel['oil_sector'])

    spec4a_formula = (
        'log_exports ~ carbon_price + carbon_price_non_eite '
        '+ eite_dummy + C(province) + C(year)'
    )
    model_spec4a = run_regression(panel, spec4a_formula, 'SPEC 4A: EITE heterogeneity', cluster_var='province')
    print_results(model_spec4a, 'SPEC 4A: log_exports ~ carbon_price + carbon_price × (1 - eite_dummy) + eite_dummy + FE')

    coef_4a_main = extract_coefs(model_spec4a, 'carbon_price')
    coef_4a_non_eite = extract_coefs(model_spec4a, 'carbon_price_non_eite')
    coef_4a_eite = extract_coefs(model_spec4a, 'eite_dummy')

    print("\n>>> INTERPRETATION (SPEC 4A – EITE)")
    print(f"    Marginal effect in EITE sectors      : {coef_4a_main['coef']*100:.3f}% per $1/tCO₂e")
    print(f"    Additional effect in non-EITE sectors: {coef_4a_non_eite['coef']*100:.3f}% per $1/tCO₂e")
    print(f"    Total effect in non-EITE             : {(coef_4a_main['coef'] + coef_4a_non_eite['coef'])*100:.3f}% per $1/tCO₂e")
    print(f"    EITE dummy (level shift)             : {coef_4a_eite['coef']*100:.3f}%")

    spec4b_formula = (
        'log_exports ~ carbon_price + carbon_price_non_oil '
        '+ oil_sector + C(province) + C(year)'
    )
    model_spec4b = run_regression(panel, spec4b_formula, 'SPEC 4B: Oil-sector heterogeneity', cluster_var='province')
    print_results(model_spec4b, 'SPEC 4B: log_exports ~ carbon_price + carbon_price × (1 - oil_sector) + oil_sector + FE')

    coef_4b_main = extract_coefs(model_spec4b, 'carbon_price')
    coef_4b_non_oil = extract_coefs(model_spec4b, 'carbon_price_non_oil')
    coef_4b_oil = extract_coefs(model_spec4b, 'oil_sector')

    print("\n>>> INTERPRETATION (SPEC 4B – Oil & heavy sectors)")
    print(f"    Marginal effect in oil/heavy sectors : {coef_4b_main['coef']*100:.3f}% per $1/tCO₂e")
    print(f"    Additional effect outside oil/heavy  : {coef_4b_non_oil['coef']*100:.3f}% per $1/tCO₂e")
    print(f"    Total effect outside oil/heavy       : {(coef_4b_main['coef'] + coef_4b_non_oil['coef'])*100:.3f}% per $1/tCO₂e")

    append_summary_rows(summary_rows, model_spec4a, 'spec_4a_eite_interaction')
    append_summary_rows(summary_rows, model_spec4b, 'spec_4b_oil_interaction')

    print("\n" + "="*100)
    print("SPECIFICATION 5: CARBON PRICE × PROVINCE/COHORT INTERACTIONS")
    print("Tests effects by treatment timing (staggered rollout)")
    print("="*100)

    cohorts_present = panel['cohort'].unique()
    if len(cohorts_present) < 2:
        print('  ⚠ SPEC 5A skipped: fewer than two cohorts available.')
    else:
        ref_cohort = 'Early (pre-2010)' if 'Early (pre-2010)' in cohorts_present else cohorts_present[0]
        spec5a_formula = (
            f'log_exports ~ carbon_price * C(cohort, Treatment(reference="{ref_cohort}")) '
            '+ C(province) + C(year) + C(naics_3digit)'
        )
        model_spec5a = run_regression(panel, spec5a_formula, 'SPEC 5A: Cohort heterogeneity', cluster_var='province')
        print_results(model_spec5a, f'SPEC 5A: carbon_price × cohort (ref={ref_cohort}) + FE')
        append_summary_rows(summary_rows, model_spec5a, 'spec_5a_cohort_interaction')

        base_effect = extract_coefs(model_spec5a, 'carbon_price')['coef']
        print("\n>>> INTERPRETATION (SPEC 5A – Cohorts)")
        print(f"    Reference cohort ({ref_cohort}) effect: {base_effect*100:.3f}% per $1/tCO₂e")
        for coh in sorted(cohorts_present):
            if coh == ref_cohort:
                continue
            term = f'carbon_price:C(cohort, Treatment(reference="{ref_cohort}"))[T.{coh}]'
            if term in model_spec5a.params.index:
                delta = model_spec5a.params[term]
                total = base_effect + delta
                print(f"    {coh}: delta vs {ref_cohort} = {delta*100:.3f} pp, total = {total*100:.3f}% per $1/tCO₂e")

    print("\n  Running province-specific models...")
    prov_results = []
    top_provs = panel['province'].value_counts().nlargest(5).index
    for prov in sorted(panel['province'].unique()):
        sub = panel[panel['province'] == prov].copy()
        if len(sub) < 20:
            continue
        try:
            prov_formula = 'log_exports ~ carbon_price + C(year) + C(naics_3digit)'
            model_prov = smf.ols(prov_formula, data=sub).fit(cov_type='cluster', cov_kwds={'groups': sub.index})
            coef_prov = model_prov.params.get('carbon_price', np.nan)
            se_prov = model_prov.bse.get('carbon_price', np.nan)
            pval_prov = model_prov.pvalues.get('carbon_price', np.nan)
            n_obs = len(sub)
            prov_results.append({
                'province': prov,
                'coef': coef_prov,
                'se': se_prov,
                'pval': pval_prov,
                'n_obs': n_obs,
                'n_years': sub['year'].nunique(),
            })
            if prov in top_provs:
                print(f"    {prov}: coef = {coef_prov*100:.3f}% (p={pval_prov:.3f}, n={n_obs})")
        except Exception as e:
            print(f"    {prov}: Skipped due to error ({e})")

    prov_df = pd.DataFrame(prov_results).sort_values('n_obs', ascending=False)
    print("\nProvince-Specific Summary Table:")
    print(prov_df.round(4).to_string(index=False))

    prov_path = Path(OUTPUT_DIR) / 'province_specific_results.csv'
    prov_df.to_csv(prov_path, index=False)
    print(f"  ✓ Saved: {prov_path}")

    append_summary_rows(summary_rows, None, 'spec_5b_province_loop', extra_data=prov_df.to_dict('records'))

    print("\n" + "="*100)
    print("ROBUSTNESS: STACKED DID EVENT STUDY (cohort-stacked)")
    print("="*100)

    stacked = build_stacked_event_data(panel, event_window=5)
    if stacked.empty:
        print('  ⚠ No stacked DID sample could be constructed.')
        event_df = pd.DataFrame(columns=['event_time', 'coef', 'se', 'pval', 'ci_low', 'ci_high'])
    else:
        stacked_ref = choose_event_study_reference(stacked, preferred='-1')
        if stacked_ref != '-1':
            print(f"  ⚠ Event-study reference '-1' unavailable; using '{stacked_ref}' instead.")
        es_formula = make_event_study_formula(stacked_ref)
        model_stacked = run_regression(stacked, es_formula, 'STACKED DID EVENT STUDY', cluster_var='province')
        print_results(model_stacked, 'STACKED DID: Event-time indicators + cohort/year/province/sector FE')
        append_summary_rows(summary_rows, model_stacked, 'spec_stacked_did_event_study')

        event_rows = []
        for k in range(-5, 6):
            if k == -1:
                continue
            term = make_event_study_term(k, stacked_ref)
            if term in model_stacked.params.index:
                event_rows.append({
                    'event_time': k,
                    'coef': model_stacked.params[term],
                    'se': model_stacked.wb_se[term],
                    'pval': model_stacked.wb_pvalues[term],
                    'ci_low': model_stacked.wb_conf_int.loc[term, 0],
                    'ci_high': model_stacked.wb_conf_int.loc[term, 1],
                })

        event_df = pd.DataFrame(event_rows).sort_values('event_time')
        print("\nEvent-study plot data (stacked DID):")
        print(event_df.to_string(index=False, float_format=lambda x: f"{x:0.6f}"))

        print("\n  Cohort-stratified event-study (top cohorts only)...")
        for coh in ['Early (pre-2010)', 'Late (2019+)']:
            sub_stacked = stacked[stacked['cohort'] == coh]
            if len(sub_stacked) > 50:
                sub_ref = choose_event_study_reference(sub_stacked, preferred='-1')
                if sub_ref != '-1':
                    print(f"    ⚠ {coh}: reference '-1' unavailable; using '{sub_ref}' instead.")
                es_sub_formula = make_event_study_formula(sub_ref)
                model_sub = run_regression(sub_stacked, es_sub_formula, f'Sub-cohort {coh}', cluster_var='province')
                print(f"    {coh} event coefs (selected):")
                for k in [-2, 0, 2]:
                    term = make_event_study_term(k, sub_ref)
                    if term in model_sub.params.index:
                        print(f"      t{k}: {model_sub.params[term]*100:.3f}% (p={model_sub.wb_pvalues[term]:.3f})")

    summary_table_path = Path(OUTPUT_DIR) / 'summary_table.csv'
    event_path = Path(OUTPUT_DIR) / 'event_study_stacked.csv'
    pd.DataFrame(summary_rows).to_csv(summary_table_path, index=False)
    event_df.to_csv(event_path, index=False)

    all_models = {
        'spec_1a': model_spec1a,
        'spec_1b': model_spec1b,
        'spec_1c': model_spec1c,
        'intensity_spec_1a': intensity_1a,
        'intensity_spec_1b': intensity_1b,
        'intensity_spec_1c': intensity_1c,
        'spec_2': model_spec2,
        'spec_3': model_spec3 if 'model_spec3' in locals() else None,
        'spec_4a': model_spec4a,
        'spec_4b': model_spec4b,
        'stacked_event_study': model_stacked if 'model_stacked' in locals() else None,
    }
    all_models_csv, all_models_json = export_all_model_results(
        models_dict=all_models,
        output_dir=OUTPUT_DIR,
        prefix='carbon_pricing_exports_did_2026',
    )

    print("\n" + "="*100)
    print('OUTPUT FILES WRITTEN')
    print("="*100)
    print(f"summary: {summary_table_path}")
    print(f"stacked event-study: {event_path}")
    print(f"all-models CSV: {all_models_csv}")
    print(f"all-models JSON: {all_models_json}")

    print("\nValidation checks:")
    table_p_1a = model_spec1a.wb_pvalues.get('carbon_price', np.nan)
    print(f"  Spec 1a table p-value == RESULT p-value: {np.isclose(table_p_1a, coef_1a['pval'], equal_nan=True)} ({table_p_1a:.6f} vs {coef_1a['pval']:.6f})")
    print(f"  Spec 1b residual DoF: {model_spec1b.df_resid:.1f}")
    print(f"  Spec 1c residual DoF: {model_spec1c.df_resid:.1f}")

    print("\n" + "="*100)
    print("END OF ANALYSIS")
    print("="*100 + "\n")
