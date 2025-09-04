"""
BIOGEME Extended Model Estimation

This script provides a comprehensive implementation for estimating extended destination choice models using BIOGEME.
It includes data preprocessing, utility specification, and model estimation. This file provides a biogeme model results in output files.
"""
import pandas as pd
import numpy as np
from biogeme.expressions import Beta, log
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
import time
import os

# Configuration
COST_PER_KM = [0.21, 30]  # [cost per km, minimum trip length km]

# Get the script directory and navigate to data folder
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(script_dir), 'data')
DATA_PATH = os.path.join(data_dir, 'final_wide_format', 'wide_data_filtered_v6.csv')

SAMPLE_SIZE = 1983
RANDOM_SEED = 123
WORK_SECTORS = ['Care', 'Education', 'Hospitality_Culture_Sports', 
               'Industry_Construction', 'Office_Services', 'Remaining', 'Shop']

# Optional parameters - set to False to exclude from analysis
INCLUDE_FLEXIBILITY = True  
INCLUDE_TRAVEL_COST = False 

def create_variable_data(df, zones, cost_config):
    """Create cost and urbanity variables"""
    cost_data = {}
    urbanity_data = {}
    
    # Cost variables
    for zone in zones:
        distance_col = f'afstand_{zone}'
        cost_col = f'travel_cost_{zone}'
        if distance_col in df.columns:
            cost_data[cost_col] = np.where(
                df[distance_col] / 1000 >= cost_config[1],
                df[distance_col] / 1000 * cost_config[0], 0)
        else:
            cost_data[cost_col] = 0
    
    # Urbanity dummy variables
    urbanity_columns = [col for col in df.columns if col.startswith('urbanity_')]
    for col in urbanity_columns:
        zone = col.split('_')[1]
        for level in [1, 2, 3, 4, 5]:
            dummy_col_name = f'urbanity_level_{level}_{zone}'
            urbanity_data[dummy_col_name] = (df[col] == level).astype(int)
    
    return cost_data, urbanity_data

def create_availability_variables(df, zones, sectors):
    """Create availability variables for all zones"""
    av_data = {}
    
    for zone in zones:
        tt_col = f'tt_{zone}'
        av_col = f'AV{zone}'
        
        if tt_col in df.columns:
            time_available = (df[tt_col] < 160).astype(int)
            sector_available = pd.Series(0, index=df.index)
            
            for sector in sectors:
                sector_var = f'sector_cluster_{sector}'
                job_var = f'work_sector_{sector}_jobs_{zone}'
                
                if sector_var in df.columns and job_var in df.columns:
                    sector_condition = ((df[sector_var] == 1) & (df[job_var] > 0)).astype(int)
                    sector_available = sector_available | sector_condition
            
            total_jobs_col = f'total_jobs_{zone}'
            has_total_jobs = (df[total_jobs_col] > 0).astype(int) if total_jobs_col in df.columns else pd.Series(1, index=df.index)
            av_data[av_col] = (time_available & sector_available & has_total_jobs).astype(int)
        else:
            av_data[av_col] = 0
    
    return av_data

def validate_availability(df):
    """Validate that all chosen zones are available"""
    def chosen_is_available(row):
        choice = int(row['CHOICE'])
        av_col = f'AV{choice}'
        return (av_col in row.index) and (row[av_col] == 1)
    
    availability_mask = df.apply(chosen_is_available, axis=1)
    n_failed = (~availability_mask).sum()
    
    if n_failed > 0:
        raise RuntimeError(f"ERROR: {n_failed} chosen zones marked unavailable")

def define_model_parameters():
    """Define all BIOGEME parameters"""
    # Helper function to create Beta parameters with default values
    def create_beta(name):
        return Beta(name, 0, None, None, 0)
    
    # Define core parameter names
    param_names = [
        ('beta_tt', 'B_tt'),
        ('beta_tt_age_C', 'B_tt_age_C'),
        ('beta_tt_age_D', 'B_tt_age_D'),
        ('beta_tt_age_E', 'B_tt_age_E'),
        ('beta_tt_gender_man', 'B_tt_gender_man'),
        ('beta_hh_tt_with_child', 'B_hh_tt_with_child'),
        ('beta_ln_total_jobs', 'B_ln_total_jobs')
    ]
    
    # Add optional parameters based on configuration
    if INCLUDE_FLEXIBILITY:
        param_names.append(('beta_tt_flexibility_flex', 'B_tt_flexibility_flex'))
    
    if INCLUDE_TRAVEL_COST:
        param_names.append(('beta_travel_cost', 'B_travel_cost'))
    
    # Add urbanity parameters
    for level in range(2, 6):
        param_names.append((f'beta_urban_{level}', f'B_urban_{level}'))
    
    # Create parameters dictionary
    return {key: create_beta(value) for key, value in param_names}

def create_utility_functions(zones, database_vars, params):
    """Create utility functions for all zones"""
    u_functions = {}
    av = {}
    
    for zone in zones:
        terms = []
        
        # Travel time effects
        tt_var_name = f'tt_{zone}'
        if tt_var_name in database_vars:
            tt_var = database_vars[tt_var_name]
            terms.append(params['beta_tt'] * tt_var)
            
            # Age interactions
            for age_cat in ['age_category_C', 'age_category_D', 'age_category_E']:
                if age_cat in database_vars:
                    param_name = f"beta_tt_age_{age_cat.split('_')[2]}"
                    terms.append(params[param_name] * tt_var * database_vars[age_cat])
            
            # Other interactions
            if 'hh_compund_grouped_with_child' in database_vars:
                terms.append(params['beta_hh_tt_with_child'] * tt_var * database_vars['hh_compund_grouped_with_child'])

            if 'gender_1.0' in database_vars:
                terms.append(params['beta_tt_gender_man'] * tt_var * database_vars['gender_1.0'])
            
            # Flexibility interaction (optional)
            if INCLUDE_FLEXIBILITY and 'work_flexibility_flexible' in database_vars:
                terms.append(params['beta_tt_flexibility_flex'] * tt_var * database_vars['work_flexibility_flexible'])
        
        # Travel cost effects (optional)
        if INCLUDE_TRAVEL_COST:
            travel_cost_var_name = f'travel_cost_{zone}'
            if travel_cost_var_name in database_vars:
                travel_cost_var = database_vars[travel_cost_var_name]
                terms.append(params['beta_travel_cost'] * travel_cost_var)

        # Urbanity effects
        for level in [2, 3, 4, 5]:
            urbanity_dummy_name = f'urbanity_level_{level}_{zone}'
            if urbanity_dummy_name in database_vars:
                terms.append(params[f'beta_urban_{level}'] * database_vars[urbanity_dummy_name])
        
        # Job opportunities
        job_var_name = f'total_jobs_{zone}'
        if job_var_name in database_vars:
            job_var = database_vars[job_var_name] * 0.001
            log_job_var = log(job_var)
            terms.append(params['beta_ln_total_jobs'] * log_job_var)
        
        # Sector-specific job shares
        for sector in WORK_SECTORS:
            job_var_name = f'work_sector_{sector}_jobs_{zone}'
            sector_var_name = f'sector_cluster_{sector}'
            
            if job_var_name in database_vars and sector_var_name in database_vars:
                job_var = database_vars[job_var_name]
                total_jobs_var = database_vars[f'total_jobs_{zone}']
                job_share = job_var / (total_jobs_var + 0.001)
                
                if sector == 'Remaining':
                    sector_share_interaction = Beta(f"B_share_{sector}", 1, None, None, 1)
                else:
                    sector_share_interaction = Beta(f"B_share_{sector}", 0, None, None, 0)
                
                terms.append(sector_share_interaction * job_share * database_vars[sector_var_name])
        
        # Create utility function
        u_functions[zone] = sum(terms) if terms else Beta(f"ASC_{zone}", 0, None, None, 0)
        
        # Availability
        av_var_name = f'AV{zone}'
        av[zone] = database_vars.get(av_var_name, 1)
    
    return u_functions, av

def process_and_save_results(results, biogeme):
    """Process and save estimation results"""
    pandasResults = results.getEstimatedParameters()
    print("\nESTIMATED PARAMETERS:")
    print("=" * 50)
    print(pandasResults)
    
    output_file = f"BIOGEME_results_{biogeme.modelName}.csv"
    pandasResults.to_csv(output_file)
    print(f"Results saved to: {output_file}")
    
    # Model statistics
    print(f"\nModel Statistics:")
    print(f"Log-likelihood: {results.data.logLike:.2f}")
    print(f"Observations: {results.data.sampleSize}")
    print(f"Parameters: {results.data.nparam}")
    
    if hasattr(results.data, 'rhoSquared'):
        print(f"Rho-squared: {results.data.rhoSquared:.3f}")
    if hasattr(results.data, 'rhoSquaredBar'):
        print(f"Adjusted rho-squared: {results.data.rhoSquaredBar:.3f}")


# Main execution
start_time = time.time()
print("Starting destination choice model estimation...")

print("1. Loading and preparing data...")
df = pd.read_csv(DATA_PATH).dropna()
chosen_alternatives = df[['id_nr', 'destination_pc4']].drop_duplicates()
chosen_alternatives.rename(columns={'destination_pc4': 'chosen_dest'}, inplace=True)
df_combined = pd.merge(df, chosen_alternatives, on='id_nr', how='left')
df_combined['CHOICE'] = df_combined['chosen_dest']

print("2. Creating variables...")
tt_columns = [col for col in df_combined.columns if col.startswith('tt_')]
pc4_zones = sorted([int(col.replace('tt_', '')) for col in tt_columns])

cost_data, urbanity_data = create_variable_data(df_combined, pc4_zones, COST_PER_KM)
df_combined = pd.concat([df_combined, 
                        pd.DataFrame(cost_data, index=df_combined.index),
                        pd.DataFrame(urbanity_data, index=df_combined.index)], axis=1)

print("3. Creating availability variables...")
av_data = create_availability_variables(df_combined, pc4_zones, WORK_SECTORS)
av_df = pd.DataFrame(av_data, index=df_combined.index)

# Ensure chosen alternatives are available
np.random.seed(RANDOM_SEED)
av_array = av_df.values
zone_to_idx = {col: idx for idx, col in enumerate(av_df.columns)}

for i, idx in enumerate(df_combined.index):
    chosen_zone = df_combined.loc[idx, 'CHOICE']
    chosen_av_col = f'AV{int(chosen_zone)}'
    if chosen_av_col in zone_to_idx:
        av_array[i, zone_to_idx[chosen_av_col]] = 1

av_df = pd.DataFrame(av_array, index=df_combined.index, columns=av_df.columns)
df_combined = pd.concat([df_combined, av_df], axis=1).copy()

validate_availability(df_combined)

print("4. Preparing data for estimation...")
df_combined['WEIGHT'] = 1.0

if len(df_combined) > 100:
    df_sample = df_combined.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)
else:
    df_sample = df_combined.copy()

df_sample = df_sample.dropna(subset=['CHOICE'])
bool_columns = df_sample.select_dtypes(include=['bool']).columns
for col in bool_columns:
    df_sample[col] = df_sample[col].astype(int)

print("5. Creating BIOGEME database...")
database = db.Database("destination_choice_wide", df_sample)
globals().update(database.variables)

print("6. Building model...")
params = define_model_parameters()
u_functions, av = create_utility_functions(pc4_zones, database.variables, params)

logprob = models.loglogit(u_functions, av, database.variables['CHOICE'])
formulas = {'loglike': logprob, 'weight': database.variables['WEIGHT']}

biogeme = bio.BIOGEME(database, formulas)
biogeme.modelName = "destination_choice_wide"

print("7. Estimating model...")
try:
    results = biogeme.estimate()
    print("Model estimation completed successfully!")
    
    process_and_save_results(results, biogeme)

    
except Exception as e:
    print(f"Model estimation failed: {str(e)}")

# Final timing
total_time = time.time() - start_time
print(f"\n{'='*60}")
print(f"PROCESS COMPLETED - Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print(f"{'='*60}")
