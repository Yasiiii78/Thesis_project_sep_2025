"""
BIOGEME Base Model Estimation

This script provides a clean implementation for estimating basic destination choice, with few explanatory variables
using BIOGEME. It includes data preprocessing, utility specification, and model estimation. This file provides a biogeme model results in output files.
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

def create_variable_data(df, zones, cost_config):
    """Create cost and urbanity variables efficiently"""
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

def create_availability_variables(df, zones):
    """Create availability variables for all zones"""
    av_data = {}
    
    for zone in zones:
        tt_col = f'tt_{zone}'
        job_col = f'total_jobs_{zone}'
        av_col = f'AV{zone}'

        # Filtering on travel time
        if tt_col in df.columns and job_col in df.columns:
            av_data[av_col] = ((df[tt_col] < 160) & (df[job_col] > 0)).astype(int)
        elif tt_col in df.columns:
            av_data[av_col] = (df[tt_col] < 160).astype(int)
        else:
            av_data[av_col] = 0
    
    return av_data

def define_model_parameters():
    """Define all BIOGEME parameters"""
    params = {
        'beta_tt': Beta("B_tt", 0, None, None, 0),
        'beta_tt_gender': Beta("B_tt_gender", 0, None, None, 0),
        # The coefficient for total jobs is fixed to 1
        'beta_ln_total_jobs': Beta("B_ln_total_jobs", 1, None, None, 1)
    }

    # Cost parameter (optional)
    # params['beta_cost'] = Beta("B_cost", 0, None, None, 0)

    # Age interaction parameters
    for age_cat in ['C', 'D', 'E']:
        params[f'beta_tt_age_{age_cat}'] = Beta(f"B_tt_age_{age_cat}", 0, None, None, 0)
    
    # Urbanity parameters
    for level in range(2, 6):
        params[f'beta_urban_{level}'] = Beta(f"B_urban_{level}", 0, None, None, 0)
        
    return params

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
av_data = create_availability_variables(df_combined, pc4_zones)
av_df = pd.DataFrame(av_data, index=df_combined.index)
df_combined = pd.concat([df_combined, av_df], axis=1).copy()

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

# Create utility functions
u_functions = {}
av = {}

for zone in pc4_zones:
    terms = []
    
    # Travel time effects
    tt_var_name = f'tt_{zone}'
    if tt_var_name in database.variables:
        tt_var = database.variables[tt_var_name]
        terms.append(params['beta_tt'] * tt_var)
        
        # Age interactions
        for age_cat in ['age_category_C', 'age_category_D', 'age_category_E']:
            if age_cat in database.variables:
                param_name = f"beta_tt_age_{age_cat.split('_')[2]}"
                terms.append(params[param_name] * tt_var * database.variables[age_cat])
        
        # Gender interaction
        if 'gender_2.0' in database.variables:
            terms.append(params['beta_tt_gender'] * tt_var * database.variables['gender_2.0'])
            
    # Cost variable (optional)
    # cost_var_name = f'cost_{zone}'
    # if cost_var_name in database.variables:
    #     cost_var = database.variables[cost_var_name]
    #     terms.append(params['beta_cost'] * cost_var)

    # Urbanity effects
    for level in [2, 3, 4, 5]:
        urbanity_dummy_name = f'urbanity_level_{level}_{zone}'
        if urbanity_dummy_name in database.variables:
            terms.append(params[f'beta_urban_{level}'] * database.variables[urbanity_dummy_name])
    
    # Job opportunities
    job_var_name = f'total_jobs_{zone}'
    if job_var_name in database.variables:
        # Total jobs is scaled down with a factor of 0.001
        job_var = database.variables[job_var_name] * 0.001
        # Apply log transformation for total jobs
        log_job_var = log(job_var)
        terms.append(params['beta_ln_total_jobs'] * log_job_var)
    
    # Create utility function
    u_functions[zone] = sum(terms) if terms else Beta(f"ASC_{zone}", 0, None, None, 0)
    
    # Availability
    av_var_name = f'AV{zone}'
    av[zone] = database.variables.get(av_var_name, 1)

logprob = models.loglogit(u_functions, av, database.variables['CHOICE'])
formulas = {'loglike': logprob, 'weight': database.variables['WEIGHT']}

biogeme = bio.BIOGEME(database, formulas)
biogeme.modelName = "destination_choice_wide"

print("7. Estimating model...")
try:
    results = biogeme.estimate()
    print("Model estimation completed successfully!")
    
    # Display and save results
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

except Exception as e:
    print(f"Model estimation failed: {str(e)}")

# Final timing
total_time = time.time() - start_time
print(f"\n{'='*60}")
print(f"PROCESS COMPLETED - Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print(f"{'='*60}")
