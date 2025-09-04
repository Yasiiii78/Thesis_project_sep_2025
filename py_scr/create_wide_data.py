import pandas as pd
import geopandas as gpd
import unidecode
import os

# Get the script directory and navigate to data folder
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(script_dir), 'data')

# Load data
df_tt = pd.read_csv(os.path.join(data_dir, 'travel_time', 'final_Asymmetric_interzonal_applied_tt_v3.csv'))
df_jobs = pd.read_csv(os.path.join(data_dir, 'new_job_data', 'jobs_pc5.csv')) 
df_agent = pd.read_csv(os.path.join(data_dir, 'LRO 2023 met ww afstand', 'df_agent_v5.csv'))
df_wide_tt = pd.read_csv(os.path.join(data_dir, 'travel_time', 'wide_data_tt.csv'))
df_new_jobs = gpd.read_file(os.path.join(data_dir, 'new_job_data', 'buurten_2022.gpkg'))
df_coupling = pd.read_excel(os.path.join(data_dir, 'geo_coupled', 'koppeling buurten pc4.xlsx'))

# Merge new jobs with coupling data
df_new_jobs = df_new_jobs.merge(df_coupling[['gwb_code', 'pst_mvp', 'urbanity']], 
                               how='left', left_on='zone_id', right_on='gwb_code')

# Rename job columns
rename_dict = {
    'arbeidsplaatsen_totaal': 'total_jobs',
    'arbeidsplaatsen_woonfunctie': 'residential_jobs',
    'arbeidsplaatsen_standplaats': 'standplace_jobs',
    'arbeidsplaatsen_ligplaats': 'berth_jobs',
    'arbeidsplaatsen_celfunctie': 'cell_function_jobs',
    'arbeidsplaatsen_onderwijsfunctie': 'education_function_jobs',
    'arbeidsplaatsen_industriefunctie': 'industry_function_jobs',
    'arbeidsplaatsen_kantoorfunctie': 'office_function_jobs',
    'arbeidsplaatsen_overige_winkelfunctie': 'other_shop_function_jobs',
    'arbeidsplaatsen_overige_voedselwinkelfunctie': 'other_food_shop_function_jobs',
    'arbeidsplaatsen_supermarkt_voedselwinkelfunctie': 'supermarket_food_shop_function_jobs',
    'arbeidsplaatsen_gezondheidszorgfunctie': 'healthcare_function_jobs',
    'arbeidsplaatsen_sportfunctie': 'sports_function_jobs',
    'arbeidsplaatsen_bijeenkomstfunctie': 'meeting_function_jobs',
    'arbeidsplaatsen_logiesfunctie': 'lodging_function_jobs',
    'arbeidsplaatsen_boerderijfunctie': 'farm_function_jobs',
    'arbeidsplaatsen_terminalfunctie': 'terminal_function_jobs',
    'arbeidsplaatsen_distributiecentrumfunctie': 'distribution_center_function_jobs',
    'arbeidsplaatsen_overige_gebruiksfunctie': 'other_usage_function_jobs',
    'arbeidsplaatsen_gecombineerde_functies': 'combined_function_jobs'
}
df_new_jobs = df_new_jobs.rename(columns=rename_dict)

# Keep only job columns and zonal data
columns_to_keep = [col for col in df_new_jobs.columns if col.endswith('_jobs')] + ['gwb_code', 'pst_mvp', 'urbanity']
df_new_jobs = df_new_jobs[columns_to_keep]

# Aggregate by PC4
job_columns = [col for col in df_new_jobs.columns if col.endswith('_jobs')]
agg_dict = {col: 'sum' for col in job_columns}
agg_dict['urbanity'] = 'mean'
df_jobs_pc4 = df_new_jobs.groupby('pst_mvp').agg(agg_dict).reset_index()

# Convert travel times from seconds to minutes
df_wide_tt.iloc[:, 2:] = df_wide_tt.iloc[:, 2:] / 60

# Round urbanity to integer
df_jobs_pc4['urbanity'] = df_jobs_pc4['urbanity'].round().astype(int)

# Formalize column names
def formalize_column(col):
    col = unidecode.unidecode(col)
    col = col.replace('-', '').replace(',', '').replace(' ', '_').replace('&', 'en')
    col = col.replace('__', '_').replace('(', '').replace(')', '')
    return col.lower()

formal_columns = [formalize_column(col) for col in df_jobs_pc4.columns]
rename_dict = dict(zip(df_jobs_pc4.columns, formal_columns))
df_jobs_pc4 = df_jobs_pc4.rename(columns=rename_dict)

# Group jobs by sector
job_to_sector_group = {
    'residential_jobs': 'work_sector_Remaining',
    'standplace_jobs': 'work_sector_Remaining',
    'berth_jobs': 'work_sector_Remaining',
    'cell_function_jobs': 'work_sector_Remaining',
    'education_function_jobs': 'work_sector_Education',
    'industry_function_jobs': 'work_sector_Industry_Construction',
    'office_function_jobs': 'work_sector_Office_Services',
    'other_shop_function_jobs': 'work_sector_Shop',
    'other_food_shop_function_jobs': 'work_sector_Shop',
    'supermarket_food_shop_function_jobs': 'work_sector_Shop',
    'healthcare_function_jobs': 'work_sector_Care',
    'sports_function_jobs': 'work_sector_Hospitality_Culture_Sports',
    'meeting_function_jobs': 'work_sector_Hospitality_Culture_Sports',
    'lodging_function_jobs': 'work_sector_Hospitality_Culture_Sports',
    'farm_function_jobs': 'work_sector_Remaining',
    'terminal_function_jobs': 'work_sector_Remaining',
    'distribution_center_function_jobs': 'work_sector_Remaining',
    'other_usage_function_jobs': 'work_sector_Remaining',
    'combined_function_jobs': 'work_sector_Remaining',
}

unique_sectors = list(set(job_to_sector_group.values()))
for sector in unique_sectors:
    cols = [col for col, grp in job_to_sector_group.items() if grp == sector]
    df_jobs_pc4[f'{sector}_jobs'] = df_jobs_pc4[cols].sum(axis=1)

# Select relevant columns
work_sector_columns = [col for col in df_jobs_pc4.columns if col.startswith('work_sector_')] + ['total_jobs', 'urbanity']

# Clean PC4 codes
df_jobs['pc4_code'] = pd.to_numeric(df_jobs['pc4_code'], errors='coerce').dropna().astype(int)
df_jobs_pc4['pst_mvp'] = pd.to_numeric(df_jobs_pc4['pst_mvp'], errors='coerce').dropna().astype(int)
df_jobs_pc4 = df_jobs_pc4[df_jobs_pc4['pst_mvp'].isin(df_jobs['pc4_code'].unique())]

# Create wide format for jobs
new_columns = {}
for _, row in df_jobs_pc4.iterrows():
    pc4_code = int(row['pst_mvp']) if not pd.isna(row['pst_mvp']) else None
    for sector in work_sector_columns:
        new_col_name = f"{sector}_{pc4_code}"
        new_columns[new_col_name] = row[sector]

df_jobs_wide = pd.DataFrame([new_columns])

# Rename travel time columns
df_wide_tt.rename(columns=lambda col: f'tt_{col}' if col not in ['id_nr', 'living_adress'] else col, inplace=True)

# Replicate jobs data for all agents
df_jobs_wide = pd.concat([df_jobs_wide]*len(df_wide_tt), ignore_index=True)

# Combine travel time and jobs data
df_combined = pd.concat([df_wide_tt, df_jobs_wide], axis=1)

# Process agent data
formal_columns = [formalize_column(col) for col in df_agent.columns]
rename_dict = dict(zip(df_agent.columns, formal_columns))
df_agent = df_agent.rename(columns=rename_dict)

# Group household composition
HH_compound_map = {
    'child_<12': 'with_child',
    'child_>12': 'with_child',
    'other_people': 'without_child',
    'without_child': 'without_child', 
    'alone': 'without_child',
    'unknown': 'without_child',
}
df_agent['hh_compund_grouped'] = df_agent['hh_compound'].map(HH_compound_map)

# Map work sectors
sector_mapping = {
    1: 'Agriculture_and_fisheries',
    2: 'Industry',
    3: 'Construction',
    4: 'Utilities',
    5: 'Commerce',
    6: 'Hospitality',
    7: 'Transportation',
    8: 'ICT',
    9: 'Financial_services',
    10: 'Business_services',
    11: 'Public_Administration',
    12: 'Education',
    13: 'Health_and_welfare',
    14: 'Culture',
    15: 'Other_services'
}
df_agent['work_sector_name'] = df_agent['work_sector'].map(sector_mapping)

sector_group_map = {
    'Agriculture_and_fisheries': 'Remaining',
    'Industry': 'Industry_Construction',
    'Construction': 'Industry_Construction',
    'Utilities': 'Remaining',
    'Commerce': 'Shop',
    'Hospitality': 'Hospitality_Culture_Sports',
    'Transportation': 'Industry_Construction',
    'ICT': 'Office_Services',
    'Financial_services': 'Office_Services',
    'Business_services': 'Office_Services',
    'Public_Administration': 'Office_Services',
    'Education': 'Education',
    'Health_and_welfare': 'Care',
    'Culture': 'Hospitality_Culture_Sports',
    'Other_services': 'Remaining'
}
df_agent['sector_cluster'] = df_agent['work_sector_name'].map(sector_group_map)

# Create dummy variables
agent_char = ['gender', 'work_flexibility', 'sector_cluster', 'hh_compund_grouped', 'age_category']
df_agent_dummies = pd.get_dummies(df_agent, columns=agent_char, prefix=agent_char, prefix_sep='_')

# Keep relevant columns
dummy_columns = [col for col in df_agent_dummies.columns if any(var in col for var in agent_char)]
columns_to_keep = ['id_nr', 'destination_pc4'] + dummy_columns
df_agent_filtered = df_agent_dummies[columns_to_keep]

# Merge with combined data
df_wide = df_agent_filtered.merge(df_combined, on='id_nr', how='left')

# Create distance wide format
df_tt['afstand'] = df_tt['afstand'] * 2  # 2-way distance
df_distance_wide = df_tt.pivot_table(
    index='pc4_from', 
    columns='pc4_to', 
    values='afstand', 
    fill_value=0
)
df_distance_wide = df_distance_wide.reset_index()
distance_columns = ['living_adress'] + ['afstand_' + str(col) for col in df_distance_wide.columns[1:]]
df_distance_wide.columns = distance_columns

# Merge with distance data
df_wide_with_distance = df_wide.merge(df_distance_wide, on='living_adress', how='left')

# Filter observations that choose zones with relevant jobs 
work_sectors = ['Care', 'Education', 'Hospitality_Culture_Sports', 
                'Industry_Construction', 'Office_Services', 'Remaining', 'Shop']

valid_observations = []
for idx, row in df_wide_with_distance.iterrows():
    chosen_zone = int(row['destination_pc4'])
    keep_observation = False
    
    for sector in work_sectors:
        sector_var = f'sector_cluster_{sector}'
        job_var = f'work_sector_{sector}_jobs_{chosen_zone}'
        
        if (sector_var in df_wide_with_distance.columns and 
            job_var in df_wide_with_distance.columns and
            row[sector_var] == 1 and 
            row[job_var] > 0):
            keep_observation = True
            break
    
    valid_observations.append(keep_observation)

df_filtered = df_wide_with_distance[valid_observations].copy()

print(f"Filtered dataset: {len(df_filtered)} observations")

# Save outputs
# Save the filtered and unfiltered data
# df_filtered.to_csv(os.path.join(data_dir, 'final_wide_format', 'wide_data_filtered_v6.csv'), index=False)
# df_wide_with_distance.to_csv(os.path.join(data_dir, 'final_wide_format', 'wide_data_UNfiltered_v6.csv'), index=False)




