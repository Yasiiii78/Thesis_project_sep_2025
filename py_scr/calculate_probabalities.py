"""
BIOGEME Model Comparison Tool

This script provides a comprehensive implementation for evaluating destination choice models
estimated with BIOGEME. It processes wide-format data to calculate choice probabilities
for different destinations based on estimated model parameters.

Key Features:
- Data preprocessing matching BIOGEME estimation pipeline
- Availability matrix creation based on travel time and sector-specific job constraints
- Utility calculation for basic and sectoral destination choice models
- Choice probability computation using multinomial logit framework
- Performance evaluation metrics (log-likelihood, top-k accuracy)
- CSV output of calculated probabilities for further analysis

Usage:
The script requires manually inputting estimated parameters from BIOGEME model results.
It then calculates destination choice probabilities for each agent and exports them
as CSV files for use in downstream applications or validation studies.
"""

import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')

class ModelComparison:
    """Class for comparing BIOGEME destination choice models."""
    
    def __init__(self, data_path=None, df=None):
        if df is not None:
            self.df = df
        elif data_path is not None:
            self.df = pd.read_csv(data_path).dropna()
        else:
            raise ValueError("Either data_path or df must be provided")
            
        self.zones = None
        self.sectors = ['Care', 'Education', 'Hospitality_Culture_Sports', 
                       'Industry_Construction', 'Office_Services', 'Remaining', 'Shop']
        self.availability_matrix = None
        self.train_data = None
        self.test_data = None
        self.train_availability = None
        self.test_availability = None
        
        # Storage for model probabilities
        self.probabilities_basic = None
        self.probabilities_sectoral = None
        self.probabilities_basic_test = None
        self.probabilities_sectoral_test = None
    
    def preprocess_data(self):
        # Step 1: Create chosen alternatives
        chosen_alternatives = self.df[['id_nr', 'destination_pc4']].drop_duplicates()
        chosen_alternatives.rename(columns={'destination_pc4': 'chosen_dest'}, inplace=True)
        
        # Merge chosen destinations with wide data
        df_combined = pd.merge(self.df, chosen_alternatives, on='id_nr', how='left')
        df_combined['CHOICE'] = df_combined['chosen_dest']
        
        # Step 2: Create travel cost variables
        cost_per_km = [0.21, 30]  # [cost per km, minimum trip length km]
        
        # Get PC4 zones from travel time columns
        tt_columns = [col for col in df_combined.columns if col.startswith('tt_')]
        pc4_zones = [int(col.replace('tt_', '')) for col in tt_columns]
        pc4_zones = sorted(pc4_zones)
        self.zones = pc4_zones
        
        # Create travel cost columns
        cost_data = {}
        for zone in pc4_zones:
            distance_col = f'afstand_{zone}'
            cost_col = f'travel_cost_{zone}'
            
            if distance_col in df_combined.columns:
                cost_data[cost_col] = np.where(
                    df_combined[distance_col] / 1000 >= cost_per_km[1],
                    df_combined[distance_col] / 1000 * cost_per_km[0],
                    0
                )
            else:
                cost_data[cost_col] = 0
        
        cost_df = pd.DataFrame(cost_data, index=df_combined.index)
        df_combined = pd.concat([df_combined, cost_df], axis=1)
        
        # Step 3: Create urbanity dummy variables
        urbanity_levels = [1, 2, 3, 4, 5]
        urbanity_columns = [col for col in df_combined.columns if col.startswith('urbanity_')]
        
        urbanity_data = {}
        for col in urbanity_columns:
            zone = col.split('_')[1] 
            for level in urbanity_levels:
                dummy_col_name = f'urbanity_level_{level}_{zone}'
                urbanity_data[dummy_col_name] = (df_combined[col] == level).astype(int)
        
        urbanity_df = pd.DataFrame(urbanity_data, index=df_combined.index)
        df_combined = pd.concat([df_combined, urbanity_df], axis=1)
        
        # Step 4: Critical data filtering and processing
        df_combined['WEIGHT'] = 1.0
        
        # Convert boolean columns to integers
        bool_columns = df_combined.select_dtypes(include=['bool']).columns
        for col in bool_columns:
            df_combined[col] = df_combined[col].astype(int)
        
        # Remove missing CHOICE values
        df_combined = df_combined.dropna(subset=['CHOICE'])
        
        self.df = df_combined
        
        return self
    
    def create_availability_matrix(self):
        if self.zones is None:
            raise ValueError("Must run preprocess_data() first to extract zones")
        
        av_data = {}
        for i, zone in enumerate(self.zones):
            tt_col = f'tt_{zone}'
            av_col = f'AV{zone}'
            
            if tt_col in self.df.columns:
                # Base condition: travel time < 160 minutes
                time_available = (self.df[tt_col] < 160).astype(int)
                
                # Sector-specific job availability condition
                sector_available = pd.Series(0, index=self.df.index)
                
                for sector in self.sectors:
                    sector_var = f'sector_cluster_{sector}'
                    job_var = f'work_sector_{sector}_jobs_{zone}'
                    
                    if sector_var in self.df.columns and job_var in self.df.columns:
                        sector_condition = (
                            (self.df[sector_var] == 1) &
                            (self.df[job_var] > 0)
                        ).astype(int)
                        sector_available = sector_available | sector_condition
                
                # Additional constraint: zone must have total jobs > 0
                total_jobs_col = f'total_jobs_{zone}'
                if total_jobs_col in self.df.columns:
                    has_total_jobs = (self.df[total_jobs_col] > 0).astype(int)
                else:
                    has_total_jobs = pd.Series(1, index=self.df.index)
                
                # Final availability
                av_data[av_col] = (time_available & sector_available & has_total_jobs).astype(int)
            else:
                av_data[av_col] = 0
        
        av_df = pd.DataFrame(av_data, index=self.df.index)
        
        # Ensure chosen alternative is included in availability
        np.random.seed(123)
        
        # Convert to numpy array for faster processing
        av_array = av_df.values
        av_columns = av_df.columns.tolist()
        n_agents, n_zones = av_array.shape
        
        # Create zone index mapping
        zone_to_idx = {col: idx for idx, col in enumerate(av_columns)}
        
        # Process agents to ensure chosen alternative is available
        for i, idx in enumerate(self.df.index):
            chosen_zone = self.df.loc[idx, 'CHOICE']
            chosen_av_col = f'AV{int(chosen_zone)}'
            
            if chosen_av_col in zone_to_idx:
                chosen_idx = zone_to_idx[chosen_av_col]
                av_array[i, chosen_idx] = 1
        
        # Update the DataFrame
        av_df = pd.DataFrame(av_array, index=self.df.index, columns=av_columns)
        
        self.availability_matrix = av_df
        
        return self
    
    def create_train_test_split(self, target_train_obs=1983, random_seed=123):
        if len(self.df) > 100:
            df_sample = self.df.sample(n=target_train_obs, random_state=random_seed)
        else:
            df_sample = self.df.copy()
        
        df_sample = df_sample.dropna(subset=['CHOICE'])
        
        # Convert boolean columns to integers
        bool_columns = df_sample.select_dtypes(include=['bool']).columns
        for col in bool_columns:
            df_sample[col] = df_sample[col].astype(int)
        
        self.train_data = df_sample.copy()
        
        # Create test data from remaining observations
        df_remaining = self.df.drop(df_sample.index)
        if len(df_remaining) >= 300:
            # the sample size is adjustable
            self.test_data = df_remaining.sample(len(df_remaining), random_state=456)
            self.test_data = self.test_data.dropna(subset=['CHOICE'])
            
            bool_columns_test = self.test_data.select_dtypes(include=['bool']).columns
            for col in bool_columns_test:
                self.test_data[col] = self.test_data[col].astype(int)
        else:
            self.test_data = None
        
        # Create corresponding availability matrices
        if self.availability_matrix is not None:
            self.train_availability = self.availability_matrix.loc[self.train_data.index]
            if self.test_data is not None:
                self.test_availability = self.availability_matrix.loc[self.test_data.index]
            else:
                self.test_availability = None
        
        return self
    
    def calculate_utilities_fixed(self, data, availability_matrix, parameters, model_type="basic"):
        """
        Calculate utility values for each available zone.
        """
        utilities = pd.DataFrame(index=data.index)
        
        for zone in self.zones:
            utility_terms = []
            
            # Travel time effects
            tt_col = f'tt_{zone}'
            if tt_col in data.columns:
                tt_value = (data[tt_col])
                
                # Main travel time effect
                if 'B_tt' in parameters:
                    utility_terms.append(parameters['B_tt'] * tt_value)
                
                # Travel time interactions with age categories
                for suffix in ['B', 'C', 'D', 'E']:
                    param_key = f'B_tt_age_{suffix}'
                    col_key = f'age_category_{suffix}'

                    if param_key in parameters and col_key in data.columns:
                        age_value = data[col_key]
                        if age_value.dtype == bool:
                            age_value = age_value.astype(int)
                        utility_terms.append(parameters[param_key] * tt_value * age_value)

                
                # Travel time interaction with gender
                if 'B_tt_gender' in parameters:
                    gender_col_options = ['gender_2.0', 'gender_male', 'gender']
                    for gender_col in gender_col_options:
                        if gender_col in data.columns:
                            gender_value = data[gender_col]
                            if gender_value.dtype == bool:
                                gender_value = gender_value.astype(int)
                            utility_terms.append(parameters['B_tt_gender'] * tt_value * gender_value)
                            break
                
                # Flexibility interaction (sectoral model)
                if 'beta_tt_flexibility_flex' in parameters:
                    flex_col_options = ['work_flexibility_flexible', 'flexibility_flexible', 'flexible']
                    for flex_col in flex_col_options:
                        if flex_col in data.columns:
                            flex_value = data[flex_col]
                            if flex_value.dtype == bool:
                                flex_value = flex_value.astype(int)
                            utility_terms.append(parameters['beta_tt_flexibility_flex'] * tt_value * flex_value)
                            break
                
                # Household interaction (sectoral model)
                if 'hh_tt_with_child' in parameters:
                    child_col_options = ['hh_compund_grouped_with_child', 'hh_with_child', 'with_child']
                    for child_col in child_col_options:
                        if child_col in data.columns:
                            child_value = data[child_col]
                            if child_value.dtype == bool:
                                child_value = child_value.astype(int)
                            utility_terms.append(parameters['hh_tt_with_child'] * tt_value * child_value)
                            break
            
            # Urbanity effects (with level 1 as reference)
            for level in [2, 3, 4, 5]:
                param_name = f'B_urban_{level}'
                if param_name in parameters:
                    # Try different urbanity column naming conventions
                    urbanity_col_options = [
                        f'urbanity_level_{level}_{zone}',
                        f'urbanity_{level}_{zone}',
                        f'urban_level_{level}_{zone}'
                    ]
                    
                    found_urbanity = False
                    for urbanity_col in urbanity_col_options:
                        if urbanity_col in data.columns:
                            urbanity_value = data[urbanity_col]
                            utility_terms.append(parameters[param_name] * urbanity_value)
                            found_urbanity = True
                            break
                    
                    # If no dummy variables found, try base urbanity column
                    if not found_urbanity:
                        base_urbanity_col = f'urbanity_{zone}'
                        if base_urbanity_col in data.columns:
                            # Create dummy from base urbanity level
                            urbanity_dummy = (data[base_urbanity_col] == level).astype(int)
                            utility_terms.append(parameters[param_name] * urbanity_dummy)
            
            # Job opportunities effects
            if model_type == "basic":
                # Total jobs model
                job_col = f'total_jobs_{zone}'
                if job_col in data.columns and 'B_ln_total_jobs' in parameters:
                    job_value = data[job_col] * 0.001
                    log_job_value = np.log(job_value)
                    utility_terms.append(parameters['B_ln_total_jobs'] * log_job_value)
                    
            elif model_type == "sectoral":
                # Total jobs effect 
                job_col = f'total_jobs_{zone}'
                if job_col in data.columns and 'B_ln_total_jobs' in parameters:
                    job_value = data[job_col] * 0.001
                    log_job_value = np.log(job_value)
                    utility_terms.append(parameters['B_ln_total_jobs'] * log_job_value)
                
                # Sectoral jobs model
                for sector in self.sectors:
                    job_col = f'work_sector_{sector}_jobs_{zone}'
                    sector_var = f'sector_cluster_{sector}'
                    
                    if job_col in data.columns and sector_var in data.columns:
                        job_value = data[job_col]
                        sector_indicator = data[sector_var]
                        
                        # Convert boolean to numeric if needed
                        if sector_indicator.dtype == bool:
                            sector_indicator = sector_indicator.astype(int)
                        
                        # Calculate job share
                        total_jobs_col = f'total_jobs_{zone}'
                        if total_jobs_col in data.columns:
                            total_jobs = data[total_jobs_col]
                            job_share = job_value / (total_jobs + 0.001)
                            
                            if sector == 'Remaining':
                                utility_terms.append(1.0 * job_share * sector_indicator)
                            else:
                                param_name = f'B_share_{sector}'
                                if param_name in parameters:
                                    utility_terms.append(parameters[param_name] * job_share * sector_indicator)
            
            # Sum all utility terms
            if utility_terms:
                total_utility = sum(utility_terms)
                # Apply availability constraint
                av_col = f'AV{zone}'
                if av_col in availability_matrix.columns:
                    utilities[f'U_{zone}'] = total_utility * availability_matrix[av_col]
                else:
                    utilities[f'U_{zone}'] = total_utility
            else:
                utilities[f'U_{zone}'] = 0
        
        return utilities
    
    def calculate_probabilities(self, utilities, availability_matrix):
        """Calculate choice probabilities using logit model."""
        probabilities = pd.DataFrame(index=utilities.index)
        
        for idx in utilities.index:
            # Get available zones for this agent
            av_row = availability_matrix.loc[idx]
            available_zones = [col.replace('AV', '') for col in av_row.index if av_row[col] == 1]
            
            if len(available_zones) == 0:
                # No available zones - assign zero probabilities
                for zone in self.zones:
                    probabilities.loc[idx, f'P_{zone}'] = 0
                continue
            
            # Get utilities for available zones
            available_utilities = {}
            for zone in available_zones:
                util_col = f'U_{zone}'
                if util_col in utilities.columns:
                    available_utilities[zone] = utilities.loc[idx, util_col]
            
            if len(available_utilities) == 0:
                # No utilities calculated
                for zone in self.zones:
                    probabilities.loc[idx, f'P_{zone}'] = 0
                continue
            
            # Apply softmax (logit choice probabilities)
            utilities_array = np.array(list(available_utilities.values()))
            max_utility = np.max(utilities_array)
            exp_utilities = np.exp(utilities_array - max_utility)
            sum_exp = np.sum(exp_utilities)
            
            if sum_exp == 0:
                prob_value = 1.0 / len(available_zones)
                for zone in available_zones:
                    probabilities.loc[idx, f'P_{zone}'] = prob_value
            else:
                for i, zone in enumerate(available_zones):
                    probabilities.loc[idx, f'P_{zone}'] = exp_utilities[i] / sum_exp
            
            # Set probabilities to 0 for unavailable zones
            for zone in self.zones:
                if str(zone) not in available_zones:
                    probabilities.loc[idx, f'P_{zone}'] = 0
        
        return probabilities
    
    def calculate_model_performance(self, data, probabilities, chosen_col='CHOICE'):
        """Calculate model performance metrics."""
        metrics = {}
        
        # Log-likelihood
        log_likelihood = 0
        chosen_probs = []
        
        for idx in data.index:
            chosen_zone = str(int(data.loc[idx, chosen_col]))
            prob_col = f'P_{chosen_zone}'
            
            if prob_col in probabilities.columns:
                prob = probabilities.loc[idx, prob_col]
                if prob > 0:
                    log_likelihood += np.log(prob)
                    chosen_probs.append(prob)
                else:
                    log_likelihood += np.log(1e-10)
                    chosen_probs.append(1e-10)
        
        metrics['log_likelihood'] = log_likelihood
        metrics['mean_chosen_prob'] = np.mean(chosen_probs)
        
        # Top-k accuracy
        for k in [1, 5, 10]:
            correct = 0
            for idx in data.index:
                chosen_zone = str(int(data.loc[idx, chosen_col]))
                
                agent_probs = {}
                for col in probabilities.columns:
                    if col.startswith('P_'):
                        zone = col.replace('P_', '')
                        prob = probabilities.loc[idx, col]
                        if prob > 0:
                            agent_probs[zone] = prob
                
                if len(agent_probs) > 0:
                    sorted_zones = sorted(agent_probs.items(), key=lambda x: x[1], reverse=True)
                    top_k_zones = [zone for zone, prob in sorted_zones[:k]]
                    
                    if chosen_zone in top_k_zones:
                        correct += 1
            
            metrics[f'top_{k}_accuracy'] = (correct / len(data)) * 100
        
        metrics['n_obs'] = len(data)
        
        return metrics
    
    def calculate_sectoral_probabilities(self, params_sectoral, sample_size=None, use_test_data=False):
        """Calculate probabilities for the sectoral model."""
        # Select data to use
        if use_test_data and self.test_data is not None:
            data_sample = self.test_data
            availability_sample = self.test_availability
        else:
            data_sample = self.train_data if self.train_data is not None else self.df
            availability_sample = self.train_availability if self.train_availability is not None else self.availability_matrix

        # Sample data if requested
        if sample_size is not None and len(data_sample) > sample_size:
            data_sample = data_sample.sample(n=sample_size, random_state=42)
            availability_sample = availability_sample.loc[data_sample.index]

        utilities_sectoral = self.calculate_utilities_fixed(data_sample, availability_sample, params_sectoral, "sectoral")
        probabilities_sectoral = self.calculate_probabilities(utilities_sectoral, availability_sample)

        # Store probabilities
        if use_test_data and self.test_data is not None:
            self.probabilities_sectoral_test = probabilities_sectoral.copy()
        else:
            self.probabilities_sectoral = probabilities_sectoral.copy()

        # Calculate performance metrics
        metrics_sectoral = self.calculate_model_performance(data_sample, probabilities_sectoral)

        return metrics_sectoral, probabilities_sectoral
    


    def save_probabilities_to_csv(self, output_dir="./model_outputs"):
        """Save stored probabilities to CSV files."""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = []
        
        if self.probabilities_basic is not None:
            basic_train_path = os.path.join(output_dir, "probabilities_basic_train.csv")
            self.probabilities_basic.to_csv(basic_train_path, index=True)
            saved_files.append(basic_train_path)
        
        if self.probabilities_sectoral is not None:
            sectoral_train_path = os.path.join(output_dir, "probabilities_sectoral_train.csv")
            self.probabilities_sectoral.to_csv(sectoral_train_path, index=True)
            saved_files.append(sectoral_train_path)
        
        if self.probabilities_basic_test is not None:
            basic_test_path = os.path.join(output_dir, "probabilities_basic_test.csv")
            self.probabilities_basic_test.to_csv(basic_test_path, index=True)
            saved_files.append(basic_test_path)
        
        if self.probabilities_sectoral_test is not None:
            sectoral_test_path = os.path.join(output_dir, "probabilities_sectoral_test.csv")
            self.probabilities_sectoral_test.to_csv(sectoral_test_path, index=True)
            saved_files.append(sectoral_test_path)
        
        return saved_files
    
    def get_probabilities(self, model_type="both", data_type="both"):
        """Get stored probabilities."""
        results = {}
        
        if model_type in ["basic", "both"]:
            if data_type in ["train", "both"] and self.probabilities_basic is not None:
                results['basic_train'] = self.probabilities_basic
            if data_type in ["test", "both"] and self.probabilities_basic_test is not None:
                results['basic_test'] = self.probabilities_basic_test
        
        if model_type in ["sectoral", "both"]:
            if data_type in ["train", "both"] and self.probabilities_sectoral is not None:
                results['sectoral_train'] = self.probabilities_sectoral
            if data_type in ["test", "both"] and self.probabilities_sectoral_test is not None:
                results['sectoral_test'] = self.probabilities_sectoral_test
        
        return results
    
    

def main():
    """Example usage of the ModelComparison class."""
    
    # Sectoral model parameters
    params_sectoral = {
        'B_ln_total_jobs': 0.773,
        'B_share_Care': 2.65,
        'B_share_Education': 4.42,
        'B_share_Hospitality_Culture_Sports': 1.63,
        'B_share_Industry_Construction': 1.72,
        'B_share_Office_Services': 1.93,
        'B_share_Shop': -0.14,
        'B_tt': -0.0558,
        'B_tt_age_B': 9.23e-05,
        'B_tt_age_D' : -0.0047,
        'B_tt_age_E' : -0.0077,
        'B_tt_gender': -0.011,
        'B_urban_2': 0.327,
        'B_urban_3': 0.415,
        'B_urban_4': 0.327,
        'B_urban_5': 0.0991,
        'beta_tt_flexibility_flex': 0.149,
        'hh_tt_with_child': -0.00621
    }

    # Get the script directory and navigate to data folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), 'data')
    data_path = os.path.join(data_dir, 'final_wide_format', 'wide_data_filtered_v6.csv')

    mc = ModelComparison(data_path=data_path)
    mc.preprocess_data()
    mc.create_availability_matrix()
    mc.create_train_test_split(target_train_obs=1983, random_seed=123)

    # Calculate sectoral probabilities
    metrics, probabilities = mc.calculate_sectoral_probabilities(params_sectoral, sample_size=None, use_test_data=False)

    # Save probabilities to CSV files
    saved_files = mc.save_probabilities_to_csv(output_dir="./model_outputs")

    return metrics, saved_files

if __name__ == "__main__":
    main()