# Data Preparation Pipeline

This deliverable contains multiple components: data files, Python scripts (py_scr), notebooks, SQL files, and outputs.

The modeling framework uses three main datasets:
- LRO survey dataset (provided in metadata format)
- Travel time data 
- Zonal data (primarily job count data for this project)

All raw and processed data are available in the `/data` directory.

Multiple files work together to clean and filter data, and format it appropriately for the BIOGEME library estimation process.

Code is provided in two formats: interactive notebooks and clean Python (.py) files.

## LRO Filtering - Agent Data

The notebook `LRO_filtering_exploring.ipynb` documents the process of filtering and processing the LRO dataset, including variable selection for analysis. Supporting files include:
- [Variable explanations](data/LRO 2023 met ww afstand/variable_explanations.txt)
- [Complete LRO documentation](data/LRO 2023 met ww afstand/landelijk-reizigersonderzoek-2023.PDF) - an extensive document containing detailed questions and categories

This notebook creates filtered agent data for subsequent use. The processed data is saved as [df_agent_v5.csv](data/LRO 2023 met ww afstand/df_agent_v5.csv).

## Geographic Coupling

The notebook [geo_coupling.ipynb](notebooks/geo_coupling.ipynb) matches zone IDs (neighborhoods) to PC4 codes (larger administrative areas). The process uses a geographic referencing method:

1. Takes centroids of zones (finer administrative levels)
2. Creates a 1000m buffer around each centroid
3. Finds intersections with PC4 zones
4. Matches to the PC4 zone with the largest intersection area

The output includes PC4 codes matched to each row of neighborhood-level zonal data, saved as [df_coupling.csv](data/geo_coupled/df_coupling.csv). An alternative coupling file from MobilitySpectrum (using a different methodology) is also available: [koppeling buurten pc4.xlsx](data/geo_coupled/koppeling buurten pc4.xlsx).



## Wide Data Format Creation

The notebook [create_wide_data.ipynb](notebooks/create_wide_data.ipynb) transforms filtered data into wide format, which is required by the Biogeme library for Multinomial Logit (MNL) model estimation. In wide format, each row represents a single decision maker with all choice alternatives as columns.

The notebook performs the following tasks:
- Imports and processes filtered agent data
- Structures data with person-specific and alternative-specific variables
- Includes detailed comments explaining each transformation step

A Python script version is available at [py_scr/create_wide_data.py](py_scr/create_wide_data.py) for command-line execution.

The process generates wide dataset output used for Biogeme estimation in both extended and basic models: [wide_data_filtered_v6.csv](data/final_wide_format/wide_data_filtered_v6.csv).

### Travel Time Data Processing

The process incorporates aggregated travel time data (PC4 to PC4) as variables for each data point. The travel time data was originally provided by Goudappel in [d260_01_weerstanden_skims_auto_met_grensovergangen.sql](sql/d260_01_weerstanden_skims_auto_met_grensovergangen.sql), containing GID-to-GID travel times.

To make this data usable, the coupling dataset `df_coupling.csv` connects GID to zone ID and ultimately to PC4 codes. Travel time aggregation to PC4 level is performed in [non_symmetric_tt_v2.sql](sql/non_symmetric_tt_v2.sql), which calculates average travel times across all uniquely coupled PC4-to-PC4 combinations directly in the database.

The raw travel time data contains:
- Origin and destination GIDs
- Evening, morning, and normal travel times
- Distance measurements

The SQL code transforms this to show PC4 origins and destinations, filtering to include only records where at least one PC4 matches the declared residential PC4 of agents in the `df_agent` dataset.

The notebook [create_wide_data.ipynb](notebooks/create_wide_data.ipynb) imports pre-processed travel time data from [wide_data_tt.csv](data/travel_time/wide_data_tt.csv). This file was generated through a process that:

1. Extracts morning travel times from origin (residence) to various destinations
2. Adds evening travel times from destination back to origin
3. Stores combined round-trip travel time as a single column for each person with a known residence

This processed travel time data is incorporated into the wider dataset for destination choice modeling.

# Biogeme Model Estimation

This project includes two separate estimation files for Multinomial Logit (MNL) models:

## 1. Basic Model: [`MNL_BIOGEME_basic.py`](py_scr/MNL_BIOGEME_basic.py)
- Contains a limited set of explanatory variables
- Includes availability conditions based solely on travel time
- Replicates the Octavius work destination choice model for car travel

## 2. Extended Model: [`MNL_BIOGEME_extended.py`](py_scr/MNL_BIOGEME_extended.py)
- Builds upon the basic model with additional variables
- Incorporates sectoral variables and agent characteristics
- Implements enhanced availability conditions (detailed in file comments)

Both files perform additional data processing and sampling for training sets, using identical random seeds to ensure model comparability. While these models could be combined into a single file, they are separated for clarity and ease of use.

Estimation results are available as standard Biogeme outputs in the [outputs](outputs) directory.

## Probability Calculation

The probability calculation file [calculate_probabilities.py](py_scr/calculate_probabilities.py) uses the same samples and filtering to provide probabilities for both training and test samples across all alternatives for each observed agent, for both basic and extended models. This file supports further project analysis, with training set outputs available in [outputs/probabilities](outputs/probabilities).

