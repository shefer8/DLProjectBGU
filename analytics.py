import importlib
import inspect
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import *

def load_and_preprocess_data(file_path):
    # Load the CSV file
    data = pd.read_csv(file_path)
    print("data_statistics after load :")
    print_data_statistics(data)

    # Convert relevant columns to numeric, forcing non-numeric values to NaN
    data['Estrogen'] = pd.to_numeric(data['Estrogen'], errors='coerce')
    if WITH_LH:
        data['LH'] = pd.to_numeric(data['LH'], errors='coerce')
        # move the LH column to the end
        data = data[['Medical_Record', 'ResultDate', 'Estrogen', 'Progesterone', 'LH']]
    else:
        data = data[['Medical_Record', 'ResultDate', 'Estrogen', 'Progesterone']]
    data['Progesterone'] = pd.to_numeric(data['Progesterone'], errors='coerce')
    # data['DayInCycle'] = pd.to_numeric(data['DayInCycle'], errors='coerce')
    data['ResultDate'] = pd.to_datetime(data['ResultDate'], format = "%d/%m/%Y")
    
    # rans steps:
    #
    # # Sort the dataframe by Medical_Record and ResultDate
    # data = data.sort_values(by=['Medical_Record', 'ResultDate'])
    #
    # # Function to calculate day_in_cycle based on actual differences
    # def calculate_day_in_cycle(group):
    #     group = group.copy()
    #     group['day_in_cycle'] = (group['ResultDate'] - group['ResultDate'].min()).dt.days
    #     return group
    #
    # # Apply the function to each group of Medical_Record
    # data = data.groupby('Medical_Record').apply(calculate_day_in_cycle).reset_index(drop=True)
    #
    # # Function to clean hormone values
    # def clean_hormone_value(value):
    #     try:
    #         # Remove non-numeric characters and convert to float
    #         cleaned_value = float(''.join(filter(lambda x: x.isdigit() or x == '.', str(value))))
    #     except ValueError:
    #         # If conversion fails, return NaN
    #         cleaned_value = pd.NA
    #     return cleaned_value
    #
    # # Apply the cleaning function to hormone columns
    # data['Estrogen'] = data['Estrogen'].apply(clean_hormone_value)
    # data['LH'] = data['LH'].apply(clean_hormone_value)
    # data['Progesterone'] = data['Progesterone'].apply(clean_hormone_value)
    #
    # # until here ran
    
    # Ensure each Medical_Record and DayInCycle combination is unique by taking the max values
    if WITH_LH:
        data = data.groupby(['Medical_Record', 'ResultDate']).agg({
            'Estrogen': 'max',
            'LH': 'max',
            'Progesterone': 'max'
        }).reset_index()
    else:
        data = data.groupby(['Medical_Record', 'ResultDate']).agg({
            'Estrogen': 'max',
            'Progesterone': 'max'
        }).reset_index()

    # set DayInCycle as the order of ResultDate within each Medical_Record
    # data = data.sort_values(by=['Medical_Record', 'ResultDate'])
    # data['DayInCycle'] = data.groupby('Medical_Record').rank() #cumcount()
    data['DayInCycle'] = data.groupby('Medical_Record')['ResultDate'].transform(lambda x: (x - x.min()).dt.days)

    # add empty rows for the data, according to the cut_off varaible
    all_days = pd.DataFrame({'DayInCycle': np.arange(MIN_DAYINCYCLE, MAX_DAYINCYCLE + 1)})
    complete_data = data[['Medical_Record']].drop_duplicates().merge(all_days, how='cross')

    # Merge the complete days range with the original data
    data = complete_data.merge(data, on=['Medical_Record', 'DayInCycle'], how='left')
    data = data[['Medical_Record', 'DayInCycle', 'Estrogen', 'Progesterone', 'LH']] if WITH_LH else data[['Medical_Record', 'DayInCycle', 'Estrogen', 'Progesterone']]

    print_data_statistics(data)
    print(data.head(10))

    # # plot histograms of number of existing values in each feature, to decide on the cut_off var
    # plot_histogram_of_records_per_day(data, 'Estrogen')
    # plot_histogram_of_records_per_day(data, 'LH')
    # plot_histogram_of_records_per_day(data, 'Progesterone')

    print(f'based on the histograms, data will be filtered to only records up to day {MAX_DAYINCYCLE}')
    print(f'percent of nan before cut = {data.isna().mean() * 100}')
    data = data[data['DayInCycle'] <= MAX_DAYINCYCLE]
    print(f'percent of nan values = {data.isna().mean() * 100}')

    # exit()
    return data

def normalize_data(data_array):
    # Calculate min and max for each feature separately
    min_val = np.nanmin(data_array, axis=(0,1), keepdims=True) # when used on the df column, used only on axis 0
    max_val = np.nanmax(data_array, axis=(0,1), keepdims=True)

    # Normalize each feature separately
    data_array = (data_array - min_val) / (max_val - min_val)

    return data_array

def plot_histogram_of_records_per_day(data, hormone):
    # Plot histogram
    day_value_counts = data.groupby('DayInCycle')[hormone].count().sort_index()

    # Plot the bar chart
    plt.figure(figsize=(10, 6))
    day_value_counts.plot(kind='bar')
    plt.xlabel('DayInCycle')
    plt.ylabel(f'Number of {hormone} values')
    plt.title(f'Number of {hormone} values per DayInCycle normalized')
    plt.show()

def pivot_data(data):
    # Pivot the data to create sequences for each Medical_Record
    values_for_pivot = ['Estrogen', 'Progesterone', 'LH'] if WITH_LH else ['Estrogen', 'Progesterone']
    pivoted_data = data.pivot(index='Medical_Record', columns='DayInCycle', values=values_for_pivot)
    return pivoted_data

def initialize_missing_values(data_array, heuristic_mean=True):
    if heuristic_mean:
        # # Initialize missing values with column mean (simple heuristic)
        # for feature in range(data_array.shape[2]):
        #     feature_mean = np.nanmean(data_array[:, :, feature])
        #     data_array[np.isnan(data_array[:, :, feature]), feature] = feature_mean
        #
        # return data_array
        return initialize_missing_values_with_defaults(data_array)
    else:
        return  initialize_missing_values_by_prior(data_array)

def initialize_missing_values_by_prior(data_array):
    # Load the estrogen and lh data
    estro = pd.read_csv('estradiol and progesterone data.csv')
    estro['days_from_lh_peak'] = estro['days_from_lh_peak'] + 15     # shift the days_from_lh_peak values by 15 to align with the day in cycle from our data
    prog = estro.iloc[:, [0, 7]]
    estro = estro.iloc[:, [0, 2]]
    # Create dictionaries for fast look-up
    estro_dict = estro.set_index('days_from_lh_peak').to_dict()['estradiol_pmol_l_mean']
    prog_dict = prog.set_index('days_from_lh_peak').to_dict()['progesterone_nmol_l_mean']

    if WITH_LH:
        lh = pd.read_csv('lh and fsh data.csv').iloc[:, [0, 2]]
        lh['days_from_lh_peak'] = lh['days_from_lh_peak'] + 15     # shift the days_from_lh_peak values by 15 to align with the day in cycle from our data
        lh_dict = lh.set_index('days_from_lh_peak').to_dict()['lh_iu_l_mean']     # Create dictionaries for fast look-up

    # Replace NaN values in data_array with prior mean values
    for day in range(0, MAX_DAYINCYCLE):
        if day in estro_dict:
            data_array[:, day, 0] = np.where(np.isnan(data_array[:, day, 0]), estro_dict[day], data_array[:, day, 0])
        if day in prog_dict:
            data_array[:, day, 1] = np.where(np.isnan(data_array[:, day, 1]), prog_dict[day], data_array[:, day, 1])
        if day in lh_dict and WITH_LH:
            data_array[:, day, 2] = np.where(np.isnan(data_array[:, day, 2]), lh_dict[day], data_array[:, day, 2])
    return data_array

def initialize_missing_values_with_defaults(data_array, default_estrogen=0, default_lh=0, default_progesterone=0):
    """
    Initialize missing values in the data array with default values for Estrogen, LH, and Progesterone.

    Parameters:
    data_array (numpy.ndarray): The input data array with shape (n_samples, n_days, n_features).
    default_estrogen (float): The default value to use for missing Estrogen values.
    default_lh (float): The default value to use for missing LH values.
    default_progesterone (float): The default value to use for missing Progesterone values.

    Returns:
    numpy.ndarray: The data array with missing values filled in.
    """
    # Define default values for each feature
    default_values = [default_estrogen, default_lh, default_progesterone] if WITH_LH else [default_estrogen, default_progesterone]

    # Iterate over each feature and fill NaN values with the corresponding default value
    for feature_index in range(data_array.shape[2]):
        data_array[:, :, feature_index] = np.where(
            np.isnan(data_array[:, :, feature_index]),
            default_values[feature_index],
            data_array[:, :, feature_index]
        )
    #print('**********  data_statistics after default value set:')
    #print_data_statistics(data_array)
    return data_array

def calculating_std_for_gmm(data, full_hormone_name):
    return (data[f'{full_hormone_name}_l_95th_percentile'] - data[f'{full_hormone_name}_l_5th_percentile'])/2*1.645

def means_and_std_into_array(prog, estro, lh = None):
    # calculating std not in a very good way (assuming normal dist)
    prog['std'] = calculating_std_for_gmm(prog, 'progesterone_nmol')
    estro['std'] = calculating_std_for_gmm(estro, 'estradiol_pmol')
    # turn the columns into arrays
    means_prog = np.array(prog.loc[prog['days_from_lh_peak'] <= MAX_DAYINCYCLE, 'progesterone_nmol_l_mean'])
    stds_prog = np.array(prog.loc[prog['days_from_lh_peak'] <= MAX_DAYINCYCLE, 'std'])

    means_estro = np.array(estro.loc[estro['days_from_lh_peak'] <= MAX_DAYINCYCLE, 'estradiol_pmol_l_mean'])
    stds_estro = np.array(estro.loc[estro['days_from_lh_peak'] <= MAX_DAYINCYCLE, 'std'])

    if WITH_LH:
        lh['std'] = calculating_std_for_gmm(lh, 'lh_iu')
        # turn the columns into arrays
        means_lh = np.array(lh.loc[lh['days_from_lh_peak'] <= MAX_DAYINCYCLE, 'lh_iu_l_mean'])
        stds_lh = np.array(lh.loc[lh['days_from_lh_peak'] <= MAX_DAYINCYCLE, 'std'])

    # Combine means and stds into a single array
    means = np.column_stack((means_prog, means_estro, means_lh)).T if WITH_LH else np.column_stack((means_prog, means_estro)).T
    stds = np.column_stack((stds_prog, stds_estro, stds_lh)).T if WITH_LH else np.column_stack((stds_prog, stds_estro)).T

    return means, stds
def means_covs_for_gmm(use_prior_mean_and_std):
    if use_prior_mean_and_std:
        estro = pd.read_csv('estradiol and progesterone data.csv')
        # shift the days_from_lh_peak values by 15 to align with the day in cycle from our data
        estro['days_from_lh_peak'] = estro['days_from_lh_peak'] + 15
        prog = estro.iloc[:, [0, 6,7,8,9,10]]
        estro = estro.iloc[:, 0:6]

        if WITH_LH:
            lh = pd.read_csv('lh and fsh data.csv')#.iloc[:, [0, 2]]
            lh['days_from_lh_peak'] = lh['days_from_lh_peak'] + 15
            lh = lh.iloc[:, 0:6]

        means, stds = means_and_std_into_array(prog, estro, lh) if WITH_LH else means_and_std_into_array(prog, estro)
    else:
        means, stds = None, None
    return means, stds #np.array(covariances)

def plot_one_hormone(pivoted_data, data_array, hormone, hormone_number):
    xlabel = 'DayInCycle'
    plt.subplot(3, 2, hormone_number * 2 + 1)
    plt.scatter(pivoted_data.index.get_level_values(0), data_array[:, :, hormone_number].flatten(), alpha=0.5)
    plt.title(f'{hormone} Levels by {xlabel} (Scatter)')
    plt.xlabel(xlabel)
    plt.ylabel(hormone)

    plt.subplot(3, 2, hormone_number * 2 + 2)
    sns.violinplot(x=xlabel, y=hormone, data=pivoted_data.reset_index().melt(id_vars=['Medical_Record'], value_vars=[hormone]), inner='quartile')
    plt.title(f'{hormone} Levels by {xlabel} (Violin)')
    plt.xlabel(xlabel)
    plt.ylabel(hormone)


def plot_data(pivoted_data):
    # Convert the pivoted DataFrame to a numpy array with the required shape
    data_array = pivoted_data.values.reshape(pivoted_data.shape[0], 28, N_FEATURES)

    # Plotting
    plt.figure(figsize=(18, 10))

    # Estrogen scatter and violin plot
    plot_one_hormone(pivoted_data, data_array, 'Estrogen', 0)

    # Progesterone scatter and violin plot
    plot_one_hormone(pivoted_data, data_array, 'Progesterone', 1)

    if WITH_LH:
        # LH scatter and violin plot
        plot_one_hormone(pivoted_data, data_array, 'LH', 2)

    plt.tight_layout()
    plt.show()

def check_for_nan(data_array):
    # Check for any NaN values in the initialized data
    print("Number of NaN values in the initialized data:", np.isnan(data_array).sum())

# for logging config global parameters
def print_variables_from_module(module_name):
    # Import the module
    module = importlib.import_module(module_name)

    # Get all attributes from the module
    attributes = inspect.getmembers(module)

    # Filter out functions, modules, and built-in attributes
    variables = {name: value for name, value in attributes if
                 not (name.startswith("__") or inspect.ismodule(value) or inspect.isfunction(value))}

    # Print all variable names and their values
    print('Global parameters values are:')
    logging.info('Global parameters values are:')

    for var_name, var_value in variables.items():
        print(f"{var_name} = {var_value}")
        logging.info(f"{var_name} = {var_value}")

def print_data_statistics(data):
    # Distinct count of Medical Records
    distinct_medical_records = data['Medical_Record'].nunique()
    print(f"Distinct Medical Records: {distinct_medical_records}")

    # Number of non-NaN values for each measurement
    columns = ['Estrogen', 'Progesterone', 'LH'] if WITH_LH else ['Estrogen', 'Progesterone']
    non_nan_counts = data[columns].notna().sum()
    print("Number of non-NaN values for each measurement:")
    print(non_nan_counts)
    #
    # # Number of distinct DayInCycle values for each measurement per Medical Record
    # distinct_days_estrogen = data.dropna(subset=['Estrogen']).groupby('Medical_Record')['DayInCycle'].nunique()
    # distinct_days_lh = data.dropna(subset=['LH']).groupby('Medical_Record')['DayInCycle'].nunique()
    # distinct_days_progesterone = data.dropna(subset=['Progesterone']).groupby('Medical_Record')['DayInCycle'].nunique()
    #
    # # Print descriptive statistics for each measurement
    # print("Number of distinct DayInCycle values for each Medical Record (Estrogen):")
    # print(distinct_days_estrogen.describe())
    # print("Number of distinct DayInCycle values for each Medical Record (LH):")
    # print(distinct_days_lh.describe())
    # print("Number of distinct DayInCycle values for each Medical Record (Progesterone):")
    # print(distinct_days_progesterone.describe())
    #
    # # Create histograms for each measurement
    # plt.figure(figsize=(18, 6))
    #
    # plt.subplot(1, 3, 1)
    # distinct_days_estrogen.hist(bins=range(1, 18), edgecolor='black', align='left')
    # plt.title('Histogram of Distinct DayInCycle Values (Estrogen)')
    # plt.xlabel('Number of Distinct DayInCycle Values')
    # plt.ylabel('Number of Medical Records')
    # plt.xticks(range(1, 18))
    #
    # plt.subplot(1, 3, 2)
    # distinct_days_lh.hist(bins=range(1, 18), edgecolor='black', align='left')
    # plt.title('Histogram of Distinct DayInCycle Values (LH)')
    # plt.xlabel('Number of Distinct DayInCycle Values')
    # plt.ylabel('Number of Medical Records')
    # plt.xticks(range(1, 18))
    #
    # plt.subplot(1, 3, 3)
    # distinct_days_progesterone.hist(bins=range(1, 18), edgecolor='black', align='left')
    # plt.title('Histogram of Distinct DayInCycle Values (Progesterone)')
    # plt.xlabel('Number of Distinct DayInCycle Values')
    # plt.ylabel('Number of Medical Records')
    # plt.xticks(range(1, 18))

    plt.tight_layout()
    plt.show()