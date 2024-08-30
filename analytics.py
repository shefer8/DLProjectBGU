import importlib
import inspect
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import *
from sklearn.preprocessing import StandardScaler

def plot_histogram_of_records_per_day(data, hormone):
    # Plot histogram
    day_value_counts = data.groupby('DayInCycle')[hormone].count().sort_index()

    # Plot the bar chart
    plt.figure(figsize=(10, 6))
    day_value_counts.plot(kind='bar')
    plt.xlabel('DayInCycle')
    plt.ylabel(f'Number of {hormone} values')
    plt.title(f'Number of {hormone} values per DayInCycle normalized')
    # plt.show() #hadas

def plot_one_hormone(pivoted_data, data_array, hormone, hormone_number):
    xlabel = 'DayInCycle'
    plt.subplot(3, 2, hormone_number * 2 + 1)
    plt.scatter(pivoted_data.index.get_level_values(0), data_array[:, :, hormone_number].flatten(), alpha=0.5)
    plt.title(f'{hormone} Levels by {xlabel} (Scatter)')
    plt.xlabel(xlabel)
    plt.ylabel(hormone)

    plt.subplot(3, 2, hormone_number * 2 + 2)
    sns.violinplot(x=xlabel, y=hormone,
                   data=pivoted_data.reset_index().melt(id_vars=['Medical_Record'], value_vars=[hormone]),
                   inner='quartile')
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
    # plt.show() #hadas


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
    # plt.show() #hadas


