from config import *
from analytics import *
from sklearn.preprocessing import StandardScaler

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
    data['ResultDate'] = pd.to_datetime(data['ResultDate'], format="%d/%m/%Y")

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
    data = data[['Medical_Record', 'DayInCycle', 'Estrogen', 'Progesterone', 'LH']] if WITH_LH else data[
        ['Medical_Record', 'DayInCycle', 'Estrogen', 'Progesterone']]

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


def normalize_data(data, method='Min_Max'):
    if method == 'z_score':
        mean = np.mean(data, axis=(0, 1))
        std = np.std(data, axis=(0, 1))
        normalized_data = (data - mean) / std
        return normalized_data
    elif method == 'Min_Max':
        min_val = np.nanmin(data, axis=(0, 1), keepdims=True)  # when used on the df column, used only on axis 0
        max_val = np.nanmax(data, axis=(0, 1), keepdims=True)
        # Normalize each feature separately
        normalized_data = (data - min_val) / (max_val - min_val)
        return normalized_data
    elif method == 'feature_wise_normalize':
        scaler = StandardScaler()
        for i in range(data.shape[2]):
            data[:, :, i] = scaler.fit_transform(data[:, :, i])
        return data
    elif method.startswith('baseline_correction'):
        # mean_baseline, std_baseline = means_stds_for_gmm(True)
        # mean_baseline = mean_baseline.T[np.newaxis, :, :]
        # std_baseline = std_baseline.T[np.newaxis, :, :]
        # normalized_data = (data - mean_baseline) / std_baseline
        means, stds = means_stds_for_gmm(True)
        hormones_dict = means_stds_array_to_dicts(means, stds)
        prog_dict = hormones_dict[0]
        estro_dict = hormones_dict[1]
        lh_dict = hormones_dict[2] if WITH_LH else None
        for day in range(0, MAX_DAYINCYCLE):
            if day in estro_dict['means']:
                data[:, day, 0] = (data[:, day, 0] - estro_dict['means'][day]) / estro_dict['stds'][day]
            if day in prog_dict['means']:
                data[:, day, 0] = (data[:, day, 0] - prog_dict['means'][day]) / prog_dict['stds'][day]
            if WITH_LH and day in lh_dict['means']:
                data[:, day, 0] = (data[:, day, 0] - lh_dict['means'][day]) / lh_dict['stds'][day]
        if not method.endswith('baseline_correction'):
            data = normalize_data(data, method=method.replace('baseline_correction_and_', ''))
        return data

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
        return initialize_missing_values_by_prior(data_array)


def initialize_missing_values_by_prior(data_array):
    estro_dict, prog_dict, lh_dict = prepare_prior_hormones_dict()
    # Replace NaN values in data_array with prior mean values
    for day in range(0, MAX_DAYINCYCLE):
        if day in estro_dict:
            data_array[:, day, 0] = np.where(np.isnan(data_array[:, day, 0]), estro_dict[day], data_array[:, day, 0])
        if day in prog_dict:
            data_array[:, day, 1] = np.where(np.isnan(data_array[:, day, 1]), prog_dict[day], data_array[:, day, 1])
        if WITH_LH and day in lh_dict:
            data_array[:, day, 2] = np.where(np.isnan(data_array[:, day, 2]), lh_dict[day], data_array[:, day, 2])
    return data_array

def prepare_prior_hormones_dict():
    # Load the estrogen and lh data
    estro = pd.read_csv('estradiol and progesterone data.csv')
    estro['days_from_lh_peak'] = estro[
                                     'days_from_lh_peak'] + 15  # shift the days_from_lh_peak values by 15 to align with the day in cycle from our data
    prog = estro.iloc[:, [0, 7]]
    estro = estro.iloc[:, [0, 2]]
    # Create dictionaries for fast look-up
    estro_dict = estro.set_index('days_from_lh_peak').to_dict()['estradiol_pmol_l_mean']
    prog_dict = prog.set_index('days_from_lh_peak').to_dict()['progesterone_nmol_l_mean']

    if WITH_LH:
        lh = pd.read_csv('lh and fsh data.csv').iloc[:, [0, 2]]
        lh['days_from_lh_peak'] = lh[
                                      'days_from_lh_peak'] + 15  # shift the days_from_lh_peak values by 15 to align with the day in cycle from our data
        lh_dict = lh.set_index('days_from_lh_peak').to_dict()['lh_iu_l_mean']  # Create dictionaries for fast look-up
    else:
        lh_dict = None
    return estro_dict, prog_dict, lh_dict

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
    default_values = [default_estrogen, default_lh, default_progesterone] if WITH_LH else [default_estrogen,
                                                                                           default_progesterone]

    # Iterate over each feature and fill NaN values with the corresponding default value
    for feature_index in range(data_array.shape[2]):
        data_array[:, :, feature_index] = np.where(
            np.isnan(data_array[:, :, feature_index]),
            default_values[feature_index],
            data_array[:, :, feature_index]
        )

    return data_array

def means_stds_array_to_dicts(means, stds):
    # order of hormones in the dict = ['Progesterone', 'Estrogen', 'LH']
    n_hormones, n_days = means.shape  # Get the number of hormones and days

    # Create a dictionary for each hormone
    dicts = {i: {'means': {}, 'stds': {}} for i in range(n_hormones)}

    # Populate the dictionaries with the mean and std values
    for i in range(n_hormones):
        for day in range(n_days):
            dicts[i]['means'][day] = means[i, day]
            dicts[i]['stds'][day] = stds[i, day]
    return dicts

def calculating_std_for_gmm(data, full_hormone_name):
    return (data[f'{full_hormone_name}_l_95th_percentile'] - data[f'{full_hormone_name}_l_5th_percentile']) / 2 * 1.645


def means_and_std_into_array(prog, estro, lh=None):
    # calculating std not in a very good way (assuming normal dist)
    # because the priors data starts in day 0, unlike the ivf days cycle data, we are filtering days < from max_dayincycle (and not =<)
    prog['std'] = calculating_std_for_gmm(prog, 'progesterone_nmol')
    estro['std'] = calculating_std_for_gmm(estro, 'estradiol_pmol')
    # turn the columns into arrays
    means_prog = np.array(prog.loc[prog['days_from_lh_peak'] < MAX_DAYINCYCLE, 'progesterone_nmol_l_mean'])
    stds_prog = np.array(prog.loc[prog['days_from_lh_peak'] < MAX_DAYINCYCLE, 'std'])

    means_estro = np.array(estro.loc[estro['days_from_lh_peak'] < MAX_DAYINCYCLE, 'estradiol_pmol_l_mean'])
    stds_estro = np.array(estro.loc[estro['days_from_lh_peak'] < MAX_DAYINCYCLE, 'std'])

    if WITH_LH:
        lh['std'] = calculating_std_for_gmm(lh, 'lh_iu')
        # turn the columns into arrays
        means_lh = np.array(lh.loc[lh['days_from_lh_peak'] < MAX_DAYINCYCLE, 'lh_iu_l_mean'])
        stds_lh = np.array(lh.loc[lh['days_from_lh_peak'] < MAX_DAYINCYCLE, 'std'])

    # Combine means and stds into a single array
    means = np.column_stack((means_prog, means_estro, means_lh)).T if WITH_LH else np.column_stack(
        (means_prog, means_estro)).T
    stds = np.column_stack((stds_prog, stds_estro, stds_lh)).T if WITH_LH else np.column_stack(
        (stds_prog, stds_estro)).T

    return means, stds


def means_stds_for_gmm(use_prior_mean_and_std):
    if use_prior_mean_and_std:
        estro = pd.read_csv('estradiol and progesterone data.csv')
        # shift the days_from_lh_peak values by 15 to align with the day in cycle from our data
        estro['days_from_lh_peak'] = estro['days_from_lh_peak'] + 15
        prog = estro.iloc[:, [0, 6, 7, 8, 9, 10]]
        estro = estro.iloc[:, 0:6]

        if WITH_LH:
            lh = pd.read_csv('lh and fsh data.csv')  # .iloc[:, [0, 2]]
            lh['days_from_lh_peak'] = lh['days_from_lh_peak'] + 15
            lh = lh.iloc[:, 0:6]

        means, stds = means_and_std_into_array(prog, estro, lh) if WITH_LH else means_and_std_into_array(prog, estro)
    else:
        means, stds = None, None
    return means, stds  # np.array(covariances)


def check_for_nan(data_array):
    # Check for any NaN values in the initialized data
    print("Number of NaN values in the initialized data:", np.isnan(data_array).sum())

