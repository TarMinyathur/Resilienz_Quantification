# disparity.py

import numpy as np
import pandas as pd
from functools import reduce
epsilon = 1e-10

def calculate_disparity_space(net_temp, generation_factors):
    """
    Calculate the disparity matrix and maximum integral disparity value for power generation units.

    Args:
        net_temp (object): Network object containing sgen, storage, and gen DataFrames.
        generation_factors (dict): Dictionary of generation factors keyed by type.

    Returns:
        dict: {
            'disparity_matrix': pd.DataFrame,  # Euclidean distance matrix between buses
            'max_integral_value': float         # Maximum integral value of disparity
        }
    """

    # Ensure required columns and set NaN or missing values to 0
    for df_name in ['sgen', 'storage', 'gen']:
        df = getattr(net_temp, df_name)
        for col in ['p_mw', 'q_mvar', 'sn_mva']:
            if col not in df.columns:
                df[col] = 0
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Check for missing 'type' column and add if missing
    for df_name in ['sgen', 'storage', 'gen']:
        df = getattr(net_temp, df_name)
        if 'type' not in df.columns:
            df['type'] = 'default'

    # Sum p_mw, q_mvar, and sn_mva*generation_factor over all sgen at each bus
    if not net_temp.sgen.empty:
        net_temp.sgen['effective_sn_mva'] = net_temp.sgen.apply(
            lambda row: row['sn_mva'] * generation_factors.get(row['type'], 1), axis=1)
        sgen_sums = net_temp.sgen.groupby('bus')[['p_mw', 'q_mvar', 'effective_sn_mva', 'sn_mva']].sum().reset_index()
    else:
        sgen_sums = pd.DataFrame(columns=['bus', 'p_mw', 'q_mvar', 'effective_sn_mva', 'sn_mva'])

    # Sum p_mw, q_mvar, and sn_mva*generation_factor over all storage at each bus
    if not net_temp.storage.empty:
        net_temp.storage['effective_sn_mva'] = net_temp.storage.apply(
            lambda row: row['sn_mva'] * generation_factors.get(row['type'], 1), axis=1)
        storage_sums = net_temp.storage.groupby('bus')[['p_mw', 'q_mvar', 'effective_sn_mva', 'sn_mva']].sum().reset_index()
    else:
        storage_sums = pd.DataFrame(columns=['bus', 'p_mw', 'q_mvar', 'effective_sn_mva', 'sn_mva'])

    # Sum p_mw and sn_mva*generation_factor over all gen at each bus
    if not net_temp.gen.empty:
        net_temp.gen['effective_sn_mva'] = net_temp.gen.apply(
            lambda row: row['sn_mva'] * generation_factors.get(row['type'], 1), axis=1)
        gen_sums = net_temp.gen.groupby('bus')[['p_mw', 'effective_sn_mva', 'sn_mva']].sum().reset_index()
        gen_sums['q_mvar'] = 0  # Add a zero q_mvar column to match other dataframes
    else:
        gen_sums = pd.DataFrame(columns=['bus', 'p_mw', 'q_mvar', 'effective_sn_mva', 'sn_mva'])

    # Efficient Merging using reduce
    sum_dfs = [sgen_sums, storage_sums, gen_sums]
    total_sums = reduce(lambda left, right: pd.merge(left, right, on='bus', how='outer').fillna(0), sum_dfs)

    # Sum relevant columns
    total_sums['p_mw'] = total_sums['p_mw_x'] + total_sums.get('p_mw_y', 0) + total_sums.get('p_mw', 0)
    total_sums['q_mvar'] = total_sums['q_mvar_x'] + total_sums.get('q_mvar_y', 0) + total_sums.get('q_mvar', 0)
    total_sums['effective_sn_mva'] = (
        total_sums['effective_sn_mva_x'] +
        total_sums.get('effective_sn_mva_y', 0) +
        total_sums.get('effective_sn_mva', 0)
    )
    total_sums['sn_mva'] = total_sums['sn_mva_x'] + total_sums.get('sn_mva_y', 0) + total_sums.get('sn_mva', 0)

    # Select relevant columns
    total_sums = total_sums[['bus', 'p_mw', 'q_mvar', 'effective_sn_mva', 'sn_mva']]

    # Vectorized Disparity Matrix Calculation
    n = len(total_sums)
    p_diff = total_sums['p_mw'].values[:, None] - total_sums['p_mw'].values
    q_diff = total_sums['q_mvar'].values[:, None] - total_sums['q_mvar'].values
    sn_diff = total_sums['sn_mva'].values[:, None] - total_sums['sn_mva'].values
    eff_sn_diff = total_sums['effective_sn_mva'].values[:, None] - total_sums['effective_sn_mva'].values

    # Vectorized calculation of Euclidean distance
    disparity_matrix = np.sqrt(p_diff**2 + q_diff**2 + sn_diff**2 + eff_sn_diff**2)
    np.fill_diagonal(disparity_matrix, 0)  # Set diagonal to zero

    # Convert to DataFrame for easier handling
    disparity_df = pd.DataFrame(disparity_matrix, index=total_sums['bus'], columns=total_sums['bus'])

    # Maximum Disparity and Integral Calculation
    max_p = total_sums['p_mw'].max()
    max_q = total_sums['q_mvar'].max()
    max_sn = total_sums['sn_mva'].max()
    max_eff = total_sums['effective_sn_mva'].max()
    max_disparity = np.sqrt(max_p ** 2 + max_q ** 2 + max_eff ** 2 + max_sn ** 2)

    # Use a small epsilon instead of max(1, value)
    max_integral_value = (n * (n - 1) / 2) * max_disparity
    max_integral_value = max(epsilon, max_integral_value)

    return disparity_df, max_integral_value

def calculate_load_disparity(net_temp):
    """
    Calculate the disparity matrix and maximum integral disparity value for load characteristics.

    Args:
        net_temp (object): Network object containing load DataFrame with relevant columns.

    Returns:
        dict: {
            'disparity_matrix': pd.DataFrame,  # Euclidean distance matrix between loads
            'max_integral_value': float         # Maximum integral value of disparity
        }
    """

    # Ensure required columns and set NaN or missing values to 0
    required_columns = ['p_mw', 'q_mvar', 'sn_mva', 'cos_phi', 'mode']
    for col in required_columns:
        if col not in net_temp.load.columns:
            net_temp.load[col] = np.nan

    # Calculate missing p_mw and q_mvar if cos_phi and sn_mva are available
    if net_temp.load['p_mw'].isnull().any() or net_temp.load['q_mvar'].isnull().any():
        net_temp.load['p_mw'] = net_temp.load.apply(
            lambda row: row['sn_mva'] * row['cos_phi'] if pd.notna(row['sn_mva']) and pd.notna(row['cos_phi']) else row['p_mw'],
            axis=1
        )
        net_temp.load['q_mvar'] = net_temp.load.apply(
            lambda row: row['sn_mva'] * np.sqrt(1 - row['cos_phi'] ** 2)
            if pd.notna(row['sn_mva']) and pd.notna(row['cos_phi']) else row['q_mvar'],
            axis=1
        )

    # Calculate sn_mva if missing
    if net_temp.load['sn_mva'].isnull().any():
        net_temp.load['sn_mva'] = net_temp.load.apply(
            lambda row: np.sqrt(row['p_mw'] ** 2 + row['q_mvar'] ** 2)
            if pd.notna(row['p_mw']) and pd.notna(row['q_mvar']) else row['sn_mva'],
            axis=1
        )

    # Replace remaining NaN values with zeros
    net_temp.load.fillna(0, inplace=True)

    # Prepare data for disparity calculation
    load_data = net_temp.load[['p_mw', 'q_mvar', 'sn_mva']].copy()

    # Vectorized Disparity Matrix Calculation
    n = len(load_data)
    p_diff = load_data['p_mw'].values[:, None] - load_data['p_mw'].values
    q_diff = load_data['q_mvar'].values[:, None] - load_data['q_mvar'].values
    sn_diff = load_data['sn_mva'].values[:, None] - load_data['sn_mva'].values

    # Vectorized calculation of Euclidean distance
    disparity_matrix = np.sqrt(p_diff**2 + q_diff**2 + sn_diff**2)
    np.fill_diagonal(disparity_matrix, 0)  # Set diagonal to zero

    # Convert to DataFrame for easier handling
    disparity_df = pd.DataFrame(disparity_matrix, index=load_data.index, columns=load_data.index)

    # Maximum Disparity and Integral Calculation
    max_p = load_data['p_mw'].max()
    max_q = load_data['q_mvar'].max()
    max_sn = load_data['sn_mva'].max()
    max_disparity = np.sqrt(max_p ** 2 + max_q ** 2 + max_sn ** 2)

    # Use a small epsilon instead of max(1, value)
    max_integral_value = (n * (n - 1) / 2) * max_disparity
    max_integral_value = max(epsilon, max_integral_value)

    return disparity_df, max_integral_value


def calculate_transformer_disparity(net_temp, debug=False):
    """
    Calculate the disparity matrix and maximum integral disparity value for transformer characteristics.

    Args:
        net_temp (object): Network object containing transformer DataFrame with relevant columns.
        debug (bool): If True, print debug information. Default is False.

    Returns:
        tuple: disparity_df (pd.DataFrame), max_integral_value (float)
    """

    # Ensure that transformer has the required columns
    required_columns = ['sn_mva', 'vn_hv_kv', 'vn_lv_kv', 'vkr_percent',
                        'vk_percent', 'pfe_kw', 'i0_percent', 'shift_degree']
    for column in required_columns:
        if column not in net_temp.trafo.columns:
            raise ValueError(f"Ensure that '{column}' column exists in net_temp.trafo")

    # Data Cleaning: Ensure all required columns are numeric and fill NaN with zero
    for column in required_columns:
        net_temp.trafo[column] = pd.to_numeric(net_temp.trafo[column], errors='coerce').fillna(0)

    # Combine the metrics into a single dataframe
    trafo_data = net_temp.trafo[required_columns].copy()

    if debug:
        print("\nDebug: Transformer Data After Cleaning")
        print(trafo_data.head())

    # Vectorized Disparity Matrix Calculation
    n = len(trafo_data)
    metrics = trafo_data.values
    diff = metrics[:, None, :] - metrics[None, :, :]
    disparity_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(disparity_matrix, 0)  # Set diagonal to zero

    # Convert to DataFrame for easier handling
    disparity_df = pd.DataFrame(disparity_matrix, index=trafo_data.index, columns=trafo_data.index)

    # Maximum Disparity and Integral Calculation
    max_values = trafo_data.max()
    max_disparity = np.sqrt(np.sum(max_values ** 2))

    max_integral_value = (n * (n - 1) / 2) * max_disparity
    max_integral_value = max(epsilon, max_integral_value)

    return disparity_df, max_integral_value


def calculate_line_disparity(net_temp, debug=False):
    """
    Calculate the disparity matrix and maximum integral disparity value for line characteristics.

    Args:
        net_temp (object): Network object containing line DataFrame with relevant columns.
        debug (bool): If True, print debug information. Default is False.

    Returns:
        tuple: disparity_df (pd.DataFrame), max_integral_value (float)
    """

    # Ensure that line has the required columns
    required_columns = ['length_km', 'from_bus', 'to_bus', 'type',
                        'r_ohm_per_km', 'x_ohm_per_km', 'max_i_ka']
    for column in required_columns:
        if column not in net_temp.line.columns:
            raise ValueError(f"Ensure that '{column}' column exists in net_temp.line")

    # Data Cleaning and Type Consistency
    net_temp.line['from_bus'] = net_temp.line['from_bus'].astype(str).str.lower().str.strip().fillna("unknown")
    net_temp.line['to_bus'] = net_temp.line['to_bus'].astype(str).str.lower().str.strip().fillna("unknown")
    net_temp.line['type'] = net_temp.line['type'].astype(str).str.lower().str.strip().fillna("unknown")

    # Normalize categorical columns
    net_temp.line['from_bus_norm'] = normalize_categorical(net_temp.line['from_bus'])
    net_temp.line['to_bus_norm'] = normalize_categorical(net_temp.line['to_bus'])
    net_temp.line['cable_type_norm'] = normalize_categorical(net_temp.line['type'])

    if debug:
        print("\nDebug: Normalization Results")
        print(net_temp.line[['from_bus', 'from_bus_norm', 'to_bus',
                        'to_bus_norm', 'type', 'cable_type_norm']].head())

    # Combine the metrics into a single dataframe
    line_data = net_temp.line[['length_km', 'from_bus_norm', 'to_bus_norm',
                          'cable_type_norm', 'r_ohm_per_km',
                          'x_ohm_per_km', 'max_i_ka']].copy()

    # Vectorized Disparity Matrix Calculation
    n = len(line_data)
    metrics = line_data.values
    diff = metrics[:, None, :] - metrics[None, :, :]
    disparity_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(disparity_matrix, 0)  # Set diagonal to zero

    # Convert to DataFrame for easier handling
    disparity_df = pd.DataFrame(disparity_matrix, index=line_data.index, columns=line_data.index)

    # Maximum Disparity and Integral Calculation
    max_values = line_data.max()
    max_disparity = np.sqrt(np.sum(max_values ** 2))

    max_integral_value = (n * (n - 1) / 2) * max_disparity
    max_integral_value = max(epsilon, max_integral_value)

    return disparity_df, max_integral_value


def normalize_categorical(data):
    """
    Normalizes categorical data to numeric values starting from 0.

    Args:
        data (pd.Series): Series containing categorical data.

    Returns:
        pd.Series: Series of normalized numeric values.
    """
    # Input Validation
    if not isinstance(data, pd.Series):
        raise TypeError("Input must be a Pandas Series.")

    if data.empty:
        return pd.Series([], dtype='int')

    # Clean Data: Convert to string, lower, strip whitespace, and fill NaN
    data = data.astype(str).str.lower().str.strip().fillna("unknown")

    # Using pd.Categorical for faster and more memory-efficient normalization
    normalized = pd.Series(pd.Categorical(data).codes, index=data.index)

    # Re-map any -1 (NaNs) to 0
    normalized = normalized.replace(-1, 0)

    return normalized