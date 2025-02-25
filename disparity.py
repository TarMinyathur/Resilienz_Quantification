# disparity.py

import numpy as np
import pandas as pd
from functools import reduce
epsilon = 1e-10

def calculate_disparity_space(net, generation_factors):
    """
    Calculate the disparity matrix and maximum integral disparity value for power generation units.

    Args:
        net (object): Network object containing sgen, storage, and gen DataFrames.
        generation_factors (dict): Dictionary of generation factors keyed by type.

    Returns:
        dict: {
            'disparity_matrix': pd.DataFrame,  # Euclidean distance matrix between buses
            'max_integral_value': float         # Maximum integral value of disparity
        }
    """

    # Ensure required columns and set NaN or missing values to 0
    for df_name in ['sgen', 'storage', 'gen']:
        df = getattr(net, df_name)
        for col in ['p_mw', 'q_mvar', 'sn_mva']:
            if col not in df.columns:
                df[col] = 0
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Check for missing 'type' column and add if missing
    for df_name in ['sgen', 'storage', 'gen']:
        df = getattr(net, df_name)
        if 'type' not in df.columns:
            df['type'] = 'default'

    # Sum p_mw, q_mvar, and sn_mva*generation_factor over all sgen at each bus
    if not net.sgen.empty:
        net.sgen['effective_sn_mva'] = net.sgen.apply(
            lambda row: row['sn_mva'] * generation_factors.get(row['type'], 1), axis=1)
        sgen_sums = net.sgen.groupby('bus')[['p_mw', 'q_mvar', 'effective_sn_mva', 'sn_mva']].sum().reset_index()
    else:
        sgen_sums = pd.DataFrame(columns=['bus', 'p_mw', 'q_mvar', 'effective_sn_mva', 'sn_mva'])

    # Sum p_mw, q_mvar, and sn_mva*generation_factor over all storage at each bus
    if not net.storage.empty:
        net.storage['effective_sn_mva'] = net.storage.apply(
            lambda row: row['sn_mva'] * generation_factors.get(row['type'], 1), axis=1)
        storage_sums = net.storage.groupby('bus')[['p_mw', 'q_mvar', 'effective_sn_mva', 'sn_mva']].sum().reset_index()
    else:
        storage_sums = pd.DataFrame(columns=['bus', 'p_mw', 'q_mvar', 'effective_sn_mva', 'sn_mva'])

    # Sum p_mw and sn_mva*generation_factor over all gen at each bus
    if not net.gen.empty:
        net.gen['effective_sn_mva'] = net.gen.apply(
            lambda row: row['sn_mva'] * generation_factors.get(row['type'], 1), axis=1)
        gen_sums = net.gen.groupby('bus')[['p_mw', 'effective_sn_mva', 'sn_mva']].sum().reset_index()
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

def calculate_load_disparity(net):
    """
    Calculate the disparity matrix and maximum integral disparity value for load characteristics.

    Args:
        net (object): Network object containing load DataFrame with relevant columns.

    Returns:
        dict: {
            'disparity_matrix': pd.DataFrame,  # Euclidean distance matrix between loads
            'max_integral_value': float         # Maximum integral value of disparity
        }
    """

    # Ensure required columns and set NaN or missing values to 0
    required_columns = ['p_mw', 'q_mvar', 'sn_mva', 'cos_phi', 'mode']
    for col in required_columns:
        if col not in net.load.columns:
            net.load[col] = np.nan

    # Calculate missing p_mw and q_mvar if cos_phi and sn_mva are available
    if net.load['p_mw'].isnull().any() or net.load['q_mvar'].isnull().any():
        net.load['p_mw'] = net.load.apply(
            lambda row: row['sn_mva'] * row['cos_phi'] if pd.notna(row['sn_mva']) and pd.notna(row['cos_phi']) else row['p_mw'],
            axis=1
        )
        net.load['q_mvar'] = net.load.apply(
            lambda row: row['sn_mva'] * np.sqrt(1 - row['cos_phi'] ** 2)
            if pd.notna(row['sn_mva']) and pd.notna(row['cos_phi']) else row['q_mvar'],
            axis=1
        )

    # Calculate sn_mva if missing
    if net.load['sn_mva'].isnull().any():
        net.load['sn_mva'] = net.load.apply(
            lambda row: np.sqrt(row['p_mw'] ** 2 + row['q_mvar'] ** 2)
            if pd.notna(row['p_mw']) and pd.notna(row['q_mvar']) else row['sn_mva'],
            axis=1
        )

    # Replace remaining NaN values with zeros
    net.load.fillna(0, inplace=True)

    # Prepare data for disparity calculation
    load_data = net.load[['p_mw', 'q_mvar', 'sn_mva']].copy()

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


def calculate_transformer_disparity(net, debug=False):
    """
    Calculate the disparity matrix and maximum integral disparity value for transformer characteristics.

    Args:
        net (object): Network object containing transformer DataFrame with relevant columns.
        debug (bool): If True, print debug information. Default is False.

    Returns:
        tuple: disparity_df (pd.DataFrame), max_integral_value (float)
    """

    # Ensure that transformer has the required columns
    required_columns = ['sn_mva', 'vn_hv_kv', 'vn_lv_kv', 'vkr_percent',
                        'vk_percent', 'pfe_kw', 'i0_percent', 'shift_degree']
    for column in required_columns:
        if column not in net.trafo.columns:
            raise ValueError(f"Ensure that '{column}' column exists in net.trafo")

    # Data Cleaning: Ensure all required columns are numeric and fill NaN with zero
    for column in required_columns:
        net.trafo[column] = pd.to_numeric(net.trafo[column], errors='coerce').fillna(0)

    # Combine the metrics into a single dataframe
    trafo_data = net.trafo[required_columns].copy()

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

def calculate_line_disparity(net, debug=False):
    # Ensure that line has the required columns
    required_columns = ['length_km', 'from_bus', 'to_bus', 'type',
                        'r_ohm_per_km', 'x_ohm_per_km', 'max_i_ka']
    for column in required_columns:
        if column not in net.line.columns:
            print(f"Ensure that '{column}' column exists in net.line")
            return None

    # Normalize categorical columns
    net.line['from_bus_norm'] = normalize_categorical(net.line['from_bus'])
    net.line['to_bus_norm'] = normalize_categorical(net.line['to_bus'])
    net.line['cable_type_norm'] = normalize_categorical(net.line['type'])

    if debug:
        print("\nDebug: Line Data After Normalization")
        print(net.line)

    # Combine the metrics into a single dataframe
    line_data = net.line[['length_km', 'from_bus_norm', 'to_bus_norm',
                          'cable_type_norm', 'r_ohm_per_km', 'x_ohm_per_km', 'max_i_ka']].copy()

    # Vectorized Disparity Matrix Calculation
    n = len(line_data)
    metrics = line_data.values

    # Calculate pairwise Euclidean distance using vectorized broadcasting
    diff = metrics[:, None, :] - metrics[None, :, :]
    disparity_matrix = np.sqrt(np.sum(diff ** 2, axis=2))

    # Set the diagonal to zero (distance to itself)
    np.fill_diagonal(disparity_matrix, 0)

    # Convert to DataFrame for easier handling
    disparity_df = pd.DataFrame(disparity_matrix, index=line_data.index, columns=line_data.index)

    # Print both counts in one row
    print("Bus | Original | Normed ")
    print("-" * 45)
    for k in range(len(net.line)):
        normed = net.line['to_bus_norm'][k]
        original = net.line['to_bus'][k]
        print(f"{k:<12} | {original:<14} | {normed:<20}")

    # Calculate max disparity and integral value
    max_values = line_data.max()
    max_disparity = np.sqrt(np.sum(max_values ** 2))
    max_integral_value = (n * (n - 1) / 2) * max_disparity

    # **Remove normalized columns after calculation**
    columns_to_drop = ['from_bus_norm', 'to_bus_norm', 'cable_type_norm']
    net.line.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    return disparity_df, max_integral_value


def normalize_categorical(data):
    """
    Normalizes categorical data to numeric values starting from 0.
    Handles missing values by assigning them a unique number.
    """
    # Fill NaN values with a placeholder
    data = data.fillna('Missing')

    # Get unique values and create a mapping
    unique_values = data.unique()
    value_map = {value: idx for idx, value in enumerate(unique_values)}

    # Map the values and return as a numeric series
    return data.map(value_map).astype(float)