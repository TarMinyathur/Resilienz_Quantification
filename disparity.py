# disparity.py

import numpy as np
import pandas as pd
from functools import reduce
epsilon = 1e-10

def calculate_disparity_space(net_temp_disp, generation_factors):
    """
    Calculate the disparity matrix and maximum integral disparity value for power generation units.

    Args:
        net_temp_disp (object): net_temp_dispwork object containing sgen, storage, and gen DataFrames.
        generation_factors (dict): Dictionary of generation factors keyed by type.

    Returns:
        dict: {
            'disparity_matrix': pd.DataFrame,  # Euclidean distance matrix between buses
            'max_integral_value': float         # Maximum integral value of disparity
        }
    """

    # Ensure required columns and set NaN or missing values to 0
    for df_name in ['sgen', 'storage', 'gen']:
        df = getattr(net_temp_disp, df_name)
        for col in ['p_mw', 'q_mvar', 'sn_mva']:
            if col not in df.columns:
                df[col] = 0
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Check for missing 'type' column and add if missing
    for df_name in ['sgen', 'storage', 'gen']:
        df = getattr(net_temp_disp, df_name)
        if 'type' not in df.columns:
            df['type'] = 'default'

    # Sum p_mw, q_mvar, and sn_mva*generation_factor over all sgen at each bus
    if not net_temp_disp.sgen.empty:
        net_temp_disp.sgen['effective_sn_mva'] = net_temp_disp.sgen.apply(
            lambda row: row['sn_mva'] * generation_factors.get(row['type'], 1), axis=1)
        sgen_sums = net_temp_disp.sgen.groupby('bus')[['p_mw', 'q_mvar', 'effective_sn_mva', 'sn_mva']].sum().reset_index()
    else:
        sgen_sums = pd.DataFrame(columns=['bus', 'p_mw', 'q_mvar', 'effective_sn_mva', 'sn_mva'])

    # Sum p_mw, q_mvar, and sn_mva*generation_factor over all storage at each bus
    if not net_temp_disp.storage.empty:
        net_temp_disp.storage['effective_sn_mva'] = net_temp_disp.storage.apply(
            lambda row: row['sn_mva'] * generation_factors.get(row['type'], 1), axis=1)
        storage_sums = net_temp_disp.storage.groupby('bus')[['p_mw', 'q_mvar', 'effective_sn_mva', 'sn_mva']].sum().reset_index()
    else:
        storage_sums = pd.DataFrame(columns=['bus', 'p_mw', 'q_mvar', 'effective_sn_mva', 'sn_mva'])

    # Sum p_mw and sn_mva*generation_factor over all gen at each bus
    if not net_temp_disp.gen.empty:
        net_temp_disp.gen['effective_sn_mva'] = net_temp_disp.gen.apply(
            lambda row: row['sn_mva'] * generation_factors.get(row['type'], 1), axis=1)
        gen_sums = net_temp_disp.gen.groupby('bus')[['p_mw', 'effective_sn_mva', 'sn_mva']].sum().reset_index()
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

    # --- Incorporate Geodata ---
    # If generation units do not have geodata, merge bus geodata into total_sums.
    # Assumes net_temp_disp.bus has columns 'bus', 'x', and 'y'
    if not net_temp_disp.bus.empty and {'bus', 'x', 'y'}.issubset(net_temp_disp.bus.columns):
        total_sums = pd.merge(total_sums, net_temp_disp.bus[['bus', 'x', 'y']], on='bus', how='left')
        total_sums['x'] = total_sums['x'].fillna(0)
        total_sums['y'] = total_sums['y'].fillna(0)
    else:
        # If no bus geodata available, assign 0 coordinates.
        total_sums['x'] = 0
        total_sums['y'] = 0

    # Vectorized Disparity Matrix Calculation
    n = len(total_sums)
    p_diff = total_sums['p_mw'].values[:, None] - total_sums['p_mw'].values
    q_diff = total_sums['q_mvar'].values[:, None] - total_sums['q_mvar'].values
    sn_diff = total_sums['sn_mva'].values[:, None] - total_sums['sn_mva'].values
    eff_sn_diff = total_sums['effective_sn_mva'].values[:, None] - total_sums['effective_sn_mva'].values

    # Differences for geodata (bus coordinates)
    x_diff = total_sums['x'].values[:, None] - total_sums['x'].values
    y_diff = total_sums['y'].values[:, None] - total_sums['y'].values

    # Euclidean distance considering both generation and spatial factors
    disparity_matrix = np.sqrt( p_diff**2 + q_diff**2 + sn_diff**2 + eff_sn_diff**2 + x_diff**2 + y_diff**2 )
    np.fill_diagonal(disparity_matrix, 0)  # Set diagonal to zero

    # Convert to DataFrame for easier handling
    disparity_df = pd.DataFrame(disparity_matrix, index=total_sums['bus'], columns=total_sums['bus'])

    # Maximum Disparity and Integral Calculation
    max_p = total_sums['p_mw'].max()
    max_q = total_sums['q_mvar'].max()
    max_sn = total_sums['sn_mva'].max()
    max_eff = total_sums['effective_sn_mva'].max()
    max_x = total_sums['x'].max()
    max_y = total_sums['y'].max()

    max_disparity = np.sqrt(max_p ** 2 + max_q ** 2 + max_sn ** 2 + max_eff ** 2 + max_x ** 2 + max_y ** 2)

    # Use a small epsilon instead of max(1, value)
    max_integral_value = (n * (n - 1) / 2) * max_disparity
    max_integral_value = max(epsilon, max_integral_value)

    return disparity_df, max_integral_value

def calculate_load_disparity(net_temp_disp):
    """
    Calculate the disparity matrix and maximum integral disparity value for load characteristics.

    Args:
        net_temp_disp (object): net_temp_dispwork object containing load DataFrame with relevant columns.

    Returns:
        dict: {
            'disparity_matrix': pd.DataFrame,  # Euclidean distance matrix between loads
            'max_integral_value': float         # Maximum integral value of disparity
        }
    """

    # Ensure required columns and set NaN or missing values to 0
    required_columns = ['p_mw', 'q_mvar', 'sn_mva', 'cos_phi', 'mode', 'controllable ']
    for col in required_columns:
        if col not in net_temp_disp.load.columns:
            net_temp_disp.load[col] = np.nan

    # Calculate missing p_mw and q_mvar if cos_phi and sn_mva are available
    if net_temp_disp.load['p_mw'].isnull().any() or net_temp_disp.load['q_mvar'].isnull().any():
        net_temp_disp.load['p_mw'] = net_temp_disp.load.apply(
            lambda row: row['sn_mva'] * row['cos_phi'] if pd.notna(row['sn_mva']) and pd.notna(row['cos_phi']) else row['p_mw'],
            axis=1
        )
        net_temp_disp.load['q_mvar'] = net_temp_disp.load.apply(
            lambda row: row['sn_mva'] * np.sqrt(1 - row['cos_phi'] ** 2)
            if pd.notna(row['sn_mva']) and pd.notna(row['cos_phi']) else row['q_mvar'],
            axis=1
        )

    # Calculate sn_mva if missing
    if net_temp_disp.load['sn_mva'].isnull().any():
        net_temp_disp.load['sn_mva'] = net_temp_disp.load.apply(
            lambda row: np.sqrt(row['p_mw'] ** 2 + row['q_mvar'] ** 2)
            if pd.notna(row['p_mw']) and pd.notna(row['q_mvar']) else row['sn_mva'],
            axis=1
        )

    # Replace remaining NaN values with zeros
    net_temp_disp.load.fillna(0, inplace=True)

    # Prepare data for disparity calculation
    load_data = net_temp_disp.load[['p_mw', 'q_mvar', 'sn_mva']].copy()

    # --- Incorporate Geodata ---
    # If load-specific geodata is available, you could use it.
    # Otherwise, merge bus geodata into load_data. We assume net_temp_disp.bus has 'bus', 'x', and 'y' columns.
    if not net_temp_disp.bus.empty and {'bus', 'x', 'y'}.issubset(net_temp_disp.bus.columns):
        load_data = pd.merge(load_data, net_temp_disp.bus[['bus', 'x', 'y']], on='bus', how='left')
        load_data['x'] = load_data['x'].fillna(0)
        load_data['y'] = load_data['y'].fillna(0)
    else:
        load_data['x'] = 0
        load_data['y'] = 0

    # Vectorized Disparity Matrix Calculation
    n = len(load_data)
    p_diff = load_data['p_mw'].values[:, None] - load_data['p_mw'].values
    q_diff = load_data['q_mvar'].values[:, None] - load_data['q_mvar'].values
    sn_diff = load_data['sn_mva'].values[:, None] - load_data['sn_mva'].values
    # Differences for geodata (bus coordinates)
    x_diff = load_data['x'].values[:, None] - load_data['x'].values
    y_diff = load_data['y'].values[:, None] - load_data['y'].values

    # Vectorized calculation of Euclidean distance
    disparity_matrix = np.sqrt(p_diff**2 + q_diff**2 + sn_diff**2 + x_diff**2 + y_diff**2)
    np.fill_diagonal(disparity_matrix, 0)  # Set diagonal to zero

    # Convert to DataFrame for easier handling
    disparity_df = pd.DataFrame(disparity_matrix, index=load_data.index, columns=load_data.index)

    # Maximum Disparity and Integral Calculation
    max_p = load_data['p_mw'].max()
    max_q = load_data['q_mvar'].max()
    max_sn = load_data['sn_mva'].max()
    max_x = load_data['x'].max()
    max_y = load_data['y'].max()
    max_disparity = np.sqrt(max_p ** 2 + max_q ** 2 + max_sn ** 2 + max_x ** 2 + max_y ** 2)

    # Use a small epsilon instead of max(1, value)
    max_integral_value = (n * (n - 1) / 2) * max_disparity
    max_integral_value = max(epsilon, max_integral_value)

    return disparity_df, max_integral_value


def calculate_transformer_disparity(net_temp_disp, debug=False):
    """
    Calculate the disparity matrix and maximum integral disparity value for transformer characteristics.

    Args:
        net_temp_disp (object): net_temp_dispwork object containing transformer DataFrame with relevant columns.
        debug (bool): If True, print debug information. Default is False.

    Returns:
        tuple: disparity_df (pd.DataFrame), max_integral_value (float)
    """

    # Ensure that transformer has the required columns
    required_columns = ['sn_mva', 'vn_hv_kv', 'vn_lv_kv', 'vkr_percent',
                        'vk_percent', 'pfe_kw', 'i0_percent', 'shift_degree']
    for column in required_columns:
        if column not in net_temp_disp.trafo.columns:
            raise ValueError(f"Ensure that '{column}' column exists in net_temp_disp.trafo")

    # Data Cleaning: Ensure all required columns are numeric and fill NaN with zero
    for column in required_columns:
        net_temp_disp.trafo[column] = pd.to_numeric(net_temp_disp.trafo[column], errors='coerce').fillna(0)

    # *** Incorporate Bus Geodata ***
    # Assuming net_temp_disp.bus should contain columns: 'bus_id', 'x', 'y'
    # And net_temp_disp.trafo should have a column 'hv_bus' that corresponds to bus IDs.

    # Check for 'hv_bus' in net_temp_disp.trafo. If missing, set a default value.
    if 'hv_bus' not in net_temp_disp.trafo.columns:
        print("Warning: 'hv_bus' column missing in net_temp_disp.trafo. Defaulting to 0 for all entries.")
        net_temp_disp.trafo['hv_bus'] = 0

    # Check if net_temp_disp.bus contains the required geodata columns
    print(net_temp_disp.bus)
    if net_temp_disp.bus_geodata.empty:
        print("Warning: net_temp_disp.bus does not contain 'bus_id', 'x', and 'y' columns. Setting geodata to 0 for all buses.")
        # Create an empty DataFrame or a default mapping where every bus gets geodata of 0
        # If bus_id exists in net_temp_disp.bus, use it; otherwise, assume an empty mapping.
        if 'bus_id' in net_temp_disp.bus.columns:
            bus_ids = net_temp_disp.bus['bus_id']
            bus_geo = pd.DataFrame({'x': 0, 'y': 0}, index=bus_ids)
        else:
            bus_geo = pd.DataFrame({'x': [], 'y': []})
    else:
       bus_geo = net_temp_disp.bus_geodata[['x', 'y']]  # Falls bus_id als Index bereits existiert


    # Map transformer high-voltage bus IDs to their coordinates
    # Use .get to handle missing bus entries safely (default to 0)
    net_temp_disp.trafo['geo_x'] = net_temp_disp.trafo['hv_bus'].map(lambda b: bus_geo.at[b, 'x'] if b in bus_geo.index else 0)
    net_temp_disp.trafo['geo_y'] = net_temp_disp.trafo['hv_bus'].map(lambda b: bus_geo.at[b, 'y'] if b in bus_geo.index else 0)

    if debug:
        print("\nDebug: Transformer Data After Cleaning and Adding Bus Geodata")
        print(net_temp_disp.trafo[[*required_columns, 'hv_bus', 'geo_x', 'geo_y']].head())

    # Combine numerical and geographic features
    trafo_data = net_temp_disp.trafo[required_columns + ['geo_x', 'geo_y']].copy()

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

    # Clean up: remove added geodata columns from net_temp_disp.trafo to preserve original structure
    net_temp_disp.trafo.drop(columns=['geo_x', 'geo_y'], inplace=True, errors='ignore')

    return disparity_df, max_integral_value

def calculate_line_disparity(net_temp_disp, debug=False):
    # Ensure that line has the required columns
    required_columns = ['length_km', 'from_bus', 'to_bus', 'type',
                        'r_ohm_per_km', 'x_ohm_per_km', 'max_i_ka']
    for column in required_columns:
        if column not in net_temp_disp.line.columns:
            print(f"Ensure that '{column}' column exists in net_temp_disp.line")
            return None

    # Normalize categorical columns
    net_temp_disp.line['from_bus_norm'] = normalize_categorical(net_temp_disp.line['from_bus'])
    net_temp_disp.line['to_bus_norm'] = normalize_categorical(net_temp_disp.line['to_bus'])
    net_temp_disp.line['cable_type_norm'] = normalize_categorical(net_temp_disp.line['type'])

    # Process line_geodata if available
    if hasattr(net_temp_disp, 'line_geodata') and not net_temp_disp.line_geodata.empty:
        # Extract centroid (average of coordinates)
        net_temp_disp.line['geo_x'] = net_temp_disp.line_geodata['coords'].apply(lambda coords: np.mean([p[0] for p in coords]) if coords else 0)
        net_temp_disp.line['geo_y'] = net_temp_disp.line_geodata['coords'].apply(lambda coords: np.mean([p[1] for p in coords]) if coords else 0)
    else:
        # Fallback: Assign 0 if no geodata available
        net_temp_disp.line['geo_x'] = 0
        net_temp_disp.line['geo_y'] = 0

    if debug:
        print("\nDebug: Line Data After Normalization")
        print(net_temp_disp.line)

    # Combine the metrics into a single dataframe
    line_data = net_temp_disp.line[['length_km', 'from_bus_norm', 'to_bus_norm',
                          'cable_type_norm', 'r_ohm_per_km', 'x_ohm_per_km', 'max_i_ka', 'geo_x', 'geo_y']].copy()

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

    if debug:
        # Print both counts in one row
        print("Bus | Original | Normed ")
        print("-" * 45)
        for k in range(len(net_temp_disp.line)):
            normed = net_temp_disp.line['to_bus_norm'][k]
            original = net_temp_disp.line['to_bus'][k]
            print(f"{k:<12} | {original:<14} | {normed:<20}")

    # Calculate max disparity and integral value
    max_values = line_data.max()
    max_disparity = np.sqrt(np.sum(max_values ** 2))
    max_integral_value = (n * (n - 1) / 2) * max_disparity

    # **Remove normalized columns after calculation**
    columns_to_drop = ['from_bus_norm', 'to_bus_norm', 'cable_type_norm', 'geo_x', 'geo_y']
    net_temp_disp.line.drop(columns=columns_to_drop, inplace=True, errors='ignore')

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