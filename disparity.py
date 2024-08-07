# disparity.py

import numpy as np
import pandas as pd

def calculate_disparity_space(net, generation_factors):
    # Ensure that sgen, gen, and storage have the required columns
    required_columns_sgen_storage = ['bus', 'p_mw', 'q_mvar', 'sn_mva', 'type']
    required_columns_gen = ['bus', 'p_mw', 'sn_mva', 'type']

    # Ensure required columns and set NaN or missing values to 0
    for df_name in ['sgen', 'storage', 'gen']:
        df = getattr(net, df_name)
        for col in ['p_mw', 'q_mvar', 'sn_mva']:
            if col not in df.columns:
                df[col] = 0
            else:
                df[col] = df[col].fillna(0)

    # Check for missing 'type' column and add if missing
    for df_name in ['sgen', 'storage', 'gen']:
        df = getattr(net, df_name)
        if 'type' not in df.columns:
            df['type'] = 'default'  # You can change 'default' to any default type you prefer

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
        net.gen['effective_sn_mva'] = net.gen.apply(lambda row: row['sn_mva'] * generation_factors.get(row['type'], 1),
                                                    axis=1)
        gen_sums = net.gen.groupby('bus')[['p_mw', 'effective_sn_mva', 'sn_mva']].sum().reset_index()
        gen_sums['q_mvar'] = 0  # Add a zero q_mvar column to match other dataframes
    else:
        gen_sums = pd.DataFrame(columns=['bus', 'p_mw', 'q_mvar', 'effective_sn_mva', 'sn_mva'])

    print(gen_sums)
    print(sgen_sums)
    print(storage_sums)

    # Merge the sums from sgen, storage, and gen
    total_sums = pd.merge(sgen_sums, storage_sums, on='bus', how='outer', suffixes=('_sgen', '_storage')).fillna(0)
    total_sums = pd.merge(total_sums, gen_sums, on='bus', how='outer', suffixes=('', '_gen')).fillna(0)

    print(total_sums)

    # Sum the relevant columns
    total_sums['p_mw'] = total_sums['p_mw'] + total_sums.get('p_mw_storage', 0) + total_sums.get('p_mw_gen', 0)
    total_sums['q_mvar'] = total_sums['q_mvar'] + total_sums.get('q_mvar_storage', 0) + total_sums.get('q_mvar_gen', 0)
    total_sums['effective_sn_mva'] = (total_sums['effective_sn_mva'] + total_sums.get('effective_sn_mva_storage', 0) + total_sums.get('effective_sn_mva_gen', 0))
    total_sums['sn_mva'] = total_sums['sn_mva'] + total_sums.get('sn_mva_storage', 0) + total_sums.get('sn_mva_gen', 0)

    print(total_sums)

    # Select only the relevant columns
    total_sums = total_sums[['bus', 'p_mw_sgen', 'q_mvar_sgen', 'effective_sn_mva_sgen', 'sn_mva_sgen']]
    total_sums = total_sums.rename(columns={
        'p_mw_sgen': 'p_mw',
        'q_mvar_sgen': 'q_mvar',
        'effective_sn_mva_sgen': 'effective_sn_mva',
        'sn_mva_sgen': 'sn_mva'
    })

    print(total_sums)

    # Create disparity matrix (Euclidean distance between summed p_mw, q_mvar, and effective_sn_mva)
    n = len(total_sums)
    disparity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                disparity_matrix[i, j] = np.sqrt(
                    (total_sums.p_mw.iloc[i] - total_sums.p_mw.iloc[j]) ** 2 +
                    (total_sums.q_mvar.iloc[i] - total_sums.q_mvar.iloc[j]) ** 2 +
                    (total_sums.effective_sn_mva.iloc[i] - total_sums.effective_sn_mva.iloc[j]) ** 2 +
                    (total_sums.sn_mva.iloc[i] - total_sums.sn_mva.iloc[j]) ** 2
                )

    # Convert to DataFrame for easier handling
    disparity_df = pd.DataFrame(disparity_matrix, index=total_sums.bus, columns=total_sums.bus)

    # Calculate maximum disparity
    max_p = total_sums['p_mw'].max()
    max_q = total_sums['q_mvar'].max()
    max_sn = total_sums['sn_mva'].max()
    max_eff = total_sums['effective_sn_mva'].max()
    max_disparity = np.sqrt(max_p ** 2 + max_q ** 2 + max_eff ** 2 + max_sn ** 2)


    # Calculate theoretical maximum integral value of disparity
    max_integral_value = (n * (n - 1) / 2) * max_disparity
    max_integral_value = max(1, max_integral_value)

    return disparity_df, max_integral_value

def calculate_load_disparity(net):
    # Ensure load DataFrame has the required columns
    # Ensure the required columns exist
    if 'p_mw' not in net.load.columns or 'q_mvar' not in net.load.columns:
        # Calculate missing p_mw and q_mvar if cos_phi and mode are available
        if 'cos_phi' in net.load.columns and 'mode' in net.load.columns:
            net.load['p_mw'] = net.load.apply(lambda row: row['sn_mva'] * row['cos_phi'], axis=1)
            net.load['q_mvar'] = net.load.apply(lambda row: row['sn_mva'] * np.sqrt(1 - row['cos_phi'] ** 2), axis=1)
        else:
            raise ValueError("Missing required columns and unable to calculate due to missing cos_phi or mode.")

    # Replace missing sn_mva or NaN with zero and calculate if necessary
    if 'sn_mva' not in net.load.columns or net.load['sn_mva'].isnull().any():
        if 'cos_phi' in net.load.columns:
            net.load['sn_mva'] = net.load['sn_mva'].fillna(0)
            net.load.loc[net.load['sn_mva'] == 0, 'sn_mva'] = net.load.apply(
                lambda row: row['p_mw'] / row['cos_phi'], axis=1)
        else:
            net.load['sn_mva'] = np.sqrt(net.load['p_mw'] ** 2 + net.load['q_mvar'] ** 2)
            # net.load['cos_phi'] = 1  # Assume cos_phi is 1 when not existing
            # net.load['sn_mva'] = net.load['sn_mva'].fillna(0)
            # net.load.loc[net.load['sn_mva'] == 0, 'sn_mva'] = net.load[
            #     'p_mw']  # Use p_mw directly if cos_phi is assumed 1

    # Ensure required columns are present after potential calculations
    required_columns = ['sn_mva', 'p_mw', 'q_mvar']

    for column in required_columns:
        if column not in net.load.columns:
            print(f"Ensure that '{column}' column exists in net.load DataFrame")
            return None

    # Prepare data for disparity calculation
    load_data = net.load[['sn_mva', 'p_mw', 'q_mvar']].copy()

    # Select relevant columns for disparity calculation
    load_data = load_data[['p_mw', 'q_mvar', 'sn_mva']]

    # Calculate disparity matrix (Euclidean distance between load characteristics)
    n = len(load_data)
    disparity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                disparity_matrix[i, j] = np.sqrt((load_data.p_mw.iloc[i] - load_data.p_mw.iloc[j]) ** 2 +
                                                 (load_data.q_mvar.iloc[i] - load_data.q_mvar.iloc[j]) ** 2 +
                                                 (load_data.sn_mva.iloc[i] - load_data.sn_mva.iloc[j]) ** 2 )

    # Convert to DataFrame for easier handling
    disparity_df = pd.DataFrame(disparity_matrix, index=load_data.index, columns=load_data.index)

    max_p = load_data['p_mw'].max()
    max_q = load_data['q_mvar'].max()
    max_sn = load_data['sn_mva'].max()
    max_disparity = np.sqrt(max_p ** 2 + max_q ** 2 + max_sn ** 2)
    n = len(load_data)
    max_integral_value = (n * (n - 1) / 2) * max_disparity

    return disparity_df, max_integral_value

def calculate_transformer_disparity(net):
    # Ensure that transformer has the required columns
    required_columns = ['sn_mva', 'vn_hv_kv', 'vn_lv_kv', 'vkr_percent',
                        'vk_percent', 'pfe_kw', 'i0_percent', 'shift_degree']
    for column in required_columns:
        if column not in net.trafo.columns:
            print(f"Ensure that '{column}' column exists in net.trafo")
            return None

    # Combine the metrics into a single dataframe
    trafo_data = net.trafo[required_columns]

    # Create disparity matrix (Euclidean distance between combined metrics)
    n = len(trafo_data)
    disparity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                disparity_matrix[i, j] = np.sqrt(
                    (trafo_data.sn_mva.iloc[i] - trafo_data.sn_mva.iloc[j]) ** 2 +
                    (trafo_data.vn_hv_kv.iloc[i] - trafo_data.vn_hv_kv.iloc[j]) ** 2 +
                    (trafo_data.vn_lv_kv.iloc[i] - trafo_data.vn_lv_kv.iloc[j]) ** 2 +
                    (trafo_data.vkr_percent.iloc[i] - trafo_data.vkr_percent.iloc[j]) ** 2 +
                    (trafo_data.vk_percent.iloc[i] - trafo_data.vk_percent.iloc[j]) ** 2 +
                    (trafo_data.pfe_kw.iloc[i] - trafo_data.pfe_kw.iloc[j]) ** 2 +
                    (trafo_data.i0_percent.iloc[i] - trafo_data.i0_percent.iloc[j]) ** 2 +
                    (trafo_data.shift_degree.iloc[i] - trafo_data.shift_degree.iloc[j]) ** 2
                )

    # Convert to DataFrame for easier handling
    disparity_df = pd.DataFrame(disparity_matrix, index=trafo_data.index, columns=trafo_data.index)

    # max disparity
    max_sn = trafo_data['sn_mva'].max()
    max_vnhv = trafo_data['vn_hv_kv'].max()
    max_vnlv = trafo_data['vn_lv_kv'].max()
    max_vkr = trafo_data['vkr_percent'].max()
    max_vk = trafo_data['vk_percent'].max()
    max_pfe = trafo_data['pfe_kw'].max()
    max_i0 = trafo_data['i0_percent'].max()
    max_sh = trafo_data['shift_degree'].max()

    max_disparity = np.sqrt(max_sn ** 2 + max_vnhv ** 2 + max_vnlv **2 + max_vkr **2 + max_vk **2 + max_pfe **2 + max_i0 **2 + max_sh **2)
    n = len(trafo_data)
    max_integral_value = (n * (n - 1) / 2) * max_disparity

    # # Visualize the disparity matrix
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(disparity_df, cmap='viridis', annot=True, fmt='.2f')
    # plt.title('Disparity Space for Transformers')
    # plt.xlabel('Transformer Index')
    # plt.ylabel('Transformer Index')
    # plt.show()

    return disparity_df,max_integral_value

def calculate_line_disparity(net):
    # Ensure that line has the required columns
    required_columns = ['length_km', 'from_bus', 'to_bus', 'type', 'r_ohm_per_km', 'x_ohm_per_km', 'max_i_ka']
    for column in required_columns:
        if column not in net.line.columns:
            print(f"Ensure that '{column}' column exists in net.line")
            return None

    net.line['from_bus_norm'] = normalize_categorical(net.line['from_bus'])
    net.line['to_bus_norm'] = normalize_categorical(net.line['to_bus'])
    net.line['cable_type_norm'] = normalize_categorical(net.line['type'])

    # Combine the metrics into a single dataframe
    line_data = net.line[['length_km', 'from_bus_norm', 'to_bus_norm', 'cable_type_norm', 'r_ohm_per_km', 'x_ohm_per_km', 'max_i_ka']]

    # Create disparity matrix (Euclidean distance between combined metrics)
    n = len(line_data)
    disparity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                disparity_matrix[i, j] = np.sqrt(
                    (line_data.length_km.iloc[i] - line_data.length_km.iloc[j]) ** 2 +
                    (line_data.from_bus_norm.iloc[i] - line_data.from_bus_norm.iloc[j]) ** 2 +
                    (line_data.to_bus_norm.iloc[i] - line_data.to_bus_norm.iloc[j]) ** 2 +
                    (line_data.cable_type_norm.iloc[i] - line_data.cable_type_norm.iloc[j]) ** 2 +
                    (line_data.r_ohm_per_km.iloc[i] - line_data.r_ohm_per_km.iloc[j]) ** 2 +
                    (line_data.x_ohm_per_km.iloc[i] - line_data.x_ohm_per_km.iloc[j]) ** 2 +
                    (line_data.max_i_ka.iloc[i] - line_data.max_i_ka.iloc[j]) ** 2
                )

    # Convert to DataFrame for easier handling
    disparity_df = pd.DataFrame(disparity_matrix, index=line_data.index, columns=line_data.index)

    # Print both counts in one row
    print("Bus | Original | Normed ")
    print("-" * 45)
    for k in range(len(net.line)):
        normed = net.line['to_bus_norm'][k]
        original = net.line['to_bus'][k]
        print(f"{k:<12} | {original:<14} | {normed:<20}")

    # max disparity
    max_length = net.line['length_km'].max()
    max_fbm = net.line['from_bus_norm'].max()
    max_tbm = net.line['to_bus_norm'].max()
    max_ctn = net.line['cable_type_norm'].max()
    max_ohm = net.line['r_ohm_per_km'].max()
    max_xohm = net.line['x_ohm_per_km'].max()
    max_ika = net.line['max_i_ka'].max()

    max_disparity = np.sqrt(max_length ** 2 + max_fbm ** 2 + max_tbm **2 + max_ctn **2 + max_ohm **2 + max_xohm **2 + max_ika **2)
    n = len(net.line)
    max_integral_value = (n * (n - 1) / 2) * max_disparity

    return disparity_df,max_integral_value

def normalize_categorical(data):
    """
    Normalizes categorical data to numeric values starting from 0.
    """
    unique_values = data.unique()
    value_map = {value: idx for idx, value in enumerate(unique_values)}
    return data.map(value_map)