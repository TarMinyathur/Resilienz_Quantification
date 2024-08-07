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