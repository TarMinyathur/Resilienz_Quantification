# diversity.py

import numpy as np
import pandas as pd

def calculate_shannon_evenness_and_variety(data, max_known_types):
    """
    Calculate Shannon evenness and variety for a given DataFrame based on the 'type' attribute.

    :param data: The DataFrame containing the component data
    :param max_known_types: The maximum number of known types for the component
    :return: Tuple of (Shannon evenness, variety)
    """
    if 'type' in data.columns:
        component_types = data['type'].value_counts()
    else:
        print("No 'type' column found in the data")
        return None, None

    # Calculate the proportion of each type

    total_components = len(data)
    if total_components > 0:
        proportions = np.minimum(1, component_types / total_components)
    else:
        proportions = pd.Series(0, index=component_types.index)  # or just proportions = 0

    # Calculate the Shannon entropy
    shannon_entropy = -np.sum(proportions * np.log(proportions))

    # Calculate the Shannon evenness
    max_entropy = np.log(len(component_types)) if len(component_types) > 0 else 0
    shannon_evenness = shannon_entropy / max_entropy if max_entropy > 0 else 0

    # Calculate the variety
    variety = len(component_types)
    max_variety = max_known_types
    variety_scaled = min(1, variety / max_variety)

    return shannon_evenness, variety, variety_scaled, max_variety, shannon_entropy

def calculate_transformer_voltage_diversity(transformers, buses, max_known_voltage_levels):
    """
    Calculate Shannon evenness, variety, and entropy for transformer high-voltage sides,
    using voltage levels from connected buses.

    :param transformers: DataFrame of transformers, must contain 'hv_bus' column
    :param buses: DataFrame of buses, must contain 'bus_id' and 'vn_kv' columns
    :param max_known_voltage_levels: Integer representing the max number of distinct voltage levels possible
    :return: Tuple (shannon_evenness, variety, variety_scaled, max_variety, shannon_entropy)
    """

    hv_data = transformers.merge(buses[['vn_kv']], left_on='hv_bus', right_index=True, how='left')

    # Check if 'vn_kv' is present after merge
    if 'vn_kv' not in hv_data.columns:
        print("Voltage level (vn_kv) data not found after merge.")
        return None, None, None, None, None

    # Count voltage level frequencies
    voltage_counts = hv_data['vn_kv'].value_counts()
    total_transformers = len(hv_data)

    if total_transformers > 0:
        proportions = voltage_counts / total_transformers
    else:
        proportions = pd.Series(0, index=voltage_counts.index)

    # Shannon entropy
    shannon_entropy = -np.sum(proportions * np.log(proportions))

    # Shannon evenness
    max_entropy = np.log(len(voltage_counts)) if len(voltage_counts) > 0 else 0
    shannon_evenness = shannon_entropy / max_entropy if max_entropy > 0 else 0

    # Variety and scaled variety
    variety = len(voltage_counts)
    max_variety = max_known_voltage_levels
    variety_scaled = variety / max_variety if max_variety > 0 else 0

    return shannon_evenness, variety, variety_scaled, max_variety, shannon_entropy