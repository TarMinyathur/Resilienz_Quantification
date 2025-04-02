# diversity.py

import numpy as np

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
    proportions = component_types / total_components if total_components > 0 else 0

    # Calculate the Shannon entropy
    shannon_entropy = -np.sum(proportions * np.log(proportions))

    # Calculate the Shannon evenness
    max_entropy = np.log(len(component_types)) if len(component_types) > 0 else 0
    shannon_evenness = shannon_entropy / max_entropy if max_entropy > 0 else 0

    # Calculate the variety
    variety = len(component_types)
    max_variety = max_known_types
    variety_scaled = variety / max_variety

    return shannon_evenness, variety, variety_scaled, max_variety, shannon_entropy