import pandapower as pp
import pandapower.networks as pn
import itertools
import math
import networkx as nx
from networkx.algorithms import community
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


net = pn.create_cigre_network_mv(with_der="all")

'''
This pandapower network includes the following parameter tables:
  - switch (8 elements)
  - load (18 elements)
  - ext_grid (1 elements)
  - sgen (15 elements)
  - line (15 elements)
  - trafo (2 elements)
  - bus (15 elements)
  - bus_geodata (15 elements)
'''
#changes
def set_storage_max_e_mwh(net):
    for idx, storage in net.storage.iterrows():
        net.storage.at[idx, 'max_e_mwh'] = storage['p_mw'] * 24

# Set max_e_mwh for all storage units
set_storage_max_e_mwh(net)

pp.runpp(net)
#print(net.line)
#print(net.load)
#print(net.gen)
#print(net.storage)

# Convert Pandapower network to NetworkX graph
#G = pp.to_networkx(net)
# Create an empty NetworkX graph
G = nx.Graph()

# Add nodes from Pandapower network
for bus in net.bus.index:
    G.add_node(bus)

# Add edges from Pandapower network
for idx, line in net.line.iterrows():
    G.add_edge(line.from_bus, line.to_bus)

def count_elements(net):
    counts = {
        "switch": len(net.switch),
        "load": len(net.load),
        "sgen": len(net.sgen),
        "line": len(net.line),
        "trafo": len(net.trafo),
        "bus": len(net.bus),
        "storage": len(net.storage) if "storage" in net else 0  # Handle networks without storage elements
    }

    # Multiply counts by 0.3 and round down
    scaled_counts = {}
    for element_type, count in counts.items():
        scaled_count = math.floor(count * 0.3)
        scaled_counts[element_type] = scaled_count

    # Create a dictionary containing both counts and scaled counts
    element_counts = {
        "original_counts": counts,
        "scaled_counts": scaled_counts
    }

    return element_counts


# Count elements and scaled elements
element_counts = count_elements(net)

# Print both counts in one row
print("Element Type | Original Count | Scaled Count (0.3)")
print("-" * 45)
for element_type in element_counts["original_counts"]:
    original_count = element_counts["original_counts"][element_type]
    scaled_count = element_counts["scaled_counts"][element_type]
    print(f"{element_type.capitalize():<12} | {original_count:<14} | {scaled_count:<20}")


def n_3_redundancy_check(net):
    results = {
        "line": {"Success": 0, "Failed": 0},
        "switch": {"Success": 0, "Failed": 0},
        "load": {"Success": 0, "Failed": 0},
        "sgen": {"Success": 0, "Failed": 0},
        "trafo": {"Success": 0, "Failed": 0},
        "bus": {"Success": 0, "Failed": 0},
        "storage": {"Success": 0, "Failed": 0}
    }

    # Create combinations of three elements for each type
    line_triples = itertools.combinations(net.line.index, element_counts["scaled_counts"]["line"])
    switch_triples = itertools.combinations(net.switch.index, element_counts["scaled_counts"]["switch"])
    load_triples = itertools.combinations(net.load.index, element_counts["scaled_counts"]["load"])
    sgen_triples = itertools.combinations(net.sgen.index, element_counts["scaled_counts"]["sgen"])
    trafo_triples = itertools.combinations(net.trafo.index, element_counts["scaled_counts"]["trafo"])
    bus_triples = itertools.combinations(net.bus.index, element_counts["scaled_counts"]["bus"])
    storage_triples = itertools.combinations(net.storage.index, element_counts["scaled_counts"]["storage"])

    element_triples = {
        "line": line_triples,
        "switch": switch_triples,
        "load": load_triples,
        "sgen": sgen_triples,
        "trafo": trafo_triples,
        "bus": bus_triples,
        "storage": storage_triples
    }

    # Function to process each triple and update results
    def process_triples(element_type, triples):
        for triple in triples:
            # Create a copy of the network to simulate the failures
            net_temp = net.deepcopy()
            # Set the elements out of service
            for element_id in triple:
                net_temp[element_type].at[element_id, 'in_service'] = False
            # Run the load flow calculation
            try:
                pp.runpp(net_temp, calculate_voltage_angles=True, tolerance_mva=1e-10)
                results[element_type]["Success"] += 1
            except:
                results[element_type]["Failed"] += 1

    # Process each element type
    for element_type, triples in element_triples.items():
        process_triples(element_type, triples)

    return results


# Perform N-3 redundancy check
n3_redundancy_results = n_3_redundancy_check(net)

# Print the results
for element_type, counts in n3_redundancy_results.items():
    print(f"{element_type.capitalize()} - Success count: {counts['Success']}, Failed count: {counts['Failed']}")

# Ensure graph is connected (or handle disconnected components)
if not nx.is_connected(G):
    # Get largest connected component
    largest_component = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_component).copy()

# Now calculate average shortest path length for the largest connected component
if G.number_of_nodes() > 1:
    avg_path_length = nx.average_shortest_path_length(G)
    print(f"Average Shortest Path Length: {avg_path_length}")
else:
    print("Graph has only one node, cannot calculate average shortest path length.")

# Calculate degree centrality
degree_centrality = nx.degree_centrality(G)

# Calculate average degree centrality
avg_degree_centrality = sum(degree_centrality.values()) / len(degree_centrality)

print(f"Degree Centrality:")
for node, centrality in degree_centrality.items():
    print(f"Bus {node}: {centrality}")

print(f"\nAverage Degree Centrality: {avg_degree_centrality}")


def calculate_modularity_index(G, communities):
    modularity_index = 0.0

    # Calculate total number of edges in the graph
    total_edges = G.number_of_edges()

    # Calculate modularity components for each community
    for community_nodes in communities:
        # Calculate e_ii: Fraction of edges within the community
        e_ii = sum(1 for u, v in G.edges(community_nodes) if v in community_nodes) / total_edges

        # Calculate a_i: Total fraction of edges from nodes in the community
        a_i = sum(1 for u, v in G.edges(community_nodes) if v not in community_nodes) / total_edges

        # Calculate (e_ii - a_i)^2 and accumulate to modularity index
        modularity_index += (e_ii - a_i) ** 2

    return modularity_index

# Detect communities (optional): Using Louvain method
communities = community.greedy_modularity_communities(G)

# Calculate modularity index
modularity_index = calculate_modularity_index(G, communities)

print(f"Modularity Index (Q): {modularity_index}")

# Check if generation meets demand on each bus
for bus in net.bus.index:
    generation_p = sum(net.gen[net.gen.bus == bus].p_mw) + sum(net.sgen[net.sgen.bus == bus].p_mw) \
                   + sum(net.storage[net.storage.bus == bus].p_mw)
    #generation_q = sum(net.gen[net.gen.bus == bus].q_mvar.fillna(0)) + sum(net.sgen[net.sgen.bus == bus].q_mvar.fillna(0))
    generation_q = sum(net.sgen[net.sgen.bus == bus].q_mvar.fillna(0))
    generation_s = sum((net.sgen[net.sgen.bus == bus].p_mw ** 2 + net.sgen[net.sgen.bus == bus].q_mvar ** 2) ** 0.5)

    demand_p = sum(net.load[net.load.bus == bus].p_mw)
    demand_q = sum(net.load[net.load.bus == bus].q_mvar)
    demand_s = sum((net.load[net.load.bus == bus].p_mw ** 2 + net.load[net.load.bus == bus].q_mvar ** 2) ** 0.5)

    print(f"Bus {bus}:")
    print(f"   Active Power:   Generation = {generation_p} MW, Demand = {demand_p} MW")
    print(f"   Reactive Power: Generation = {generation_q} Mvar, Demand = {demand_q} Mvar")
    print(f"   Apparent Power: Generation = {generation_s} MVA, Demand = {demand_s} MVA")

# Remove external grid (optional, if needed)
# pruefen ob sinnvoll oder nicht
    # Create a copy of the network to simulate the failures
    #net_temp_ext = net.deepcopy()
    #pp.drop_elements(net_temp_ext, [0], element="ext_grid")
    # Deactivate the external grid
    #net_temp_ext.ext_grid.at[0, 'in_service'] = False

# Run power flow
#pp.runpp(net_temp_ext)

# Check convergence
#if net_temp_ext.converged:
#    print("Load flow converged without external grid.")
#else:
#    print("Load flow did not converge.")

def calculate_shannon_evenness(net):
    # Step 1: Identify and count different types of sgen
    if 'type' in net.sgen.columns:
        sgen_types = net.sgen['type'].value_counts()
    else:
        # If 'type' column doesn't exist, return None or appropriate value
        print("No 'type' column found in net.sgen")
        return None

    # Step 2: Calculate the proportion of each type
    total_sgen = len(net.sgen)
    proportions = sgen_types / total_sgen

    # Step 3: Calculate the Shannon entropy
    shannon_entropy = -np.sum(proportions * np.log(proportions))

    # Step 4: Calculate the Shannon evenness
    max_entropy = np.log(len(sgen_types))
    shannon_evenness = shannon_entropy / max_entropy if max_entropy > 0 else 0

    #print(f"Variety: {sgen_types}")
    print(f"Variety: {len(sgen_types)}")
    return shannon_evenness

# Calculate Shannon evenness
shannon_evenness = calculate_shannon_evenness(net)
print(f"Shannon Evenness: {shannon_evenness}")


def calculate_disparity_space(net, generation_factors):
    # Ensure that sgen and storage have the required columns
    required_columns = ['bus', 'p_mw', 'q_mvar', 'sn_mva', 'type']
    for column in required_columns:
        if column not in net.sgen.columns or column not in net.storage.columns:
            print(f"Ensure that '{column}' column exists in both net.sgen and net.storage")
            return None

    # Sum p_mw, q_mvar, and sn_mva*generation_factor over all sgen at each bus
    net.sgen['effective_sn_mva'] = net.sgen.apply(lambda row: row['sn_mva'] * generation_factors.get(row['type'], 0),
                                                  axis=1)
    sgen_sums = net.sgen.groupby('bus')[['p_mw', 'q_mvar', 'effective_sn_mva']].sum().reset_index()

    # Sum p_mw, q_mvar, and sn_mva*generation_factor over all storage at each bus
    net.storage['effective_sn_mva'] = net.storage.apply(
        lambda row: row['sn_mva'] * generation_factors.get(row['type'], 0), axis=1)
    storage_sums = net.storage.groupby('bus')[['p_mw', 'q_mvar', 'effective_sn_mva']].sum().reset_index()

    # Merge the sums from sgen and storage
    total_sums = pd.merge(sgen_sums, storage_sums, on='bus', how='outer', suffixes=('_sgen', '_storage')).fillna(0)
    total_sums['p_mw'] = total_sums['p_mw_sgen'] + total_sums['p_mw_storage']
    total_sums['q_mvar'] = total_sums['q_mvar_sgen'] + total_sums['q_mvar_storage']
    total_sums['effective_sn_mva'] = total_sums['effective_sn_mva_sgen'] + total_sums['effective_sn_mva_storage']

    # Select only the relevant columns
    total_sums = total_sums[['bus', 'p_mw', 'q_mvar', 'effective_sn_mva']]

    # Create disparity matrix (Euclidean distance between summed p_mw, q_mvar, and effective_sn_mva)
    n = len(total_sums)
    disparity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                disparity_matrix[i, j] = np.sqrt((total_sums.p_mw.iloc[i] - total_sums.p_mw.iloc[j]) ** 2 +
                                                 (total_sums.q_mvar.iloc[i] - total_sums.q_mvar.iloc[j]) ** 2 +
                                                 (total_sums.effective_sn_mva.iloc[i] -
                                                  total_sums.effective_sn_mva.iloc[j]) ** 2)

    # Convert to DataFrame for easier handling
    disparity_df = pd.DataFrame(disparity_matrix, index=total_sums.bus, columns=total_sums.bus)

    # Visualize the disparity matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(disparity_df, cmap='viridis', annot=True, fmt='.2f')
    plt.title('Disparity Space for Summed Static Generators and Storage at Each Bus (3D)')
    plt.xlabel('Bus Index')
    plt.ylabel('Bus Index')
    plt.show()

    return disparity_df

def calculate_generation_factors(net):
    # Initialize generation factors
    generation_factors = {}

    # Calculate for static generators (sgen)
    sgen_types = net.sgen['type'].unique()
    for sgen_type in sgen_types:
        if sgen_type == 'pv':
            generation_factors[sgen_type] = 0.15  # Example factor
        elif sgen_type == 'wind':
            generation_factors[sgen_type] = 0.25  # Example factor
        elif sgen_type == 'biomass':
            generation_factors[sgen_type] = 0.8  # Example factor

    # Calculate for batteries (storage)
    for idx, row in net.storage.iterrows():
        capacity = row['sn_mva']
        p_mw = row['p_mw']
        generation_factors['battery'] = (capacity / p_mw) / 24 if p_mw != 0 else 0

    return generation_factors

# Calculate generation factors
generation_factors = calculate_generation_factors(net)

# Calculate disparity space
disparity_df_gen = calculate_disparity_space(net, generation_factors)
print(disparity_df_gen)

# Compute the integral (sum) over the entire DataFrame
integral_value_gen = disparity_df_gen.values.sum()
print(f"Disparity Integral Generators: {integral_value_gen}")

def calculate_load_disparity(net):
    # Ensure load DataFrame has the required columns
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

    # Visualize the disparity matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(disparity_df, cmap='viridis', annot=True, fmt='.2f')
    plt.title('Disparity Space for Loads (5D)')
    plt.xlabel('Load Index')
    plt.ylabel('Load Index')
    plt.show()

    return disparity_df

def calculate_theoretical_max_disparity(load_data):
    # Prepare data for disparity calculation
    load_data = net.load[['sn_mva', 'p_mw', 'q_mvar']].copy()

    max_p = load_data['p_mw'].max()
    max_q = load_data['q_mvar'].max()
    max_sn = load_data['sn_mva'].max()
    max_disparity = np.sqrt(max_p**2 + max_q**2 + max_sn**2 )
    n = len(load_data)
    max_integral_value = (n * (n - 1) / 2) * max_disparity

    return max_integral_value


# Calculate disparity space for loads
disparity_df_load = calculate_load_disparity(net)
print(disparity_df_load)

# Compute the integral (sum) over the entire DataFrame
integral_value_load = disparity_df_load.values.sum()
max_integral_load = calculate_theoretical_max_disparity(net.load)
print(f"Disparity Integral Loads: {integral_value_load}")
print(f"Disparity Integral max Loads: {max_integral_load }")


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

    # Visualize the disparity matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(disparity_df, cmap='viridis', annot=True, fmt='.2f')
    plt.title('Disparity Space for Transformers')
    plt.xlabel('Transformer Index')
    plt.ylabel('Transformer Index')
    plt.show()

    return disparity_df


# Calculate disparity space for transformers
disparity_df_trafo = calculate_transformer_disparity(net)
print(disparity_df_trafo)

# Compute the integral (sum) over the entire DataFrame
integral_value_trafo = disparity_df_trafo.values.sum()
print(f"Disparity Integral Transformers: {integral_value_trafo}")

def normalize_categorical(data):
    """
    Normalizes categorical data to numeric values starting from 0.
    """
    unique_values = data.unique()
    value_map = {value: idx for idx, value in enumerate(unique_values)}
    return data.map(value_map)


def calculate_line_disparity(net):
    # Ensure that line has the required columns
    required_columns = ['length_km', 'from_bus', 'to_bus', 'std_type', 'r_ohm_per_km']
    for column in required_columns:
        if column not in net.line.columns:
            print(f"Ensure that '{column}' column exists in net.line")
            return None

    # Normalize categorical columns
    net.line['from_bus_norm'] = normalize_categorical(net.line['from_bus'])
    net.line['to_bus_norm'] = normalize_categorical(net.line['to_bus'])
    net.line['cable_type_norm'] = normalize_categorical(net.line['std_type'])

    # Combine the metrics into a single dataframe
    line_data = net.line[['length_km', 'from_bus_norm', 'to_bus_norm', 'cable_type_norm', 'r_ohm_per_km']]

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
                    (line_data.r_ohm_per_km.iloc[i] - line_data.r_ohm_per_km.iloc[j]) ** 2
                )

    # Convert to DataFrame for easier handling
    disparity_df = pd.DataFrame(disparity_matrix, index=line_data.index, columns=line_data.index)

    # Visualize the disparity matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(disparity_df, cmap='viridis', annot=True, fmt='.2f')
    plt.title('Disparity Space for Lines')
    plt.xlabel('Line Index')
    plt.ylabel('Line Index')
    plt.show()

    return disparity_df


# Calculate disparity space for lines
disparity_df_lines = calculate_line_disparity(net)
print(disparity_df_lines)

# Compute the integral (sum) over the entire DataFrame
integral_value_line = disparity_df_lines.values.sum()
print(f"Disparity Integral Lines: {integral_value_line}")
