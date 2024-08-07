import pandapower as pp
import pandapower.networks as pn
#import itertools
#import math
import networkx as nx
from networkx.algorithms import community
#import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from count_elements import count_elements
from modularity import calculate_modularity_index
from diversity import calculate_shannon_evenness_and_variety
from disparity import calculate_disparity_space, calculate_line_disparity, calculate_transformer_disparity, calculate_load_disparity
from GenerationFactors import calculate_generation_factors
from Redundancy import n_3_redundancy_check
from visualize import plot_spider_chart

#net = pn.create_cigre_network_lv()
net = pn.create_cigre_network_mv(False)
# False, 'pv_wind', 'all'

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

# Close all switches in the network
net.switch['closed'] = True

pp.runpp(net)
print(net.switch)

# Initialize an empty DataFrame
dfinalresults = pd.DataFrame(columns=['Indicator', 'Value'])
ddisparity = pd.DataFrame(columns=['Name', 'Value', 'max Value', 'Verhaeltnis'])

# Function to add data to the DataFrame
def add_indicator(indicator_name, value):
    global dfinalresults
    dfinalresults = dfinalresults.append({'Indicator': indicator_name, 'Value': value}, ignore_index=True)

def add_disparity(indicator_name, value, max_value, verhaeltnis):
    global ddisparity
    ddisparity = ddisparity.append({'Indicator': indicator_name, 'Value': value, 'max Value': max_value, 'Verhaeltnis': verhaeltnis}, ignore_index=True)

# Convert Pandapower network to NetworkX graph
#G = pp.to_networkx(net)
# Create an empty NetworkX graph
G = nx.Graph()

# Add nodes from Pandapower network
for bus in net.bus.index:
    G.add_node(bus)

for idx, line in net.line.iterrows():
    from_bus = line.from_bus
    to_bus = line.to_bus

    # Check if there is a switch between from_bus and to_bus
    switch_exists = False
    for _, switch in net.switch.iterrows():
        if switch.bus == from_bus and switch.element == to_bus and switch.et == 'l':
            switch_exists = True
            switch_closed = switch.closed
            break
        elif switch.bus == to_bus and switch.element == from_bus and switch.et == 'l':
            switch_exists = True
            switch_closed = switch.closed
            break

    # Only add the edge if there is no switch or if the switch is closed
    if not switch_exists or (switch_exists and switch_closed):
        length = line.length_km
        G.add_edge(from_bus, to_bus, weight=length)

# Count elements and scaled elements
element_counts = count_elements(net)

# Print both counts in one row
print("Element Type | Original Count | Scaled Count (0.3)")
print("-" * 45)
for element_type in element_counts["original_counts"]:
    original_count = element_counts["original_counts"][element_type]
    scaled_count = element_counts["scaled_counts"][element_type]
    print(f"{element_type.capitalize():<12} | {original_count:<14} | {scaled_count:<20}")

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

num_nodes = G.number_of_nodes()
num_nodes = (num_nodes - 1) if num_nodes > 1 else 0
norm_avg_pl = max(0, 1 - (avg_path_length / num_nodes ))
print(f"Datatype of norm_avg_pl: {type(norm_avg_pl)}")
print(f"Normalized Average Path Length: {norm_avg_pl}")
add_indicator('Average Shortest Path Length',norm_avg_pl)

# Calculate degree centrality
degree_centrality = nx.degree_centrality(G)

# Calculate average degree centrality
avg_degree_centrality = sum(degree_centrality.values()) / len(degree_centrality)

print(f"Degree Centrality:")
for node, centrality in degree_centrality.items():
    print(f"Bus {node}: {centrality}")

print(f"\nAverage Degree Centrality: {avg_degree_centrality}")

add_indicator('Average Degree Centrality',max(0,avg_degree_centrality))

# Detect communities (optional): Using Louvain method
communities = community.greedy_modularity_communities(G)

# Calculate modularity index
modularity_index = calculate_modularity_index(G, communities)

print(f"Modularity Index (Q): {modularity_index}")

add_indicator('Modularity Index',max(0,modularity_index))

erfolg = 0
misserfolg = 0
ergebnis = 0

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

    if generation_p >= demand_p:
        erfolg += 1
    else:
        misserfolg += 1

    if generation_q >= demand_q:
        erfolg += 1
    else:
        misserfolg += 1

    if generation_s >= demand_s:
        erfolg += 1
    else:
        misserfolg += 1


selfsuff = erfolg / (erfolg + misserfolg)
print(f" self sufficiency: {selfsuff}")
add_indicator('self sufficiency at bus level',selfsuff)


# Define the maximum known types for each component
max_known_types = {
    'generation': 8,  # Adjust this based on your actual known types (sgen: solar, wind, biomass, gen: gas, coal, nuclear, storage: battery, hydro
    'line': 2,  # "ol" (overhead line) and "cs" (cable system)
    'load': 10  # Example: 4 known types of loads (residential, commercial, industrial, agricultaral, transport, municipal, dynamic, static, critical, non-critical
}

# Combine sgen, gen, and storage into one DataFrame
generation_data = pd.concat([net.sgen, net.gen, net.storage], ignore_index=True)

# Calculate and print Shannon evenness and variety for combined generation units
evenness, variety, variety_scaled, max_variety = calculate_shannon_evenness_and_variety(generation_data, max_known_types['generation'])
print(f"Generation - Shannon Evenness: {evenness}, Variety: {variety}, Max Variety: {max_variety}, Scaled Variety: {variety_scaled}")

add_indicator("Generation Shannon Evenness", evenness)
add_indicator("Generation Variety", variety_scaled)
#add_indicator("Generation Max Variety", max_variety)

# Calculate and print Shannon evenness and variety for lines
evenness, variety, variety_scaled, max_variety = calculate_shannon_evenness_and_variety(net.line, max_known_types['line'])
print(f"Line - Shannon Evenness: {evenness}, Variety: {variety}, Max Variety: {max_variety}, Scaled Variety: {variety_scaled}")

add_indicator("Line Shannon Evenness", evenness)
add_indicator("Line Variety", variety_scaled)
#add_indicator("Line Max Variety", max_variety)

# Calculate and print Shannon evenness and variety for loads
evenness, variety, variety_scaled, max_variety = calculate_shannon_evenness_and_variety(net.load, max_known_types['load'])
print(f"Load - Shannon Evenness: {evenness}, Variety: {variety}, Max Variety: {max_variety}, Scaled Variety: {variety_scaled}")

add_indicator("Load Shannon Evenness", evenness)
add_indicator("Load Variety", variety_scaled)
#add_indicator("Load Max Variety", max_variety)
    # add_indicator('Shannon Evenness',shannon_evenness)
    # add_indicator('Variety',Variety)

# Calculate generation factors
generation_factors = calculate_generation_factors(net)

# Calculate disparity space
disparity_df_gen, max_integral_gen = calculate_disparity_space(net, generation_factors)

# Compute the integral (sum) over the entire DataFrame
integral_value_gen = disparity_df_gen.values.sum()
print(f"Disparity Integral Generators: {integral_value_gen}")
print(f"Disparity Integral max Loads: {max_integral_gen}")

add_disparity('Generators', integral_value_gen, max_integral_gen, integral_value_gen / max_integral_gen)

# Calculate disparity space for loads
disparity_df_load, max_integral_load = calculate_load_disparity(net)
#print(disparity_df_load)

# Compute the integral (sum) over the entire DataFrame
integral_value_load = disparity_df_load.values.sum()
print(f"Disparity Integral Loads: {integral_value_load}")
print(f"Disparity Integral max Loads: {max_integral_load }")

add_disparity('Load', integral_value_load, max_integral_load, integral_value_load/ max_integral_load)

# Calculate disparity space for transformers
disparity_df_trafo,max_int_trafo = calculate_transformer_disparity(net)
#print(disparity_df_trafo)

# Compute the integral (sum) over the entire DataFrame
integral_value_trafo = disparity_df_trafo.values.sum()
print(f"Disparity Integral Transformers: {integral_value_trafo}")
print(f"max theoretical Disparity Integral Transformers: {max_int_trafo}")

add_disparity('Trafo', integral_value_trafo, max_int_trafo, integral_value_trafo / max_int_trafo)

# Calculate disparity space for lines
disparity_df_lines,max_int_disp_lines = calculate_line_disparity(net)

# Compute the integral (sum) over the entire DataFrame
integral_value_line = disparity_df_lines.values.sum()
print(f"Disparity Integral Lines: {integral_value_line}")
print(f"max theoretical Disparity Integral Lines: {max_int_disp_lines}")

add_disparity('Lines',integral_value_line,max_int_disp_lines,integral_value_line / max_int_disp_lines)

print(ddisparity)

add_indicator('Disparity Generators',ddisparity.loc[ddisparity['Indicator'] == 'Generators', 'Verhaeltnis'].values[0])
add_indicator('Disparity Load',ddisparity.loc[ddisparity['Indicator'] == 'Load', 'Verhaeltnis'].values[0])
add_indicator('Disparity Trafo',ddisparity.loc[ddisparity['Indicator'] == 'Trafo', 'Verhaeltnis'].values[0])
add_indicator('Disparity Lines',ddisparity.loc[ddisparity['Indicator'] == 'Lines', 'Verhaeltnis'].values[0])
#add_indicator('Disparity',ddisparity['Verhaeltnis'].mean())

def count_elements(net):
    counts = {
        "switch": len(net.switch),
        "load": len(net.load),
        "sgen": len(net.sgen),
        "line": len(net.line),
        "trafo": len(net.trafo),
        "bus": len(net.bus),
        "storage": len(net.storage) if "storage" in net else 0  # Some networks might not have storage elements
    }
    return counts

element_counts = count_elements(net)
element_counts["scaled_counts"] = {k: int(v * 0.3) for k, v in element_counts.items()}

# Perform N-3 redundancy check
n3_redundancy_results = n_3_redundancy_check(net,element_counts)

# Initialize Success and Failed counters
Success = 0
Failed = 0
total_checks = 0

# Print the results
for element_type, counts in n3_redundancy_results.items():
    Success += counts['Success']
    Failed += counts['Failed']
    total_checks = Success + Failed
    rate = Success / total_checks if total_checks != 0 else 0
    print(f"{element_type.capitalize()} - Success count: {counts['Success']}, Failed count: {counts['Failed']}")

add_indicator('Overall 70% Redundancy', rate)

plot_spider_chart(dfinalresults)
# Save the DataFrame to an Excel file
dfinalresults.to_excel("dfinalresults.xlsx", sheet_name="Results", index=False)
print(dfinalresults)
