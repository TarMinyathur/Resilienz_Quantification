# import pandapower as pp
import pandapower.networks as pn
#import itertools
#import math
import networkx as nx
#import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from count_elements import count_elements
from diversity import calculate_shannon_evenness_and_variety
from disparity import calculate_disparity_space, calculate_line_disparity, calculate_transformer_disparity, calculate_load_disparity
from GenerationFactors import calculate_generation_factors
from Redundancy import n_3_redundancy_check
from visualize import plot_spider_chart
from initialize import add_indicator
from initialize import add_disparity
from indi_gt import GraphenTheorieIndicator
from adjustments import set_missing_limits
from self_sufficiency import selfsuff

#initialize test grids from CIGRE; either medium voltage including renewables or the low voltage grid
#net = pn.create_cigre_network_lv()
net = pn.create_cigre_network_mv('all')
# False, 'pv_wind', 'all'

net = set_missing_limits(net)

#pp.runopp(net)
#print(net.switch)

# Initialize an empty DataFrame

dfinalresults = pd.DataFrame(columns=['Indicator', 'Value'])
ddisparity = pd.DataFrame(columns=['Name', 'Value', 'max Value', 'Verhaeltnis'])

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

#checks if G is complete connected, otherwise the largest subgraph is analyzed going forward
if not nx.is_connected(G):
    # Get largest connected component
    largest_component = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_component).copy()

GraphenTheorieIndicator(G, dfinalresults)

indi_selfsuff = float(selfsuff(net))
print(f" self sufficiency: {indi_selfsuff}")
dfinalresults = add_indicator(dfinalresults,'self sufficiency at bus level',indi_selfsuff)


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

dfinalresults = add_indicator(dfinalresults,"Generation Shannon Evenness", evenness)
dfinalresults = add_indicator(dfinalresults,"Generation Variety", variety_scaled)
#add_indicator("Generation Max Variety", max_variety)

# Calculate and print Shannon evenness and variety for lines
evenness, variety, variety_scaled, max_variety = calculate_shannon_evenness_and_variety(net.line, max_known_types['line'])
print(f"Line - Shannon Evenness: {evenness}, Variety: {variety}, Max Variety: {max_variety}, Scaled Variety: {variety_scaled}")

dfinalresults = add_indicator(dfinalresults,"Line Shannon Evenness", evenness)
dfinalresults = add_indicator(dfinalresults,"Line Variety", variety_scaled)
#add_indicator("Line Max Variety", max_variety)

# Calculate and print Shannon evenness and variety for loads
evenness, variety, variety_scaled, max_variety = calculate_shannon_evenness_and_variety(net.load, max_known_types['load'])
print(f"Load - Shannon Evenness: {evenness}, Variety: {variety}, Max Variety: {max_variety}, Scaled Variety: {variety_scaled}")

dfinalresults = add_indicator(dfinalresults,"Load Shannon Evenness", evenness)
dfinalresults = add_indicator(dfinalresults,"Load Variety", variety_scaled)

# Calculate generation factors
generation_factors = calculate_generation_factors(net)

# Calculate disparity space
disparity_df_gen, max_integral_gen = calculate_disparity_space(net, generation_factors)

# Compute the integral (sum) over the entire DataFrame
integral_value_gen = disparity_df_gen.values.sum()
print(f"Disparity Integral Generators: {integral_value_gen}")
print(f"Disparity Integral max Loads: {max_integral_gen}")

ddisparity = add_disparity(ddisparity,'Generators', integral_value_gen, max_integral_gen, integral_value_gen / max_integral_gen)

# Calculate disparity space for loads
disparity_df_load, max_integral_load = calculate_load_disparity(net)
#print(disparity_df_load)

# Compute the integral (sum) over the entire DataFrame
integral_value_load = disparity_df_load.values.sum()
print(f"Disparity Integral Loads: {integral_value_load}")
print(f"Disparity Integral max Loads: {max_integral_load }")

ddisparity =add_disparity(ddisparity,'Load', integral_value_load, max_integral_load, integral_value_load/ max_integral_load)

# Calculate disparity space for transformers
disparity_df_trafo,max_int_trafo = calculate_transformer_disparity(net)
#print(disparity_df_trafo)

# Compute the integral (sum) over the entire DataFrame
integral_value_trafo = disparity_df_trafo.values.sum()
print(f"Disparity Integral Transformers: {integral_value_trafo}")
print(f"max theoretical Disparity Integral Transformers: {max_int_trafo}")

ddisparity = add_disparity(ddisparity,'Trafo', integral_value_trafo, max_int_trafo, integral_value_trafo / max_int_trafo)

# Calculate disparity space for lines
disparity_df_lines,max_int_disp_lines = calculate_line_disparity(net)

# Compute the integral (sum) over the entire DataFrame
integral_value_line = disparity_df_lines.values.sum()
print(f"Disparity Integral Lines: {integral_value_line}")
print(f"max theoretical Disparity Integral Lines: {max_int_disp_lines}")

ddisparity = add_disparity(ddisparity,'Lines',integral_value_line,max_int_disp_lines,integral_value_line / max_int_disp_lines)

print(ddisparity)

dfinalresults = add_indicator(dfinalresults,'Disparity Generators',ddisparity.loc[ddisparity['Indicator'] == 'Generators', 'Verhaeltnis'].values[0])
dfinalresults = add_indicator(dfinalresults,'Disparity Load',ddisparity.loc[ddisparity['Indicator'] == 'Load', 'Verhaeltnis'].values[0])
dfinalresults = add_indicator(dfinalresults,'Disparity Trafo',ddisparity.loc[ddisparity['Indicator'] == 'Trafo', 'Verhaeltnis'].values[0])
dfinalresults = add_indicator(dfinalresults,'Disparity Lines',ddisparity.loc[ddisparity['Indicator'] == 'Lines', 'Verhaeltnis'].values[0])
# dfinalresults = add_indicator('Disparity',ddisparity['Verhaeltnis'].mean())

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

dfinalresults = add_indicator(dfinalresults,'Overall 70% Redundancy', rate)


#Output

plot_spider_chart(dfinalresults)
# Save the DataFrame to an Excel file
# dfinalresults.to_excel("dfinalresults.xlsx", sheet_name="Results", index=False)
print(dfinalresults)
