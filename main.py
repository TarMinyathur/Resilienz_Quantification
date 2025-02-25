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
from adjustments import determine_minimum_ext_grid_power
from self_sufficiency import selfsuff
from self_sufficiency import selfsufficiency_neu

dfinalresults = pd.DataFrame(columns=['Indicator', 'Value'])
ddisparity = pd.DataFrame(columns=['Name', 'Value', 'max Value', 'Verhaeltnis'])

# Basic initialisation of different test grids via "Grid": "mv_all"; "mv_pv_wind"; "mv_no_renew"; "lv"; "new"; "mv_all_high5"; "mv_all_high10"
#initialize test grids from CIGRE; either medium voltage including renewables or the low voltage grid

basic = {
    "Grid": "mv_all_high10",
    "Adjustements": True,
    "Overview_Grid": True
}

if basic["Grid"] == "mv_all_high10":
    net = pn.create_cigre_network_mv('all')
elif basic["Grid"] == "mv_all_high10":
    net = pn.create_cigre_network_mv('all')
    print("Verteilte Erzeugung und Speicher um den Faktor 10 erhöht.")

    # 1. Erhöhung für gen (zentraler Generator)
    for idx, gen in net.gen.iterrows():
        net.gen.at[idx, 'p_mw'] *= 10
        net.gen.at[idx, 'q_mvar'] *= 10

    # 2. Erhöhung für sgen (verteilte Erzeugung)
    for idx, sgen in net.sgen.iterrows():
        net.sgen.at[idx, 'p_mw'] *= 10
        net.sgen.at[idx, 'q_mvar'] *= 10

    # 3. Erhöhung für storage (Speicher)
    for idx, storage in net.storage.iterrows():
        net.storage.at[idx, 'p_mw'] *= 10
        net.storage.at[idx, 'q_mvar'] *= 10
elif basic["Grid"] == "mv_all_high5":
    net = pn.create_cigre_network_mv('all')
    print("Verteilte Erzeugung und Speicher um den Faktor 5 erhöht.")

    # 1. Erhöhung für gen (zentraler Generator)
    for idx, gen in net.gen.iterrows():
        net.gen.at[idx, 'p_mw'] *= 5
        net.gen.at[idx, 'q_mvar'] *= 5

    # 2. Erhöhung für sgen (verteilte Erzeugung)
    for idx, sgen in net.sgen.iterrows():
        net.sgen.at[idx, 'p_mw'] *= 5
        net.sgen.at[idx, 'q_mvar'] *= 5

    # 3. Erhöhung für storage (Speicher)
    for idx, storage in net.storage.iterrows():
        net.storage.at[idx, 'p_mw'] *= 5
        net.storage.at[idx, 'q_mvar'] *= 5

elif basic["Grid"] == "mv_pv_wind":
    net = pn.create_cigre_network_mv('pv_wind')
elif basic["Grid"] == "mv_no_renew":
    net = pn.create_cigre_network_mv(False)
elif basic["Grid"] == "cigre_lv":
    net = pn.create_cigre_network_lv()
elif basic["Grid"] == "kerber":
    net = pn.create_kerber_dorfnetz()
elif basic["Grid"] == "dickert":
    net = pn.create_dickert_lv_network()
else:
    raise ValueError(f"Unbekannter Grid-Typ: {basic['Grid']}")

if basic["Overview_Grid"]:
    # Count elements and scaled elements
    element_counts = count_elements(net)
    # Print both counts in one row
    print(net)
    print("Element Type | Original Count | Scaled Count (* 0.15)")
    print("-" * 45)
    for element_type in element_counts["original_counts"]:
        original_count = element_counts["original_counts"][element_type]
        scaled_count = element_counts["scaled_counts"][element_type]
        print(f"{element_type.capitalize():<12} | {original_count:<14} | {scaled_count:<20}")

if basic["Adjustements"]:
    net, required_p_mw, required_q_mvar = determine_minimum_ext_grid_power(net)
    net = set_missing_limits(net, required_p_mw, required_q_mvar)


selected_indicators = {
    "all": False,
    "self_sufficiency": True,
    "show_self_sufficiency_at_bus": False,
    "system_self_sufficiency": False,
    "generation_shannon_evenness": False,
    "generation_variety": False,
    "line_shannon_evenness": False,
    "line_variety": False,
    "load_shannon_evenness": False,
    "load_variety": False,
    "disparity_generators": True,
    "disparity_load": True,
    "disparity_trafo": True,
    "disparity_lines": True,
    "n_3_redundancy": True,
    "n_3_redundancy_print": True,
    "GraphenTheorie": False,
    "show_spider_plot": False,
    "print_results": True,
    "output_excel": False
}

if selected_indicators["all"]:
    # Setze alle anderen Indikatoren auf True
    for key in selected_indicators:
        if key != "all":  # 'all' selbst bleibt unverändert
            selected_indicators[key] = True

if selected_indicators["self_sufficiency"]:
    # Calculate generation factors
    generation_factors = calculate_generation_factors(net, "Fraunhofer ISE (2024)")
    #print(f"{generation_factors}")
    indi_selfsuff = float(selfsuff(net,generation_factors, selected_indicators["show_self_sufficiency_at_bus"]))
    dfinalresults = add_indicator(dfinalresults, 'self sufficiency at bus level', indi_selfsuff)

if selected_indicators["system_self_sufficiency"]:
    indi_selfsuff_neu = selfsufficiency_neu(net)
    dfinalresults = add_indicator(dfinalresults, 'System Self Sufficiency', indi_selfsuff_neu)

if selected_indicators["generation_shannon_evenness"] or selected_indicators["line_shannon_evenness"] or selected_indicators["load_shannon_evenness"]:
    # Define the maximum known types for each component
    max_known_types = {
        'generation': 8,
        # Adjust this based on your actual known types (sgen: solar, wind, biomass, gen: gas, coal, nuclear, storage: battery, hydro
        'line': 2,  # "ol" (overhead line) and "cs" (cable system)
        'load': 10
        # Example: 4 known types of loads (residential, commercial, industrial, agricultaral, transport, municipal, dynamic, static, critical, non-critical
    }

if selected_indicators["generation_shannon_evenness"]:
    # Combine sgen, gen, and storage into one DataFrame
    generation_data = pd.concat([net.sgen, net.gen, net.storage], ignore_index=True)
    evenness, variety, variety_scaled, max_variety = calculate_shannon_evenness_and_variety(generation_data, max_known_types['generation'])
    dfinalresults = add_indicator(dfinalresults, "Generation Shannon Evenness", evenness)
    if selected_indicators["generation_variety"]:
        dfinalresults = add_indicator(dfinalresults, "Generation Variety", variety_scaled)

if selected_indicators["line_shannon_evenness"]:
    evenness, variety, variety_scaled, max_variety = calculate_shannon_evenness_and_variety(net.line, max_known_types['line'])
    dfinalresults = add_indicator(dfinalresults, "Line Shannon Evenness", evenness)
    if selected_indicators["line_variety"]:
        dfinalresults = add_indicator(dfinalresults, "Line Variety", variety_scaled)

if selected_indicators["load_shannon_evenness"]:
    evenness, variety, variety_scaled, max_variety = calculate_shannon_evenness_and_variety(net.load, max_known_types['load'])
    dfinalresults = add_indicator(dfinalresults, "Load Shannon Evenness", evenness)
    if selected_indicators["load_variety"]:
        dfinalresults = add_indicator(dfinalresults, "Load Variety", variety_scaled)

if selected_indicators["GraphenTheorie"]:
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

    # checks if G is complete connected, otherwise the largest subgraph is analyzed going forward
    if not nx.is_connected(G):
        # Get largest connected component
        largest_component = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_component).copy()

    GraphenTheorieIndicator(G, dfinalresults)

if selected_indicators["disparity_generators"]:
    if not selected_indicators["self_sufficiency"]:
        # Calculate generation factors
        generation_factors = calculate_generation_factors(net, "Fraunhofer ISE (2024)")

    # Calculate disparity space
    disparity_df_gen, max_integral_gen = calculate_disparity_space(net, generation_factors)

    # Compute the integral (sum) over the entire DataFrame
    integral_value_gen = disparity_df_gen.values.sum()
    #print(f"Disparity Integral Generators: {integral_value_gen}")
    #print(f"Disparity Integral max Loads: {max_integral_gen}")

    ddisparity = add_disparity(ddisparity,'Generators', integral_value_gen, max_integral_gen, integral_value_gen / max_integral_gen)
    dfinalresults = add_indicator(dfinalresults, 'Disparity Generators',
                                  ddisparity.loc[ddisparity['Indicator'] == 'Generators', 'Verhaeltnis'].values[0])

if selected_indicators["disparity_load"]:
    disparity_df_load, max_integral_load = calculate_load_disparity(net)
    integral_value_load = disparity_df_load.values.sum()
    ddisparity = add_disparity(ddisparity, 'Load', integral_value_load, max_integral_load,integral_value_load / max_integral_load)
    dfinalresults = add_indicator(dfinalresults, 'Disparity Load',ddisparity.loc[ddisparity['Indicator'] == 'Load', 'Verhaeltnis'].values[0])

if selected_indicators["disparity_trafo"]:
    disparity_df_trafo, max_int_trafo = calculate_transformer_disparity(net)
    integral_value_trafo = disparity_df_trafo.values.sum()
    if integral_value_trafo == 0 or ddisparity[ddisparity['Name'] == 'Trafo'].empty:
        print("Disperity Berechnung für Trafos war fehlerhaft und wird mit 0 ersetzt")
        ddisparity = add_disparity(ddisparity, 'Trafo', 0, max_int_trafo,0)
    else:
        ddisparity = add_disparity(ddisparity, 'Trafo', integral_value_trafo, max_int_trafo, integral_value_trafo / max_int_trafo)

    dfinalresults = add_indicator(dfinalresults, 'Disparity Trafo',ddisparity.loc[ddisparity['Indicator'] == 'Trafo', 'Verhaeltnis'].values[0])

if selected_indicators["disparity_lines"]:
    disparity_df_lines, max_int_disp_lines = calculate_line_disparity(net)
    integral_value_line = disparity_df_lines.values.sum()
    ddisparity = add_disparity(ddisparity, 'Lines', integral_value_line, max_int_disp_lines,integral_value_line / max_int_disp_lines)
    dfinalresults = add_indicator(dfinalresults, 'Disparity Lines',ddisparity.loc[ddisparity['Indicator'] == 'Lines', 'Verhaeltnis'].values[0])

if selected_indicators["n_3_redundancy"]:
    if not basic["Overview_Grid"]:
        # Count elements and scaled elements
        element_counts = count_elements(net)
    n3_redundancy_results = n_3_redundancy_check(net, element_counts)
    Success = sum(counts['Success'] for counts in n3_redundancy_results.values())
    Failed = sum(counts['Failed'] for counts in n3_redundancy_results.values())
    total_checks = Success + Failed
    rate = Success / total_checks if total_checks != 0 else 0
    for element_type, counts in n3_redundancy_results.items():
        Success += counts['Success']
        Failed += counts['Failed']
        total_checks = Success + Failed
        rate = Success / total_checks if total_checks != 0 else 0
        if selected_indicators["n_3_redundancy_print"]:
            print(f"{element_type.capitalize()} - Success count: {counts['Success']}, Failed count: {counts['Failed']}")
    dfinalresults = add_indicator(dfinalresults, 'Overall 70% Redundancy', rate)

if selected_indicators["show_spider_plot"]:
    plot_spider_chart(dfinalresults)
if selected_indicators["print_results"]:
    print(dfinalresults)
if selected_indicators["output_excel"]:
    dfinalresults.to_excel("dfinalresults.xlsx", sheet_name="Results", index=False)
