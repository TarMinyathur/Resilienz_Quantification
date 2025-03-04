# import pandapower as pp
import pandapower.networks as pn
#import itertools
#import math
import networkx as nx
#import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import time
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

# Dictionary to including all grid names to functions, including special cases for test grids, whose opp converges
grids = {
    "GBreducednetwork": pn.GBreducednetwork,
    "case118": pn.case118,
    "case14": pn.case14,
    "case24_ieee_rts": pn.case24_ieee_rts,
    "case30": pn.case30,
    "case33bw": pn.case33bw,
    "case39": pn.case39,
    "case5": pn.case5,
    "case6ww": pn.case6ww,
    "case9": pn.case9,
    "create_cigre_network_lv": pn.create_cigre_network_lv,
    "create_cigre_network_mv": pn.create_cigre_network_mv,
    "create_cigre_network_mv_all": lambda: pn.create_cigre_network_mv(with_der="all"),
    "create_cigre_network_mv_pv_wind": lambda: pn.create_cigre_network_mv(with_der="pv_wind"),
    "ieee_european_lv_asymmetric": pn.ieee_european_lv_asymmetric,

    # Special Cases with Adjustments
    "mv_all_high10": lambda: increase_generation(pn.create_cigre_network_mv(with_der="all"), factor=10),
    "mv_all_high5": lambda: increase_generation(pn.create_cigre_network_mv(with_der="all"), factor=5)
}

# Function to increase generation and storage capacities
def increase_generation(net, factor):
    print(f"Verteilte Erzeugung und Speicher um den Faktor {factor} erhöht.")

    # 1. Increase for gen (zentraler Generator)
    for idx, gen in net.gen.iterrows():
        net.gen.at[idx, 'p_mw'] *= factor
        net.gen.at[idx, 'q_mvar'] *= factor

    # 2. Increase for sgen (verteilte Erzeugung)
    for idx, sgen in net.sgen.iterrows():
        net.sgen.at[idx, 'p_mw'] *= factor
        net.sgen.at[idx, 'q_mvar'] *= factor

    # 3. Increase for storage (Speicher)
    for idx, storage in net.storage.iterrows():
        net.storage.at[idx, 'p_mw'] *= factor
        net.storage.at[idx, 'q_mvar'] *= factor

    return net

# Configuration
basic = {
    "Grid": "create_cigre_network_mv_pv_wind",  # Change this to select the grid
    "Adjustments": True,
    "Overview_Grid": False
}

selected_indicators = {
    "all": False,
    "self_sufficiency": True,
    "show_self_sufficiency_at_bus": True,
    "system_self_sufficiency": True,
    "generation_shannon_evenness": True,
    "generation_variety": True,
    "line_shannon_evenness": True,
    "line_variety": True,
    "load_shannon_evenness": True,
    "load_variety": True,
    "disparity_generators": True,
    "disparity_load": True,
    "disparity_trafo": True,
    "disparity_lines": True,
    "n_3_redundancy": True,
    "n_3_redundancy_print": False,
    "GraphenTheorie": False,
    "show_spider_plot": False,
    "print_results": True,
    "output_excel": False
}

# Main Function
def main():

    dfinalresults = pd.DataFrame(columns=['Indicator', 'Value'])
    ddisparity = pd.DataFrame(columns=['Name', 'Value', 'max Value', 'Verhaeltnis'])

    # Select and create the grid dynamically
    if basic["Grid"] in grids:
        net = grids[basic["Grid"]]()
    else:
        raise ValueError(f"Unknown Grid Type: {basic['Grid']}")

    if basic["Overview_Grid"]:
        # Count elements and scaled elements
        element_counts = count_elements(net)
        # Print both counts in one row
        print(net)
        print("Voltage Limits:")

        print("External Grid Settings:")
        print(net.ext_grid)

        print("Generators:")
        print(net.gen)
        print(net.sgen)
        print(net.storage)

        print("Element Type | Original Count | Scaled Count (* 0.15)")
        print("-" * 45)
        for element_type in element_counts["original_counts"]:
            original_count = element_counts["original_counts"][element_type]
            scaled_count = element_counts["scaled_counts"][element_type]
            print(f"{element_type.capitalize():<12} | {original_count:<14} | {scaled_count:<20}")

    if basic["Adjustments"]:
        net, required_p_mw, required_q_mvar = determine_minimum_ext_grid_power(net)
        net = set_missing_limits(net, required_p_mw, required_q_mvar)

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
            # Example: 10 known types of loads (residential, commercial, industrial, agricultaral, transport, municipal, dynamic, static, critical, non-critical
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

    # if selected_indicators["n_3_redundancy"]:
    #     if not basic["Overview_Grid"]:
    #         element_counts = count_elements(net)
    #     start_time = time.time()
    #     n3_redundancy_results = n_3_redundancy_check(net, element_counts, start_time)
    #
    #     # Correctly compute Success and Failed only once
    #     Success = sum(counts['Success'] for counts in n3_redundancy_results.values())
    #     Failed = sum(counts['Failed'] for counts in n3_redundancy_results.values())
    #     total_checks = Success + Failed
    #     rate = Success / total_checks if total_checks != 0 else 0
    #
    #     # Print without altering the Success/Failed values
    #     for element_type, counts in n3_redundancy_results.items():
    #         if selected_indicators["n_3_redundancy_print"]:
    #             print(
    #                 f"{element_type.capitalize()} - Success count: {counts['Success']}, Failed count: {counts['Failed']}")
    #
    #     dfinalresults = add_indicator(dfinalresults, 'Overall 70% Redundancy', rate)
    #
    # if selected_indicators["n_3_redundancy"]:
    #     if not basic["Overview_Grid"]:
    #         # Count elements and scaled elements
    #         element_counts = count_elements(net)
    #     start_time = time.time()
    #     n3_redundancy_results = n_3_redundancy_check(net, element_counts, start_time)
    #     Success = sum(counts['Success'] for counts in n3_redundancy_results.values())
    #     Failed = sum(counts['Failed'] for counts in n3_redundancy_results.values())
    #     total_checks = Success + Failed
    #     rate = Success / total_checks if total_checks != 0 else 0
    #     for element_type, counts in n3_redundancy_results.items():
    #         Success += counts['Success']
    #         Failed += counts['Failed']
    #         total_checks = Success + Failed
    #         rate = Success / total_checks if total_checks != 0 else 0
    #         if selected_indicators["n_3_redundancy_print"]:
    #             print(f"{element_type.capitalize()} - Success count: {counts['Success']}, Failed count: {counts['Failed']}")
    #     dfinalresults = add_indicator(dfinalresults, 'Overall 70% Redundancy', rate)

    if selected_indicators["n_3_redundancy"]:
        if not basic["Overview_Grid"]:
            # Count elements and scaled elements
            element_counts = count_elements(net)

        # Liste der zu prüfenden Elementtypen
        element_types = ["sgen", "gen", "line", "trafo", "bus", "storage"]

        n3_redundancy_results = {}
        Success = 0
        Failed = 0
        timeout = 300

        # Über alle relevanten Elementtypen iterieren
        for element_type in element_types:
            start_time = time.time()
            results = n_3_redundancy_check(net, element_counts, start_time, element_type, timeout)
            n3_redundancy_results[element_type] = results[element_type]

            # Summiere die Ergebnisse
            Success += results[element_type]['Success']
            Failed += results[element_type]['Failed']

            # Optional: Ergebnis für jeden Typ ausgeben
            if selected_indicators["n_3_redundancy_print"]:
                print(
                    f"{element_type.capitalize()} - Success count: {results[element_type]['Success']}, Failed count: {results[element_type]['Failed']}")

        # Gesamtrate berechnen
        total_checks = Success + Failed
        rate = Success / total_checks if total_checks != 0 else 0

        # Ergebnis in DataFrame speichern
        dfinalresults = add_indicator(dfinalresults, 'Overall 70% Redundancy', rate)

    if selected_indicators["n_3_redundancy_print"]:
        for element_type, counts in n3_redundancy_results.items():
            print(f"{element_type.capitalize()} - Success count: {counts['Success']}, Failed count: {counts['Failed']}")
    if selected_indicators["show_spider_plot"]:
        plot_spider_chart(dfinalresults)
    if selected_indicators["print_results"]:
        print(dfinalresults)
    if selected_indicators["output_excel"]:
        dfinalresults.to_excel("dfinalresults.xlsx", sheet_name="Results", index=False)

if __name__ == '__main__':
    main()
