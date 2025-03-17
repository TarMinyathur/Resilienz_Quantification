import pandapower.networks as pn
import networkx as nx
import numpy as np
import pandas as pd
import time
from count_elements import count_elements
from diversity import calculate_shannon_evenness_and_variety
from disparity import calculate_disparity_space, calculate_line_disparity, calculate_transformer_disparity, calculate_load_disparity
from GenerationFactors import calculate_generation_factors
from Redundancy_new import Redundancy
from Redundancy import n_3_redundancy_check
from visualize import plot_spider_chart
from initialize import add_indicator
from initialize import add_disparity
from indi_gt import GraphenTheorieIndicator
from adjustments import set_missing_limits
from adjustments import determine_minimum_ext_grid_power
from self_sufficiency import selfsuff
from self_sufficiency import selfsufficiency_neu
from flexibility import calculate_flexibility
from buffer import calculate_buffer
from fxor import flexibility_fxor
from stressors import stress_scenarios
from Evaluate_scenario import run_scenario

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
        net.gen.at[idx, 'sn_mva'] = np.sqrt(net.gen.at[idx, 'p_mw']**2 + net.gen.at[idx, 'q_mvar']**2)

    # 2. Increase for sgen (verteilte Erzeugung)
    for idx, sgen in net.sgen.iterrows():
        net.sgen.at[idx, 'p_mw'] *= factor
        net.sgen.at[idx, 'q_mvar'] *= factor
        net.sgen.at[idx, 'sn_mva'] = np.sqrt(net.sgen.at[idx, 'p_mw']**2 + net.sgen.at[idx, 'q_mvar']**2)

    # 3. Increase for storage (Speicher)
    for idx, storage in net.storage.iterrows():
        net.storage.at[idx, 'p_mw'] *= factor
        net.storage.at[idx, 'q_mvar'] *= factor
        net.storage.at[idx, 'sn_mva'] = np.sqrt(net.storage.at[idx, 'p_mw']**2 + net.storage.at[idx, 'q_mvar']**2)

    return net

# Configuration
basic = {
    "Grid": "create_cigre_network_mv_all",  # Change this to select the grid
    "Adjustments": True,
    "Overview_Grid": True
}

selected_indicators = {
    "all": True,
    "self_sufficiency": True,
    "show_self_sufficiency_at_bus": False,
    "system_self_sufficiency": False,
    "generation_shannon_evenness": False,
    "generation_variety": False,
    "line_shannon_evenness": False,
    "line_variety": False,
    "load_shannon_evenness": False,
    "load_variety": False,
    "disparity_generators": False,
    "disparity_load": False,
    "disparity_trafo": False,
    "disparity_lines": False,
    "n_3_redundancy": False,
    "n_3_redundancy_print": False,
    "Redundancy": False,
    "GraphenTheorie": True,
    "Flexibility": True,
    "Flexibility_fxor": True,
    "Buffer": False,
    "show_spider_plot": False,
    "print_results": True,
    "output_excel": False
}

selected_scenario = {
    "stress_scenario": True,
    "all": False,
    "Flood": True,
    "Earthquake": False,
    "Dunkelflaute": True,
    "Storm": False,
    "Fire": False,
    "Line Overload": False,
    "IT-Attack": False,
    "Geopolitical": False,
    "High EE generation": False,
    "High Load": False,
    "Sabotage": True,
    "print_results": True,
    "output_excel": True
}

# Main Function
def main():

    dfinalresults = pd.DataFrame(columns=['Indicator', 'Value'])
    ddisparity = pd.DataFrame(columns=['Name', 'Value', 'max Value', 'Verhaeltnis'])

    dfinalresults = add_indicator(dfinalresults, basic["Grid"] , 0)

    # Select and create the grid dynamically
    if basic["Grid"] in grids:
        net = grids[basic["Grid"]]()
    else:
        raise ValueError(f"Unknown Grid Type: {basic['Grid']}")

    if basic["Overview_Grid"]:
        # Count elements and scaled elements
        element_counts = count_elements(net)
        # Print both counts in one row
        # print(net)
        print("Voltage Limits:")

        print("External Grid Settings:")
        print(net.ext_grid)

        # print("Generators:")
        # print(net.gen)
        # print(net.sgen)
        # print(net.storage)

        print("Element Type    | Original Count |")
        print("-" * 20)
        for element_type in element_counts["original_counts"]:
            original_count = element_counts["original_counts"][element_type]
            print(f"{element_type.capitalize():<12}    | {original_count:<12} ")

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
        indi_selfsuff = float(selfsuff(net,generation_factors, selected_indicators["show_self_sufficiency_at_bus"]))
        dfinalresults = add_indicator(dfinalresults, 'self sufficiency at bus level', indi_selfsuff)

    if selected_indicators["system_self_sufficiency"]:
        netsa = net.deepcopy()
        indi_selfsuff_neu = selfsufficiency_neu(netsa)
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

        # 1) Busse als Knoten hinzufügen
        for bus_id in net.bus.index:
            G.add_node(bus_id)

        # 2) Leitungen als Kanten hinzufügen (unter Berücksichtigung geschlossener Schalter)
        for idx, line in net.line.iterrows():
            from_bus = line.from_bus
            to_bus = line.to_bus

            # Prüfen, ob ein Schalter (switch.et == 'l') zwischen den Bussen liegt
            switch_exists = False
            switch_closed = True  # wird nur dann False, wenn wir tatsächlich einen offenen Switch finden

            for _, sw in net.switch.iterrows():
                if sw.et == 'l':
                    # Bus- und Element-Kombination checken
                    if (sw.bus == from_bus and sw.element == to_bus) or (sw.bus == to_bus and sw.element == from_bus):
                        switch_exists = True
                        switch_closed = sw.closed
                        break

            # Nur Kante hinzufügen, wenn kein Switch existiert ODER er geschlossen ist
            if not switch_exists or (switch_exists and switch_closed):
                # Als Gewicht nehmen wir hier exemplarisch die Leitungslänge
                length = line.length_km
                G.add_edge(from_bus, to_bus, weight=length)

        # 3) Trafos als Kanten hinzufügen (ebenfalls optional mit Schalter-Check)
        for idx, trafo in net.trafo.iterrows():
            hv_bus = trafo.hv_bus
            lv_bus = trafo.lv_bus

            # Prüfen, ob ein Schalter (switch.et == 't') zum Trafo existiert
            switch_exists = False
            switch_closed = True

            for _, sw in net.switch.iterrows():
                if sw.et == 't':
                    # Bei Trafos ist meist bus = hv_bus oder lv_bus und element = trafo.id
                    # Hier einfacher Check: falls bus einer der beiden ist und switch.element == diesem Trafo
                    if sw.bus in [hv_bus, lv_bus] and sw.element == idx:
                        switch_exists = True
                        switch_closed = sw.closed
                        break

            # Nur Kante hinzufügen, wenn kein Trafo-Switch existiert ODER dieser geschlossen ist
            if not switch_exists or (switch_exists and switch_closed):
                # Beispiel: Als Gewicht kannst du beliebig etwas hinterlegen (z. B. trafo.sn_mva)
                G.add_edge(hv_bus, lv_bus, weight=1.0)

        # 4) Prüfen, ob der Graph zusammenhängend ist
        if not nx.is_connected(G):
            # Größte zusammenhängende Komponente extrahieren
            largest_component = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_component).copy()

        dfinalresults = GraphenTheorieIndicator(G, dfinalresults)

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

    if selected_indicators["n_3_redundancy"]:
        if not basic["Overview_Grid"]:
            # Count elements and scaled elements
            element_counts = count_elements(net)

        # Liste der zu prüfenden Elementtypen
        element_types = ["line", "sgen", "gen", "trafo", "bus", "storage", "switch", "load"]

        n3_redundancy_results = {}
        Success = 0
        Failed = 0
        timeout = 180

        # Über alle relevanten Elementtypen iterieren
        for element_type in element_types:
            start_time = time.time()
            results = n_3_redundancy_check(net, start_time, element_type, timeout)
            n3_redundancy_results[element_type] = results[element_type]

            # Summiere die Ergebnisse
            Success += results[element_type]['Success']
            Failed += results[element_type]['Failed']
            print(time.time() - start_time)

        # Gesamtrate berechnen
        total_checks = Success + Failed
        rate = Success / total_checks if total_checks != 0 else 0

        # Ergebnis in DataFrame speichern
        dfinalresults = add_indicator(dfinalresults, 'Overall n-3 Redundancy', rate)


    if selected_indicators["Redundancy"]:
        Lastfluss, n2_Redundanz, kombi, component_indicators, red_results = Redundancy(net)
        dfinalresults = add_indicator(dfinalresults, "Loadflow Redundancy", Lastfluss)
        dfinalresults = add_indicator(dfinalresults, "N-2 Redundancy", n2_Redundanz)
        dfinalresults = add_indicator(dfinalresults, "Combined Redundancy", kombi)

        #dfinalresults = add_indicator(dfinalresults, "Load Shannon Evenness", evenness)
        # Ausgabe der Indikatoren pro Komponente:
        print("Komponentenindikatoren (1 = optimal, 0 = schlecht):")
        for comp, inds in component_indicators.items():
            print(f"{comp.capitalize()}:")
            print(f"  Lastfluss: {inds['lf']:.3f}")
            print(f"  Redundanz: {inds['red']:.3f}")
            print(f"  Kombiniert: {inds['combined']:.3f}")

        # Ergebnisse ausgeben
        print("\nErgebnisse der N-2-Redundanzprüfung:")
        for element, stats in red_results.items():
            print(f"{element.capitalize()}: Erfolg: {stats['Success']}, Fehlgeschlagen: {stats['Failed']}")

        print("\nGesamtindikatoren:")
        print(f"  Lastfluss Gesamt: {Lastfluss:.3f}")
        print(f"  N-2 Redundanz Gesamt: {n2_Redundanz:.3f}")
        print(f"  Kombinierter Gesamtindikator: {kombi:.3f}")

    if selected_indicators["Flexibility"]:
        dflexiresults = calculate_flexibility(net)
        dfinalresults = add_indicator(dfinalresults, 'Flex Netzreserve', dflexiresults.loc[dflexiresults['Indicator'] == 'Flex Netzreserve', 'Value'].values[0])
        dfinalresults = add_indicator(dfinalresults, 'Flex Reserve krit Leitungen', dflexiresults.loc[dflexiresults['Indicator'] == 'Flex Reserve krit Leitungen', 'Value'].values[0])
        dfinalresults = add_indicator(dfinalresults, 'Flexibilität Gesamt', dflexiresults.loc[dflexiresults['Indicator'] == 'Flexibilität Gesamt', 'Value'].values[0])

    if selected_indicators["Buffer"]:
        Speicher = calculate_buffer(net)
        dfinalresults = add_indicator(dfinalresults, 'Buffer Capacity', Speicher)

    if selected_indicators["Flexibility_fxor"]:
        Flex_fxor = flexibility_fxor(net, False)
        dfinalresults = add_indicator(dfinalresults, 'Feasible operating region', Flex_fxor)

    if selected_indicators["n_3_redundancy_print"]:
        print("Results of N-3 Redundancy")
        for element_type, counts in n3_redundancy_results.items():
            print(f"{element_type.capitalize()} - Success count: {counts['Success']}, Failed count: {counts['Failed']}")    
    if selected_indicators["show_spider_plot"]:
        plot_spider_chart(dfinalresults)
    if selected_indicators["print_results"]:
        print("Results for Indicators:")
        print(dfinalresults)
    if selected_indicators["output_excel"]:
        dfinalresults.T.to_excel("dfinalresults.xlsx", sheet_name="Results Indicator", index=False)

    if selected_scenario["stress_scenario"]:

        if selected_scenario["all"]:
            # Setze alle anderen Indikatoren auf True
            for key in selected_scenario:
                if key != "all":  # 'all' selbst bleibt unverändert
                    selected_scenario[key] = True

        dfresultsscenario = pd.DataFrame()
        dfresultsscenario = add_indicator(dfresultsscenario, basic["Grid"], 0)

        for scenario in ["Flood", "Earthquake", "Dunkelflaute", "Storm", "Fire", "Line Overload",
                             "IT-Attack", "Geopolitical", "High EE generation", "High Load", "Sabotage"]:
            if selected_scenario[scenario]:
                stressor = scenario.lower()
                for n in range(10):
                    modified_nets = stress_scenarios(net, [stressor])
                    res_scenario = run_scenario(modified_nets, scenario)
                    dfresultsscenario = add_indicator(dfresultsscenario, scenario, res_scenario)

        if not dfresultsscenario.empty:
            # Compute the average of all values excluding the first row
            if len(dfresultsscenario) > 1:  # Ensure there are enough rows to calculate an average
                scenario_average_value = dfresultsscenario["Value"].iloc[1:].mean()  # Exclude the first row

            # Add the average as a new row
            dfresultsscenario = add_indicator(dfresultsscenario, "Overall Scenario Resilience Score", scenario_average_value)
            if selected_scenario["print_results"]:
                print(dfresultsscenario)

            if selected_scenario["output_excel"]:
                dfresultsscenario.T.to_excel("dfresultsscenario.xlsx", sheet_name="Results Scenario", index=False)

if __name__ == '__main__':
    main()
