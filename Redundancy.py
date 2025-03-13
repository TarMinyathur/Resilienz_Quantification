# redundancy.py

import pandapower as pp
import itertools
import networkx as nx
import pandapower.networks as pn
from concurrent.futures import ProcessPoolExecutor
import time
import random

"Aufgrund der CPU-lastigen Natur der OPF-Berechnungen würde man ohne GIL-Freigabe der Libraries eindeutig zum ProcessPoolExecutor greifen. Falls deine numerischen Routinen aber tatsächlich den GIL freigeben (was sehr oft der Fall ist), können Threads einen guten Kompromiss aus geringerem Overhead und echter Parallelität darstellen."

# Idee: Redundanz über senken von max external messen?
# generell: max ext grid = Summe Erzeugung?
def n_3_redundancy_check(net_temp_red, start_time, element_type, timeout):
    if element_type not in ["line", "sgen", "gen", "trafo", "bus", "storage", "switch", "load"]:
        raise ValueError(f"Invalid element type: {element_type}")

    results = {element_type: {"Success": 0, "Failed": 0}}

    # If the table for this element type is empty or has fewer than 3 rows, no combinations can be made
    if net_temp_red[element_type].empty or len(net_temp_red[element_type].index) < 3:
        return results

    # Extract indices and shuffle them to achieve a pseudo-random iteration of combinations
    index_list = list(net_temp_red[element_type].index)
    random.shuffle(index_list)

    # Create a combinations generator instead of a full list
    element_triples_gen = itertools.combinations(index_list, 3)

    should_stop = False

    # Process the selected element type in parallel
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = []

        for triple in element_triples_gen:
            if should_stop:
                break

            net_temp_red_copy = net_temp_red.deepcopy()
            futures.append(executor.submit(process_triple, element_type, triple, net_temp_red_copy))

            # Check timeout after each task submission
            if (time.time() - start_time) > timeout:
                print("Timeout reached. Ending process.")
                should_stop = True
                break

        # Collect results
        for future in futures:
            element_type_returned, status = future.result()
            results[element_type_returned][status] += 1

    return results


# Function to process each triple and update results
def process_triple(element_type, triple, net_temp_red):
    # Set the elements out of service
    for element_id in triple:
        net_temp_red[element_type].at[element_id, 'in_service'] = False

    # Check if the grid is still connected
    out_of_service_elements = {element_type: triple}
    if not is_graph_connected(net_temp_red, out_of_service_elements):
        return element_type, "Failed"

    # Run the load flow calculation
    try:
        # First attempt with init="pf"
        pp.runopp(
            net_temp_red,
            init="pf",
            calculate_voltage_angles=True,  # Compute voltage angles
            enforce_q_lims=True,  # Enforce reactive power (Q) limits
            distributed_slack=True  # Distribute slack among multiple sources
        )
        return element_type, "Success"

    except (pp.optimal_powerflow.OPFNotConverged, pp.powerflow.LoadflowNotConverged):
        #print(f"OPF did not converge with init='pf' for {element_type}, retrying with init='flat'")
        try:
            # Retry with init="flat"
            pp.runopp(
                net_temp_red,
                init="flat",
                calculate_voltage_angles=True,  # Compute voltage angles
                enforce_q_lims=True,  # Enforce reactive power (Q) limits
                distributed_slack=True  # Distribute slack among multiple sources
            )
            return element_type, "Success"
        except (pp.optimal_powerflow.OPFNotConverged, pp.powerflow.LoadflowNotConverged):
            #print(f"OPF did not converge with init='flat' for {element_type}")
            return element_type, "Failed"
        except Exception as e:
            #print(f"Unexpected error for {element_type} with triple {triple} using init='flat': {e}")
            return element_type, "Failed"
    except Exception as e:
        #print(f"Unexpected error for {element_type} with triple {triple} using init='pf': {e}")
        return element_type, "Failed"


def is_graph_connected(net_temp_red, out_of_service_elements):
    """
    Prüft, ob das Netz nach dem Rausnehmen bestimmter Elemente (bspw. n-3-Fälle) noch zusammenhängend ist.
    out_of_service_elements ist ein Dictionary, z.B. {"line": [2, 5], "bus": [3], ...}.
    """

    # Neues Graph-Objekt
    G = nx.Graph()

    # 1) Buses hinzufügen
    #    Nur die Busse, die als in_service=True gelten UND nicht in out_of_service_elements['bus'] stehen
    for bus_id, bus in net_temp_red.bus.iterrows():
        if bus.in_service and bus_id not in out_of_service_elements.get('bus', []):
            G.add_node(bus_id)

    # 2) Lines hinzufügen
    #    Auch hier schauen wir, ob die Leitung in_service=True ist und nicht in out_of_service_elements['line'] steht.
    #    Zusätzlich prüfen wir, ob eventuell ein Switch zwischen den Bussen liegt und geschlossen/in Service ist.
    for line_id, line in net_temp_red.line.iterrows():
        if line.in_service and line_id not in out_of_service_elements.get('line', []):
            from_bus = line.from_bus
            to_bus = line.to_bus

            # Erst prüfen, ob die entsprechenden Busse noch im Graphen sind
            if from_bus not in G or to_bus not in G:
                continue

            # Schauen, ob es einen Switch gibt, der diese Verbindung öffnet
            # (switch.et == 'l' und bus-element-Kombination == (from_bus, to_bus) oder umgekehrt).
            switch_exists = False
            switch_closed = True

            for sw_id, sw in net_temp_red.switch.iterrows():
                if sw.et == 'l':
                    # Ein Switch koppelt dieselben Busse? (bus == from_bus, element == to_bus) oder umgekehrt
                    if ((sw.bus == from_bus and sw.element == to_bus) or
                        (sw.bus == to_bus and sw.element == from_bus)):

                        switch_exists = True

                        # Prüfen, ob der Switch geschlossen und in_service ist und NICHT in out_of_service_elements
                        if (not sw.closed or
                            not sw.in_service or
                            sw_id in out_of_service_elements.get('switch', [])):
                            switch_closed = False
                        break

            # Nur Kante hinzufügen, wenn entweder kein Switch existiert,
            # oder er existiert und ist tatsächlich geschlossen/in service
            if (not switch_exists) or (switch_exists and switch_closed):
                G.add_edge(from_bus, to_bus)

    # 3) Trafos hinzufügen (analog zu Lines)
    for trafo_id, trafo in net_temp_red.trafo.iterrows():
        if trafo.in_service and trafo_id not in out_of_service_elements.get('trafo', []):
            hv_bus = trafo.hv_bus
            lv_bus = trafo.lv_bus

            # Nur add_edge, wenn diese Busse noch im Graph sind
            if hv_bus in G and lv_bus in G:
                G.add_edge(hv_bus, lv_bus)

    # Abschließende Prüfung, ob der Graph zusammenhängend ist.
    # Wenn G leer ist (z.B. alle Busse entfernt), führt nx.is_connected(G) zu einem Fehler.
    # Deswegen vorher abfangen:
    if len(G) == 0:
        # Je nach Definition könnte ein leeres Netz auch als "nicht verbunden" gewertet werden
        return False

    return nx.is_connected(G)