# redundancy.py

import pandapower as pp
import itertools
import networkx as nx
import pandapower.networks as pn
from concurrent.futures import ThreadPoolExecutor
import time

# Idee: Redundanz über senken von max external messen?
# generell: max ext grid = Summe Erzeugung?
def n_3_redundancy_check(net_temp, element_counts):
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
    element_triples = {
        "line": list(itertools.combinations(net_temp.line.index, min(3, element_counts["scaled_counts"]["line"]))) if not net_temp.line.empty else [],
        #"switch": list(itertools.combinations(net_temp.switch.index, min(3, element_counts["scaled_counts"]["switch"]))) if not net_temp.switch.empty else [],
        #"load": list(itertools.combinations(net_temp.load.index, min(3, element_counts["scaled_counts"]["load"]))) if not net_temp.load.empty else [],
        "sgen": list(itertools.combinations(net_temp.sgen.index, min(3, element_counts["scaled_counts"]["sgen"]))) if not net_temp.sgen.empty else [],
        "trafo": list(itertools.combinations(net_temp.trafo.index, min(3, element_counts["scaled_counts"]["trafo"]))) if not net_temp.trafo.empty else [],
        "bus": list(itertools.combinations(net_temp.bus.index, min(3, element_counts["scaled_counts"]["bus"]))) if not net_temp.bus.empty else [],
        "storage": list(itertools.combinations(net_temp.storage.index, min(3, element_counts["scaled_counts"]["storage"]))) if not net_temp.storage.empty else []

    }

    # Process each element type in parallel
    with ThreadPoolExecutor() as executor:
        futures = []
        for element_type, triples in element_triples.items():
            for triple in triples:
                # Create a shallow copy of the network to simulate the failures
                net_temp = net_temp.deepcopy()

                # pass copied net_temp
                futures.append(executor.submit(process_triple, element_type, triple, net_temp))

        for future in futures:
            element_type, status = future.result()
            results[element_type][status] += 1

    return results

# Function to process each triple and update results
def process_triple(element_type, triple, net_temp):

    # Set the elements out of service
    for element_id in triple:
        net_temp[element_type].at[element_id, 'in_service'] = False

    # Check if the grid is still connected
    out_of_service_elements = {element_type: triple}
    if not is_graph_connected(net_temp, out_of_service_elements):
        return element_type, "Failed"

    # Run the load flow calculation
    try:
        # First attempt with init="pf"
        pp.runopp(
            net_temp,
            init="pf",
            calculate_voltage_angles=True,  # Compute voltage angles
            enforce_q_lims=True,  # Enforce reactive power (Q) limits
            distributed_slack=True  # Distribute slack among multiple sources
        )
        return element_type, "Success"
        # total_generation_q = net_temp.res_gen.q_mvar.sum()
        # total_load_q = net_temp.res_load.q_mvar.sum()
        # print(f"Total Q Generation: {total_generation_q} MVar")
        # print(f"Total Q Load: {total_load_q} MVar")
        # print(f"Total Q Externes Netz: {net_temp.res_ext_grid.q_mvar} MVar")
        # Debugging for inductive behavior. The external grid pushes more reactive power in the system, than the loads need in total, which makes sense, as the external grid stabilizes the voltages, where the generators were not able to.
    except (pp.optimal_powerflow.OPFNotConverged, pp.powerflow.LoadflowNotConverged):
        print(f"OPF did not converge with init='pf' for {element_type}, retrying with init='flat'")
        try:
            # Retry with init="flat"
            pp.runopp(
                net_temp,
                init="flat",
                calculate_voltage_angles=True,  # Compute voltage angles
                enforce_q_lims=True,  # Enforce reactive power (Q) limits
                distributed_slack=True  # Distribute slack among multiple sources
            )
            return element_type, "Success"
        except (pp.optimal_powerflow.OPFNotConverged, pp.powerflow.LoadflowNotConverged):
            print(f"OPF did not converge with init='flat' for {element_type}")
            return element_type, "Failed"
        except Exception as e:
            print(f"Unexpected error for {element_type} with triple {triple} using init='flat': {e}")
            return element_type, "Failed"
    except Exception as e:
        print(f"Unexpected error for {element_type} with triple {triple} using init='pf': {e}")
        return element_type, "Failed"

def is_graph_connected(net_temp, out_of_service_elements):
    # Create NetworkX graph manually
    G = nx.Graph()

    # Add buses
    for bus in net_temp.bus.itertuples():
        if bus.Index not in out_of_service_elements.get('line', []):
            G.add_nodes_from(net_temp.bus.index)

    # Add lines
    for line in net_temp.line.itertuples():
        if line.Index not in out_of_service_elements.get('line', []):
            #G.add_edge(line.from_bus, line.to_bus)
            for idx, line in net_temp.line.iterrows():
                from_bus = line.from_bus
                to_bus = line.to_bus

                # Check if there is a switch between from_bus and to_bus
                switch_exists = False
                for _, switch in net_temp.switch.iterrows():
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
                    G.add_edge(from_bus, to_bus)

    # Add transformers
    for trafo in net_temp.trafo.itertuples():
        if trafo.Index not in out_of_service_elements.get('trafo', []):
            G.add_edge(trafo.hv_bus, trafo.lv_bus)

    # Check if the graph is still connected
    return nx.is_connected(G)