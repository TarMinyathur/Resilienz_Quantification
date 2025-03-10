# redundancy_process.py

import pandapower as pp
import itertools
import networkx as nx
import pandapower.networks as pn
from concurrent.futures import ProcessPoolExecutor
import time
import random


def n_3_redundancy_check(net_temp, start_time, element_type, timeout):
    results = {element_type: {"Success": 0, "Failed": 0}}

    triples = (list(itertools.combinations(net_temp[element_type].index,3)) if not net_temp[element_type].empty else [])
    random.shuffle(triples)

    print(f"Original network ID before copies: {id(net_temp)}")

    should_stop = False

    # for triple in triples:
    #     if (time.time() - start_time) > timeout:
    #         print("Timeout reached. Ending process.")
    #         break
    #
    #     net_copy = net_temp.deepcopy()  # Ensure a fresh copy
    #     print(f"Processing bus triple: {triple} in network copy ID {id(net_temp)}")
    #     print(f"Processing {element_type} triple: {triple} in network copy ID {id(net_copy)}")
    #
    #     etype, status = process_triple(element_type, triple, net_copy)
    #     results[etype][status] += 1
    #
    # return results

    # Process the selected element type in parallel
    with ProcessPoolExecutor() as executor:
        futures = []

        for triple in triples:
            if should_stop:
                break

            net_temp_copy = net_temp.deepcopy()
            futures.append(executor.submit(process_triple, element_type, triple, net_temp_copy))

            # Check timeout after each task submission
            if (time.time() - start_time) > timeout:
                print("Timeout reached. Ending process.")
                should_stop = True
                break

        for future in futures:
            element_type, status = future.result()
            results[element_type][status] += 1

    return results


def process_triple(element_type, triple, net_temp):
    # net_id = id(net_temp)  # Get unique identifier for this network instance
    # print(f"Processing {element_type} triple: {triple} in net ID {net_id}")
    #
    # # Ensure that we are working on an independent network
    # initial_in_service = net_temp[element_type]['in_service'].sum()
    # print(f"Initial {element_type} in-service count: {initial_in_service}")

    for element_id in triple:
        net_temp[element_type].at[element_id, 'in_service'] = False

    after_out_of_service = net_temp[element_type]['in_service'].sum()
    print(f"After deactivating {triple}, in-service count: {after_out_of_service}")

    # Run connectivity check
    if not is_graph_connected(net_temp, {element_type: triple}):
        #print(f"Graph not connected for {element_type} triple: {triple}")
        return element_type, "Failed"

    try:
        pp.runopp(
            net_temp,
            init="pf",
            calculate_voltage_angles=True,
            enforce_q_lims=True,
            distributed_slack=True
        )
        return element_type, "Success"
    except (pp.optimal_powerflow.OPFNotConverged, pp.powerflow.LoadflowNotConverged):
        #print(f"OPF did not converge with init='pf' for {element_type} triple: {triple}, retrying with init='flat'")
        try:
            pp.runopp(
                net_temp,
                init="flat",
                calculate_voltage_angles=True,
                enforce_q_lims=True,
                distributed_slack=True
            )
            return element_type, "Success"
        except (pp.optimal_powerflow.OPFNotConverged, pp.powerflow.LoadflowNotConverged):
            #print(f"OPF did not converge with init='flat' for {element_type} triple: {triple}")
            return element_type, "Failed"
        except Exception as e:
            #print(f"Unexpected error for {element_type} triple: {triple} using init='flat': {e}")
            return element_type, "Failed"
    except Exception as e:
        #print(f"Unexpected error for {element_type} triple: {triple} using init='pf': {e}")
        return element_type, "Failed"


def is_graph_connected(net_temp, out_of_service_elements):
    G = nx.Graph()
    # For bus removals, mimic Version 1 behavior: add all buses even if "removed"
    if "bus" not in out_of_service_elements:
        G.add_nodes_from(net_temp.bus.index)
    else:
        for bus in net_temp.bus.itertuples():
            if bus.Index not in out_of_service_elements.get('bus', []):
                G.add_node(bus.Index)

    for line in net_temp.line.itertuples():
        if line.Index in out_of_service_elements.get('line', []):
            continue
        from_bus = line.from_bus
        to_bus = line.to_bus
        switch_exists = False
        switch_closed = False
        for _, switch in net_temp.switch.iterrows():
            if ((switch.bus == from_bus and switch.element == to_bus) or
                (switch.bus == to_bus and switch.element == from_bus)) and switch.et == 'l':
                switch_exists = True
                switch_closed = switch.closed
                break
        if not switch_exists or (switch_exists and switch_closed):
            G.add_edge(from_bus, to_bus)

    for trafo in net_temp.trafo.itertuples():
        if trafo.Index in out_of_service_elements.get('trafo', []):
            continue
        G.add_edge(trafo.hv_bus, trafo.lv_bus)

    return nx.is_connected(G)


# if __name__ == '__main__':
#     # Example usage:
#     start = time.time()
#     # Create your network here, e.g.:
#     net = pn.simple_four_bus_system()  # or any pandapower network
#     element_counts = {
#         "scaled_counts": {
#             "bus": len(net.bus),
#             "line": len(net.line),
#             "sgen": len(net.sgen),
#             "trafo": len(net.trafo),
#             "storage": len(net.storage)
#         }
#     }
#
#     # For example, run for bus elements:
#     result = n_3_redundancy_check(net, element_counts, start, "bus", timeout=900)
#     print(result)