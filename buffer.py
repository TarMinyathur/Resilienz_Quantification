import pandapower.networks as pn


def calculate_buffer(net):
    
    total_load = net.load["p_mw"].sum()
    print(f"Total net load: {total_load} MW")

    buffer_sum = []  

    buffer_sum = check_battery_capacity(net, buffer_sum)
    buffer_sum = get_sgen(net, buffer_sum)

    total_buffer_capacity = sum(entry["capacity"] for entry in buffer_sum) if buffer_sum else 0
    print(f"Total buffer capacity: {total_buffer_capacity} MW")

    buffer_ratio = (total_buffer_capacity / total_load) if total_load > 0 else 0  # to discuss value for buffer
    print(f"Buffer: {buffer_ratio:.4f}")

    return buffer_ratio   


def check_battery_capacity(net, buffer_sum):
    if "storage" in net and not net.storage.empty:
        battery_capacity = net.storage["p_mw"].sum()
        buffer_sum.append({"type": "storage", "capacity": battery_capacity}) 
    else:
        print("No batteries in net")
    return buffer_sum 


def get_sgen(net, buffer_sum):
    sgen_types = ["Residential fuel cell", "CHP diesel", "Fuel cell"] # tdb if further sgen types to include

    for sgen_type in sgen_types:
        filtered_sgen = net.sgen[net.sgen["name"].str.contains(sgen_type, case=False, na=False)]
        sgen_capacity = filtered_sgen["p_mw"].sum()

        if sgen_capacity > 0:
            buffer_sum.append({"type": sgen_type, "capacity": sgen_capacity})

    return buffer_sum 


if __name__ == "__main__":
    net = pn.create_cigre_network_mv(with_der="all")
    calculate_buffer(net)
