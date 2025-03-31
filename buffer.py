
def calculate_buffer(net_buff):

    total_load = net_buff.load["p_mw"].sum()
    print(f"Total net_buff load: {total_load} MW")

    buffer_sum = []

    buffer_sum = check_battery_capacity(net_buff, buffer_sum)
    buffer_sum = get_sgen(net_buff, buffer_sum)
    buffer_sum = get_gen(net_buff, buffer_sum)
    buffer_sum = get_flexible_loads(net_buff, buffer_sum)

    total_buffer_capacity = sum(entry["capacity"] for entry in buffer_sum) if buffer_sum else 0
    print(f"Total buffer capacity: {total_buffer_capacity} MW")

    buffer_ratio = (total_buffer_capacity / total_load) if total_load > 0 else 0  # to discuss value for buffer
    print(f"Buffer: {buffer_ratio:.4f}")

    return buffer_ratio


def check_battery_capacity(net_buff, buffer_sum):
    if "storage" in net_buff and not net_buff.storage.empty:
        battery_capacity = net_buff.storage["p_mw"].sum()
        buffer_sum.append({"type": "storage", "capacity": battery_capacity})
    else:
        print("No batteries in net_buff")
    return buffer_sum


def get_sgen(net_buff, buffer_sum):
    sgen_types = ["Residential fuel cell", "CHP diesel", "Fuel cell"]  # tdb if further sgen types to include

    for sgen_type in sgen_types:
        filtered_sgen = net_buff.sgen[net_buff.sgen["name"].str.contains(sgen_type, case=False, na=False)]
        sgen_capacity = filtered_sgen["p_mw"].sum()

        if sgen_capacity > 0:
            buffer_sum.append({"type": sgen_type, "capacity": sgen_capacity})

    return buffer_sum


def get_gen(net_buff, buffer_sum):
    gen_types = ["sync", "async"]

    for gen_type in gen_types:
        filtered_gen = net_buff.gen[net_buff.gen["name"].str.contains(gen_type, case=False, na=False)]
        gen_capacity = filtered_gen["p_mw"].sum()

        if gen_capacity > 0:
            buffer_sum.append({"type": gen_type, "capacity": gen_capacity})

    return buffer_sum


def get_flexible_loads(net_buff, buffer_sum):
    if "load" in net_buff and not net_buff.load.empty:
        if "controllable" not in net_buff.load.columns:
            print("Warning: 'controllable' column not found in net.load. Skipping flexible loads.")
            return buffer_sum
        flexible_loads = net_buff.load[net_buff.load["controllable"] == True]

        private_loads = flexible_loads[flexible_loads["type"] == "private"]["p_mw"].sum() * 1.0  # 100 % flexible
        business_loads = flexible_loads[flexible_loads["type"] == "business"]["p_mw"].sum() * 0.15  # 15 % flexible

        if private_loads > 0:
            buffer_sum.append({"type": "private load flexibility", "capacity": private_loads})

        if business_loads > 0:
            buffer_sum.append({"type": "business load flexibility", "capacity": business_loads})

    return buffer_sum