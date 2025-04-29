
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

    if total_load <= 0:
        print("Warning: total_load is zero or negative — buffer ratio set to 0.")
        buffer_ratio = 0
    else:
        buffer_ratio = total_buffer_capacity / total_load * 24  # to discuss value for buffer

    print(f"Buffer: {buffer_ratio:.4f}")

    return buffer_ratio, total_buffer_capacity


def check_battery_capacity(net_buff, buffer_sum):
    if "storage" in net_buff and not net_buff.storage.empty:
        filtered_storage = net_buff.storage[net_buff.storage["p_mw"] > 0]
        battery_capacity = filtered_storage["p_mw"].sum()
        buffer_sum.append({"type": "storage", "capacity": battery_capacity})
    else:
        print("No batteries in net_buff")
    return buffer_sum

def get_sgen(net_buff, buffer_sum):
    sgen_types = ["Residential fuel cell", "CHP diesel"]  # tdb if further sgen types to include

    for sgen_type in sgen_types:
        filtered_sgen = net_buff.sgen[net_buff.sgen["name"].str.contains(sgen_type, case=False, na=False)]
        filtered_sgen = filtered_sgen[filtered_sgen["p_mw"] > 0]
        sgen_capacity = filtered_sgen["p_mw"].sum() * 24

        if sgen_capacity > 0:
            buffer_sum.append({"type": sgen_type, "capacity": sgen_capacity})

    # Static types (only need one keyword and a fixed scaling)
    static_sgen_categories = {
        "chp": 0.5,
        "biomass": 0.5
    }

    # Identify buses with electrolyzers
    buses_with_electrolyzer = set()
    if "load" in net_buff and not net_buff.load.empty:
        if "type" in net_buff.load.columns and "name" in net_buff.load.columns:
            type_matches = net_buff.load["type"].astype(str).str.contains("electrolyzer", case=False, na=False)
            name_matches = net_buff.load["name"].astype(str).str.contains("electrolyzer", case=False, na=False)

            is_electrolyzer = type_matches | name_matches
            buses_with_electrolyzer = set(net_buff.load[is_electrolyzer]["bus"].unique())

    # Normalize sgen names once for easy matching
    net_buff.sgen["name_lower"] = net_buff.sgen["name"].astype(str).str.lower()

    # --- Handle static types like CHP diesel ---
    for keyword, scaling in static_sgen_categories.items():
        filtered_sgen = net_buff.sgen[net_buff.sgen["name_lower"].str.contains(keyword, case=False, na=False)]
        filtered_sgen = filtered_sgen[filtered_sgen["p_mw"] > 0]

        sgen_capacity = filtered_sgen["p_mw"].sum() * scaling
        if sgen_capacity > 0:
            buffer_sum.append({"type": keyword, "capacity": sgen_capacity})

    # --- Handle fuel cells with context ---
    fuel_cells = net_buff.sgen[net_buff.sgen["name_lower"].str.contains("fuel cell", case=False, na=False)]
    fuel_cells = fuel_cells[fuel_cells["p_mw"] > 0]

    if not fuel_cells.empty:
        # Fuel cells co-located with electrolyzers
        fuel_cell_el = fuel_cells[fuel_cells["bus"].isin(buses_with_electrolyzer)]
        fuel_cell_el_capacity = fuel_cell_el["p_mw"].sum() * 1.0  # Full contribution
        if fuel_cell_el_capacity > 0:
            buffer_sum.append({"type": "fuel cell el", "capacity": fuel_cell_el_capacity})

        # Remaining fuel cells without electrolyzers nearby
        fuel_cell_rest = fuel_cells[~fuel_cells["bus"].isin(buses_with_electrolyzer)]
        fuel_cell_rest_capacity = fuel_cell_rest["p_mw"].sum() * 0.5  # Reduced contribution
        if fuel_cell_rest_capacity > 0:
            buffer_sum.append({"type": "fuel cell", "capacity": fuel_cell_rest_capacity})

    return buffer_sum


def get_gen(net_buff, buffer_sum):
    gen_types = ["sync", "async"]

    for gen_type in gen_types:
        filtered_gen = net_buff.gen[net_buff.gen["name"].str.contains(gen_type, case=False, na=False)]
        filtered_gen = filtered_gen[filtered_gen["p_mw"] > 0]
        gen_capacity = filtered_gen["p_mw"].sum() * 24

        if gen_capacity > 0:
            buffer_sum.append({"type": gen_type, "capacity": gen_capacity})

    return buffer_sum


def get_flexible_loads(net_buff, buffer_sum):
    if "load" in net_buff and not net_buff.load.empty:
        if "controllable" not in net_buff.load.columns:
            print("Warning: 'controllable' column not found in net.load. Skipping flexible loads.")
            return buffer_sum
        flexible_loads = net_buff.load[net_buff.load["controllable"] == True]
        flexible_loads = flexible_loads[flexible_loads["p_mw"] > 0]

        private_loads = flexible_loads[flexible_loads["type"] == "private"]["p_mw"].sum() * 1.0 * 24 # 100 % flexible
        business_loads = flexible_loads[flexible_loads["type"] == "business"]["p_mw"].sum() * 0.15 * 24  # 15 % flexible

        if private_loads > 0:
            buffer_sum.append({"type": "private load flexibility", "capacity": private_loads})

        if business_loads > 0:
            buffer_sum.append({"type": "business load flexibility", "capacity": business_loads})

    return buffer_sum