import pandapower as pp
import pandapower.networks as pn

# TBD
# andere Speicher / Buffer ? z.B. Wasserstoff Speicher (net.sgen), aber keine in cigre nets enthalten?!
# keine negative lasten? wie fließt batterie mit ein?
# woher kommt wasserstoff für Brennstoffzelle?
# rest in flexibiltät


# buffer gesamt load zu buffer verhältnis setzen , CHP + h2 dazuzählen


def main():
    net = pn.create_cigre_network_mv(with_der="all")

     # get sum load of net
    total_load = net.load['p_mw'].sum()
    print(f"Total net load: {total_load} MW")

    # print(net)
    # print(net.gen)

    battery_availabiliy = check_battery_availability(net)

    if battery_availabiliy == "yes":
        buffer_capacity = analyze_buffer_at_bus_level(net)
        print(f"buffer capacity: {buffer_capacity}")


def check_battery_availability(net):
    if "storage" in net and not net.storage.empty:
        print(f"{len(net.storage)} batteries in net\n")
        battery_availability = "yes"
    else:
        print("no batteries in net")
        battery_availability = "no"
    return battery_availability

def analyze_buffer_at_bus_level(net):
    batteries = net.storage
    # print(batteries)
    loads = net.load
    buffer_capacity_per_bus = {}  # store capacities per bus

    for index, battery in batteries.iterrows(): # get batteries, buses and their loads 
        battery_name = battery['name']
        connected_bus = battery['bus']
        battery_power = battery['p_mw']
        loads_at_bus = loads[loads['bus'] == connected_bus]
        total_load_at_bus = loads_at_bus['p_mw'].sum()

        print(f"{battery_name}, {battery_power} MW")
        print(f"connected bus: {connected_bus}, total loads: {total_load_at_bus} MW")
       
        # check if loads at bus are covered by battery
        if battery_power >= total_load_at_bus:
            print(f"battery able to cover loads at bus\n")
            buffer_capacity = 1 #tbd
        else:
            missing_power_at_bus_level = total_load_at_bus - battery_power
            print(f"battery unable to cover loads at bus, {missing_power_at_bus_level:.2f} MW missing\n")
            buffer_capacity = battery_power / total_load_at_bus # rate how much of loads able to cover tbd

            # Store buffer capacity for each bus
        buffer_capacity_per_bus[connected_bus] = buffer_capacity

    return buffer_capacity_per_bus  # Return collected data



if __name__ == "__main__":
    main()
