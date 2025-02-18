import pandapower as pp
import pandapower.networks as pn


#to do: buffer anteilig und indicator berechnen
# call buffer.py by main and return indicator (in code integrieren)

buffer_sum = []

def main():
    net = pn.create_cigre_network_mv(with_der="all")

    total_load = net.load["p_mw"].sum()
    print(f"Total net load: {total_load} MW")

    battery_capacity = check_battery_capacity(net)
    
    # SGen-Kapazitäten berechnen & zur Liste hinzufügen
    get_sgen(net)

    total_buffer_capacity = sum(entry["capacity"] for entry in buffer_sum)
    print(f"Total buffer capacity: {total_buffer_capacity} MW")




def check_battery_capacity(net):
    """ Berechnet die Gesamt-Speicherkapazität und fügt sie zur Liste hinzu. """
    if "storage" in net and not net.storage.empty:
        battery_capacity = net.storage["p_mw"].sum()
        buffer_sum.append({"type": "storage", "capacity": battery_capacity}) 
        return battery_capacity  # Rückgabe für die Anzeige
    else:
        print("No batteries in net")
        return None


def get_sgen(net):
    sgen_types = ["Residential fuel cell", "CHP diesel", "Fuel cell"]  # potentially unify for further adaption to nets

    for sgen_type in sgen_types:
        # Filtern der Generatoren nach Typ
        filtered_sgen = net.sgen[net.sgen["name"].str.contains(sgen_type, case=False, na=False)]
        sgen_capacity = filtered_sgen["p_mw"].sum()

        # Falls Kapazität vorhanden, zur Liste hinzufügen
        if sgen_capacity > 0:
            buffer_sum.append({"type": sgen_type, "capacity": sgen_capacity})

if __name__ == "__main__":
    main()
