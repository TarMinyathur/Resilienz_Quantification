# self_sufficiency.py
# Checks if generation of real, reactive and apparent power meets the corresponding demand on each bus

# selfsuff über subgraphs und conversion of loadflow? →

import pandapower as pp
import time

def selfsuff(net):

    erfolg = 0
    misserfolg = 0

    for bus in net.bus.index:
        generation_p = sum(net.gen[net.gen.bus == bus].p_mw) + sum(net.sgen[net.sgen.bus == bus].p_mw) \
                       + sum(net.storage[net.storage.bus == bus].p_mw)
        # generation_q = sum(net.gen[net.gen.bus == bus].q_mvar.fillna(0)) + sum(net.sgen[net.sgen.bus == bus].q_mvar.fillna(0))
        generation_q = sum(net.sgen[net.sgen.bus == bus].q_mvar.fillna(0))
        generation_s = sum((net.sgen[net.sgen.bus == bus].p_mw ** 2 + net.sgen[net.sgen.bus == bus].q_mvar ** 2) ** 0.5)

        demand_p = sum(net.load[net.load.bus == bus].p_mw)
        demand_q = sum(net.load[net.load.bus == bus].q_mvar)
        demand_s = sum((net.load[net.load.bus == bus].p_mw ** 2 + net.load[net.load.bus == bus].q_mvar ** 2) ** 0.5)

        print(f"Bus {bus}:")
        print(f"   Active Power:   Generation = {generation_p} MW, Demand = {demand_p} MW")
        print(f"   Reactive Power: Generation = {generation_q} Mvar, Demand = {demand_q} Mvar")
        print(f"   Apparent Power: Generation = {generation_s} MVA, Demand = {demand_s} MVA")

        if generation_p >= demand_p:
            erfolg += 1
        else:
            misserfolg += 1

        if generation_q >= demand_q:
            erfolg += 1
        else:
            misserfolg += 1

        if generation_s >= demand_s:
            erfolg += 1
        else:
            misserfolg += 1

    selfsuff = float(erfolg) / (float(erfolg) + float(misserfolg))
    return float(selfsuff)

def selfsufficiency_neu(grid, reduction_factor=0.95, min_threshold=0.5):
    """
    Testet, bis zu welchem Grad die Netzgrenzen reduziert werden können, bevor OPF nicht mehr konvergiert.

    Parameter:
    - net: pandapower Netz
    - reduction_factor: Pro Schritt Reduktion der Netzgrenzen (Standard: 95% der vorherigen Werte)
    - min_threshold: Untere Grenze für die Reduktion (Standard: 95% der ursprünglichen Werte)

    Gibt zurück:
    - Liste mit (Reduktionswert, Zeit bis zur Konvergenz)
    """
    results = []

    # Originalwerte speichern
    #net = grid.copy
    ext_grid = grid.ext_grid.copy
    tempresults = 1

    while True:
        try:
            # Startzeit messen
            start_time = time.time()

            # OPF ausführen
            pp.runopp(
                grid,
                init="pf",
                calculate_voltage_angles=True,
                enforce_q_lims=True,
                distributed_slack=True
            )

            # Dauer messen
            duration = time.time() - start_time
            print(f"OPF konvergiert in {duration:.4f} Sekunden.")

            # Ergebnisse speichern
            tempresults *= reduction_factor

            # Netzgrenzen weiter reduzieren
            grid.ext_grid.loc[:, 'max_p_mw'] *= reduction_factor
            grid.ext_grid.loc[:, 'max_q_mvar'] *= reduction_factor
            grid.ext_grid.loc[:, 'min_p_mw'] *= reduction_factor
            grid.ext_grid.loc[:, 'min_q_mvar'] *= reduction_factor

        except pp.optimal_powerflow.OPFNotConverged:
            print("OPF konvergiert nichtmehr bei folgenden Werten!")
            break

    results = 1 - tempresults
    print(f"  max_p_mw: {grid.ext_grid.get('max_p_mw')} MW")
    print(f"  min_p_mw: {grid.ext_grid.get('min_p_mw')} MW")
    print(f"  max_q_mvar: {grid.ext_grid.get('max_q_mvar')} MVAR")
    print(f"  min_q_mvar: {grid.ext_grid.get('min_q_mvar')} MVAR")

    return results
