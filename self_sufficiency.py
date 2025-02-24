# self_sufficiency.py
# Checks if generation of real, reactive and apparent power meets the corresponding demand on each bus

# selfsuff über subgraphs und conversion of loadflow? →

import pandapower as pp
import numpy as np
import time


def selfsuff(net, gen_factor, show_self_sufficiency_at_bus):
    """
    Berechnet die Eigenversorgung eines Stromnetzes effizienter durch Vektorisierung und optimierte Berechnungen.
    Berücksichtigt dabei die gesicherten Erzeugungswerte anhand der Faktoren in gen_factor, inklusive Batterien.

    Parameter:
        net (pandapowerNet): Das Netzobjekt mit den Informationen zu Erzeugern, Lasten und Speichern.
        gen_factor (dict): Ein Dictionary mit Faktoren für die gesicherten Erzeugungswerte der verschiedenen Erzeugungstechnologien.
    """
    erfolg = 0
    misserfolg = 0

    # Vektorisierte Berechnung für jeden Bus
    for bus in net.bus.index:
        # Gesicherte Erzeugungsleistung für gen und sgen berechnen
        gen_bus = net.gen[net.gen.bus == bus]
        sgen_bus = net.sgen[net.sgen.bus == bus]
        storage_bus = net.storage[net.storage.bus == bus]

        # Gesicherte Leistung für gen
        gen_p = np.sum([row.p_mw * gen_factor.get(row.type, 1.0) for _, row in gen_bus.iterrows()])

        # Gesicherte Leistung für sgen
        sgen_p = np.sum([row.p_mw * gen_factor.get(row.type, 1.0) for _, row in sgen_bus.iterrows()])
        sgen_q = sgen_bus.q_mvar.fillna(0).sum()

        # Scheinleistung für sgen
        sgen_s = np.sqrt((sgen_bus.p_mw ** 2 + sgen_bus.q_mvar ** 2)).sum()

        # Gesicherte Leistung für Batteriespeicher
        battery_factor = gen_factor.get("Battery", 1.0)
        storage_p = storage_bus.p_mw.sum() * battery_factor
        storage_s = storage_bus.p_mw.sum() * battery_factor

        # Gesamte Erzeugung am Bus
        generation_p = gen_p + sgen_p + storage_p
        generation_q = sgen_q
        generation_s = sgen_s + storage_s

        # Lasten berechnen
        load_bus = net.load[net.load.bus == bus]
        demand_p = load_bus.p_mw.sum()
        demand_q = load_bus.q_mvar.sum()
        demand_s = np.sqrt((load_bus.p_mw ** 2 + load_bus.q_mvar ** 2)).sum()

        if show_self_sufficiency_at_bus:
            # Debug-Informationen
            print(f"Bus {bus}:")
            print(f"   Active Power:   Generation = {generation_p:.2f} MW, Demand = {demand_p:.2f} MW")
            print(f"   Reactive Power: Generation = {generation_q:.2f} Mvar, Demand = {demand_q:.2f} Mvar")
            print(f"   Apparent Power: Generation = {generation_s:.2f} MVA, Demand = {demand_s:.2f} MVA")

        # Überprüfung der Versorgungssicherheit
        erfolg += generation_p >= demand_p
        erfolg += generation_q >= demand_q
        erfolg += generation_s >= demand_s

        misserfolg += generation_p < demand_p
        misserfolg += generation_q < demand_q
        misserfolg += generation_s < demand_s

    # Eigenversorgungsgrad berechnen
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
