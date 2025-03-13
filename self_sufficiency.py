# self_sufficiency.py
# Checks if generation of real, reactive and apparent power meets the corresponding demand on each bus

# selfsuff über subgraphs und conversion of loadflow? →

import pandapower as pp
import numpy as np
import time


def selfsuff(net_temp_selfsuff, gen_factor, show_self_sufficiency_at_bus):
    """
    Berechnet die Eigenversorgung eines Stromnetzes effizienter durch Vektorisierung und optimierte Berechnungen.
    Berücksichtigt dabei die gesicherten Erzeugungswerte anhand der Faktoren in gen_factor, inklusive Batterien.

    Parameter:
        net_temp_selfsuff (pandapowerNet): Das Netzobjekt mit den Informationen zu Erzeugern, Lasten und Speichern.
        gen_factor (dict): Ein Dictionary mit Faktoren für die gesicherten Erzeugungswerte der verschiedenen Erzeugungstechnologien.
    """
    erfolg = 0
    misserfolg = 0

    # Vektorisierte Berechnung für jeden Bus
    for bus in net_temp_selfsuff.bus.index:
        # Gesicherte Erzeugungsleistung für gen und sgen berechnen
        gen_bus = net_temp_selfsuff.gen[net_temp_selfsuff.gen.bus == bus]
        sgen_bus = net_temp_selfsuff.sgen[net_temp_selfsuff.sgen.bus == bus]
        storage_bus = net_temp_selfsuff.storage[net_temp_selfsuff.storage.bus == bus]

        # Gesicherte Leistung für gen
        gen_p = 0
        gen_q = 0
        gen_s = 0
        for _, row in gen_bus.iterrows():
            factor = gen_factor.get(row.type, 1.0)
            p = row.p_mw * factor
            pf = gen_factor.get(f"{row.type}_pf", 0.95)  # Leistungsfaktor für gen Type
            q = p * np.tan(np.arccos(pf))
            s = np.sqrt(p**2 + q**2)
            gen_p += p
            gen_q += q
            gen_s += s

        # Gesicherte Leistung für sgen
        sgen_p = 0
        sgen_q = 0
        sgen_s = 0
        for _, row in sgen_bus.iterrows():
            factor = gen_factor.get(row.type, 1.0)
            p = row.p_mw * factor
            q = row.q_mvar if not np.isnan(row.q_mvar) else 0
            pf = gen_factor.get(f"{row.type}_pf", 0.95)  # Leistungsfaktor für sgen Type
            if q == 0:  # Falls q nicht vorhanden ist, aus pf berechnen
                q = p * np.tan(np.arccos(pf))
            s = np.sqrt(p**2 + q**2)
            sgen_p += p
            sgen_q += q
            sgen_s += s

        # Gesicherte Leistung für Batteriespeicher
        storage_p = 0
        storage_q = 0
        storage_s = 0
        for _, row in storage_bus.iterrows():
            factor = gen_factor.get("Battery", 1.0)
            p = row.p_mw * factor
            pf = gen_factor.get("Battery_pf", 0.95)  # Leistungsfaktor für Batterien
            q = p * np.tan(np.arccos(pf))
            s = np.sqrt(p**2 + q**2)
            storage_p += p
            storage_q += q
            storage_s += s

        # Gesamte Erzeugung am Bus
        generation_p = gen_p + sgen_p + storage_p
        generation_q = gen_q + sgen_q + storage_q
        generation_s = np.sqrt(generation_p**2 + generation_q**2)  # Konsistente Berechnung

        # Lasten berechnen
        load_bus = net_temp_selfsuff.load[net_temp_selfsuff.load.bus == bus]
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

def selfsufficiency_neu(net_temp_selfsuff_2, reduction_factor=0.95, verbose=True, min_threshold=0.01):
    """
    Testet, bis zu welchem Grad die Netzgrenzen reduziert werden können, bevor OPF nicht mehr konvergiert.

    Parameter:
    - net_temp_selfsuff_2: pandapower Netz
    - reduction_factor: Pro Schritt Reduktion der Netzgrenzen (Standard: 95% der vorherigen Werte)
    - verbose: Gibt aus, wie lange die OPF-Berechnung dauert (Standard: True)

    Gibt zurück:
    - Ein Wert: Maximaler Reduktionsfaktor, bei dem noch Konvergenz erreicht wird
    """
    # Originalwerte speichern
    original_ext_net = net_temp_selfsuff_2.ext_grid.copy()
    current_reduction = 1.0  # Start bei 100%

    while True:
        try:
            # Zeitmessung starten, falls verbose aktiviert ist
            if verbose:
                start_time = time.time()

            # OPF ausführen
            pp.runopp(
                net_temp_selfsuff_2,
                init="pf",
                calculate_voltage_angles=True,
                enforce_q_lims=True,
                distributed_slack=True
            )

            # Zeitmessung beenden und ausgeben, falls verbose aktiviert ist
            if verbose:
                duration = time.time() - start_time
                print(f"OPF konvergiert in {duration:.4f} Sekunden.")

            # Netzgrenzen weiter reduzieren
            net_temp_selfsuff_2.ext_grid.loc[:, 'max_p_mw'] *= reduction_factor
            net_temp_selfsuff_2.ext_grid.loc[:, 'max_q_mvar'] *= reduction_factor
            net_temp_selfsuff_2.ext_grid.loc[:, 'min_p_mw'] *= reduction_factor
            net_temp_selfsuff_2.ext_grid.loc[:, 'min_q_mvar'] *= reduction_factor

            # Reduktionsfaktor aktualisieren
            current_reduction *= reduction_factor

            # Überprüfen, ob min_threshold erreicht wurde
            if (net_temp_selfsuff_2.ext_grid['max_p_mw'] < min_threshold * original_ext_net['max_p_mw']).any():
                print("Minimale Netzgrenzen erreicht.")
                break

        except pp.optimal_powerflow.OPFNotConverged:
            print("OPF konvergiert nicht mehr bei folgenden Werten!")
            break

    # Resultat berechnen
    result_selfsuff_neu = 1 - current_reduction
    print(f"System Self Sufficency: {result_selfsuff_neu}")

    return result_selfsuff_neu
