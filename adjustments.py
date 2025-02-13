# adjustments.py
# grid is adjusted

import pandapower as pp
import pandas as pd

def set_missing_limits(net):

    # Make sure no cost functions are applied
    if "poly_cost" in net:
        net.poly_cost.drop(net.poly_cost.index, inplace=True)

    # Close all switches in the network
    net.switch['closed'] = True

    """Setzt plausible P, Q und S-Grenzen für externe Netze gemäß der Netzspannungsebene, falls nicht definiert."""

    # Durchlaufe alle externen Netze (es kann mehrere geben)
    for idx, ext_grid in net.ext_grid.iterrows():
        bus_idx = ext_grid['bus']  # Bus, an dem das externe Netz angeschlossen ist
        vn_kv = net.bus.at[bus_idx, 'vn_kv']  # Spannungslevel des Busses

        # Basierend auf Spannungsebene typische Transformator-Nennleistung S_max setzen
        if vn_kv <= 20:
            S_max = 50  # Mittelspannungstrafos liegen meist zwischen 25-63 MVA
            P_factor, Q_factor = 0.9, 0.4
        elif vn_kv <= 110:
            S_max = 200  # Hochspannungstrafos liegen meist zwischen 100-300 MVA
            P_factor, Q_factor = 0.95, 0.5
        elif vn_kv <= 220:
            S_max = 500  # Höchstspannungstrafos liegen meist zwischen 300-800 MVA
            P_factor, Q_factor = 0.97, 0.6
        else:  # ≥ 380 kV
            S_max = 1500  # Höchstspannungsbereich (800-3000 MVA möglich)
            P_factor, Q_factor = 1.0, 0.7

        # Falls max_p_mw oder min_p_mw nicht gesetzt sind, setzen
        if pd.isna(ext_grid.get('max_p_mw')):
            net.ext_grid.at[idx, 'max_p_mw'] = S_max * P_factor
        if pd.isna(ext_grid.get('min_p_mw')):
            net.ext_grid.at[idx, 'min_p_mw'] = S_max * 0.5  # Sicherheitsfaktor

        # Falls max_q_mvar oder min_q_mvar nicht gesetzt sind, setzen
        if pd.isna(ext_grid.get('max_q_mvar')):
            net.ext_grid.at[idx, 'max_q_mvar'] = S_max * Q_factor * 0.5  # Begrenzte Q-Abgabe (+50%)
        if pd.isna(ext_grid.get('min_q_mvar')):
            net.ext_grid.at[idx, 'min_q_mvar'] = -S_max * Q_factor * 0.8  # Größerer Q-Bezug (-80%)

        # Apply zero cost function to external grids (`ext_grid`)
    #    if idx not in net.poly_cost["element"].values:
    #        pp.create_poly_cost(net, element=idx, et="ext_grid", cp1_eur_per_mw=0)

    # Optionale Info-Ausgabe
    print(f"Set limits for ext_grid at bus {bus_idx} (vn_kv={vn_kv} kV):")
    print(f"  max_p_mw: {net.ext_grid.at[idx, 'max_p_mw']} MW")
    print(f"  min_p_mw: {net.ext_grid.at[idx, 'min_p_mw']} MW")
    print(f"  max_q_mvar: {net.ext_grid.at[idx, 'max_q_mvar']} MVAR")
    print(f"  min_q_mvar: {net.ext_grid.at[idx, 'min_q_mvar']} MVAR")

    """Iterate through gens, sgens, and storage to set max_p_mw and max_q_mvar plus max_e_mwh for storages where missing."""
    #Synchronous generators (e.g., large hydro, gas turbines, CHP) can provide reactive power (positive & negative Q).
    #Asynchronous generators (e.g., small wind farms, induction generators) absorb reactive power from the grid.
    #PV & Wind (sgen) are assumed to support some reactive power regulation, so their max_q_mvar is set to ±30% of P.
    #Storage units can supply and absorb both active and reactive power.

    # Process conventional generators (gen)
    for idx, gen in net.gen.iterrows():
        gen_type = str(gen.get("type", "")).lower()

        # Active power limits (P)
        if pd.isna(gen.get('max_p_mw')):
            net.gen.at[idx, 'max_p_mw'] = 1.2 * gen['p_mw']
        if pd.isna(gen.get('min_p_mw')):
            net.gen.at[idx, 'min_p_mw'] = 0.5 * gen['p_mw']

        # Reactive power limits (Q) based on sync/async type
        if "sync" in gen_type or gen_type == "":  # Default to synchronous
            if pd.isna(gen.get('max_q_mvar')):
                net.gen.at[idx, 'max_q_mvar'] = 0.5 * gen['p_mw']  # Can provide Q
            if pd.isna(gen.get('min_q_mvar')):
                net.gen.at[idx, 'min_q_mvar'] = -0.5 * gen['p_mw']

        elif "async" in gen_type:  # Asynchronous generators absorb reactive power
            if pd.isna(gen.get('max_q_mvar')):
                net.gen.at[idx, 'max_q_mvar'] = -0.2 * gen['p_mw']  # Limited Q production
            if pd.isna(gen.get('min_q_mvar')):
                net.gen.at[idx, 'min_q_mvar'] = -0.5 * gen['p_mw']  # Absorbs Q from the grid

        # Apply zero cost function to generators (`gen`)
    #    if idx not in net.poly_cost["element"].values:  # Avoid overwriting existing costs
    #        pp.create_poly_cost(net, element=idx, et="gen", cp1_eur_per_mw=0)


    # Process static generators (sgen) - e.g., PV & Wind
    for idx, sgen in net.sgen.iterrows():
        if "wind" in str(sgen.get("type", "")).lower() or "pv" in str(sgen.get("type", "")).lower():
            if pd.isna(sgen.get('max_p_mw')):
                net.sgen.at[idx, 'max_p_mw'] = sgen['p_mw']
            if pd.isna(sgen.get('min_p_mw')):
                net.sgen.at[idx, 'min_p_mw'] = 0
            if pd.isna(sgen.get('max_q_mvar')):
                net.sgen.at[idx, 'max_q_mvar'] = 0.3 * sgen['p_mw']
            if pd.isna(sgen.get('min_q_mvar')):
                net.sgen.at[idx, 'min_q_mvar'] = -0.3 * sgen['p_mw']

        # Apply zero cost function to static generators (`sgen`)
    #    if idx not in net.poly_cost["element"].values:
    #        pp.create_poly_cost(net, element=idx, et="sgen", cp1_eur_per_mw=0)

    # Process storage units (storage)
    for idx, storage in net.storage.iterrows():
        if pd.isna(storage.get('max_p_mw')):
            net.storage.at[idx, 'max_p_mw'] = storage['p_mw']
        if pd.isna(storage.get('min_p_mw')):
            net.storage.at[idx, 'min_p_mw'] = -storage['p_mw']
        if pd.isna(storage.get('max_q_mvar')):
            net.storage.at[idx, 'max_q_mvar'] = 0.2 * storage['p_mw']
        if pd.isna(storage.get('min_q_mvar')):
            net.storage.at[idx, 'min_q_mvar'] = -0.2 * storage['p_mw']

        # the storable energy of all storages is changed from infinity, to a value that they can supply they peak power for one day
        if storage['max_e_mwh'] == float('inf'):
            net.storage.at[idx, 'max_e_mwh'] = storage['p_mw'] * 24

        # Apply zero cost function to storages (`storage`)
    #    if idx not in net.poly_cost["element"].values:
    #        pp.create_poly_cost(net, element=idx, et="storage", cp1_eur_per_mw=0)

    return net
