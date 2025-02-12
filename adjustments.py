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

    # Set active power (P) limits for the external grid
    net.ext_grid.at[0, 'max_p_mw'] = 150  # Maximum active power supply
    net.ext_grid.at[0, 'min_p_mw'] = -50  # Minimum active power supply

    # Set reactive power (Q) limits for the external grid
    net.ext_grid.at[0, 'max_q_mvar'] = 100  # Maximum reactive power supply
    net.ext_grid.at[0, 'min_q_mvar'] = -50  # Minimum reactive power absorption

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

    return net
