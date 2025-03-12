# adjustments.py
# grid is adjusted

import pandapower as pp
import pandas as pd


# import pandapower.networks as pn
# from networkx import grid_graph

# net = pn.create_cigre_network_mv(with_der="all")

def set_missing_limits(net, required_p_mw, required_q_mvar):
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
        if pd.isna(ext_grid.get('max_p_mw')) or ext_grid.get('max_p_mw') < S_max:
            net.ext_grid.at[idx, 'max_p_mw'] = S_max * P_factor
        if pd.isna(ext_grid.get('min_p_mw')) or ext_grid.get('min_p_mw') < -S_max:
            net.ext_grid.at[idx, 'min_p_mw'] = - (S_max * 0.5)  # Sicherheitsfaktor

        # Falls max_q_mvar oder min_q_mvar nicht gesetzt sind, setzen
        if pd.isna(ext_grid.get('max_q_mvar')) or ext_grid.get('max_p_mw') < S_max:
            net.ext_grid.at[idx, 'max_q_mvar'] = S_max * Q_factor * 0.5  # Begrenzte Q-Abgabe (+50%)
        if pd.isna(ext_grid.get('min_q_mvar')) or ext_grid.get('min_q_mvar') < -S_max:
            net.ext_grid.at[idx, 'min_q_mvar'] = - (S_max * Q_factor * 0.8)  # Größerer Q-Bezug (-80%)

    # Optionale Info-Ausgabe
    print(f"Set limits for ext_grid at bus {bus_idx} (vn_kv={vn_kv} kV):")
    print(f"Mindest-Netzbezug ermittelt: {required_p_mw:.2f} MW")
    print(f"Mindest-Netzbezug ermittelt: {required_q_mvar:.2f} MVAR")
    print(f"  max_p_mw: {net.ext_grid.at[idx, 'max_p_mw']} MW")
    print(f"  min_p_mw: {net.ext_grid.at[idx, 'min_p_mw']} MW")
    print(f"  max_q_mvar: {net.ext_grid.at[idx, 'max_q_mvar']} MVAR")
    print(f"  min_q_mvar: {net.ext_grid.at[idx, 'min_q_mvar']} MVAR")

    """Iterate through gens, sgens, and storage to set max_p_mw and max_q_mvar plus max_e_mwh for storages where missing."""
    # Synchronous generators (e.g., large hydro, gas turbines, CHP) can provide reactive power (positive & negative Q).
    # Asynchronous generators (e.g., small wind farms, induction generators) absorb reactive power from the grid.
    # PV & Wind (sgen) are assumed to support some reactive power regulation, so their max_q_mvar is set to ±30% of P.
    # Storage units can supply and absorb both active and reactive power.

    # Process conventional generators (gen)
    for idx, gen in net.gen.iterrows():
        gen_type = str(gen.get("type", "")).lower()

        # Active power limits (P)
        if pd.isna(gen.get('max_p_mw')):
            net.gen.at[idx, 'max_p_mw'] = 1.2 * gen['p_mw']
        if pd.isna(gen.get('min_p_mw')):
            net.gen.at[idx, 'min_p_mw'] = 0.2 * gen['p_mw']

        # Reactive power limits (Q) based on sync/async type
        if "sync" in gen_type or gen_type == "":  # Default to synchronous
            if pd.isna(gen.get('max_q_mvar')):
                net.gen.at[idx, 'max_q_mvar'] = gen['p_mw']  # Can provide Q
            if pd.isna(gen.get('min_q_mvar')):
                net.gen.at[idx, 'min_q_mvar'] = -0.5 * gen['p_mw']

        elif "async" in gen_type:  # Asynchronous generators absorb reactive power
            if pd.isna(gen.get('max_q_mvar')):
                net.gen.at[idx, 'max_q_mvar'] = 0.2 * gen['p_mw']  # Limited Q production
            if pd.isna(gen.get('min_q_mvar')):
                net.gen.at[idx, 'min_q_mvar'] = -0.5 * gen['p_mw']  # Absorbs Q from the grid

    # Process static generators (sgen) - e.g., PV & Wind
    for idx, sgen in net.sgen.iterrows():
        if "wind" in str(sgen.get("type", "")).lower() or "pv" in str(sgen.get("type", "")).lower() or "WP" in str(
                sgen.get("type", "")).lower() or "PV" in str(sgen.get("type", "")).lower():
            if pd.isna(sgen.get('max_p_mw')):
                net.sgen.at[idx, 'max_p_mw'] = sgen['p_mw']
            if pd.isna(sgen.get('min_p_mw')):
                net.sgen.at[idx, 'min_p_mw'] = 0
            if pd.isna(sgen.get('max_q_mvar')):
                net.sgen.at[idx, 'max_q_mvar'] = 0.3 * sgen['p_mw']
            if pd.isna(sgen.get('min_q_mvar')):
                net.sgen.at[idx, 'min_q_mvar'] = -0.1 * sgen['p_mw']

    # Process storage units (storage)
    for idx, storage in net.storage.iterrows():
        if pd.isna(storage.get('max_p_mw')):
            net.storage.at[idx, 'max_p_mw'] = storage['p_mw']
        if pd.isna(storage.get('min_p_mw')):
            net.storage.at[idx, 'min_p_mw'] = -storage['p_mw']
        if pd.isna(storage.get('max_q_mvar')):
            net.storage.at[idx, 'max_q_mvar'] = storage['p_mw']
        if pd.isna(storage.get('min_q_mvar')):
            net.storage.at[idx, 'min_q_mvar'] = -storage['p_mw']

        # the storable energy of all storages is changed from infinity, to a value that they can supply they peak power for one day
        if storage['max_e_mwh'] == float('inf'):
            net.storage.at[idx, 'max_e_mwh'] = storage['p_mw'] * 24

        # Begrenzung für Busse (`bus`)**
    for idx, bus in net.bus.iterrows():
        vn_kv = bus['vn_kv']  # Spannungsebene des Busses

        # Setze realistische Spannungsgrenzen basierend auf Netzebene
        if vn_kv <= 20:
            min_vm_pu, max_vm_pu = 0.95, 1.05  # Mittelspannung
        elif vn_kv <= 110:
            min_vm_pu, max_vm_pu = 0.94, 1.06  # Hochspannung
        elif vn_kv <= 220:
            min_vm_pu, max_vm_pu = 0.93, 1.07  # Höchstspannung 220 kV
        else:  # ≥ 380 kV
            min_vm_pu, max_vm_pu = 0.92, 1.08  # Höchstspannung 380 kV

        # Falls keine Grenzen gesetzt sind, setzen
        if pd.isna(bus.get('min_vm_pu')):
            net.bus.at[idx, 'min_vm_pu'] = min_vm_pu
        if pd.isna(bus.get('max_vm_pu')):
            net.bus.at[idx, 'max_vm_pu'] = max_vm_pu

        # Begrenzung für Kabel/Leitungen (`line`)**
    for idx, line in net.line.iterrows():
        std_type = line["std_type"]
        vn_kv = net.bus.at[line['from_bus'], 'vn_kv']

        # Check if the line has a predefined `std_type`
        if pd.notna(std_type) and std_type in net.std_types["line"]:
            line_data = net.std_types["line"][std_type]
            max_i_ka = line_data.get("imax", line_data.get("max_i_ka"))  # Get the correct key
        else:
            # Default values for lines with no `std_type`
            if vn_kv <= 20:
                max_i_ka = 0.4
            elif vn_kv <= 110:
                max_i_ka = 1.2
            elif vn_kv <= 220:
                max_i_ka = 2.0
            else:
                max_i_ka = 3.0

        if pd.isna(line.get('max_i_ka')):
            net.line.at[idx, 'max_i_ka'] = max_i_ka

    # Transformer Limits (`trafo` and `trafo3w`)**
    for idx, trafo in net.trafo.iterrows():
        vn_kv = net.bus.at[trafo['hv_bus'], 'vn_kv']

        if vn_kv <= 20:
            sn_mva = 40
        elif vn_kv <= 110:
            sn_mva = 200
        elif vn_kv <= 220:
            sn_mva = 500
        else:
            sn_mva = 1000

        if pd.isna(trafo.get('sn_mva')):
            net.trafo.at[idx, 'sn_mva'] = sn_mva

    if "in_service" not in net.switch.columns:
        net.switch["in_service"] = True  # Default-Wert für alle Switches

    return net

def set_power_limits(df, multiplier=1.4):
    """
    Vectorized function to set power limits for gen, sgen, and storage.
    """

    # Ensure required columns exist, initializing with NaN if missing
    for col in ['max_p_mw', 'min_p_mw', 'max_q_mvar', 'min_q_mvar']:
        if col not in df.columns:
            df[col] = pd.NA  # Initialize with NaN

    # Set Active Power Limits (P)
    df['max_p_mw'] = df['max_p_mw'].fillna(df['p_mw'] * multiplier)
    df['min_p_mw'] = df['min_p_mw'].fillna(0)

    # Set Reactive Power Limits (Q)
    df['max_q_mvar'] = df['max_q_mvar'].fillna(df['p_mw'])
    df['min_q_mvar'] = df['min_q_mvar'].fillna(-df['p_mw'])

def initialize_and_set_bus_voltage_limits(net, min_vm_pu=0.8, max_vm_pu=1.2):
    """
    Ensures 'min_vm_pu' and 'max_vm_pu' columns exist and sets default voltage limits.
    This avoids KeyError and handles future Pandas changes gracefully.
    """
    # Check and initialize if columns don't exist
    if 'min_vm_pu' not in net.bus.columns:
        net.bus['min_vm_pu'] = pd.NA
    if 'max_vm_pu' not in net.bus.columns:
        net.bus['max_vm_pu'] = pd.NA

    # Safely fill NaN values with defaults
    net.bus['min_vm_pu'] = net.bus['min_vm_pu'].fillna(min_vm_pu).astype(float)
    net.bus['max_vm_pu'] = net.bus['max_vm_pu'].fillna(max_vm_pu).astype(float)

def determine_minimum_ext_grid_power(net):
    """
    Runs OPF to determine the required minimum external grid power (max_p_mw).
    Then updates ext_grid max_p_mw based on the result.
    """

    # Define Constants
    DEFAULT_MAX_P_MW = 10000000
    DEFAULT_MIN_P_MW = -2000000
    DEFAULT_MAX_Q_MVAR = 10000000
    DEFAULT_MIN_Q_MVAR = -2000000

    # Apply the power limits to each component using vectorized operations
    set_power_limits(net.gen)
    set_power_limits(net.sgen)
    set_power_limits(net.storage)

    # Set bus voltage limits
    initialize_and_set_bus_voltage_limits(net)

    # External Grid Settings
    for idx, ext_grid in net.ext_grid.iterrows():
        net.ext_grid.loc[idx, 'max_p_mw'] = ext_grid.get('max_p_mw', DEFAULT_MAX_P_MW)
        net.ext_grid.loc[idx, 'min_p_mw'] = ext_grid.get('min_p_mw', DEFAULT_MIN_P_MW)
        net.ext_grid.loc[idx, 'max_q_mvar'] = ext_grid.get('max_q_mvar', DEFAULT_MAX_Q_MVAR)
        net.ext_grid.loc[idx, 'min_q_mvar'] = ext_grid.get('min_q_mvar', DEFAULT_MIN_Q_MVAR)
        net.ext_grid.loc[idx, "slack"] = True  # Set as slack bus

    # Ensure all elements have cost functions
    if "poly_cost" not in net or net.poly_cost.empty:
        net["poly_cost"] = pd.DataFrame(columns=["element", "et", "cp1_eur_per_mw"])

    add_cost_function_if_missing(net, "gen")
    add_cost_function_if_missing(net, "sgen")
    add_cost_function_if_missing(net, "storage")
    add_cost_function_if_missing(net, "ext_grid")

    # Run Optimal Power Flow (OPF)
    try:
        pp.runopp(
            net,
            init="pf",
            calculate_voltage_angles=True,
            enforce_q_lims=True,
            distributed_slack=True
        )
        print("OPF Converged Successfully!")

    except (pp.optimal_powerflow.OPFNotConverged, pp.powerflow.LoadflowNotConverged):
        print("OPF did not converge with init='pf'. Retrying with init='flat'")
        try:
            # Retry with init="flat"
            pp.runopp(
                net,
                init="flat",
                calculate_voltage_angles=True,
                enforce_q_lims=True,
                distributed_slack=True
            )
            print("OPF Converged with init='flat'")
        except pp.optimal_powerflow.OPFNotConverged:
            print("OPF failed. Debugging information:")
            print("Voltage Limits:")
            print(net.bus[["name", "vn_kv", "min_vm_pu", "max_vm_pu"]])

            print("External Grid Settings:")
            print(net.ext_grid)

            print("Generators:")
            print(net.gen)
            print(net.sgen)
            print(net.storage)

            print("Loads:")
            print(net.load)
            raise ValueError("OPF did not converge with any initialization method.")

    # Calculate required external grid power
    required_p_mw = net.res_ext_grid["p_mw"].sum() * 1.1
    required_q_mvar = net.res_ext_grid["q_mvar"].sum() * 1.1

    # **Corrected Assignment**
    net.ext_grid["max_p_mw"] = required_p_mw
    net.ext_grid["max_q_mvar"] = required_q_mvar

    return net, required_p_mw, required_q_mvar


def add_cost_function_if_missing(net, element_type):
    """Ensure every element has an associated cost function"""
    if "poly_cost" not in net or net.poly_cost.empty:
        net["poly_cost"] = pp.create_empty_network().poly_cost  # Create cost table if missing

    elements = getattr(net, element_type)

    for idx in elements.index:
        # Check if a cost function already exists for this element
        if not (net.poly_cost["element"].eq(idx) & net.poly_cost["et"].eq(element_type)).any():
            pp.create_poly_cost(net, element=idx, et=element_type, cp1_eur_per_mw=0.01)  # Example cost
