import pandapower as pp
import pandapower.networks as pn
import pandas as pd

# tests all existing test grids and checks if the opp converges

# List of all available test networks in pandapower
test_grids = [
    # Example Networks
    ("example_simple", pn.example_simple),
    ("example_multivoltage", pn.example_multivoltage),

    # # Simple pandapower Test Networks
    # ("panda_four_load_branch", pn.panda_four_load_branch),
    # ("four_loads_with_branches_out", pn.four_loads_with_branches_out),
    # ("simple_four_bus_system", pn.simple_four_bus_system),
    # ("simple_mv_open_ring_net", pn.simple_mv_open_ring_net),

    # # CIGRE Networks
    # ("create_cigre_network_hv", pn.create_cigre_network_hv),
    # ("create_cigre_network_mv", pn.create_cigre_network_mv),
    # ("create_cigre_network_mv_pv_wind", lambda: pn.create_cigre_network_mv(with_der="pv_wind")),
    # ("create_cigre_network_mv_all", lambda: pn.create_cigre_network_mv(with_der="all")),
    # ("create_cigre_network_lv", pn.create_cigre_network_lv),
    #
    # # MV Oberrhein
     ("mv_oberrhein", pn.mv_oberrhein),
    #
    # # Power System Test Cases
    # ("case4gs", pn.case4gs),
    # ("case5", pn.case5),
    # ("case6ww", pn.case6ww),
    # ("case9", pn.case9),
    # ("case14", pn.case14),
    # ("case24_ieee_rts", pn.case24_ieee_rts),
    # ("case30", pn.case30),
    # ("case33bw", pn.case33bw),
    # ("case39", pn.case39),
    # ("case57", pn.case57),
    # ("case89pegase", pn.case89pegase),
    # ("case118", pn.case118),
    # ("case145", pn.case145),
    # ("case300", pn.case300),
    # ("case1354pegase", pn.case1354pegase),
    # ("case2869pegase", pn.case2869pegase),
    # ("case9241pegase", pn.case9241pegase),
    # ("case_illinois200", pn.case_illinois200),
    # ("case_1888rte", pn.case1888rte),
    # ("case_2848rte", pn.case2848rte),
    # ("case_3120sp", pn.case3120sp),
    # ("case_6470rte", pn.case6470rte),
    # ("case_6495rte", pn.case6495rte),
    # ("case_6515rte", pn.case6515rte),
    # ("GBnetwork", pn.GBnetwork),
    # ("GBreducednetwork", pn.GBreducednetwork),
    # ("iceland", pn.iceland),

    # Synthetic Voltage Control LV Networks
    ("create_synthetic_voltage_control_lv_network", pn.create_synthetic_voltage_control_lv_network),

    # # 3-Phase Grid Data
    # ("ieee_european_lv_asymmetric", pn.ieee_european_lv_asymmetric),

    # # Average Kerber Networks
    # ("create_kerber_landnetz_freileitung_1", pn.create_kerber_landnetz_freileitung_1),
    # ("create_kerber_landnetz_freileitung_2", pn.create_kerber_landnetz_freileitung_2),
    # ("create_kerber_landnetz_kabel_1", pn.create_kerber_landnetz_kabel_1),
    # ("create_kerber_landnetz_kabel_2", pn.create_kerber_landnetz_kabel_2),
    ("create_kerber_dorfnetz", pn.create_kerber_dorfnetz),
    #("create_kerber_vorstadtnetz_kabel_1", pn.create_kerber_vorstadtnetz_kabel_1),
    #("create_kerber_vorstadtnetz_kabel_2", pn.create_kerber_vorstadtnetz_kabel_2),

    # Extreme Kerber Networks
    # ("kb_extrem_landnetz_freileitung", pn.kb_extrem_landnetz_freileitung),
    # ("kb_extrem_landnetz_kabel", pn.kb_extrem_landnetz_kabel),
    # ("kb_extrem_landnetz_freileitung_trafo", pn.kb_extrem_landnetz_freileitung_trafo),
    # ("kb_extrem_landnetz_kabel_trafo", pn.kb_extrem_landnetz_kabel_trafo),
    ("kb_extrem_dorfnetz", pn.kb_extrem_dorfnetz),
    # ("kb_extrem_dorfnetz_trafo", pn.kb_extrem_dorfnetz_trafo),
    # ("kb_extrem_vorstadtnetz_1", pn.kb_extrem_vorstadtnetz_1),
    # ("kb_extrem_vorstadtnetz_2", pn.kb_extrem_vorstadtnetz_2),
    # ("kb_extrem_vorstadtnetz_trafo_1", pn.kb_extrem_vorstadtnetz_trafo_1)
]

# Store the results
results = []

print("\n--- OPF Convergence Test for Pandapower Grids ---\n")


def run_corrected_opf(net):

        # 🔹 Fix Voltage Limits (Prevent Collapse)
    net.bus["min_vm_pu"] = 0.94  # Min Voltage = 0.9 p.u.
    net.bus["max_vm_pu"] = 1.06  # Max Voltage = 1.1 p.u.

    # 🔹 **Limit Voltage Angles**
    net.ext_grid["max_va_degree"] = 50  # Reduce angle range
    net.ext_grid["min_va_degree"] = -50  # Prevent extreme angles

    # 🔹 **Fix Generator Limits (Prevent Negative P)**
    net.gen["min_p_mw"] = net.gen["p_mw"].clip(lower=15)  # Ensure min generation ≥ 10 MW
    net.gen["max_p_mw"] = net.gen["p_mw"] * 1.3  # Allow headroom for OPF
    net.gen["min_q_mvar"] = -50  # Allow reasonable reactive range
    net.gen["max_q_mvar"] = 100

    # 🔹 Remove All Existing Cost Functions to Avoid Duplicates
    if not net.poly_cost.empty:
        net.poly_cost.drop(net.poly_cost.index, inplace=True)  # Clear existing cost functions

    # # 🔹 Add a Single Cost Function Per Generator
    for i, gen_idx in enumerate(net.gen.index):
        pp.create_poly_cost(net, gen_idx, "gen",  cp1_eur_per_mw=0.01)

    # net.poly_cost = net.poly_cost[net.poly_cost["et"] == "gen"]  # Keep only generator costs
    # net.poly_cost["c"] = [0, 1.5, 0]  # **Linear cost function instead of quadratic**

    # 🔹 Increase Transformer & Line MVA Ratings
    net.trafo["max_loading_percent"] = 130  # Allow transformers to load up to 120%
    net.line["max_loading_percent"] = 130  # Allow lines to load up to 120%

    # 🔹 Set Voltage Angle Limits (Prevent Unrealistic Fluctuations)
    net.ext_grid["max_va_degree"] = 60  # Max 60 degrees
    net.ext_grid["min_va_degree"] = -60  # Min -60 degrees

    # 🔹 **Reduce Line & Transformer Losses**
    net.line["r_ohm_per_km"] *= 0.75  # Reduce resistance (20% reduction)
    net.line["x_ohm_per_km"] *= 0.75  # Reduce reactance (20% reduction)

    return net

# Iterate through all test grids
for grid_name, grid_func in test_grids:
    print(f"Testing {grid_name} ...")
    # Create the network
    net = grid_func()
    net.switch['closed'] = True

    if grid_name == "example_multivoltage":
        net = run_corrected_opf(net)


    # Process conventional generators (gen)
    for idx, gen in net.gen.iterrows():
        # Active power limits (P)
        if pd.isna(gen.get('max_p_mw')):
            net.gen.at[idx, 'max_p_mw'] = 1.2 * gen['p_mw']
        if pd.isna(gen.get('min_p_mw')):
            net.gen.at[idx, 'min_p_mw'] = 0
        if pd.isna(gen.get('max_q_mvar')):
            net.gen.at[idx, 'max_q_mvar'] = gen['p_mw']  # Can provide Q
        if pd.isna(gen.get('min_q_mvar')):
            net.gen.at[idx, 'min_q_mvar'] = - gen['p_mw']

    # Process static generators (sgen) - e.g., PV & Wind
    for idx, sgen in net.sgen.iterrows():
        if pd.isna(sgen.get('max_p_mw')):
            net.sgen.at[idx, 'max_p_mw'] = sgen['p_mw']
        if pd.isna(sgen.get('min_p_mw')):
            net.sgen.at[idx, 'min_p_mw'] = 0
        if pd.isna(sgen.get('max_q_mvar')):
            net.sgen.at[idx, 'max_q_mvar'] = sgen['p_mw']
        if pd.isna(sgen.get('min_q_mvar')):
            net.sgen.at[idx, 'min_q_mvar'] = -sgen['p_mw']

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

    for idx, bus in net.bus.iterrows():
        min_vm_pu, max_vm_pu = 0.7, 1.3  #
        # Falls keine Grenzen gesetzt sind, setzen
        if pd.isna(bus.get('min_vm_pu')):
            net.bus.at[idx, 'min_vm_pu'] = min_vm_pu
        if pd.isna(bus.get('max_vm_pu')):
            net.bus.at[idx, 'max_vm_pu'] = max_vm_pu

    for idx, ext_grid in net.ext_grid.iterrows():
        # Falls max_p_mw oder min_p_mw nicht gesetzt sind, setzen
        if pd.isna(ext_grid.get('max_p_mw')):
            net.ext_grid.at[idx, 'max_p_mw'] = 1000000
        if pd.isna(ext_grid.get('min_p_mw')):
            net.ext_grid.at[idx, 'min_p_mw'] = -200000  # Sicherheitsfaktor

        # Falls max_q_mvar oder min_q_mvar nicht gesetzt sind, setzen
        if pd.isna(ext_grid.get('max_q_mvar')):
            net.ext_grid.at[idx, 'max_q_mvar'] = 1000000  # Begrenzte Q-Abgabe (+50%)
        if pd.isna(ext_grid.get('min_q_mvar')):
            net.ext_grid.at[idx, 'min_q_mvar'] = -200000  # Größerer Q-Bezug (-80%)

    try:
        # Run Power Flow first
        pp.runpp(net)
        print(net.trafo)
        # print(f"{grid_name}: Power Flow converged successfully!")
        #
        # print("Spannungen an den Netzknoten:")
        # print(net.res_bus)
        #
        # print("\nLeistungsflüsse auf den Leitungen:")
        # print(net.res_line)
        #
        # print("\nTransformatorleistung:")
        # print(net.res_trafo)
        #
        # print("\nGeneratorleistung:")
        # print(net.res_gen)

        # Try to run Optimal Power Flow (OPF) with init='pf'
        try:
            pp.runopp(
                net,
                init="pf",
                verbose=True,
                calculate_voltage_angles=True,
                enforce_q_lims=True,
                distributed_slack=True,
                pdipm_step_size=0.05
            )
            print(f"{grid_name}: OPF converged successfully with init='pf'!")
            results.append((grid_name, "OPF Converged with init='pf'"))

        # If OPF does not converge, retry with init='flat'
        except pp.optimal_powerflow.OPFNotConverged:
            print(f"{grid_name}: OPF did not converge with init='pf', retrying with init='flat'")

            try:
                pp.runopp(
                    net,
                    init="flat",
                    calculate_voltage_angles=True,
                    enforce_q_lims=True,
                    distributed_slack=True
                )
                print(f"{grid_name}: OPF converged successfully with init='flat'!")
                results.append((grid_name, "OPF Converged with init='flat'"))

            # If still no convergence, retry with init='results'
            except pp.optimal_powerflow.OPFNotConverged:
                print(f"{grid_name}: OPF did not converge with init='flat', retrying with init='results'")

                try:
                    pp.runopp(
                        net,
                        init="results",
                        calculate_voltage_angles=True,
                        enforce_q_lims=True,
                        distributed_slack=True
                    )
                    print(f"{grid_name}: OPF converged successfully with init='results'!")
                    results.append((grid_name, "OPF Converged with init='results'"))

                # If still no convergence, record as not converged
                except pp.optimal_powerflow.OPFNotConverged:
                    print(f"{grid_name}: OPF did not converge with any init method")
                    results.append((grid_name, "OPF Not Converged"))

    except pp.LoadflowNotConverged:
        print(f"{grid_name}: Power Flow did not converge.")
        results.append((grid_name, "PF Not Converged"))

print("\n" + "-" * 50 + "\n")

# Display the summary of results
print("\n--- Summary of OPF Convergence Tests ---\n")
df_results = pd.DataFrame(results, columns=["Grid", "Status"])

# Sort the results for better readability
df_results.sort_values(by="Grid", inplace=True)

# Display the complete summary table
print(df_results.to_string(index=False))
df_results.to_excel("Results OPP.xlsx", sheet_name="Results", index=False)
