import pandapower as pp
import pandapower.networks as pn
import pandas as pd
import pandapower.converter as pc
import os
import simbench as sb

# Example placeholders for external feeders
# You can adjust filenames/paths to match where you store the .m, .raw, .json, or .xlsx files.
# 1) Define the directory where your test grid files are located / stored:
TESTGRID_PATH = r"C:\Users\runte\Dropbox\Zwischenablage\Regression_Plots\Testnetze"

def load_case_pge69bus():
    """Lädt casePGE69BUS.m als MATPOWER-Fall (falls es wirklich ein MATPOWER-File ist)."""
    file_path = os.path.join(TESTGRID_PATH, "casePGE69BUS.m")
    return pc.from_mpc(file_path, casename_m="casePGE69BUS")

def load_case_pge69bus_dg():
    """Lädt casePGE69BUS_DG.m als MATPOWER-Fall."""
    file_path = os.path.join(TESTGRID_PATH, "casePGE69BUS_DG.m")
    return pc.from_mpc(file_path, casename_m="casePGE69BUS_DG")

def load_case_pge69bus_dg_topo():
    """Lädt casePGE69BUS_DG_topo.m als MATPOWER-Fall."""
    file_path = os.path.join(TESTGRID_PATH, "casePGE69BUS_DG_topo.m")
    return pc.from_mpc(file_path, casename_m="casePGE69BUS_DG_topo")

def load_case_ieee37_dg():
    """Lädt caseIEEE37_DG.m als MATPOWER-Fall."""
    file_path = os.path.join(TESTGRID_PATH, "caseIEEE37_DG.m")
    return pc.from_mpc(file_path, casename_m="caseIEEE37_DG")

def load_cigre_mv_network():
    """Falls CigreMvNetwork.m wirklich eine MATPOWER-Datei ist, hier laden.
       Sonst ggf. manuell erstellen oder Kommentar.
    """
    file_path = os.path.join(TESTGRID_PATH, "CigreMvNetwork.m")
    return pc.from_mpc(file_path, casename_m="CigreMvNetwork")

def load_cigre_mv_network_onefeeder():
    file_path = os.path.join(TESTGRID_PATH, "CigreMvNetwork_onefeeder.m")
    return pc.from_mpc(file_path, casename_m="CigreMvNetwork_onefeeder")

def load_cigre_mv_network_onefeeder_der():
    file_path = os.path.join(TESTGRID_PATH, "CigreMvNetwork_onefeeder_DER.m")
    return pc.from_mpc(file_path, casename_m="CigreMvNetwork_onefeeder_DER")

# ----------------------------
# Define scenarios and codes
# ----------------------------
scenarios = ["0", "1", "2"]

# Basic single-voltage grid codes:
basic_codes = []
# EHV
#basic_codes.extend([f"1-EHV-mixed--{sc}-sw" for sc in scenarios])
# HV (2 variants)
basic_codes.extend([f"1-HV-mixed--{sc}-sw" for sc in scenarios])
basic_codes.extend([f"1-HV-urban--{sc}-sw" for sc in scenarios])
# # MV (4 variants)
basic_codes.extend([f"1-MV-rural--{sc}-sw" for sc in scenarios])
# basic_codes.extend([f"1-MV-semiurb--{sc}-sw" for sc in scenarios])
basic_codes.extend([f"1-MV-urban--{sc}-sw" for sc in scenarios])
basic_codes.extend([f"1-MV-comm--{sc}-sw" for sc in scenarios])
# # LV (6 variants)
# basic_codes.extend([f"1-LV-rural1--{sc}-sw" for sc in scenarios])
# basic_codes.extend([f"1-LV-rural2--{sc}-sw" for sc in scenarios])
# basic_codes.extend([f"1-LV-rural3--{sc}-sw" for sc in scenarios])
# basic_codes.extend([f"1-LV-semiurb4--{sc}-sw" for sc in scenarios])
# basic_codes.extend([f"1-LV-semiurb5--{sc}-sw" for sc in scenarios])
#basic_codes.extend([f"1-LV-urban6--{sc}-sw" for sc in scenarios])



def load_example_power_flow():
    """Prüfen, ob ExamplePowerFlow.m MATPOWER-Struktur hat.
       Wenn nicht, kann man es nicht 1:1 via from_mpc importieren.
    """
    file_path = os.path.join(TESTGRID_PATH, "ExamplePowerFlow.m")
    return pc.from_mpc(file_path, casename_m="ExamplePowerFlow")

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
        pp.create_poly_cost(net, gen_idx, "gen", cp1_eur_per_mw=0.01)

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

# tests all existing test grids and checks if the opp converges
test_grids = [

    # ----------------------------------------------------------------------------
    # Example pandapower Networks (shipped with pandapower)
    # ----------------------------------------------------------------------------
    #("example_simple", pn.example_simple),
    #("example_multivoltage", pn.example_multivoltage),

    # Neue Einträge für gefundene *.m Dateien (nur falls MATPOWER-kompatibel):
    #("casePGE69BUS", load_case_pge69bus),
    #("casePGE69BUS_DG", load_case_pge69bus_dg),
    #("casePGE69BUS_DG_topo", load_case_pge69bus_dg_topo),
    #("caseIEEE37_DG", load_case_ieee37_dg),
    #("CigreMvNetwork", load_cigre_mv_network),
    #("CigreMvNetwork_onefeeder", load_cigre_mv_network_onefeeder),
    #("CigreMvNetwork_onefeeder_DER", load_cigre_mv_network_onefeeder_der),
    #("ExamplePowerFlow", load_example_power_flow),

    # ----------------------------------------------------------------------------
    # Simple pandapower Test Networks
    # (uncomment if you want them)
    # ("panda_four_load_branch", pn.panda_four_load_branch),
    # ("four_loads_with_branches_out", pn.four_loads_with_branches_out),
    # ("simple_four_bus_system", pn.simple_four_bus_system),
    # ("simple_mv_open_ring_net", pn.simple_mv_open_ring_net),

    # ----------------------------------------------------------------------------
    # CIGRE Networks (built into pandapower)
    # ----------------------------------------------------------------------------
    #("create_cigre_network_hv", pn.create_cigre_network_hv),
    # ("create_cigre_network_mv", pn.create_cigre_network_mv),
    # ("create_cigre_network_mv_pv_wind", lambda: pn.create_cigre_network_mv(with_der="pv_wind")),
    # ("create_cigre_network_mv_all", lambda: pn.create_cigre_network_mv(with_der="all")),
    # ("create_cigre_network_lv", pn.create_cigre_network_lv),

    # ----------------------------------------------------------------------------
    # MV Oberrhein (built into pandapower)
    # ----------------------------------------------------------------------------
    #("mv_oberrhein", pn.mv_oberrhein),

    # ----------------------------------------------------------------------------
    # Built-in MATPOWER Cases (transmission & a few radial distribution)
    # ----------------------------------------------------------------------------
    # ("case4gs", pn.case4gs),
    # ("case5", pn.case5),
    # ("case6ww", pn.case6ww),
    # ("case9", pn.case9),
    # ("case14", pn.case14),
    # ("case24_ieee_rts", pn.case24_ieee_rts),
    # ("case30", pn.case30),
    # ("case33bw", pn.case33bw),  # 33-bus radial distribution feeder
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
    #("iceland", pn.iceland),

    # ----------------------------------------------------------------------------
    # Synthetic Distribution Network from pandapower
    # ----------------------------------------------------------------------------
    #("create_synthetic_voltage_control_lv_network", pn.create_synthetic_voltage_control_lv_network),

    # ----------------------------------------------------------------------------
    # 3-Phase Grid Data
    # (uncomment if you need the asymmetric European LV feeder)
    # ----------------------------------------------------------------------------
    # ("ieee_european_lv_asymmetric", pn.ieee_european_lv_asymmetric),

    # ----------------------------------------------------------------------------
    # Kerber Networks (Average & Extreme) – built into pandapower
    # ----------------------------------------------------------------------------
    # ("create_kerber_landnetz_freileitung_1", pn.create_kerber_landnetz_freileitung_1),
    # ("create_kerber_landnetz_freileitung_2", pn.create_kerber_landnetz_freileitung_2),
    # ("create_kerber_landnetz_kabel_1", pn.create_kerber_landnetz_kabel_1),
    # ("create_kerber_landnetz_kabel_2", pn.create_kerber_landnetz_kabel_2),
    #("create_kerber_dorfnetz", pn.create_kerber_dorfnetz),
    # ("create_kerber_vorstadtnetz_kabel_1", pn.create_kerber_vorstadtnetz_kabel_1),
    # ("create_kerber_vorstadtnetz_kabel_2", pn.create_kerber_vorstadtnetz_kabel_2),

    # Extreme Kerber Networks
    # ("kb_extrem_landnetz_freileitung", pn.kb_extrem_landnetz_freileitung),
    # ("kb_extrem_landnetz_kabel", pn.kb_extrem_landnetz_kabel),
    # ("kb_extrem_landnetz_freileitung_trafo", pn.kb_extrem_landnetz_freileitung_trafo),
    # ("kb_extrem_landnetz_kabel_trafo", pn.kb_extrem_landnetz_kabel_trafo),
    # ("kb_extrem_dorfnetz", pn.kb_extrem_dorfnetz),
    # ("kb_extrem_dorfnetz_trafo", pn.kb_extrem_dorfnetz_trafo),
    # ("kb_extrem_vorstadtnetz_1", pn.kb_extrem_vorstadtnetz_1),
    # ("kb_extrem_vorstadtnetz_2", pn.kb_extrem_vorstadtnetz_2),
    # ("kb_extrem_vorstadtnetz_trafo_1", pn.kb_extrem_vorstadtnetz_trafo_1),

    # ----------------------------------------------------------------------------
    # IEEE Distribution Feeders (placeholders - you must have the files locally)
    # ----------------------------------------------------------------------------
    # ("ieee13_from_mpc", load_ieee_13_from_mpc),
    # ("ieee34_from_mpc", load_ieee_34_from_mpc),
    # ("ieee37_from_mpc", load_ieee_37_from_mpc),
    # ("ieee123_from_mpc", load_ieee_123_from_mpc),
    # ("ieee8500_from_mpc", load_ieee_8500_from_mpc),

    # ----------------------------------------------------------------------------
    # EPRI Distribution Feeders (placeholders - requires .m or .raw files)
    # ----------------------------------------------------------------------------
    # ("epri_j1", load_epri_j1),
    # ("epri_k1", lambda: pc.from_mpc("EPRI_K1.m")),
    # ("epri_l1", lambda: pc.from_mpc("EPRI_L1.m")),

    # ----------------------------------------------------------------------------
    # Additional Synthetic / Research Feeders (placeholders)
    # Provide your own load functions if you have .m/.raw/.json files
    # ----------------------------------------------------------------------------
    # ("tamu_synthetic_mv", lambda: pc.from_mpc("tamu_synthetic_mv.m")),
    # ("nrel_feeder_x", lambda: pc.from_mpc("nrel_feeder_x.m")),

]

# Get all available SimBench codes
all_codes = sb.collect_all_simbench_codes()

# # Filter for codes that represent combined MV+LV grids
# mv_lv_codes = [code for code in all_codes if "MVLV" in code]

# Optionally, you can print them to see what’s available:
#print("Found MV+LV codes:", mv_lv_codes)

# # Filter for combined HV+MV codes that only contain HV and MV (not extra levels)
hv_mv_codes = [code for code in all_codes if code.startswith("1-HVMV-")]
#
# # Filter for combined MV+LV codes that only contain MV and LV (not extra levels)
# mv_lv_codes = [code for code in all_codes if code.startswith("1-MVLV-")]

# print("Found HV+MV codes:", hv_mv_codes)
# print("Found MV+LV codes:", mv_lv_codes)

# Append these codes to the test_grids list:
for code in hv_mv_codes: #+ mv_lv_codes:
    test_grids.append((code, lambda code=code: sb.get_simbench_net(code)))
#
# # Append these MV+LV codes to your test_grids list
# for code in mv_lv_codes:
#     test_grids.append((
#         code,
#         lambda code=code: sb.get_simbench_net(code)
#     ))

# Combine all SimBench codes:
all_simbench_codes = basic_codes

# Store the results
results = []

# Now add each SimBench grid as (name, loader_function)
for sb_code in all_simbench_codes:
    test_grids.append((
        sb_code,
        # We wrap get_simbench_net in a lambda so it loads fresh each time you call it
        lambda code=sb_code: sb.get_simbench_net(code)
    ))

print("\n--- OPF Convergence Test for Pandapower Grids ---\n")

# Iterate through all test grids
for grid_name, grid_func in test_grids:
    print(f"Testing {grid_name} ...")
    # Create the network
    net = grid_func()
    net.switch['closed'] = True
    print(net.bus)

    # Process conventional generators (gen)
    for idx, gen in net.gen.iterrows():
        # Active power limits (P)
        if pd.isna(gen.get('max_p_mw')):
            net.gen.at[idx, 'max_p_mw'] = 2 * gen['p_mw']
        if pd.isna(gen.get('min_p_mw')):
            net.gen.at[idx, 'min_p_mw'] = 0
        if pd.isna(gen.get('max_q_mvar')):
            net.gen.at[idx, 'max_q_mvar'] = gen['p_mw']  # Can provide Q
        if pd.isna(gen.get('min_q_mvar')):
            net.gen.at[idx, 'min_q_mvar'] = - gen['p_mw']

    # Process static generators (sgen) - e.g., PV & Wind
    for idx, sgen in net.sgen.iterrows():
        if pd.isna(sgen.get('max_p_mw')):
            net.sgen.at[idx, 'max_p_mw'] = 2 * sgen['p_mw']
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

        # 🔹 Remove All Existing Cost Functions to Avoid Duplicates
    if not net.poly_cost.empty:
        net.poly_cost.drop(net.poly_cost.index, inplace=True)  # Clear existing cost functions

        # # 🔹 Add a Single Cost Function Per Generator
    for i, gen_idx in enumerate(net.gen.index):
        pp.create_poly_cost(net, gen_idx, "gen", cp1_eur_per_mw=0.01)

    # 🔹 Fix Voltage Limits (Prevent Collapse)
    net.bus["min_vm_pu"] = 0.9  # Min Voltage = 0.9 p.u.
    net.bus["max_vm_pu"] = 1.1  # Max Voltage = 1.1 p.u.

    # 🔹 **Reduce Line & Transformer Losses**
    net.line["max_i_ka"] *= 1.5  # increase line limits by 50 %

    if "max_loading_percent"  in net.line.columns:
        # Set a default loading percent for all lines (e.g., 100%)
        net.line["max_loading_percent"] *=1.5


    try:
        # Run Power Flow first
        pp.runpp(net)
        #print(net.trafo)
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
                    calculate_voltage_angles=False,
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
