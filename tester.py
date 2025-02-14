import pandapower as pp
import pandapower.networks as pn
import pandas as pd
import numpy as np

# Create the CIGRE MV network with all DERs
net = pn.create_cigre_network_mv(with_der="all")

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
    min_vm_pu, max_vm_pu = 0.8, 1.2  #
    # Falls keine Grenzen gesetzt sind, setzen
    if pd.isna(bus.get('min_vm_pu')):
        net.bus.at[idx, 'min_vm_pu'] = min_vm_pu
    if pd.isna(bus.get('max_vm_pu')):
        net.bus.at[idx, 'max_vm_pu'] = max_vm_pu

print(net.ext_grid.columns)
# Ensure limits are set properly in ext_grid

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
        net.ext_grid.at[idx, 'min_q_mvar'] = -200000 # Größerer Q-Bezug (-80%)

print(net.ext_grid.columns)
# Run power flow simulation
try:
    pp.runpp(net)
    convergence_status = "Power flow converged successfully!"
except pp.pandapowerException as e:
    convergence_status = f"Power flow did not converge. Error: {str(e)}"

# Check OPF feasibility (if needed)
try:
    pp.runopp(
       net,
       init="pf",
       calculate_voltage_angles=True,
       enforce_q_lims=True,
       distributed_slack=True
    )
    opf_status = "Optimal Power Flow (OPF) converged successfully!"
except pp.pandapowerException as e:
    opf_status = f"OPF did not converge. Error: {str(e)}"

print(convergence_status)
print(opf_status)

