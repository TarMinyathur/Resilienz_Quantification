import pandapower as pp
import pandapower.networks as pn
import pandapower.optimal_powerflow as opf
import numpy as np

from adjustments import set_missing_limits
from adjustments import determine_minimum_ext_grid_power
from self_sufficiency import selfsuff
from self_sufficiency import selfsufficiency_neu
from buffer import calculate_buffer

#initialize test grids from CIGRE; either medium voltage including renewables or the low voltage grid
# net = pn.create_cigre_network_lv()
net = pn.create_cigre_network_mv('all')
# False, 'pv_wind', 'all'

net, required_p_mw, required_q_mvar = determine_minimum_ext_grid_power(net)
#print(net.poly_cost)

net = set_missing_limits(net, required_p_mw, required_q_mvar)

# Anzahl der Monte-Carlo-Simulationen
N = 5

# Unsicherheitsbereich für Last & Erzeugung (+/-10%)
variation_percent = 0.00009  

# Liste zur Speicherung der Ergebnisse
flex_index_values = []

# Originalwerte für Skalierung speichern
orig_load_p = net.load.p_mw.copy()  # Original Last
orig_sgen_p = net.sgen.p_mw.copy()  # Original PV/Wind


for _ in range(N):
    # Zufällige Variation der Lasten & Einspeisungen
    net.load.p_mw = orig_load_p * (1 + np.random.uniform(-variation_percent, variation_percent, size=len(orig_load_p)))
    net.sgen.p_mw = orig_sgen_p * (1 + np.random.uniform(-variation_percent, variation_percent, size=len(orig_sgen_p)))

    # OPF durchführen (Optimal Power Flow)
    try:
        net, required_p_mw, required_q_mvar = determine_minimum_ext_grid_power(net)
        #print(net.poly_cost)

        net = set_missing_limits(net, required_p_mw, required_q_mvar)
    except:
        print("no")
        continue  # Falls OPF nicht konvergiert, überspringen

    # Scheinleistung berechnen (S = sqrt(P^2 + Q^2))
    s_total = np.sqrt(net.res_bus.p_mw**2 + net.res_bus.q_mvar**2).sum()

    # Flexibilitätsindikator speichern
    flex_index_values.append(s_total)

# Berechnung des finalen Flexibilitätsindikators
flex_index = np.sum(flex_index_values) / N

print(f"Flexibilitätsindikator: {flex_index:.9f}")