import pandapower as pp
import pandapower.networks as pn
import pandapower.optimal_powerflow as opf
import numpy as np

from adjustments import set_missing_limits
from adjustments import determine_minimum_ext_grid_power


def calculate_flexibility(net):
    net, required_p_mw, required_q_mvar = determine_minimum_ext_grid_power(net)

    net = set_missing_limits(net, required_p_mw, required_q_mvar)

    # Anzahl Monte-Carlo-Simulationen
    N = 5

    # Unsicherheitsbereich für Last & Erzeugung (+/-10%)
    variation_percent = 0.1  

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

            net = set_missing_limits(net, required_p_mw, required_q_mvar)
        except:
            print("no")
            continue  # Falls OPF nicht konvergiert, überspringen


        # check wie viel externer Bezug

        
        # Scheinleistung (S = sqrt(P^2 + Q^2))
        s_total = np.sqrt(net.res_bus.p_mw**2 + net.res_bus.q_mvar**2).sum()

        flex_index_values.append(s_total)

    # Berechnung Flexibilitätsindikators = 1/N * Sum Si
    flex_index = np.sum(flex_index_values) / N

    return flex_index


if __name__ == "__main__":
    net = pn.create_cigre_network_mv(with_der="all")
    flex_index = calculate_flexibility(net)
    print(f"Flexibilitätsindikator: {flex_index:.9f}")