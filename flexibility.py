import pandapower as pp
import numpy as np
import pandas as pd
from pandapower import runopp
from initialize import add_indicator


def calculate_flexibility(net_flex):

    dflexresults = pd.DataFrame(columns=['Indicator', 'Value'])
    pp.runpp(net_flex)
    flex2 = calculate_net_flexwork_reserve(net_flex)
    dflexresults = add_indicator(dflexresults, 'Flex Reserve Leitungen', flex2)
    flex4_normed, flex4 = calculate_loadflow_reserve(net_flex)
    dflexresults = add_indicator(dflexresults, 'Flex Reserve krit Leitungen scaled', flex4_normed)
    dflexresults = add_indicator(dflexresults, 'Flex Reserve krit Leitungen', flex4)
    flex_index= (flex2 + flex4_normed) / 2
    dflexresults = add_indicator(dflexresults, 'Flexibilität Gesamt', flex_index)

    return dflexresults

def calculate_net_flexwork_reserve(net_flexb):
    """ Berechnet_flex den Anteil der ungenutzten Kapazität im net_flexz. """
    # Lade thermische Grenzen der Leitungen
    S_max = net_flexb.line.max_i_ka * np.sqrt(3) * net_flexb.bus.vn_kv  # Annahme: U = vn_kv
    S_max_total = np.sum(S_max)

    # Berechne aktuelle Belastung
    S_current = np.sqrt(net_flexb.res_line.p_from_mw ** 2 + net_flexb.res_line.q_from_mvar ** 2)
    S_current_total = np.sum(S_current)

    # Flexibilität = Verhältnis ungenutzter Kapazität der Leitungen zu Gesamt-Kapazität
    if S_max_total > 0:
        flexibility_index = max(0, (S_max_total - S_current_total) / S_max_total)
    else:
        flexibility_index = 0  # Falls keine Kapazitätswerte vorhanden sind

    return flexibility_index

def calculate_loadflow_reserve(net_flexd):
    """ Berechnet_flex den durchschnittlichen freien Kapazitätsanteil auf kritischen Leitungen. """
    S_max = net_flexd.line.max_i_ka * np.sqrt(3) * net_flexd.bus.vn_kv  # Maximale Scheinleistung
    S_max[S_max == 0] = 1e-6  # Verhindere Division durch Null

    # Aktuelle Belastung
    p = net_flexd.res_line.p_from_mw.fillna(0)
    q = net_flexd.res_line.q_from_mvar.fillna(0)
    S_current = np.sqrt(p ** 2 + q ** 2)

    # Berechnung der Reserve
    reserve = S_max - S_current

    # Kritische Leitungen finden (z.B. jene mit mehr als 80% Auslastung)
    critical_lines = S_current / S_max > 0.8

    critical_reserves = reserve[critical_lines]

    # Durchschnittliche Reserve berechnen
    avg_reserve = np.mean(critical_reserves) if len(critical_reserves) > 0 else np.mean(reserve)

    # Normierung
    flex_index = avg_reserve / np.sum(S_max[critical_lines]) if avg_reserve > 0 else 0

    # Begrenzung auf [0,1]
    flex_index = max(0, min(1, flex_index))

    return flex_index, avg_reserve