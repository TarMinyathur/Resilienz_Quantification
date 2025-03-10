import pandapower as pp
import numpy as np
import pandas as pd
from pandapower import runopp

from adjustments import set_missing_limits
from adjustments import determine_minimum_ext_grid_power

from initialize import add_indicator


def calculate_flexibility(net_flex):

    dflexresults = pd.DataFrame(columns=['Indicator', 'Value'])
    flex1 = calculate_flexibility_monte(net_flex)
    dflexresults = add_indicator(dflexresults, 'Flex Monte Carlo', flex1)

    flex2 = calculate_net_flexwork_reserve(net_flex)
    dflexresults = add_indicator(dflexresults, 'Flex Netzreserve', flex2)
    flex3 = calculate_opf_success_rate(net_flex)
    dflexresults = add_indicator(dflexresults, 'Flex Erfolgreiche OPP', flex3)
    flex4 = calculate_loadflow_reserve(net_flex)
    dflexresults = add_indicator(dflexresults, 'Flex Reserve krit Leitungen', flex4)

    flex_index= (flex1 + flex2 + flex3 + flex4) / 4
    dflexresults = add_indicator(dflexresults, 'Flexibilität Gesamt', flex_index)

    return dflexresults


def calculate_flexibility_monte(net_flexa):
    """ Berechnet den normierten Flexibilitätsindex zwischen 0 und 1. """

    # net_flexa, required_p_mw, required_q_mvar = determine_minimum_ext_grid_power(net_flexa)
    # net_flexa = set_missing_limits(net_flexa, required_p_mw, required_q_mvar)

    # Anzahl Monte-Carlo-Simulationen
    N = 50
    variation_percent = 0.1  # +/-10% Variation

    flex_index_values = []
    orig_load_p = net_flexa.load.p_mw.copy()
    orig_sgen_p = net_flexa.sgen.p_mw.copy()

    # Berechnung der maximalen externen Einspeisung als Normierungsbasis
    max_ext_grid_power = np.abs(net_flexa.ext_grid["s_sc_max_mva"].sum()) if not net_flexa.ext_grid.empty else 1e-6

    for _ in range(N):
        # Zufällige Variation von Lasten & Einspeisungen
        net_flexa.load.p_mw = orig_load_p * (1 + np.random.uniform(-variation_percent, variation_percent, size=len(orig_load_p)))
        net_flexa.sgen.p_mw = orig_sgen_p * (1 + np.random.uniform(-variation_percent, variation_percent, size=len(orig_sgen_p)))

        # OPF durchführen
        try:
            runopp(net_flexa)
        except:
            print("no")
            continue  # Falls OPF nicht konvergiert, überspringen

        # Berechnung der Gesamtscheinleistung (Netzbelastung)
        s_total = np.sqrt(net_flexa.res_bus.p_mw ** 2 + net_flexa.res_bus.q_mvar ** 2).sum()
        flex_index_values.append(s_total)

    # Mittelwert der Flexibilität berechnen
    flexi_index = np.sum(flex_index_values) / N

    # Normierung der Flexibilität auf den Bereich [0,1]
    flexi_index_norm = max(0, min(1, 1 - (flexi_index / max_ext_grid_power)))

    return flexi_index_norm


def calculate_net_flexwork_reserve(net_flexb):
    """ Berechnet_flex den Anteil der ungenutzten Kapazität im net_flexz. """
    # Lade thermische Grenzen der Leitungen
    S_max = net_flexb.line.max_i_ka * np.sqrt(3) * net_flexb.bus.vn_kv  # Annahme: U = vn_kv
    S_max_total = np.sum(S_max)

    # Berechne aktuelle Belastung
    S_current = np.sqrt(net_flexb.res_line.p_from_mw ** 2 + net_flexb.res_line.q_from_mvar ** 2)
    S_current_total = np.sum(S_current)

    # Flexibilität = Verhältnis ungenutzter Kapazität zu Gesamt-Kapazität
    if S_max_total > 0:
        flexibility_index = (S_max_total - S_current_total) / S_max_total
    else:
        flexibility_index = 0  # Falls keine Kapazitätswerte vorhanden sind

    return flexibility_index

def calculate_opf_success_rate(net_flexc, N=100, variation_percent=0.1):
    """ Simuliert N verschiedene Lastszenarien und berechnet_flex die Erfolgsrate der OPF-Lösung. """
    successful_cases = 0

    orig_load_p = net_flexc.load.p_mw.copy()
    orig_sgen_p = net_flexc.sgen.p_mw.copy()

    for _ in range(N):
        # Zufällige Variation der Lasten & Einspeisungen
        net_flexc.load.p_mw = np.maximum(0, orig_load_p * (1 + np.random.uniform(-variation_percent, variation_percent, len(orig_load_p))))
        net_flexc.sgen.p_mw = np.maximum(0, orig_sgen_p * (1 + np.random.uniform(-variation_percent, variation_percent, len(orig_sgen_p))))

        try:
            pp.runopp(net_flexc)
            successful_cases += 1  # OPF erfolgreich
        except:
            continue  # OPF fehlgeschlagen, ignoriere dieses Szenario

    # Erfolgsrate berechnen
    success_rate = successful_cases / N
    return success_rate

def calculate_loadflow_reserve(net_flexd):
    """ Berechnet_flex den durchschnittlichen freien Kapazitätsanteil auf kritischen Leitungen. """
    S_max = net_flexd.line.max_i_ka * np.sqrt(3) * net_flexd.bus.vn_kv  # Maximale Scheinleistung
    S_max[S_max == 0] = 1e-6  # Verhindere Division durch Null

    # Aktuelle Belastung
    S_current = np.sqrt(net_flexd.res_line.p_from_mw**2 + net_flexd.res_line.q_from_mvar**2)

    # Berechnung der Reserve
    reserve = S_max - S_current

    # Kritische Leitungen finden (z.B. jene mit mehr als 80% Auslastung)
    critical_lines = S_current / S_max > 0.8

    # Durchschnittliche freie Reserve auf kritischen Leitungen
    if np.sum(critical_lines) > 0:
        avg_reserve = np.mean(reserve[critical_lines])
    else:
        avg_reserve = np.mean(reserve)  # Falls keine Engpässe vorhanden sind, gesamte Reserve nutzen

    # Normierung: Maximale Summe aller freien Kapazitäten
    max_possible_reserve = np.sum(S_max)

    if max_possible_reserve > 0:
        flex_index = avg_reserve / max_possible_reserve
    else:
        flex_index = 0  # Falls keine sinnvolle Reserve existiert, Flexibilität = 0 setzen

    # Begrenzung auf den Wertebereich [0,1]
    flex_index = max(0, min(1, flex_index))

    return flex_index