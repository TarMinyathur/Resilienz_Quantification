#!/usr/bin/env python3

import itertools
import pandapower as pp
import networkx as nx
from concurrent.futures import ProcessPoolExecutor
import time
from Redundancy import is_graph_connected
import random
import pandapower.networks as pn
from itertools import islice

# Schwellenwerte definieren
CRITICAL_THRESHOLD = 90  # in Prozent, z. B. für Leitungen und Transformatoren
UNDER_VOLTAGE = 0.95     # p.u.
OVER_VOLTAGE = 1.05      # p.u.

"""
Dieses Skript berechnet Indikatoren für einzelne Netzkomponenten (z. B. Leitungen, Transformatoren, Busse, sgen, storage) 
auf Basis der Lastfluss- und N‑2‑Redundanzanalyse. 
Die Indikatoren werden so definiert, dass 1 das bestmögliche Ergebnis darstellt und 0 das schlechtestmögliche.
Lastfluss und Redundanz werden dabei gleich gewichtet; im Lastfluss fließen sowohl die durchschnittliche Auslastung 
als auch der Anteil kritischer Elemente ein.
"""

def Redundancy(net_temp_red3, max_calls):
    # # Beispielnetz erstellen (IEEE 9-Bus-System)

    # Lastflussberechnung durchführen
    pp.runpp(net_temp_red3)

    print("\nLastflussanalyse:")

    # Ergebnisse in lf_resultsb speichern:
    lf_resultsb = {}
    # Auswertung der Auslastungen:
    lf_resultsb["line"] = analyze_loading(net_temp_red3.res_line[['loading_percent']], "Leitungen")
    lf_resultsb["trafo"] = analyze_loading(net_temp_red3.res_trafo[['loading_percent']], "Transformatoren")

    # Busspannungsauswertung:
    lf_resultsb["bus"] = analyze_buses(net_temp_red3.res_bus[['vm_pu']])

    # Erzeugeranalyse (Generator-Pmax aus net_temp_red3.gen):
    lf_resultsb["gen"] = analyze_components_gen(net_temp_red3.res_gen[['p_mw']], net_temp_red3.gen[['max_p_mw']], net_temp_red3.res_sgen[['p_mw']], net_temp_red3.sgen[['max_p_mw']], net_temp_red3.res_storage[['p_mw']], net_temp_red3.storage[['max_p_mw']])

    # Ordentliche, formatierte Ausgabe:
    print("Ergebnisse der Lastflussanalyse:\n" + "-" * 40)
    for comp, resultsb in lf_resultsb.items():
        print(f"{comp}:")
        for key, value in resultsb.items():
            print(f"  {key}: {value}")
        print("-" * 40)

    print("\nStarte N-2-Redundanzprüfung...")
    # Liste der zu prüfenden Elementtypen
    element_types = ["line", "sgen", "gen", "trafo", "bus", "storage", "switch", "load"]

    n2_redundancy_results = {}
    Success = 0
    Failed = 0
    timeout = 300

    # Über alle relevanten Elementtypen iterieren
    for element_type in element_types:
        start_time = time.time()
        results = n_2_redundancy_check(net_temp_red3, start_time, element_type, timeout, max_calls)
        n2_redundancy_results[element_type] = results[element_type]

        # Summiere die Ergebnisse
        Success += results[element_type]['Success']
        Failed += results[element_type]['Failed']
        print(time.time() - start_time)

    # Gesamtrate berechnen
    total_checks = Success + Failed
    red_resultsb  = Success / total_checks if total_checks != 0 else 0

        # Berechnung der Komponentenindikatoren:
    component_indicators = {}

    # Leitungen:
    lf_line = compute_lf_indicator(
        avg_loading=lf_resultsb["line"]["avg_loading"],
        num_crit=lf_resultsb["line"]["num_crit"],
        total=lf_resultsb["line"]["total"]
    )
    red_line = compute_red_indicator(
        Success=n2_redundancy_results["line"]["Success"],
        Fail=n2_redundancy_results["line"]["Failed"]
    )
    component_indicators["line"] = {
        "lf": lf_line,
        "red": red_line,
        "combined": combine_indicators(lf_line, red_line)
    }

    # Transformatoren:
    lf_trafo = compute_lf_indicator(
        avg_loading=lf_resultsb["trafo"]["avg_loading"],
        num_crit=lf_resultsb["trafo"]["num_crit"],
        total=lf_resultsb["trafo"]["total"]
    )
    red_trafo = compute_red_indicator(
        Success=n2_redundancy_results["trafo"]["Success"],
        Fail=n2_redundancy_results["trafo"]["Failed"]
    )
    component_indicators["trafo"] = {
        "lf": lf_trafo,
        "red": red_trafo,
        "combined": combine_indicators(lf_trafo, red_trafo)
    }

    # Busse:
    lf_bus = compute_bus_lf_indicator(
        avg_voltage=lf_resultsb["bus"]["avg_voltage"],
        under=lf_resultsb["bus"]["under"],
        over=lf_resultsb["bus"]["over"],
        total=lf_resultsb["bus"]["total"]
    )
    red_bus = compute_red_indicator(
        Success=n2_redundancy_results["bus"]["Success"],
        Fail=n2_redundancy_results["bus"]["Failed"]
    )
    component_indicators["bus"] = {
        "lf": lf_bus,
        "red": red_bus,
        "combined": combine_indicators(lf_bus, red_bus)
    }

    # sgen und storage besitzen in unserem Beispiel keine Lastflussdaten.
    # Daher verwenden wir für den Lastflussindikator hier default-mäßig 1 (optimal),
    # sodass der kombinierte Indikator allein von der Redundanz abhängt.

    for comp in ["sgen", "storage"]:
        red_comp = compute_red_indicator(
            Success=n2_redundancy_results[comp]["Success"],
            Fail=n2_redundancy_results[comp]["Failed"]
        )
        component_indicators[comp] = {
            "lf": 1.0,
            "red": red_comp,
            "combined": combine_indicators(1.0, red_comp)
        }

    # Berechnung der Gesamtindikatoren
    # Wir berechnen hier den Durchschnitt der Lastflussindikatoren und der Redundanzindikatoren
    lf_total = 0
    red_total = 0
    count_lf = 0
    count_red = 0
    for comp, inds in component_indicators.items():
        # Falls LF-Daten vorhanden sind:
        if inds["lf"] is not None:
            lf_total += inds["lf"]
            count_lf += 1
        if inds["red"] is not None:
            red_total += inds["red"]
            count_red += 1

    overall_lf = lf_total / count_lf if count_lf > 0 else 0
    overall_red = red_total / count_red if count_red > 0 else 0
    overall_combined = (overall_lf + overall_red) / 2

    return overall_lf, overall_red, overall_combined, component_indicators, n2_redundancy_results


def compute_lf_indicator(avg_loading, num_crit, total):
    """
    Berechnet den Lastflussindikator für Komponenten, bei denen eine Auslastung (in %) und kritische Elemente vorliegen.
    Dabei gilt: Je niedriger die durchschnittliche Auslastung (normiert als avg_loading/100) und je geringer der Anteil kritischer Elemente,
    desto besser. Die beiden Teildimensionen werden gleichgewichtet.

    Rückgabe:
        Ein Wert zwischen 0 (schlecht) und 1 (optimal).
    """
    if total <= 0:
        return 1  # Falls keine Daten vorliegen, wird das als optimal gewertet.
    fraction_crit = num_crit / total
    normalized_loading = avg_loading / 100.0
    return ((1 - normalized_loading) + (1 - fraction_crit)) / 2


def compute_bus_lf_indicator(avg_voltage, under, over, total, voltage_tolerance=0.05):
    """
    Berechnet den Lastflussindikator für Busse.
    Hier wird angenommen, dass 1.0 p.u. optimal ist.
    Zwei Aspekte fließen ein:
      - Eine Spannungsabweichung: Je näher avg_voltage an 1.0 liegt, desto besser.
        Der Indikatoranteil wird als: 1 - (Abweichung / voltage_tolerance) berechnet und auf [0,1] begrenzt.
      - Der Anteil der Buse, die außerhalb des zulässigen Bereichs (unter/über) liegen.

    Rückgabe:
        Ein Wert zwischen 0 (schlecht) und 1 (optimal).
    """
    if total <= 0:
        return 1
    voltage_deviation = abs(1.0 - avg_voltage)
    voltage_indicator = max(0, 1 - (voltage_deviation / voltage_tolerance))
    ratio_indicator = 1 - ((under + over) / total)
    return (voltage_indicator + ratio_indicator) / 2


def compute_red_indicator(Success, Fail):
    """
    Berechnet den Redundanzindikator aus den Ergebnissen der N-2-Prüfung.
    Der Indikator entspricht der Erfolgsquote:
        Success / (Success + Fail)
    Falls keine Prüfungen durchgeführt wurden, wird als 0 angenommen.
    """
    total = Success + Fail
    return Success / total if total > 0 else 0


def combine_indicators(lf_ind, red_ind):
    """
    Kombiniert den Lastfluss- und den Redundanzindikator zu einem Gesamtindex,
    wobei beide gleich gewichtet werden.
    """
    return (lf_ind + red_ind) / 2

"""
Dieses Skript führt eine Lastflussberechnung mit pandapower durch und gibt
die durchschnittliche Auslastung sowie die Anzahl der kritischen Elemente für:
- Leitungen
- Transformatoren
- Busspannungen
- Erzeugeranlagen
"""

def analyze_loading(loading_data, element_name):
    """
    Berechnet die durchschnittliche Auslastung, die Anzahl kritischer Elemente und die Gesamtanzahl.
    Gibt die Ergebnisse als Dictionary zurück.
    """
    if loading_data is not None and not loading_data.empty:
        avg_loading = loading_data.mean()[0]
        critical_count = (loading_data >= CRITICAL_THRESHOLD).sum()[0]
        total_count = len(loading_data)
        result = {
            "avg_loading": avg_loading,
            "num_crit": critical_count,
            "total": total_count
        }
    else:
        result = {"avg_loading": None, "num_crit": None, "total": 0}
    return result


def analyze_buses(bus_data):
    """
    Berechnet Kennzahlen zur Busspannung:
      - Durchschnittliche Spannung
      - Anzahl Busse mit Unterspannung (< UNDER_VOLTAGE)
      - Anzahl Busse mit Überspannung (> OVER_VOLTAGE)
      - Gesamtanzahl der Busse
    Gibt die Ergebnisse als Dictionary zurück.
    """
    if bus_data is not None and not bus_data.empty:
        avg_voltage = bus_data.mean()[0]
        under_voltage_count = (bus_data < UNDER_VOLTAGE).sum()[0]
        over_voltage_count = (bus_data > OVER_VOLTAGE).sum()[0]
        total_count = len(bus_data)
        result = {
            "avg_voltage": avg_voltage,
            "under": under_voltage_count,
            "over": over_voltage_count,
            "total": total_count
        }
    else:
        result = {"avg_voltage": None, "under": 0, "over": 0, "total": 0}
    return result


def analyze_components_gen(gen_data, gen_max, sgen_data=None, sgen_max=None, stor_data=None, stor_max=None):
    """
    Berechnet die prozentuale Auslastung von Generatoren, statischen Generatoren (sgen) und Speichern (storages).
    Kritisch wird eine Komponente bewertet, wenn sie über 95 % ihrer Kapazität arbeitet.
    Gibt die Ergebnisse als Dictionary zurück.
    """

    def analyze(data, max_data, name):
        if data is not None and not data.empty and max_data is not None and not max_data.empty:
            loading = (data / max_data) * 100
            avg_loading = loading.mean()[0]
            critical_threshold = 95
            critical_count = (loading >= critical_threshold).sum()[0]
            total_count = len(loading)
            return {
                "avg_loading": avg_loading,
                "num_crit": critical_count,
                "total": total_count
            }
        else:
            return {"avg_loading": None, "num_crit": None, "total": 0}

    return {
        "generators": analyze(gen_data, gen_max, "generators"),
        "sgen": analyze(sgen_data, sgen_max, "sgen"),
        "storages": analyze(stor_data, stor_max, "storages")
    }

"""
Dieses Skript führt eine N-2-Redundanzprüfung mit pandapower durch.
"""


def n_2_redundancy_check(net_temp_red2, start_time, element_type, timeout, max_calls=250):
    """Überprüft die N-2-Redundanz für verschiedene Netzkomponenten."""
    if element_type not in ["line", "sgen", "gen", "trafo", "bus", "storage", "switch", "load"]:
        raise ValueError(f"Invalid element type for n_2 redundancy: {element_type}")

    resultsa = {element_type: {"Success": 0, "Failed": 0}}

    # Early return if there are fewer than 2 elements
    if net_temp_red2[element_type].empty or len(net_temp_red2[element_type].index) < 2:
        return resultsa

    # Extract the indices and shuffle them if you want random pairs
    index_list = list(net_temp_red2[element_type].index)
    random.shuffle(index_list)

    # Create a generator for all 2-element combinations
    element_pairs_gen = itertools.combinations(index_list, 2)

    should_stop_n2 = False

    # Parallelisierung der Netzberechnungen
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = []

        for pairs in islice(element_pairs_gen, max_calls):
            if should_stop_n2:
                break

            net_temp_red2_n2_copy = net_temp_red2.deepcopy()
            futures.append(executor.submit(process_pair, element_type, pairs, net_temp_red2_n2_copy))

            # Check timeout after each task submission
            if (time.time() - start_time) > timeout:
                print("Timeout reached. Ending process.")
                should_stop_n2 = True
                break

        for future in futures:
            element_type_returned, status = future.result()
            resultsa[element_type_returned][status] += 1

    return resultsa


def process_pair(element_type, pair, net_temp_red2):
    """Setzt zwei Elemente außer Betrieb und überprüft die Netzstabilität."""
    for element_id in pair:
        net_temp_red2[element_type].at[element_id, 'in_service'] = False

    out_of_service_elements = {element_type: pair}

    # Prüfe, ob das Netz nach dem Ausfall noch verbunden ist
    if not is_graph_connected(net_temp_red2, out_of_service_elements):
        return element_type, "Failed"

    # Versuche eine Lastflussberechnung
    try:
        pp.runopp(
            net_temp_red2,
            init="pf",
            calculate_voltage_angles=True,
            enforce_q_lims=True,
            distributed_slack=True
        )
        return element_type, "Success"

    except (pp.optimal_powerflow.OPFNotConverged, pp.powerflow.LoadflowNotConverged):
        #print(f"OPF mit init='pf' für {element_type} fehlgeschlagen, versuche init='flat'")
        try:
            pp.runopp(
                net_temp_red2,
                init="flat",
                calculate_voltage_angles=True,
                enforce_q_lims=True,
                distributed_slack=True
            )
            return element_type, "Success"
        except (pp.optimal_powerflow.OPFNotConverged, pp.powerflow.LoadflowNotConverged):
            #print(f"OPF mit init='flat' für {element_type} fehlgeschlagen")
            return element_type, "Failed"
        except Exception as e:
            #print(f"Fehler bei {element_type} mit Paar {pair}: {e}")
            return element_type, "Failed"
    except Exception as e:
        #print(f"Fehler bei {element_type} mit Paar {pair}: {e}")
        return element_type, "Failed"

if __name__ == "__main__":
    net_temp_stress = pn.create_cigre_network_mv(with_der="all")
    red = Redundancy(net_temp_stress)
