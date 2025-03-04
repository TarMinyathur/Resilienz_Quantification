#!/usr/bin/env python3


import itertools
import time
import pandapower as pp
import pandapower.networks as pn
import networkx as nx
from concurrent.futures import ThreadPoolExecutor
from adjustments import set_missing_limits
from adjustments import determine_minimum_ext_grid_power

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
    Falls keine Prüfungen durchgeführt wurden, wird als optimal (1) angenommen.
    """
    total = Success + Fail
    return Success / total if total > 0 else 1


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


def analyze_generators(gen_data, gen_max):
    """
    Berechnet die prozentuale Auslastung der Generatoren anhand der aktuellen Leistung (p_mw)
    im Verhältnis zur maximalen Leistung (max_p_mw).
    Kritisch wird z. B. ein Generator bewertet, wenn er über 95 % seiner Kapazität arbeitet.
    Gibt die Ergebnisse als Dictionary zurück.
    """
    if gen_data is not None and not gen_data.empty and gen_max is not None and not gen_max.empty:
        # Berechne die Auslastung in Prozent
        loading = (gen_data / gen_max) * 100
        avg_loading = loading.mean()[0]
        critical_threshold = 95  # z. B. 95% als Schwelle für kritische Auslastung
        critical_count = (loading >= critical_threshold).sum()[0]
        total_count = len(loading)
        result = {
            "avg_loading": avg_loading,
            "num_crit": critical_count,
            "total": total_count
        }
    else:
        result = {"avg_loading": None, "num_crit": None, "total": 0}
    return result

"""
Dieses Skript führt eine N-2-Redundanzprüfung mit pandapower durch.
"""

def n_2_redundancy_check(net_temp, element_counts, timeout=900):
    """Überprüft die N-2-Redundanz für verschiedene Netzkomponenten."""
    results = {
        "line": {"Success": 0, "Failed": 0},
        "sgen": {"Success": 0, "Failed": 0},
        "trafo": {"Success": 0, "Failed": 0},
        "bus": {"Success": 0, "Failed": 0},
        "storage": {"Success": 0, "Failed": 0}
    }

    # Erstelle Kombinationen von zwei Elementen pro Kategorie (N-2-Prüfung)
    element_pairs = {
        "line": list(itertools.combinations(net_temp.line.index, min(2, element_counts["scaled_counts"]["line"]))) if not net_temp.line.empty else [],
        "sgen": list(itertools.combinations(net_temp.sgen.index, min(2, element_counts["scaled_counts"]["sgen"]))) if not net_temp.sgen.empty else [],
        "trafo": list(itertools.combinations(net_temp.trafo.index, min(2, element_counts["scaled_counts"]["trafo"]))) if not net_temp.trafo.empty else [],
        "bus": list(itertools.combinations(net_temp.bus.index, min(2, element_counts["scaled_counts"]["bus"]))) if not net_temp.bus.empty else [],
        "storage": list(itertools.combinations(net_temp.storage.index, min(2, element_counts["scaled_counts"]["storage"]))) if not net_temp.storage.empty else []
    }

    start_time = time.time()
    should_stop = False

    # Parallelisierung der Netzberechnungen
    with ThreadPoolExecutor() as executor:
        futures = []

        for element_type, pairs in element_pairs.items():
            if should_stop:
                break

            for pair in pairs:
                net_copy = net_temp.deepcopy()
                futures.append(executor.submit(process_pair, element_type, pair, net_copy))

                # Timeout-Überprüfung
                if (time.time() - start_time) > timeout:
                    print("Timeout erreicht. Beende den Prozess.")
                    should_stop = True
                    break

        for future in futures:
            element_type, status = future.result()
            results[element_type][status] += 1

    return results


def process_pair(element_type, pair, net_temp):
    """Setzt zwei Elemente außer Betrieb und überprüft die Netzstabilität."""
    for element_id in pair:
        net_temp[element_type].at[element_id, 'in_service'] = False

    out_of_service_elements = {element_type: pair}

    # Prüfe, ob das Netz nach dem Ausfall noch verbunden ist
    if not is_graph_connected(net_temp, out_of_service_elements):
        return element_type, "Failed"

    # Versuche eine Lastflussberechnung
    try:
        pp.runopp(
            net_temp,
            init="pf",
            calculate_voltage_angles=True,
            enforce_q_lims=True,
            distributed_slack=True
        )
        return element_type, "Success"

    except (pp.optimal_powerflow.OPFNotConverged, pp.powerflow.LoadflowNotConverged):
        print(f"OPF mit init='pf' für {element_type} fehlgeschlagen, versuche init='flat'")
        try:
            pp.runopp(
                net_temp,
                init="flat",
                calculate_voltage_angles=True,
                enforce_q_lims=True,
                distributed_slack=True
            )
            return element_type, "Success"
        except (pp.optimal_powerflow.OPFNotConverged, pp.powerflow.LoadflowNotConverged):
            print(f"OPF mit init='flat' für {element_type} fehlgeschlagen")
            return element_type, "Failed"
        except Exception as e:
            print(f"Fehler bei {element_type} mit Paar {pair}: {e}")
            return element_type, "Failed"
    except Exception as e:
        print(f"Fehler bei {element_type} mit Paar {pair}: {e}")
        return element_type, "Failed"


def is_graph_connected(net_temp, out_of_service_elements):
    """Prüft, ob das Netz nach dem Ausfall noch zusammenhängend ist."""
    G = nx.Graph()

    # Knoten (Busse) hinzufügen
    for bus in net_temp.bus.itertuples():
        if bus.Index not in out_of_service_elements.get('bus', []):
            G.add_node(bus.Index)

    # Leitungen hinzufügen (wenn sie nicht ausgefallen sind)
    for line in net_temp.line.itertuples():
        if line.Index not in out_of_service_elements.get('line', []):
            from_bus, to_bus = line.from_bus, line.to_bus

            # Prüfe, ob eine Schalterverbindung existiert
            switch_closed = any(
                (switch.bus == from_bus and switch.element == to_bus and switch.et == 'l' and switch.closed) or
                (switch.bus == to_bus and switch.element == from_bus and switch.et == 'l' and switch.closed)
                for _, switch in net_temp.switch.iterrows()
            )

            # Füge Kante hinzu, wenn kein offener Schalter dazwischen ist
            if not switch_closed:
                G.add_edge(from_bus, to_bus)

    # Transformatoren als Kanten hinzufügen
    for trafo in net_temp.trafo.itertuples():
        if trafo.Index not in out_of_service_elements.get('trafo', []):
            G.add_edge(trafo.hv_bus, trafo.lv_bus)

    # Prüfe, ob das Netz verbunden ist
    return nx.is_connected(G)


def main():
    # Beispielnetz erstellen (IEEE 9-Bus-System)
    net = pn.create_cigre_network_mv("all")
    net, required_p_mw, required_q_mvar = determine_minimum_ext_grid_power(net)
    net = set_missing_limits(net, required_p_mw, required_q_mvar)

    # Lastflussberechnung durchführen
    pp.runpp(net)

    print("\nLastflussanalyse:")

    # Ergebnisse in lf_results speichern:
    lf_results = {}
    # Auswertung der Auslastungen:
    lf_results["line"] = analyze_loading(net.res_line[['loading_percent']], "Leitungen")
    lf_results["trafo"] = analyze_loading(net.res_trafo[['loading_percent']], "Transformatoren")

    # Busspannungsauswertung:
    lf_results["bus"] = analyze_buses(net.res_bus[['vm_pu']])

    # Erzeugeranalyse (Generator-Pmax aus net.gen):
    lf_results["gen"] = analyze_generators(net.res_gen[['p_mw']], net.gen[['max_p_mw']])

    # Ordentliche, formatierte Ausgabe:
    print("Ergebnisse der Lastflussanalyse:\n" + "-" * 40)
    for comp, results in lf_results.items():
        print(f"{comp}:")
        for key, value in results.items():
            print(f"  {key}: {value}")
        print("-" * 40)

    element_counts = {
        "scaled_counts": {
            "line": len(net.line),
            "sgen": len(net.sgen),
            "trafo": len(net.trafo),
            "bus": len(net.bus),
            "storage": len(net.storage)
        }
    }

    print("\nStarte N-2-Redundanzprüfung...")
    red_results = n_2_redundancy_check(net, element_counts)

    # Ergebnisse ausgeben
    print("\nErgebnisse der N-2-Redundanzprüfung:")
    for element, stats in red_results.items():
        print(f"{element.capitalize()}: Erfolg: {stats['Success']}, Fehlgeschlagen: {stats['Failed']}")

        # Berechnung der Komponentenindikatoren:
    component_indicators = {}

    # Leitungen:
    lf_line = compute_lf_indicator(
        avg_loading=lf_results["line"]["avg_loading"],
        num_crit=lf_results["line"]["num_crit"],
        total=lf_results["line"]["total"]
    )
    red_line = compute_red_indicator(
        Success=red_results["line"]["Success"],
        Fail=red_results["line"]["Failed"]
    )
    component_indicators["line"] = {
        "lf": lf_line,
        "red": red_line,
        "combined": combine_indicators(lf_line, red_line)
    }

    # Transformatoren:
    lf_trafo = compute_lf_indicator(
        avg_loading=lf_results["trafo"]["avg_loading"],
        num_crit=lf_results["trafo"]["num_crit"],
        total=lf_results["trafo"]["total"]
    )
    red_trafo = compute_red_indicator(
        Success=red_results["trafo"]["Success"],
        Fail=red_results["trafo"]["Failed"]
    )
    component_indicators["trafo"] = {
        "lf": lf_trafo,
        "red": red_trafo,
        "combined": combine_indicators(lf_trafo, red_trafo)
    }

    # Busse:
    lf_bus = compute_bus_lf_indicator(
        avg_voltage=lf_results["bus"]["avg_voltage"],
        under=lf_results["bus"]["under"],
        over=lf_results["bus"]["over"],
        total=lf_results["bus"]["total"]
    )
    red_bus = compute_red_indicator(
        Success=red_results["bus"]["Success"],
        Fail=red_results["bus"]["Failed"]
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
            Success=red_results[comp]["Success"],
            Fail=red_results[comp]["Failed"]
        )
        component_indicators[comp] = {
            "lf": 1.0,
            "red": red_comp,
            "combined": combine_indicators(1.0, red_comp)
        }

    # Ausgabe der Indikatoren pro Komponente:
    print("Komponentenindikatoren (1 = optimal, 0 = schlecht):")
    for comp, inds in component_indicators.items():
        print(f"{comp.capitalize()}:")
        print(f"  Lastfluss: {inds['lf']:.3f}")
        print(f"  Redundanz: {inds['red']:.3f}")
        print(f"  Kombiniert: {inds['combined']:.3f}")

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

    overall_lf = lf_total / count_lf if count_lf > 0 else 1
    overall_red = red_total / count_red if count_red > 0 else 1
    overall_combined = (overall_lf + overall_red) / 2

    print("\nGesamtindikatoren:")
    print(f"  Lastfluss Gesamt: {overall_lf:.3f}")
    print(f"  N-2 Redundanz Gesamt: {overall_red:.3f}")
    print(f"  Kombinierter Gesamtindikator: {overall_combined:.3f}")

if __name__ == '__main__':
    main()
