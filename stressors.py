import pandapower as pp
import pandapower.networks as pn
import numpy as np
import copy
import matplotlib.pyplot as plt
from geo_data import get_geodata_coordinates, get_buses_to_disable, plot_net, components_to_disable_dynamic
import random


class Scenario:
    def __init__(self, name, mode, targets, reduction_rate, random_select=True):
        self.name = name
        self.mode = mode
        self.targets = targets
        self.reduction_rate = reduction_rate
        self.random_select = random_select

        # dictionary for accessing and "damaging" components correctly
        self.component_data = {
            "PV": {"filter": lambda net_temp_stress: net_temp_stress.sgen[net_temp_stress.sgen["type"] == "PV"],
                   "element": "sgen", "column": "p_mw"},
            "WP": {"filter": lambda net_temp_stress: net_temp_stress.sgen[net_temp_stress.sgen["type"] == "WP"],
                   "element": "sgen", "column": "p_mw"},
            "CHP": {"filter": lambda net_temp_stress: net_temp_stress.sgen[
                net_temp_stress.sgen["type"].str.contains("CHP", case=False, na=False)], "element": "sgen",
                    "column": "p_mw"},
            "fuel_cell": {"filter": lambda net_temp_stress: net_temp_stress.sgen[
                net_temp_stress.sgen["type"].str.contains("fuel cell", case=False, na=False)], "element": "sgen",
                          "column": "p_mw"},
            "sgen": {"filter": lambda net_temp_stress: net_temp_stress.sgen, "element": "sgen", "column": "p_mw"},
            "gen": {"filter": lambda net_temp_stress: net_temp_stress.gen, "element": "gen", "column": "p_mw"},
            "load": {"filter": lambda net_temp_stress: net_temp_stress.load, "element": "load", "column": "p_mw"},
            "trafo": {"filter": lambda net_temp_stress: net_temp_stress.trafo, "element": "trafo",
                      "column": "in_service"},
            "underground_lines": {"filter": lambda net_temp_stress: net_temp_stress.line[
                    net_temp_stress.line["type"].str.contains(r"(?:cs|ug|cable|underground)", case=False, na=False)],
                "element": "line",
                "column": "in_service"
            },
            "overhead_lines": {"filter": lambda net_temp_stress: net_temp_stress.line[
                    net_temp_stress.line["type"].str.contains(r"(?:ol|overhead|ohl)", case=False, na=False)],
                "element": "line",
                "column": "in_service"
            }
        }

    # adapt net_temp_stress to scenario
    def apply_modifications(self, net_temp_stress):
        for target in self.targets:
            if target == "n.a.":  # for mode = "geo"
                print("target n.a.")
            elif target not in self.component_data:
                print(f"Add target {target} to target list or select different target.")
                continue  # unknown target, continue to next iteration

            if target in self.component_data:
                df = self.component_data[target]["filter"](net_temp_stress)  # call component filters and get df
                if df.empty:
                    print(f"Target {target} does not exist in net_temp_stress. Will be skipped")
                    continue  # if empty target --> next

                # get access to components (elements to be attacked and the column, e.g. p_mw or "in_service")
                element, column = self.component_data[target]["element"], self.component_data[target]["column"]

            if self.mode == "types":
                if column == "p_mw":
                    # Reduziere p_mw und q_mvar Werte
                    net_temp_stress[element].loc[df.index, "p_mw"] *= self.reduction_rate
                    if "q_mvar" in net_temp_stress[element].columns:
                        net_temp_stress[element].loc[df.index, "q_mvar"] *= self.reduction_rate

                    # Optional: auch Grenzen anpassen, sofern vorhanden
                    if "p_mw_max" in net_temp_stress[element].columns:
                        net_temp_stress[element].loc[df.index, "p_mw_max"] *= self.reduction_rate
                    if "p_mw_min" in net_temp_stress[element].columns:
                        net_temp_stress[element].loc[df.index, "p_mw_min"] *= self.reduction_rate
                    if "q_mvar_max" in net_temp_stress[element].columns:
                        net_temp_stress[element].loc[df.index, "q_mvar_max"] *= self.reduction_rate
                    if "q_mvar_min" in net_temp_stress[element].columns:
                        net_temp_stress[element].loc[df.index, "q_mvar_min"] *= self.reduction_rate
                else:
                    False

            # mode = component -> apply changes to specific components
            elif self.mode == "component":
                num_to_disable = int(len(df) * self.reduction_rate)
                # if random_select = True, select random, if False select first ones from list
                indices = (
                    np.random.choice(df.index, size=num_to_disable, replace=False)
                    if self.random_select else df.index[:num_to_disable]
                )
                net_temp_stress[element].loc[indices, "in_service"] = False

            elif self.mode == "geo":
                # net_temp_stress = geo_referenced_destruction(net_temp_stress, self.reduction_rate, self.random_select)
                x_coords, y_coords = get_geodata_coordinates(net_temp_stress)
                buses_to_disable, x_start, y_start, side_length = get_buses_to_disable(net_temp_stress, x_coords,
                                                                                       y_coords, self.random_select,
                                                                                       self.reduction_rate)
                # plot_net(net_temp_stress, x_start, y_start, side_length) # plot function for visual control!
                # net_temp_stress = components_to_disable_static(net_temp_stress, buses_to_disable)
                net_temp_stress = components_to_disable_dynamic(net_temp_stress, buses_to_disable)

        return net_temp_stress


def scenarios(net_temp_stress, selected_scenarios):
    scenarios_list = get_scenarios()
    modified_net_temp_stresss = []

    for scenario in scenarios_list:
        if scenario.name in selected_scenarios:
            net_temp_stress_copy = copy.deepcopy(net_temp_stress)  # blanko net_temp_stress for each scenario
            modified_net_temp_stress = scenario.apply_modifications(
                net_temp_stress_copy)  # Apply scenario modifications
            modified_net_temp_stresss.append((scenario.name,
                                              modified_net_temp_stress))  # Store scenario name and modified net_temp_stress as tulple for mapping/ association

    return modified_net_temp_stresss


# Define all potential scenarios
# random_select optional, if not defined -> standard True
# geo_data optional, if no defined -> set False, see Class Scenario
def get_scenarios():
    return [
        Scenario("flood", mode="geo", targets=["n.a."], reduction_rate=random.uniform(0.3, 1), random_select=True),
        Scenario("earthquake", mode="component",
                 targets=random.sample(["overhead_lines", "underground_lines", "trafo", "load", "gen", "sgen"], k=random.randint(4,6)),
                 reduction_rate=random.uniform(0.3, 1),
                 random_select=True),
        Scenario("dunkelflaute", mode="types", targets=["PV", "WP"], reduction_rate=random.uniform(0, 0.15)),
        Scenario("storm", mode="component",
                 targets=random.sample(["overhead_lines", "underground_lines", "trafo"], k=random.randint(1,3)),
                 reduction_rate=random.uniform(0, 1),
                 random_select=True),
        Scenario("geopolitical_chp", mode="component", targets=["CHP"], reduction_rate=random.uniform(0.7, 1), random_select=True),
        Scenario("geopolitical_h2", mode="component", targets=["fuel_cell"], reduction_rate=random.uniform(0.7, 1), random_select=True),
        Scenario("high_load", mode="types", targets=["load"], reduction_rate=random.uniform(1, 5)),
        Scenario("sabotage_trafo", mode="component", targets=["trafo"], reduction_rate=random.uniform(0 , 1), random_select=True),
    ]


def stress_scenarios(net_temp_stress, selected_scenarios):
    # selected_scenarios = ["flood", "hagel"]   # einzelnes aufrufen funktioniert
    modified_net_temp_stresss = []

    scenarios_list = get_scenarios()
    valid_scenario_names = [scenario.name for scenario in scenarios_list]

    if not all(s in valid_scenario_names for s in selected_scenarios):  # Validate scenarios
        print(f"selected_scenarios")
        print("Invalid or no scenario selected. Please check selected scenario(s)!")
    else:
        modified_net_temp_stresss = scenarios(net_temp_stress, selected_scenarios)
        # print(f"Amount of modified net_temp_stresss: {len(modified_net_temp_stresss)}")

        # for name, modified_net_temp_stress in modified_net_temp_stresss:
        #     # print(f"Scenario {name} - Trafo Tabelle: \n {modified_net_temp_stress.trafo}")
        #     print(f"Scenario {name} - load Tabelle: \n {modified_net_temp_stress.load}")
    return modified_net_temp_stresss


if __name__ == "__main__":
    net_temp_stress = pn.create_cigre_network_mv(with_der="all")
    # net_temp_stress = pn.create_cigre_network_hv(length_km_6a_6b=0.1)
    selected_scenarios = ["flood", "earthquake", "storm", "sabotage_trafo", "dunkelflaute"]

    net_temp_stress_stress = stress_scenarios(net_temp_stress, selected_scenarios)
    for scenario_name, stressed_net in net_temp_stress_stress:
        print("=" * 60)
        print(f"Scenario: {scenario_name}")

        # 1) Lines out of service
        lines_off = stressed_net.line[stressed_net.line.in_service == False]
        if not lines_off.empty:
            print("Lines out of service:")
            print(lines_off[["from_bus", "to_bus", "type"]])
            print()  # blank line for readability

        # 2) Trafos out of service
        trafos_off = stressed_net.trafo[stressed_net.trafo.in_service == False]
        if not trafos_off.empty:
            print("Transformers out of service:")
            print(trafos_off[["hv_bus", "lv_bus"]])
            print()

        # 3) Loads out of service
        loads_off = stressed_net.load[stressed_net.load.in_service == False]
        if not loads_off.empty:
            print("Loads out of service:")
            print(loads_off[["bus", "p_mw", "q_mvar"]])
            print()

        # 4) Sgens out of service
        sgens_off = stressed_net.sgen[stressed_net.sgen.in_service == False]
        if not sgens_off.empty:
            print("Static generators (sgen) out of service:")
            print(sgens_off[["bus", "p_mw", "type"]])
            print()

        # 5) Gens out of service (if any)
        if "gen" in stressed_net and not stressed_net.gen.empty:
            gens_off = stressed_net.gen[stressed_net.gen.in_service == False]
            if not gens_off.empty:
                print("Generators (gen) out of service:")
                print(gens_off[["bus", "p_mw"]])
                print()

        # If none of the main elements are out of service, you could add:
        if (lines_off.empty and trafos_off.empty and loads_off.empty and
                sgens_off.empty and (not "gen" in stressed_net )):
            print("No elements were set out of service.")

        print("=" * 60, "\n")

        # Compare loads:
        load_diff = stressed_net.load["p_mw"] - net_temp_stress.load["p_mw"]
        changed_loads = load_diff[abs(load_diff) > 1e-9]  # anything not zero
        if not changed_loads.empty:
            print("Loads whose p_mw changed:")
            print(changed_loads)
            print()

        # Compare sgens:
        sgen_diff = stressed_net.sgen["p_mw"] - net_temp_stress.sgen["p_mw"]
        changed_sgens = sgen_diff[abs(sgen_diff) > 1e-9]
        if not changed_sgens.empty:
            print("Sgens whose p_mw changed:")
            print(changed_sgens)
            print()

