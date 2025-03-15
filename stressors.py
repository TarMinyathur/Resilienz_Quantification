import pandapower as pp
import pandapower.networks as pn
import numpy as np
import copy 
import matplotlib.pyplot as plt
from geo_data import get_geodata_coordinates, get_buses_to_disable, plot_net, components_to_disable_dynamic



class Scenario:
    def __init__(self, name, mode, targets, reduction_rate, random_select=True):
        self.name = name
        self.mode = mode
        self.targets = targets
        self.reduction_rate = reduction_rate
        self.random_select = random_select

        # dictionary for accessing and "damaging" components correctly
        self.component_data = {
                    "PV": {"filter": lambda net_temp_stress: net_temp_stress.sgen[net_temp_stress.sgen["type"] == "PV"], "element": "sgen", "column": "p_mw"},
                    "WP": {"filter": lambda net_temp_stress: net_temp_stress.sgen[net_temp_stress.sgen["type"] == "WP"], "element": "sgen", "column": "p_mw"},
                    "trafo": {"filter": lambda net_temp_stress: net_temp_stress.trafo, "element": "trafo", "column": "in_service"},
                    "underground_lines": {"filter": lambda net_temp_stress: net_temp_stress.line[net_temp_stress.line["type"] == "cs"], "element": "line", "column": "in_service"},
                    "overhead_lines": {"filter": lambda net_temp_stress: net_temp_stress.line[net_temp_stress.line["type"] == "ol"], "element": "line", "column": "in_service"},
        }

    # adapt net_temp_stress to scenario
    def apply_modifications(self, net_temp_stress): 
        for target in self.targets:
            if target =="n.a.":    # for mode = "geo"
                print("target n.a.")
            elif target not in self.component_data:
                print(f"Add target {target} to target list or select different target.")
                continue  # unknown target, continue to next iteration

            if target in self.component_data:
                df = self.component_data[target]["filter"](net_temp_stress)  # call component filters and get df 
                if df.empty:
                    print(f"Target {target} does not exist in net_temp_stress. Will be skipped")
                    continue  # if empty target --> next

                # get access to components (elements to be attacked and thei column, e.g. p_mw or "in_service")
                element, column = self.component_data[target]["element"], self.component_data[target]["column"]


            # mode = types -> apply changes to all components of a type
            if self.mode == "types":  #
                if column == "p_mw":
                    net_temp_stress[element].loc[df.index, column] *= self.reduction_rate
                else:
                    False

            # mode = component -> apply changes to specific components
            elif self.mode == "components":  
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
                buses_to_disable, x_start, y_start, side_length = get_buses_to_disable(net_temp_stress, x_coords,y_coords, self.random_select, self.reduction_rate)
                # plot_net_temp_stress(net_temp_stress, x_start, y_start, side_length) # plot function for visual control!
                # net_temp_stress = components_to_disable_static(net_temp_stress, buses_to_disable)
                net_temp_stress = components_to_disable_dynamic(net_temp_stress, buses_to_disable)
        
        return net_temp_stress


def scenarios(net_temp_stress, selected_scenarios):
    scenarios_list = get_scenarios()

    for scenario in scenarios_list:
        if scenario.name in selected_scenarios:
            net_temp_stress_copy = copy.deepcopy(net_temp_stress)   # blanko net_temp_stress for each scenario
            modified_net_temp_stress = scenario.apply_modifications(net_temp_stress_copy)  # Apply scenario modifications
            modified_net_temp_stresss.append((scenario.name, modified_net_temp_stress))  # Store scenario name and modified net_temp_stress as tulple for mapping/ association

    return modified_net_temp_stresss


# Define all potential scenarios
# random_select optional, if not defined -> standard True
# geo_data optional, if no defined -> set False, see Class Scenario
def get_scenarios():  
    return [
        Scenario("dunkelflaute", mode="types", targets=["PV", "WP"], reduction_rate=0.0),
        Scenario("hagel", mode="types", targets=["PV"], reduction_rate=0.5),
        Scenario("sabotage", mode="components", targets=["trafo"], reduction_rate= 0.5, random_select=False),
        Scenario("flood", mode="geo", targets=["n.a."], reduction_rate= 0.5, random_select=True),
        # Add more scenarios as needed
    ]


def stress_scenarios(net_temp_stress, selected_scenarios):
    # selected_scenarios = ["flood", "hagel"]   # einzelnes aufrufen funktioniert
    modified_net_temp_stresss = []

    scenarios_list = get_scenarios()
    valid_scenario_names = [scenario.name for scenario in scenarios_list]

    if not all(s in valid_scenario_names for s in selected_scenarios):  # Validate scenarios
        print("Invalid or no scenario selected. Please check selected scenario(s)!")
    else:
        modified_net_temp_stresss = scenarios(net_temp_stress, selected_scenarios)
        print(f"Amount of modified net_temp_stresss: {len(modified_net_temp_stresss)}")

        # for name, modified_net_temp_stress in modified_net_temp_stresss:
            # print(f"Scenario {name} - Trafo Tabelle: \n {modified_net_temp_stress.trafo}")
            # print(f"Scenario {name} - sgen Tabelle: \n {modified_net_temp_stress.sgen}")
    return modified_net_temp_stresss


# 
# if __name__ == "__main__":
#     net_temp_stress = pn.create_cigre_net_temp_stresswork_mv(with_der="all")
#     net_temp_stress_stress = stress_scenarios(net_temp_stress)

    