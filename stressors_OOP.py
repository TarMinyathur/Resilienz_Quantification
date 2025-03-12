import pandapower as pp
import pandapower.networks as pn
import numpy as np
import copy 
import matplotlib.pyplot as plt



class Scenario:
    def __init__(self, name, mode, targets, reduction_rate, random_select=True, geo_data=False):
        self.name = name
        self.mode = mode
        self.targets = targets
        self.reduction_rate = reduction_rate
        self.random_select = random_select

        # dictionary for accessing and "damaging" components correctly
        self.component_data = {
                    "PV": {"filter": lambda net: net.sgen[net.sgen["type"] == "PV"], "element": "sgen", "column": "p_mw"},
                    "WP": {"filter": lambda net: net.sgen[net.sgen["type"] == "WP"], "element": "sgen", "column": "p_mw"},
                    "trafo": {"filter": lambda net: net.trafo, "element": "trafo", "column": "in_service"},
                    "underground_lines": {"filter": lambda net: net.line[net.line["type"] == "cs"], "element": "line", "column": "in_service"},
                    "overhead_lines": {"filter": lambda net: net.line[net.line["type"] == "ol"], "element": "line", "column": "in_service"},
        }

    # adapt net to scenario
    def apply_modifications(self, net): 
        for target in self.targets:
            if target not in self.component_data:
                print(f"Add target {target} to target list or select different target.")
                continue  # unknown target, continue to next iteration

            df = self.component_data[target]["filter"](net)  # call component filters and get df 
        
            if df.empty:
                print(f"Target {target} does not exist in net. Will be skipped")
                continue  # if empty target --> next

            # get access to components (elements to be attacked and thei column, e.g. p_mw or "in_service")
            element, column = self.component_data[target]["element"], self.component_data[target]["column"]

            # mode = types -> apply changes to all components of a type
            if self.mode == "types":  #
                if column == "p_mw":
                    net[element].loc[df.index, column] *= self.reduction_rate
                else:
                    False

            # mode = component -> apply changes to specific components
            elif self.mode == "components":  
                num_to_disable = int(len(df) * self.reduction_rate)
                indices = (
                    np.random.choice(df.index, size=num_to_disable, replace=False)
                    if self.random_select else df.index[:num_to_disable]
                )
                net[element].loc[indices, "in_service"] = False

            
        return net


def scenarios(net, selected_scenarios):
    scenarios_list = get_scenarios()
    modified_nets = []

    for scenario in scenarios_list:
        if scenario.name in selected_scenarios:
            net_copy = copy.deepcopy(net)   # blanko net for each scenario
            modified_net = scenario.apply_modifications(net_copy)  # Apply scenario modifications
            modified_nets.append((scenario.name, modified_net))  # Store scenario name and modified net as tulple for mapping/ association

    return modified_nets


# Define all potential scenarios
# random_select optional, if not defined -> standard True
# geo_data optional, if no defined -> set False, see Class Scenario
def get_scenarios():  
    return [
        Scenario("dunkelflaute", mode="types", targets=["PV", "WP"], reduction_rate=0.0),
        Scenario("hagel", mode="types", targets=["PV"], reduction_rate=0.5),
        Scenario("sabotage", mode="components", targets=["trafo"], reduction_rate= 0.5, random_select=False),
        Scenario("sabotage_2", mode="components", targets=["trafo"], reduction_rate= 1, random_select=False),
        Scenario("flood", mode="components", targets=["trafo"], reduction_rate= 1, random_select=False),
        # Scenario("..., geo")
        # Add more scenarios as needed
    ]


if __name__ == "__main__":
    net = pn.create_cigre_network_mv(with_der="all")
    # net = pn.create_cigre_network_mv(with_der=False)
    # print(net.sgen)
    # print(net.line)
    # print(net.trafo)

    selected_scenarios = ["sabotage"]   # einzelnes aufrufen funktioniert

    scenarios_list = get_scenarios()
    valid_scenario_names = [scenario.name for scenario in scenarios_list]

    if not all(s in valid_scenario_names for s in selected_scenarios):  # Validate scenarios
        print("Invalid or no scenario selected. Please check selected scenario(s)!")
    else:
        modified_nets = scenarios(net, selected_scenarios)
        print(f"Amount of modified nets: {len(modified_nets)}")

        for name, modified_net in modified_nets:
            print(f"Scenario {name} - Trafo Tabelle: \n {modified_net.trafo}")
            # print(f"Scenario {name} - sgen Tabelle: \n {modified_net.sgen}")
            