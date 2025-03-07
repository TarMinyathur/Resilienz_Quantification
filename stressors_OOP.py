import pandapower as pp
import pandapower.networks as pn
import numpy as np

class Scenario:
    def __init__(self, name, mode, targets, reduction_rate, random_select=True):
        self.name = name
        self.mode = mode
        self.targets = targets
        self.reduction_rate = reduction_rate
        self.random_select = random_select

    # adapts net to scenario
    def apply_modifications(self, net): 
        # Mapping of component types to filtering logic
        component_filters = {
            "PV": lambda net: net.sgen[net.sgen["type"] == "PV"],
            "WP": lambda net: net.sgen[net.sgen["type"] == "WP"],
            "trafo": lambda net: net.trafo[net.trafo["name"].str.contains("Trafo", na=False)],
            "underground_lines": lambda net: net.line[net.line["type"] == "cs"],  # Underground line
            "overhead_lines": lambda net: net.line[net.line["type"] == "ol"],  # Overhead line
        }

        if self.mode == "types":  # Apply changes to all components of a type
            for target in self.targets:
                if target in component_filters:
                    net.sgen.loc[net.sgen["type"] == target, "p_mw"] *= self.reduction_rate
                else:
                    print(f"Warning: Unknown component type '{target}'")

        elif self.mode == "components":  # Apply changes to specific components
            for target in self.targets:
                if target in component_filters:
                    df = component_filters[target](net)  # Apply filter function
                else:
                    print(f"{target} not in net")
                    continue

                num_components = len(df)
                print(f"Amount Components in {target}: {num_components}")

                if num_components > 0:
                    num_to_disable = int(num_components * self.reduction_rate)
                    indices = (
                        np.random.choice(df.index, size=num_to_disable, replace=False)
                        if self.random_select else df.index[:num_to_disable]
                    )

                    # Deactivate the components
                    if target in ["PV", "WP"]:
                        net.sgen.loc[indices, "in_service"] = False
                    elif target in ["underground_lines", "overhead_lines"]:
                        net.line.loc[indices, "in_service"] = False  # 🔥 Fix: Direkt auf net.line schreiben
                    else:
                        net[target].loc[indices, "in_service"] = False
        return net


def get_scenarios():  # Define all potential scenarios
    return [
        Scenario("dunkelflaute", mode="types", targets=["PV", "WP"], reduction_rate=0.05),
        Scenario("hagel", mode="types", targets=["PV"], reduction_rate=0.5),
        Scenario("sabotage", mode="components", targets=["trafo"], reduction_rate=1, random_select=False),
        # Add more scenarios as needed
    ]


def scenarios(net, selected_scenarios):
    scenarios_list = get_scenarios()
    modified_nets = []

    for scenario in scenarios_list:
        if scenario.name in selected_scenarios:
            modified_net = scenario.apply_modifications(net)  # Apply scenario modifications
            modified_nets.append(modified_net)  # Store modified net

    return modified_nets


if __name__ == "__main__":
    net = pn.create_cigre_network_mv(with_der="all")
    # print(net)
    # print(net.line)
    # print(net.trafo)

    selected_scenarios = ["sabotage"]

    scenarios_list = get_scenarios()
    valid_scenario_names = [scenario.name for scenario in scenarios_list]

    if not all(s in valid_scenario_names for s in selected_scenarios):  # Validate scenarios
        print("Invalid or no scenario selected. Please check selected scenario(s)!")
    else:
        modified_nets = scenarios(net, selected_scenarios)
        print(f"Amount of modified nets: {len(modified_nets)}")

        for i, modified_net in enumerate(modified_nets):
            print(f"Netz {i+1} - sgen Tabelle:")
            print(modified_net.trafo)
