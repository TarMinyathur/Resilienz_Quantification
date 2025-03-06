import pandapower as pp
import pandapower.networks as pn
import numpy as np
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
        if self.mode == "types":     # for type "PV" (all PV affected)
            print(self.targets)
            for target in self.targets: # for more than one target
                net.sgen.loc[net.sgen["type"] == target, "p_mw"] *= self.reduction_rate

        elif self.mode == "components":     # for "PV1" (specific component in net affected)
            for target in self.targets:  
                if target in net:   # if target directly in net table, e.g. trafo, line,..
                    df = net[target]  
                elif "type" in net.sgen.columns and target in net.sgen["type"].unique():    # if target in sgen table, e.g. PV, WP
                    df = net.sgen[net.sgen["type"] == target]
                else:
                    num_components = 0
                    print(f"{target} not in net")

                num_components = len(df)
                print(f"Amount Components in {target}: {num_components}")

                if num_components > 0:
                    num_to_disable = int(num_components * self.reduction_rate)
                    if self.random_select:   # random select = True 
                        indices = np.random.choice(df.index, size=int(num_to_disable), replace=False)
                    else:   # random select = False -> first components in table to be seleceted
                        indices = df.index[:int(num_components * self.reduction_rate)]
                    print(target)

                    if "type" in net.sgen.columns and target in net.sgen["type"].unique():    # if target in sgen table, e.g. PV, WP
                        net.sgen.loc[indices, "in_service"] = False  # deactive component
                    else:   # e.g. trafo, line,..
                        net.loc[indices,"in_service"] = False # deactive component
        return net


def get_scenarios():    # define all potential scenario
    scenarios_list = [
        Scenario("dunkelflaute", mode="types", targets=["PV", "WP"], reduction_rate=0.05),
        Scenario("hagel", mode="types", targets=["PV"], reduction_rate=0.5),
        Scenario("sabotage", mode="components", targets=["PV","WP"], reduction_rate=1, random_select=False),
        # geo referenced data
        # combination different modi types / components
    ]
    return scenarios_list

def scenarios(net, selected_scenarios):    # define scenario
    scenarios_list = get_scenarios()  # Holen der Szenarienliste
    modified_nets = []  # list of modified nets

    # apply scenario
    for scenario in scenarios_list:
        if scenario.name in selected_scenarios:
            scenario.net = net  # Weist das Originalnetz dem Szenario zu
            modified_net = scenario.apply_modifications(net)  # Modifiziertes Netz zurückbekommen
            modified_nets.append(modified_net)  # Speichert das modifizierte Netz in der Liste
            # print(modified_net)
    
    return modified_nets




if __name__ == "__main__":
    net = pn.create_cigre_network_mv(with_der="all")
    # print(f"orginigal values for sgen \n {net.sgen}")
    # print(net.sgen)

    selected_scenarios = [""]


    scenarios_list = get_scenarios()
    valid_scenario_names = [scenario.name for scenario in scenarios_list]

    if not all(s in valid_scenario_names for s in selected_scenarios):  
        print("Invalid or no scenario selected. Please check selected scenario(s)!")
    else:
        modified_nets = scenarios(net, selected_scenarios)
        print(f"amount modified nets: {len(modified_nets)}")
        
        for i, net in enumerate(modified_nets):
            print(f"Netz {i+1} - sgen Tabelle:")
            print(net.sgen)





