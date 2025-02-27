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

    def apply_modifications(self, net):
        """Passt das Netz basierend auf dem Szenario an."""
        if self.mode == "types":
            print(self.targets)
            # Falls wir auf einen Typ (z. B. "PV") abzielen
            for target in self.targets: #falls mehrere targets angegeben werden
                net.sgen.loc[net.sgen["type"] == target, "p_mw"] *= self.reduction_rate

        elif self.mode == "components":
            # Falls wir auf eine bestimmte Anzahl von Komponenten abzielen
            if isinstance(self.targets, str) and self.targets in self.net:
                df = net[self.targets]  # Greife auf die Netzkomponente zu (z. B. net["trafo"])
                num_components = len(df)

                if num_components > 0:
                    if self.random_select:
                        indices = np.random.choice(df.index, size=int(num_components * self.reduction_rate), replace=False)
                    else:
                        indices = df.index[:int(num_components * self.reduction_rate)]

                    self.net[self.target].loc[indices, "in_service"] = False  # Komponenten deaktivieren
        return net


def scenarios(net, selected_scenarios):    # Szenarien definieren
    scenarios = [
        Scenario("dunkelflaute", mode="types", targets=["PV", "WP"], reduction_rate=0.05),
        Scenario("hagel", mode="types", targets=["PV"], reduction_rate=0.5),
        Scenario("sabotage", mode="components", targets=["trafo"], reduction_rate=1.0, random_select=False),
    ]

    modified_nets = []  # Liste für modifizierte Netze

    # Szenarien anwenden
    for scenario in scenarios:
        if scenario.name in selected_scenarios:
            scenario.net = net  # Weist das Originalnetz dem Szenario zu
            modified_net = scenario.apply_modifications(net)  # Modifiziertes Netz zurückbekommen
            modified_nets.append(modified_net)  # Speichert das modifizierte Netz in der Liste
            # print(modified_net)
    
    return modified_nets




if __name__ == "__main__":
    net = pn.create_cigre_network_mv(with_der="all")
    print(f"orginigal values for sgen \n {net.sgen}")

    selected_scenarios = ["dunkelflaute"]
    modified_nets = scenarios(net, selected_scenarios)
    print(f"amount modified nets: {len(modified_nets)}")
    
    for i, net in enumerate(modified_nets):
        print(f"Netz {i+1} - sgen Tabelle:")
        print(net.sgen)





