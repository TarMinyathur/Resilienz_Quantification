import pandapower.networks as pn


# to be implemented in main in some kind of loop?
# User interface to implement so scenario selection



def define_scenario(net):
    # print(net.sgen)
   

    # net = dunkelflaute(net) # done
    net = h2_shortage(net) # done
    # net = diesel_shortage(net) # done
    # net = partial_outage(net) # flood, cyberattack, storm, earthquake  




def dunkelflaute(net): 
    # PV and Wind drops under 10 % for 48 h
    # https://www.cleanenergywire.org/news/prolonged-dunkelflaute-shrinks-germanys-renewables-output-early-november?utm_source=chatgpt.com
   
    pv_p = net.sgen.loc[net.sgen["type"] == "PV", "p_mw"].sum()
    wind_p = net.sgen.loc[net.sgen["type"] == "WP", "p_mw"].sum()

    print("PV & Wind before:", (pv_p + wind_p), "MW")

    # change values for PV and wind
    net.sgen.loc[net.sgen["type"] == "PV", "p_mw"] *= 0.05
    net.sgen.loc[net.sgen["type"] == "WP", "p_mw"] *= 0.05

    print("PV & Wind after:", ((net.sgen.loc[net.sgen["type"] == "PV", "p_mw"].sum()) + (net.sgen.loc[net.sgen["type"] == "WP", "p_mw"].sum())), "MW")

    return net


def h2_shortage(net):
    fuel_cells_in_net = net.sgen["type"].str.contains("fuel cell", case=False, na=False) 
    print("Fuel Cell before: ", net.sgen.loc[fuel_cells_in_net, "p_mw"].sum())

    # set fuel cells to 0
    net.sgen.loc[fuel_cells_in_net, "p_mw"] = 0

    print("Fuel Cell after: ", net.sgen.loc[fuel_cells_in_net, "p_mw"].sum())
    return net


def diesel_shortage(net):
    chp_diesel_in_net = net.sgen["type"].str.contains("CHP diesel", case=False, na=False)

    print("CHP Diesel before: ", net.sgen.loc[chp_diesel_in_net, "p_mw"].sum())

    # set CHP diesel to 0
    net.sgen.loc[chp_diesel_in_net, "p_mw"] = 0

    print("CHP Diesel after: ", net.sgen.loc[chp_diesel_in_net, "p_mw"].sum())

    return net

def partial_outage(net):
    ...


if __name__ == "__main__":
    net = pn.create_cigre_network_mv(with_der="all")
    define_scenario(net)