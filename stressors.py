import pandapower.networks as pn
import numpy as np

# to be implemented in main in some kind of loop?
# User interface to implement so scenario selection



def define_scenario(net):
    print(net.line)
   
    
    # net = dunkelflaute(net) # done
    # # einfluss wind?
    # # einfluss pv?
    # net = h2_shortage(net) # done
    # net = diesel_shortage(net) # done
    # net = flood(net)
    # net = sabotage(net)# # cyberattack -> trafos, pv 

    # net = partial_outage(net, failure_perfenctage=0.2) 
    
    # # storm, earthquake überland leitungen (type: "ol" overhead line) gehen kaputt  

def sabotage(net):
    ...


def flood(net):
    # set PV and wind to 0
    net.sgen.loc[net.sgen["type"] == "PV", "p_mw"] *= 0
    net.sgen.loc[net.sgen["type"] == "WP", "p_mw"] *= 0
    print("PV & Wind after:", ((net.sgen.loc[net.sgen["type"] == "PV", "p_mw"].sum()) + (net.sgen.loc[net.sgen["type"] == "WP", "p_mw"].sum())), "MW")

    # set 80 % of underground lines (type cs) off (to False)
    cs_lines = net.line[net.line["type"] == "cs"]
    num_to_disable = int(len(cs_lines) * 0.88) # ~ 80 % of cs lines
    lines_to_disable = np.random.choice(cs_lines.index, size=num_to_disable, replace=False)
    net.line.loc[lines_to_disable, "in_service"] = False

    # chef if "cs" lines are turned off
    # print(f"Deaktivierte CS-Leitungen: {100 * net.line.loc[net.line['type'] == 'cs', 'in_service'].eq(False).mean():.2f}%")

    return net


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


def partial_outage(net, failure_percentage): # tb checked
    # Set a seed for the same (reproduceable) results
    np.random.seed(42)


    # Simulate partial failure in generators (e.g., reduce their output by failure_percentage)
    for index, row in net.sgen.iterrows():
        failure_factor = 0.8  
        net.sgen.at[index, 'p_mw'] *= failure_factor  # Reduce generator output
    
    # # Simulate partial failure in transformers (e.g., reduce their capacity by failure_percentage)
    # for index, row in net.trafo.iterrows():
    #     failure_factor = 1 - np.random.uniform(0, failure_percentage)  # Random partial failure
    #     net.trafo.at[index, 'sn_mva'] *= failure_factor  # Reduce transformer capacity
    
    # # Simulate partial failure in lines (e.g., reduce their capacity by failure_percentage)
    # for index, row in net.line.iterrows():
    #     failure_factor = 1 - np.random.uniform(0, failure_percentage)  # Random partial failure
    #     net.line.at[index, 'max_i_ka'] *= failure_factor  # Reduce line capacity
    
    print("Partial failure simulated with reduced capacities.")

    return net


if __name__ == "__main__":
    net = pn.create_cigre_network_mv(with_der="all")
    define_scenario(net)