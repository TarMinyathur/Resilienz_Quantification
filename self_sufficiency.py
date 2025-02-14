# self_sufficiency.py
# Checks if generation of real, reactive and apparent power meets the corresponding demand on each bus

# selfsuff über subgraphs und conversion of loadflow? →

def selfsuff(net):

    erfolg = 0
    misserfolg = 0

    for bus in net.bus.index:
        generation_p = sum(net.gen[net.gen.bus == bus].p_mw) + sum(net.sgen[net.sgen.bus == bus].p_mw) \
                       + sum(net.storage[net.storage.bus == bus].p_mw)
        # generation_q = sum(net.gen[net.gen.bus == bus].q_mvar.fillna(0)) + sum(net.sgen[net.sgen.bus == bus].q_mvar.fillna(0))
        generation_q = sum(net.sgen[net.sgen.bus == bus].q_mvar.fillna(0))
        generation_s = sum((net.sgen[net.sgen.bus == bus].p_mw ** 2 + net.sgen[net.sgen.bus == bus].q_mvar ** 2) ** 0.5)

        demand_p = sum(net.load[net.load.bus == bus].p_mw)
        demand_q = sum(net.load[net.load.bus == bus].q_mvar)
        demand_s = sum((net.load[net.load.bus == bus].p_mw ** 2 + net.load[net.load.bus == bus].q_mvar ** 2) ** 0.5)

        print(f"Bus {bus}:")
        print(f"   Active Power:   Generation = {generation_p} MW, Demand = {demand_p} MW")
        print(f"   Reactive Power: Generation = {generation_q} Mvar, Demand = {demand_q} Mvar")
        print(f"   Apparent Power: Generation = {generation_s} MVA, Demand = {demand_s} MVA")

        if generation_p >= demand_p:
            erfolg += 1
        else:
            misserfolg += 1

        if generation_q >= demand_q:
            erfolg += 1
        else:
            misserfolg += 1

        if generation_s >= demand_s:
            erfolg += 1
        else:
            misserfolg += 1

    selfsuff = float(erfolg) / (float(erfolg) + float(misserfolg))
    return float(selfsuff)