# GenerationFactors.py

def calculate_generation_factors(net):
    # Initialize generation factors
    generation_factors = {}

    # Calculate for static generators (sgen)
    sgen_types = net.sgen['type'].unique()
    for sgen_type in sgen_types:
        if sgen_type == 'pv':
            generation_factors[sgen_type] = 0.15  # Example factor
        elif sgen_type == 'wind':
            generation_factors[sgen_type] = 0.25  # Example factor
        elif sgen_type == 'biomass':
            generation_factors[sgen_type] = 0.8  # Example factor
        elif sgen_type == 'Residential fuel cell':
            generation_factors[sgen_type] = 1  # Example factor
        elif sgen_type == 'CHP diesel':
            generation_factors[sgen_type] = 1  # Example factor
        elif sgen_type == 'Fuel cell':
            generation_factors[sgen_type] = 1  # Example factor

    # Calculate for batteries (storage)
    for idx, row in net.storage.iterrows():
        capacity = row['sn_mva']
        p_mw = row['p_mw']
        generation_factors['battery'] = (capacity / p_mw) / 24 if p_mw != 0 else 0

    return generation_factors