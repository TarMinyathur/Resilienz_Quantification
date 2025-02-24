# GenerationFactors.py
import pandas as pd

def calculate_generation_factors(net,Studie):
    # Initialize generation factors
    generation_factors = {}

    # Daten in ein DataFrame übertragen
    data = {
        "Technologie": [
            "Kernkraft", "Braunkohle", "Steinkohle", "Erdgas", "Pumpspeicher", "Biomasse", "Lauf- und Speicherwasser", "Wind Onshore", "Wind Offshore", "Photovoltaik"
        ],
        "ITAS-Studie (2012)": [7289, 6814, 4547, 3183, 828, 5846, 3908, 1690, None, 739],
        "Fraunhofer ISE (2024)": [7500, 6500, 4200, 3000, None, 6000, 4000, 2200, 4500, 1000],
        "BEE-Studie (2021)": [7800, 6800, 4500, 3200, None, 6200, 4100, 2100, 4800, 1100],
        "Tagesschau (2024)": [None, None, None, None, None, None, None, "2000-2500", "bis zu 4000", None]
    }

    df = pd.DataFrame(data)

    generation_factors = umrechnung_fuer_studie(df, Studie, net)

    return generation_factors

def umrechnung_fuer_studie(daf, studie, net):
    # Calculate for static generators (sgen)
    sgen_types = net.sgen['type'].unique()
    gen_types = net.storage['type'].unique()
    generation_factors = {}

    # Auswahl der Studie
    if studie not in ["ITAS-Studie (2012)", "Fraunhofer ISE (2024)", "BEE-Studie (2021)"]:
        raise ValueError("Unbekannte Studie")

    # Zugriff auf die Spalte der ausgewählten Studie
    volllaststunden = daf.set_index("Technologie")[studie]

    # Umrechnung für die statischen Generatoren
    for sgen_type in sgen_types:
        if sgen_type == 'PV':
            generation_factors[sgen_type] = volllaststunden["Photovoltaik"] / 8760 if pd.notna(volllaststunden["Photovoltaik"]) else 0
        elif sgen_type == 'WP':
            generation_factors[sgen_type] = volllaststunden["Wind Onshore"] / 8760 if pd.notna(volllaststunden["Wind Onshore"]) else 0
        elif sgen_type == 'biomass':
            generation_factors[sgen_type] = volllaststunden["Biomasse"] / 8760 if pd.notna(volllaststunden["Biomasse"]) else 0
        elif sgen_type == 'Residential fuel cell':
            generation_factors[sgen_type] = 0.9  # Assuming that sufficient fuel is available locally
        elif sgen_type == 'CHP diesel':
            generation_factors[sgen_type] = 0.9  # Assuming that sufficient fuel is available locally

    # Umrechnung für die konventionellen Generatoren (gen)
    for gen_type in gen_types:
        if gen_type == 'gas':
            generation_factors[gen_type] = volllaststunden["Erdgas"] / 8760 if pd.notna(
                volllaststunden["Erdgas"]) else 0
        elif gen_type == 'coal':
            generation_factors[gen_type] = volllaststunden["Steinkohle"] / 8760 if pd.notna(
                volllaststunden["Steinkohle"]) else 0
        elif gen_type == 'nuclear':
            generation_factors[gen_type] = volllaststunden["Kernkraft"] / 8760 if pd.notna(
                volllaststunden["Kernkraft"]) else 0

    # Umrechnung für Batteriespeicher und Wasserkraft
    for idx, row in net.storage.iterrows():
        if row['type'] == 'Battery':
            capacity = row['sn_mva']
            p_mw = row['p_mw']
            generation_factors['Battery'] = (capacity / p_mw) / 24 if p_mw != 0 else 0
        elif row['type'] == 'Hydro':
            capacity = row['sn_mva']
            p_mw = row['p_mw']
            if not capacity ==0:
                generation_factors['Battery'] = (capacity / p_mw) / 24 if p_mw != 0 else 0
            else:
                generation_factors['Hydro'] = volllaststunden["Pumpspeicher"] / 8760 if pd.notna(
                    volllaststunden["Pumpspeicher"]) else 0

    return generation_factors
