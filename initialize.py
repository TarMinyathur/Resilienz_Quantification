import pandas as pd


#changes
# the storable energy of all storages is changed from infinity, to a value that they can supply they peak power for one day
def set_storage_max_e_mwh(net):
    for idx, storage in net.storage.iterrows():
        if storage['max_e_mwh'] == float('inf'):
            net.storage.at[idx, 'max_e_mwh'] = storage['p_mw'] * 24


# Function to add data to the DataFrame
def add_indicator(dfinalresults, indicator_name, value):
    new_row = pd.DataFrame([{'Indicator': indicator_name, 'Value': value}])
    dfinalresults = pd.concat([dfinalresults, new_row], ignore_index=True)
    return dfinalresults

def add_disparity(ddisparity, indicator_name, value, max_value, verhaeltnis):
    new_row = pd.DataFrame([{'Indicator': indicator_name, 'Value': value, 'max Value': max_value, 'Verhaeltnis': verhaeltnis}])
    ddisparity = pd.concat([ddisparity, new_row], ignore_index=True)
    return ddisparity
