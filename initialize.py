import pandas as pd

# Function to add data to the DataFrame
def add_indicator(dfinalresults, indicator_name, value):
    new_row = pd.DataFrame([{'Indicator': indicator_name, 'Value': value}])
    dfinalresults = pd.concat([dfinalresults, new_row], ignore_index=True)
    return dfinalresults

def add_disparity(ddisparity, indicator_name, value, max_value, verhaeltnis):
    new_row = pd.DataFrame([{'Indicator': indicator_name, 'Value': value, 'max Value': max_value, 'Verhaeltnis': verhaeltnis}])
    ddisparity = pd.concat([ddisparity, new_row], ignore_index=True)
    return ddisparity
