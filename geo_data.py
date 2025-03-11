import pandapower.networks as pn
import numpy as np

net = pn.create_cigre_network_mv(with_der="all")
# print(f"net before: {net}")



share_destruction = 0.25

# Lists to store all x and y coordinates
x_coords, y_coords = [], []

# Check if bus geodata is available
if not net.bus_geodata.empty:
    x_coords.extend(net.bus_geodata["x"])
    y_coords.extend(net.bus_geodata["y"])

# Check if line geodata is available
if not net.line_geodata.empty:
    # Drop NaN values # coords: array with 3 entries, first: bus a, second: bending ponint, last: bus b
    for coords in net.line_geodata["coords"].dropna():  
        x_coords.extend([c[0] for c in coords])  # create list of  x values
        y_coords.extend([c[1] for c in coords])  # create list of y values

# Compute area if we have any geodata
if x_coords and y_coords:
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Calculate the bounding box area
    area = (x_max - x_min) * (y_max - y_min)
    area_destroyed = area * share_destruction
    side_length = np.sqrt(area_destroyed)  # Square root to get a square-like area

    print(f"Max x: {x_max}, Min x: {x_min}")
    print(f"Max y: {y_max}, Min y: {y_min}")
    print(f"Total network area: {area:.2f}")
    print(f"Area to be destroyed: {area_destroyed:.2f}")

    # Select a region to "destroy" (either random or predefined)
    random_selection = True  # Set to False for a fixed region

    if random_selection:
        # Select a random x and y range
        x_start = np.random.uniform(x_min, x_max - side_length)
        x_end = x_start + side_length
        y_start = np.random.uniform(y_min, y_max - side_length)
        y_end = y_start + side_length
    else:
        # Fixed selection: take the bottom-left quarter of the network
        x_start, x_end = x_min, x_min + (x_max - x_min) * np.sqrt(share_destruction)
        y_start, y_end = y_min, y_min + (y_max - y_min) * np.sqrt(share_destruction)

    print(f"Selected region: x_start={x_start:.2f}, x_end={x_end:.2f}, y_start={y_start:.2f}, y_end={y_end:.2f}")

    # Find buses within the selected destruction area (both x and y ranges)
    buses_to_disable = net.bus_geodata[
        (net.bus_geodata["x"] >= x_start) & (net.bus_geodata["x"] <= x_end) &
        (net.bus_geodata["y"] >= y_start) & (net.bus_geodata["y"] <= y_end)
    ].index

    print(f"Buses to be disabled: {list(buses_to_disable)}")




    

    # Disable buses and all connected elements
    net.bus.loc[buses_to_disable, "in_service"] = False
    net.line.loc[net.line["from_bus"].isin(buses_to_disable) | 
                    net.line["to_bus"].isin(buses_to_disable), "in_service"] = False
    net.trafo.loc[net.trafo["hv_bus"].isin(buses_to_disable) | 
                    net.trafo["lv_bus"].isin(buses_to_disable), "in_service"] = False
    net.load.loc[net.load["bus"].isin(buses_to_disable), "in_service"] = False
    net.sgen.loc[net.sgen["bus"].isin(buses_to_disable), "in_service"] = False

    print("Network elements in the selected area have been disabled.")

    # print(f"net after: {net}")


else:
    print("No geodata available in the network.")



