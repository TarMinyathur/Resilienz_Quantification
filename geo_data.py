import pandapower.networks as pn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



def get_geodata_coordinates(net):
    # Lists to store all x and y coordinates
    x_coords, y_coords = [], []

    if net.bus_geodata.empty and net.line_geodata.empty:
        raise ValueError("No geodata available in the network.")
        

    # Check if bus geodata is available
    if not net.bus_geodata.empty:
        print("bus geo_data available")
        x_coords.extend(net.bus_geodata["x"])
        y_coords.extend(net.bus_geodata["y"])

    # handling geo data in lines available, not covered in code
    if not net.line_geodata.empty:
        raise ValueError("geo_data in lines. Not covered in code. Either pick different net or extend code")

    return x_coords, y_coords


def get_buses_to_disable(x_coords,y_coords, random_select):
    # Compute area if we have any geodata
    if x_coords and y_coords:
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Calculate the bounding box area
        area = (x_max - x_min) * (y_max - y_min)
        area_destroyed = area * reduction_rate
        side_length = np.sqrt(area_destroyed)  # Square root to get a square-like area
       
        print(f"Total network area: {area:.2f}")
        print(f"Area to be destroyed: {area_destroyed:.2f}")

    if random_select:
        # Select a random x and y range
        x_start = np.random.uniform(x_min, x_max - side_length)
        y_start = np.random.uniform(y_min, y_max - side_length)
    else:
        # Fixed selection: take the bottom-left quarter of the network
        x_start = x_min
        y_start = y_min

    x_end = x_start + side_length
    y_end = y_start + side_length


    # Find buses within the selected destruction area (both x and y ranges)
    buses_to_disable = net.bus_geodata[
        (net.bus_geodata["x"] >= x_start) & (net.bus_geodata["x"] <= x_end) &
        (net.bus_geodata["y"] >= y_start) & (net.bus_geodata["y"] <= y_end)
    ].index

    print(f"Buses to be disabled: {list(buses_to_disable)}")

    return buses_to_disable, x_start, y_start, side_length


def plot_net(net, x_start, y_start, side_length):
    # Visualize the network
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot bus coordinates
    if not net.bus_geodata.empty:
        ax.scatter(net.bus_geodata["x"], net.bus_geodata["y"], c="blue", s=50, label="Buses")
        for idx, row in net.bus_geodata.iterrows():
            ax.text(row["x"], row["y"], str(idx), fontsize=9, color='black', ha='right', va='bottom')


        # Draw the selected region as a red rectangle
        from matplotlib.patches import Rectangle
        rect = Rectangle((x_start, y_start), side_length, side_length, linewidth=2, edgecolor="red", facecolor="none", label="Destruction Area")
        ax.add_patch(rect)
        ax.set_aspect('equal')      # to guarantee axis equality
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title("Network Visualization with Destruction Area")
        ax.legend()
        plt.show()



def components_to_disable_static(net, buses_to_disable):
    # 1 step: disables the bus(es) itself
    # net.bus.loc[buses_to_disable, "in_service"] = False

    # Disable buses and all connected elements
    #  1 step: disables the bus(es) itself
    net.bus.loc[buses_to_disable, "in_service"] = False

    # 2 step
    net.line.loc[net.line["from_bus"].isin(buses_to_disable) | 
                    net.line["to_bus"].isin(buses_to_disable), "in_service"] = False
    net.trafo.loc[net.trafo["hv_bus"].isin(buses_to_disable) | 
                    net.trafo["lv_bus"].isin(buses_to_disable), "in_service"] = False
    net.load.loc[net.load["bus"].isin(buses_to_disable), "in_service"] = False
    net.sgen.loc[net.sgen["bus"].isin(buses_to_disable), "in_service"] = False
    # ... further components to be added
    print("Network elements in the selected area have been disabled.")

    # print(f"net after: {net}")
    return net


def components_to_disable_dynamic(net, buses_to_disable):
        # Liste der Komponenten mit den zu überprüfenden Bus-Spalten
    bus_columns = {
        'load': ['bus'],
        'sgen': ['bus'],
        'gen': ['bus'],
        'ext_grid': ['bus'],
        'line': ['from_bus', 'to_bus'],
        'trafo': ['hv_bus', 'lv_bus'],
        'trafo3w': ['hv_bus', 'mv_bus', 'lv_bus'],
        'dcline': ['from_bus', 'to_bus'],
        'impedance': ['from_bus', 'to_bus'],
        'switch': ['bus'],  
        'shunt': ['bus'],
        'storage': ['bus']
    }

    deactivated_components = {}  # Speichert deaktivierte Komponenten für die Ausgabe

    for comp, cols in bus_columns.items():
        if comp in net and not net[comp].empty:
            mask = pd.Series(False, index=net[comp].index)  # Startet mit False für alle Zeilen
            for col in cols:
                if col in net[comp].columns:
                    mask |= net[comp][col].isin(buses_to_disable)

            if mask.any():
                net[comp].loc[mask, "in_service"] = False
                deactivated_components[comp] = mask.sum()  # Speichert Anzahl der deaktivierten Elemente



    for comp, count in deactivated_components.items():
        print(f"{comp}: {count} deactivated")

    # Falls keine Komponenten deaktiviert wurden
    if not deactivated_components:
        print("no components deactivated")




if __name__ == "__main__":
    net = pn.create_cigre_network_mv(with_der="all")

    # print(net.bus_geodata)
    # net = pn.create_cigre_network_mv(with_der=False)
    # print(net.sgen)
    # print(net.line)

    reduction_rate = 0.01
     # Select a region to "destroy" (either random or predefined)
    random_select = True  # Set to False for a fixed region

    # please note: as of today only geo_data stored in buses will be handled
    # since geo_data of the exemplary nets for this project are only stored in buses (not lines)
    # for future application the geo_data stored in lines might be added to the code
    x_coords, y_coords = get_geodata_coordinates(net)


    buses_to_disable, x_start, y_start, side_length = get_buses_to_disable(x_coords,y_coords, random_select)
    plot_net(net, x_start, y_start, side_length)
    # net = components_to_disable_static(net, buses_to_disable)
    net = components_to_disable_dynamic(net, buses_to_disable)
    
