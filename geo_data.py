import pandapower.networks as pn
import numpy as np
import matplotlib.pyplot as plt



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

    # Check if line geodata is available
    if not net.line_geodata.empty:
        print("line geo_data available")
        # Drop NaN values # coords: array with 3 entries, first: bus a, second: bending ponint, last: bus b
        for coords in net.line_geodata["coords"].dropna():  
            x_coords.extend([c[0] for c in coords])  # create list of  x values
            y_coords.extend([c[1] for c in coords])  # create list of y values
    
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

        print(f"Max x: {x_max}, Min x: {x_min}")
        print(f"Max y: {y_max}, Min y: {y_min}")
        print(f"Total network area: {area:.2f}")
        print(f"Area to be destroyed: {area_destroyed:.2f}")



    if random_select:
        # Select a random x and y range
        x_start = np.random.uniform(x_min, x_max - side_length)
        x_end = x_start + side_length
        y_start = np.random.uniform(y_min, y_max - side_length)
        y_end = y_start + side_length
    else:
        # Fixed selection: take the bottom-left quarter of the network
        x_start, x_end = x_min, x_min + (x_max - x_min) * np.sqrt(reduction_rate)
        y_start, y_end = y_min, y_min + (y_max - y_min) * np.sqrt(reduction_rate)

    print(f"Selected region: x_start={x_start:.2f}, x_end={x_end:.2f}, y_start={y_start:.2f}, y_end={y_end:.2f}")

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


    # Plot line coordinates if available
    if not net.line_geodata.empty:
        for coords in net.line_geodata["coords"].dropna():
            x_line = [c[0] for c in coords]
            y_line = [c[1] for c in coords]
            ax.plot(x_line, y_line, color="gray", lw=1, label="Lines")
            # Compute midpoint for label
            mid_x = np.mean(x_line)
            mid_y = np.mean(y_line)
            ax.text(mid_x, mid_y, str(idx), fontsize=8, color='purple', ha='center', va='center')


        # Draw the selected region as a red rectangle
        from matplotlib.patches import Rectangle
        rect = Rectangle((x_start, y_start), side_length, side_length, linewidth=2, edgecolor="red", facecolor="none", label="Destruction Area")
        ax.add_patch(rect)
        
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title("Network Visualization with Destruction Area")
        ax.legend()
        plt.show()



def components_to_disable(net, buses_to_disable):
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
    return net






if __name__ == "__main__":
    net = pn.create_cigre_network_mv(with_der="all")
    # net = pn.create_cigre_network_mv(with_der=False)
    # print(net.sgen)
    # print(net.line)

    reduction_rate = 0.25
     # Select a region to "destroy" (either random or predefined)
    random_select = True  # Set to False for a fixed region


    x_coords, y_coords = get_geodata_coordinates(net)


    buses_to_disable, x_start, y_start, side_length = get_buses_to_disable(x_coords,y_coords, random_select)
    plot_net(net, x_start, y_start, side_length)
    net = components_to_disable(net, buses_to_disable)
    
