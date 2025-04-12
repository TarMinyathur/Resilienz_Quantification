import pandapower.networks as pn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_geodata_coordinates(net_temp_geo, debug=True):
    # Lists to store all x and y coordinates
    x_coords, y_coords = [], []

    if net_temp_geo.bus_geodata.empty and net_temp_geo.line_geodata.empty:
        raise ValueError("No geodata available in the net.")

    # Check if bus geodata is available
    if not net_temp_geo.bus_geodata.empty:
        if debug:
            print("bus geo_data available")
        x_coords.extend(net_temp_geo.bus_geodata["x"])
        y_coords.extend(net_temp_geo.bus_geodata["y"])

    # handling geo data in lines available, not covered in code
    if not net_temp_geo.line_geodata.empty and net_temp_geo.bus_geodata.empty:
        raise ValueError("geo_data only in lines. Not covered in code. Either pick different net or extend code")

    return x_coords, y_coords


def get_buses_to_disable(net_temp_geo, x_coords, y_coords, random_select, reduction_rate, debug= True):
    # Compute area if we have any geodata
    if x_coords and y_coords:
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # print(f"xmin: {x_min}, xmax: {x_max}, y_min: {y_min}, ymax: {y_max}")

        # Calculate the destroyed area
        area = (x_max - x_min) * (y_max - y_min)
        area_destroyed = area * reduction_rate
        side_length = np.sqrt(area_destroyed)  # Square root to get a square-like area

        if debug:
            print(f"Total net area: {area:.2f}")
            print(f"Area to be destroyed: {area_destroyed:.2f}")

    if random_select:
        # Select a random x and y range
        x_start = np.random.uniform(x_min - side_length, x_max)  # - side_length)
        # x_start = x_min - side_length # for testing if edge bus is affected -> yes :)
        y_start = np.random.uniform(y_min - side_length, y_max)  # - side_length)
        # print(f"xstart: {x_start}, ystart: {y_start}")
    else:
        # Fixed selection: take the bottom-left quarter of the net_temp_geowork
        x_start = x_min
        y_start = y_min

    x_end = x_start + side_length
    y_end = y_start + side_length

    # Find buses within the selected destruction area (both x and y ranges)
    buses_to_disable = net_temp_geo.bus_geodata[
        (net_temp_geo.bus_geodata["x"] >= x_start) & (net_temp_geo.bus_geodata["x"] <= x_end) &
        (net_temp_geo.bus_geodata["y"] >= y_start) & (net_temp_geo.bus_geodata["y"] <= y_end)
        ].index

    if debug:
        print(f"Buses to be disabled: {list(buses_to_disable)}")

    return buses_to_disable, x_min, x_start, x_max, y_min, y_start, y_max, side_length


def get_buses_to_disable_circle(net_temp_geo, x_coords, y_coords, random_select, reduction_rate, debug = True):
    if not x_coords or not y_coords:
        # No data, bail out
        return [], 0, 0, 0

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Center of bounding box
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    # Outer circle radius + margin
    half_diagonal = 0.5 * np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)
    margin = 0.2 * half_diagonal  # 20% margin
    outer_radius = half_diagonal + margin

    # Flooded area = bounding circle area * reduction_rate
    # OR your own formula for how big the flooded area should be.
    # If you want the same logic as the rectangle case:
    bounding_area = np.pi * (outer_radius ** 2)
    area_destroyed = bounding_area * reduction_rate

    # Convert to flood circle radius
    flood_radius = np.sqrt(area_destroyed / np.pi)

    if debug:
        print(f"Total bounding circle area: {bounding_area:.2f}")
        print(f"Area to be destroyed: {area_destroyed:.2f}")
        print(f"Flood circle radius: {flood_radius:.2f}")

    # Pick random center for the flood circle
    if random_select:
        # We want it fully inside the big circle, so max radial offset is:
        max_offset = outer_radius - flood_radius
        if max_offset < 0:
            # If flood_radius is bigger than outer_radius, just flood everything
            flood_center_x, flood_center_y = center_x, center_y
            flood_radius = outer_radius
        else:
            rho = np.random.uniform(0, max_offset)
            theta = np.random.uniform(0, 2 * np.pi)
            flood_center_x = center_x + rho * np.cos(theta)
            flood_center_y = center_y + rho * np.sin(theta)
    else:
        # For a fixed scenario, you could just choose the bounding circle center
        flood_center_x, flood_center_y = center_x, center_y

    # Find buses within the flood circle
    dist_sq = (net_temp_geo.bus_geodata["x"] - flood_center_x) ** 2 + \
              (net_temp_geo.bus_geodata["y"] - flood_center_y) ** 2
    flood_radius_sq = flood_radius ** 2

    buses_to_disable = net_temp_geo.bus_geodata[dist_sq <= flood_radius_sq].index

    if debug:
        print(f"Buses to be disabled: {list(buses_to_disable)}")

    # Return the center coords and radius if you want to plot
    return buses_to_disable, flood_center_x, flood_center_y, flood_radius


def plot_net(net_temp_geo, x_min, x_start, x_max, y_min, y_start, y_max, side_length):
    # Visualize the net_temp_geowork
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot bus coordinates
    if not net_temp_geo.bus_geodata.empty:
        ax.scatter(net_temp_geo.bus_geodata["x"], net_temp_geo.bus_geodata["y"], c="blue", s=50, label="Buses")
        for idx, row in net_temp_geo.bus_geodata.iterrows():
            ax.text(row["x"], row["y"], str(idx), fontsize=9, color='black', ha='right', va='bottom')

        # Draw the selected region as a red rectangle
        from matplotlib.patches import Rectangle
        rect = Rectangle((x_start, y_start), side_length, side_length, linewidth=2, edgecolor="red", facecolor="none",
                         label="Destruction Area")
        ax.add_patch(rect)
        ax.set_aspect('equal')  # to guarantee axis equality
        # ax.set_xlabel("X Coordinate")
        # ax.set_ylabel("Y Coordinate")
        plt.xlim(x_min - side_length, x_max + side_length)
        plt.ylim(y_min - side_length, y_max + side_length)
        ax.set_title("Georeferenced Destruction Area")
        ax.legend()


def plot_area(net_temp_geo, x_min, x_max, y_min, y_max, side_length):
    fig, ax = plt.subplots(figsize=(10, 8))
    import matplotlib.patches as patches
    # Flächen als Rechtecke zeichnen
    area_rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                  edgecolor='black', facecolor='gray', alpha=0.3, label='Net Area')
    target_rect = patches.Rectangle((x_min - side_length, y_min - side_length), ((x_max + side_length) - x_min),
                                    (y_max + side_length) - y_min,
                                    edgecolor='green', facecolor='gray', alpha=0.3, hatch='/',
                                    label='Possible Target Area')
    destructed_rect = patches.Rectangle((x_min - side_length, y_min - side_length),
                                        (x_max + side_length) - (x_min - side_length),
                                        (y_max + side_length) - (y_min - side_length),
                                        edgecolor='red', facecolor='blue', alpha=0.1, hatch='X',
                                        label='Possible Destruction Area')

    ax.add_patch(area_rect)
    ax.add_patch(target_rect)
    ax.add_patch(destructed_rect)

    # Beschriftung der Flächen
    # ax.text((x_min + x_max) / 2, (y_min + y_max) / 2, 'Total Area', fontsize=12, color='black', ha='center', va='center')
    # ax.text((x_min + x_max - side_length) / 2, (y_min + y_max - side_length) / 2, 'Target Area', fontsize=12, color='black', ha='center', va='center')
    # ax.text((x_min + x_max + side_length) / 2, (y_min + y_max + side_length) / 2, 'Destruction Area', fontsize=12, color='black', ha='center', va='center')

    # Busse plotten
    if not net_temp_geo.bus_geodata.empty:
        ax.scatter(net_temp_geo.bus_geodata["x"], net_temp_geo.bus_geodata["y"], c="blue", s=50, label="Buses")
        for idx, row in net_temp_geo.bus_geodata.iterrows():
            ax.text(row["x"], row["y"], str(idx), fontsize=9, color='black', ha='right', va='bottom')

    ax.set_aspect('equal')
    ax.set_xlim(x_min - side_length, x_max + side_length)
    ax.set_ylim(y_min - side_length, y_max + side_length)
    ax.set_title("Georeferenced Destruction Area")
    ax.legend()
    plt.show()


def components_to_disable_static(net_temp_geo, buses_to_disable):
    # 1 step: disables the bus(es) itself
    # net_temp_geo.bus.loc[buses_to_disable, "in_service"] = False

    # Disable buses and all connected elements
    #  1 step: disables the bus(es) itself
    net_temp_geo.bus.loc[buses_to_disable, "in_service"] = False

    # 2 step
    net_temp_geo.line.loc[net_temp_geo.line["from_bus"].isin(buses_to_disable) |
                          net_temp_geo.line["to_bus"].isin(buses_to_disable), "in_service"] = False
    net_temp_geo.trafo.loc[net_temp_geo.trafo["hv_bus"].isin(buses_to_disable) |
                           net_temp_geo.trafo["lv_bus"].isin(buses_to_disable), "in_service"] = False
    net_temp_geo.load.loc[net_temp_geo.load["bus"].isin(buses_to_disable), "in_service"] = False
    net_temp_geo.sgen.loc[net_temp_geo.sgen["bus"].isin(buses_to_disable), "in_service"] = False
    # ... further components to be added
    print("net elements in the selected area have been disabled.")

    # print(f"net_temp_geo after: {net_temp_geo}")
    return net_temp_geo


def components_to_disable_dynamic(net_temp_geo, buses_to_disable, debug = False):
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
        if comp in net_temp_geo and not net_temp_geo[comp].empty:
            mask = pd.Series(False, index=net_temp_geo[comp].index)  # Startet mit False für alle Zeilen
            for col in cols:
                if col in net_temp_geo[comp].columns:
                    mask |= net_temp_geo[comp][col].isin(buses_to_disable)

            if mask.any():
                net_temp_geo[comp].loc[mask, "in_service"] = False
                deactivated_components[comp] = mask.sum()  # Speichert Anzahl der deaktivierten Elemente

    if debug:
        for comp, count in deactivated_components.items():
            print(f"{comp}: {count} deactivated")

        # Falls keine Komponenten deaktiviert wurden
        if not deactivated_components:
            print("no components deactivated")

    return net_temp_geo


def geo_referenced_destruction(net_temp_geo, reduction_rate, random_select):
    # please note: as of today only geo_data stored in buses will be handled
    # since geo_data of the exemplary net_temp_geos for this project are only stored in buses (not lines)
    # for future application the geo_data stored in lines might be added to the code
    x_coords, y_coords = get_geodata_coordinates(net_temp_geo)

    buses_to_disable, x_min, x_start, x_max, y_min, y_start, y_max, side_length = get_buses_to_disable(net_temp_geo,
                                                                                                       x_coords,
                                                                                                       y_coords,
                                                                                                       random_select,
                                                                                                       reduction_rate)
    # buses_to_disable, x_min, x_start, x_max, y_min, y_start, y_max, side_length = get_buses_to_disable_circle(net_temp_geo, x_coords, y_coords,random_select, reduction_rate)

    plot_net(net_temp_geo, x_min, x_start, x_max, y_min, y_start, y_max, side_length)
    plot_area(net_temp_geo, x_min, x_max, y_min, y_max, side_length)
    # net_temp_geo = components_to_disable_static(net_temp_geo, buses_to_disable)
    net_temp_geo = components_to_disable_dynamic(net_temp_geo, buses_to_disable)

    return net_temp_geo


if __name__ == "__main__":
    net_temp_geo = pn.create_cigre_network_mv(with_der="all")

    reduction_rate = 0.1
    # Select a region to "destroy" (either random or predefined)
    random_select = True  # Set to False for a fixed region

    n = 1
    for _ in range(n):
        net_temp_geo = geo_referenced_destruction(net_temp_geo, reduction_rate, random_select)
    plt.show()