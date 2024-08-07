# visualize.py

import numpy as np
import matplotlib.pyplot as plt

def calculate_polygon_area(values, angles):
    """
    Calculate the area of a polygon using the Shoelace formula.
    """
    x = np.cos(angles) * values
    y = np.sin(angles) * values
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return area

def calculate_polygon_centroid(values, angles):
    """
    Calculate the centroid (center of mass) of a polygon defined by its vertices.
    """
    x = np.cos(angles) * values
    y = np.sin(angles) * values
    area = calculate_polygon_area(values, angles)
    cx = np.dot(x, np.roll(y, 1)) + np.dot(np.roll(x, 1), y)
    cy = np.dot(y, np.roll(x, 1)) + np.dot(np.roll(y, 1), x)
    centroid_x = cx / (6 * area)
    centroid_y = cy / (6 * area)
    return centroid_x, centroid_y

def plot_spider_chart(df, title="Resilience Score"):
    num_vars = len(df)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    values = df['Value'].tolist()
    values += values[:1]  # Complete the loop

    # Calculate the area of the polygon
    area = calculate_polygon_area(df['Value'], angles[:-1])

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Draw one axe per variable and add labels
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], df['Indicator'], color='grey', size=12)

    # Draw y-labels
    ax.set_rscale('linear')
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=7)
    plt.ylim(0, 1)

    # Plot data
    ax.plot(angles, values, linewidth=1, linestyle='solid', label='Resilience Score', color='b')
    ax.fill(angles, values, 'b', alpha=0.1)

    # Add text annotations for each value
    for i in range(num_vars):
        angle_rad = angles[i]
        value = df['Value'][i]
        ax.text(angle_rad, value + 0.05, f'{value:.2f}', horizontalalignment='center', size=10, color='black')

    # Calculate centroid of the polygon
    centroid_x, centroid_y = calculate_polygon_centroid(values, angles)

    # Display the area at the centroid of the polygon
    ax.text(centroid_x, centroid_y, f'{area:.2f}', horizontalalignment='center', verticalalignment='center', fontsize=12, color='black')

    # Title and legend
    plt.title(f"{title} (Area: {area:.2f})", size=20, color='b', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    plt.show()
