#

import shapely.geometry as geom
from shapely.ops import orient
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
import numpy as np

def flexibility_fxor (net_flex, visualize):

    bus_polygons = {}
    for b in net_flex.bus.index:
        poly = bus_flex_polygon(net_flex, b)
        if not poly.is_empty:
            bus_polygons[b] = poly
            print(f"Bus {b} '{net_flex.bus.at[b, 'name']}': polygon area = {poly.area:.3f}")
        else:
            print(f"Bus {b} '{net_flex.bus.at[b, 'name']}': EMPTY polygon")

    # 1) Flächen der Bus-Polygone aufsummieren
    sum_area_polygons = 0.0
    for bus_idx, poly in bus_polygons.items():
        area = poly.area
        sum_area_polygons += area

    print(f"Summe der Polygonflächen aller Busse: {sum_area_polygons:.3f}")

    # 2) Polygon aus der Summe der Lasten bilden
    #    Wir nehmen an, dass net_flex.load.p_mw und net_flex.load.q_mvar existieren,
    #    und summieren alle Lasten auf, um (Sum_P, Sum_Q) zu erhalten.
    sum_p = net_flex.load["p_mw"].sum()
    sum_q = net_flex.load["q_mvar"].sum()

    # 3) Vergleich ausgeben
    flexibility = min(1 , sum_area_polygons / np.sqrt(sum_p**2 + sum_q**2))
    print(f"Verhältnis (Summe Bus-Polygone / Summe Last): {flexibility:.3f}")

    if visualize:
        # Plot
        plt.figure(figsize=(8, 6))
        for b, poly in bus_polygons.items():
            x, y = poly.exterior.xy
            plt.plot(x, y, label=f"Bus {b}: {net_flex.bus.at[b, 'name']}")
        plt.xlabel("P [MW]")
        plt.ylabel("Q [MVAr]")
        plt.title("Local (P,Q) Polygons for Each Bus (CIGRE MV)")
        plt.grid(True)
        plt.legend()
        plt.show()
    
    return flexibility

# -------------------------------------------------------------------------
# Helper: device_polygon
# -------------------------------------------------------------------------
def device_polygon(p_min, p_max, q_min, q_max, sn_mva=None, circle_resolution=32):
    """
    Build a shapely Polygon representing all feasible (P,Q) for one device,
    given bounding box constraints and optional s_n (MVA) rating.
    Then take its convex hull to ensure convexity.
    """
    # 1) Rectangle: [p_min, p_max] x [q_min, q_max]
    rect_coords = [
        (p_min, q_min),
        (p_min, q_max),
        (p_max, q_max),
        (p_max, q_min),
        (p_min, q_min)  # close
    ]
    rect_poly = geom.Polygon(rect_coords)

    # 2) If sn_mva is given, intersect with circle sqrt(P^2 + Q^2) <= sn_mva
    if sn_mva is not None and sn_mva > 0:
        circle = geom.Point(0, 0).buffer(sn_mva, resolution=circle_resolution)
        feasible_poly = rect_poly.intersection(circle)
    else:
        feasible_poly = rect_poly

    # 3) Force convexity by taking the polygon’s convex hull
    feasible_poly = feasible_poly.convex_hull

    return feasible_poly

# -------------------------------------------------------------------------
# Helper: Minkowski sum of two convex polygons (linear time)
# -------------------------------------------------------------------------
def minkowski_sum(polyA, polyB):
    """
    Computes the Minkowski sum of two *convex* polygons polyA and polyB
    in O(n + m) time using a rotating-calipers-like approach.

    Conditions:
      1) polyA, polyB are convex.
      2) Their exterior coordinates are in CCW order.
         (Use shapely.ops.orient(poly, sign=1.0) if needed.)

    Returns a shapely Polygon representing polyA ⊕ polyB.
    """
    # 1) Handle empties
    if polyA.is_empty:
        return polyB
    if polyB.is_empty:
        return polyA

    # 2) Ensure each is convex and in CCW orientation
    polyA = orient(polyA.convex_hull, sign=1.0)
    polyB = orient(polyB.convex_hull, sign=1.0)

    # 3) Extract coordinates (excluding repeated last point)
    coordsA = list(polyA.exterior.coords)[:-1]
    coordsB = list(polyB.exterior.coords)[:-1]
    lenA = len(coordsA)
    lenB = len(coordsB)
    if lenA == 0:
        return polyB
    if lenB == 0:
        return polyA

    # Helper: find index of "lowest-leftmost" vertex
    def start_index(coords):
        idx = 0
        for i in range(1, len(coords)):
            y, x = coords[i][1], coords[i][0]
            y0, x0 = coords[idx][1], coords[idx][0]
            if (y < y0) or (y == y0 and x < x0):
                idx = i
        return idx

    startA = start_index(coordsA)
    startB = start_index(coordsB)

    # Current indices & current coordinate = sum of start vertices
    iA = startA
    iB = startB
    current_x = coordsA[iA][0] + coordsB[iB][0]
    current_y = coordsA[iA][1] + coordsB[iB][1]
    result = [(current_x, current_y)]

    # 2D cross product
    def cross(u, v):
        return u[0]*v[1] - u[1]*v[0]

    # next index in ring
    def next_idx(i, length):
        return (i + 1) % length

    # 4) "Walk" around both polygons
    steps = 0
    while steps < (lenA + lenB):
        iA_next = next_idx(iA, lenA)
        iB_next = next_idx(iB, lenB)
        # Edge vectors
        edgeA = (coordsA[iA_next][0] - coordsA[iA][0],
                 coordsA[iA_next][1] - coordsA[iA][1])
        edgeB = (coordsB[iB_next][0] - coordsB[iB][0],
                 coordsB[iB_next][1] - coordsB[iB][1])

        # Compare cross product
        cross_val = cross(edgeA, edgeB)
        if cross_val >= 0:
            # Advance polygon A
            iA = iA_next
            current_x += edgeA[0]
            current_y += edgeA[1]
        else:
            # Advance polygon B
            iB = iB_next
            current_x += edgeB[0]
            current_y += edgeB[1]

        result.append((current_x, current_y))
        steps += 1

    sum_poly = Polygon(result)
    # Typically already convex, but to be safe:
    sum_poly = sum_poly.convex_hull
    return sum_poly

# -------------------------------------------------------------------------
# Helper: build aggregated (P,Q) polygon from all devices on a bus
# -------------------------------------------------------------------------
def bus_flex_polygon(net_flex, bus_idx):
    """
    Build a shapely Polygon representing the aggregated (P,Q) region
    for *all* flexible devices at bus_idx in the pandapower net_flex.

    We look at net_flex.gen, net_flex.sgen, net_flex.storage.
    We assume each row has columns:
      - min_p_mw, max_p_mw
      - min_q_mvar, max_q_mvar
      - sn_mva (optional)

    Returns a shapely Polygon (could be empty if no flex devices).
    """
    import pandas as pd
    device_polys = []

    element_tables = [
        ('gen',    net_flex.gen),
        ('sgen',   net_flex.sgen),
        ('storage', net_flex.storage)
    ]

    for elem_type, df in element_tables:
        if df.empty:
            continue

        # find rows for this bus
        these_devs = df[df.bus == bus_idx]
        for idx, row in these_devs.iterrows():
            pmin = row.get('min_p_mw', 0.0)
            pmax = row.get('max_p_mw', 0.0)
            qmin = row.get('min_q_mvar', 0.0)
            qmax = row.get('max_q_mvar', 0.0)
            sn   = row.get('sn_mva', None)

            # build device polygon
            poly_dev = device_polygon(pmin, pmax, qmin, qmax, sn_mva=sn)
            device_polys.append(poly_dev)

    # Minkowski sum across all device polygons
    bus_poly = geom.Polygon()  # empty
    for p in device_polys:
        bus_poly = minkowski_sum(bus_poly, p)

    return bus_poly
