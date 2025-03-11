import numpy as np
import pandas as pd
import pandapower as pp
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


############################################################
# Helper: sample points uniformly *inside* a convex polygon
# polygon_vertices is a list of (x,y) tuples for the polygon
############################################################
def sample_points_in_polygon(polygon_vertices, n_samples):
    """
    Randomly sample n_samples points within the polygon defined by polygon_vertices.
    polygon_vertices : array-like of shape (n_vertices, 2)

    Returns: array of shape (n_samples, 2) with [P, Q]
    """
    poly = Polygon(polygon_vertices)
    minx, miny, maxx, maxy = poly.bounds

    # We’ll use a simple rejection sampling approach:
    pts = []
    count = 0
    while count < n_samples:
        # random point in bounding box
        rx = np.random.uniform(minx, maxx)
        ry = np.random.uniform(miny, maxy)
        if poly.contains(Point(rx, ry)):
            pts.append([rx, ry])
            count += 1

    return np.array(pts)


############################################################
# Helper: run power flow with given P/Q injection per bus
############################################################
def run_pf_with_injections(net, flex_buses, p_injections, q_injections):
    """
    Updates net.load or net.sgen elements at the specified flex_buses with
    the given active/reactive power injections p_injections, q_injections (MW, MVAr).
    Then runs a power flow.
    Returns True if constraints are satisfied, else False.
    """
    # For simplicity, assume each flexible bus has exactly one "controllable" element,
    # e.g. a single static generator (sgen) or negative load.
    # You should adapt to your actual data structure.
    for i, b in enumerate(flex_buses):
        # here we assume sgen for demonstration
        # find the sgen row for this bus:
        sgen_idx = net.sgen[net.sgen.bus == b].index
        if len(sgen_idx) != 1:
            raise ValueError("This example assumes exactly 1 sgen at each flexible bus!")
        # update sgen p (MW) and q (MVAr). Pandapower uses net.sgen['p_mw'] for active power,
        # but for Q we often need to set sgen['sn_mva'] + sgen['current_source'] or
        # sgen['scaling'] for Q control, or use net.gen. For simplicity, we store Q in 'q_mvar'.
        net.sgen.loc[sgen_idx, 'p_mw'] = p_injections[i]
        net.sgen.loc[sgen_idx, 'q_mvar'] = q_injections[i]

    # Now run the power flow
    try:
        pp.runpp(net)
    except:
        # If power flow doesn't converge, treat it as an infeasible point
        return False

    # Check constraints: voltage, line loading, etc.
    # For example, let's ensure all voltages are within +/- 5%
    # and lines are below 100% loading. You can adapt as needed.
    v_ok = net.res_bus.vm_pu.between(0.95, 1.05).all()
    loading_ok = (net.res_line.loading_percent < 100).all()

    return bool(v_ok and loading_ok)


############################################################
# Helper: compute the convex hull of a set of (P,Q) points
############################################################
def convex_hull_points(points):
    """
    Takes an array of shape (N, 2) of valid (P,Q) points.
    Returns the list of hull vertices in order.
    """
    hull = ConvexHull(points)
    vertices = points[hull.vertices]
    return vertices


def compute_FOR_via_random_sampling(net, flex_buses, bus_polygons,
                                    n_samples_per_bus=50,  # how many random draws per bus
                                    show_plot=True):
    """
    net           : pandapower network
    flex_buses    : list of bus indices that have flexible generation or load
    bus_polygons  : list of polygon vertices (array-like) for each bus in flex_buses
    n_samples_per_bus: how many random draws to sample per bus's polygon
    show_plot     : whether to display the final FOR plot

    Returns:
      feasible_points: array of shape (N, 2) storing feasible (P_in, Q_in) at the
                       coupling bus (slack bus or any bus you define as "boundary")
      hull_vertices  : array of shape (M, 2) with vertices of the convex hull
                       around feasible_points
    """
    # Just for demonstration, let's define the "interconnection" or "boundary"
    # as the slack bus power injection. Usually net.bus.slack= True or net.ext_grid.
    # We'll read net.res_ext_grid.p_mw and net.res_ext_grid.q_mvar after each PF.

    # For storing all feasible (P_in, Q_in) points.
    feasible_points = []

    # The naive approach below will pick random points for *each bus independently*
    # and do a single PF. That means total combinations = (n_samples_per_bus)^(#flex_buses).
    # That can explode quickly if you have many flexible buses.
    # In practice, you might do more advanced sampling or dimension reduction.
    # For demonstration, let's do a simple multi-loop.

    from itertools import product

    # For each bus, sample random points inside the polygon
    bus_samples = []
    for poly_verts in bus_polygons:
        bus_samples.append(sample_points_in_polygon(poly_verts, n_samples_per_bus))

    # bus_samples[i] is shape (n_samples_per_bus, 2).
    # We want to try all permutations => product(*bus_samples)
    # But be careful: product(...) of big lists can be huge.
    # For demonstration, let's just do a small example.
    # If you truly want random draws across all buses simultaneously,
    # see "Quadrants" or "Vertices Combination" approach in the paper.

    # We'll do a direct product for clarity. For large grids, you'll want a more targeted approach.
    all_combinations = list(product(*bus_samples))

    # all_combinations is a list of length (n_samples_per_bus^(len(flex_buses))).
    # Each item is a tuple like ((P1, Q1), (P2, Q2), ..., (Pn, Qn)).

    for combo in all_combinations:
        # combo is ((p1, q1), (p2, q2), ...)
        p_vec = np.array([c[0] for c in combo])
        q_vec = np.array([c[1] for c in combo])

        feasible = run_pf_with_injections(net, flex_buses, p_vec, q_vec)
        if feasible:
            # record (P_in, Q_in) at the external grid / slack
            # In pandapower, net.res_ext_grid.p_mw and .q_mvar are the total slack bus injection
            p_in = net.res_ext_grid.p_mw.values[0]
            q_in = net.res_ext_grid.q_mvar.values[0]
            feasible_points.append([p_in, q_in])

    feasible_points = np.array(feasible_points)
    if feasible_points.shape[0] == 0:
        print("No feasible operating points found!")
        return None, None

    # Now compute the convex hull
    hull_vertices = convex_hull_points(feasible_points)

    if show_plot:
        plt.figure()
        plt.scatter(feasible_points[:, 0], feasible_points[:, 1], marker='.', label='Feasible Points')
        # close the hull polygon for plotting
        closed_hull = np.vstack([hull_vertices, hull_vertices[0]])
        plt.plot(closed_hull[:, 0], closed_hull[:, 1], 'r-', label='FOR hull')
        plt.xlabel('P_in (MW)')
        plt.ylabel('Q_in (MVAr)')
        plt.title('Feasible Operating Region (Random Sampling)')
        plt.legend()
        plt.grid(True)
        plt.show()

    return feasible_points, hull_vertices


if __name__ == "__main__":
    # 1) Create a minimal pandapower net
    net = pp.create_empty_network(sn_mva=1.0)
    #net = pp.networks.create_cigre_network_mv()

    # Create 3 buses
    b0 = pp.create_bus(net, vn_kv=110, name="Grid Bus")
    b1 = pp.create_bus(net, vn_kv=110, name="Flex Bus 1")
    b2 = pp.create_bus(net, vn_kv=110, name="Flex Bus 2")

    # External grid at bus 0
    pp.create_ext_grid(net, b0, vm_pu=1.00)
    # Lines from b0->b1, b0->b2 (just placeholders)
    pp.create_line_from_parameters(net, from_bus=b0, to_bus=b1, length_km=1.0,
                                   r_ohm_per_km=0.1, x_ohm_per_km=0.1,
                                   c_nf_per_km=0, max_i_ka=1)
    pp.create_line_from_parameters(net, from_bus=b0, to_bus=b2, length_km=1.0,
                                   r_ohm_per_km=0.1, x_ohm_per_km=0.1,
                                   c_nf_per_km=0, max_i_ka=1)

    # 2) Create flexible "generators" at buses 1 and 2.
    #    You could also create loads with negative p_mw, or a battery model, etc.
    sg1 = pp.create_sgen(net, bus=b1, p_mw=0.0, q_mvar=0.0, name="Flex1", index=None)
    sg2 = pp.create_sgen(net, bus=b2, p_mw=0.0, q_mvar=0.0, name="Flex2", index=None)

    flex_buses = [b1, b2]

    # 3) Define polygon vertices for each bus's feasible region, e.g. triangles
    #    Suppose bus 1 can do:
    #       - P from 0 to +2 MW
    #       - Q from -1 to +1 MVAr
    #    But let's pick a triangular shape to illustrate irregular polygons
    poly1 = np.array([
        [0.0, 0.0],
        [2.0, 1.0],
        [2.0, -1.0]
    ])
    # Suppose bus 2 can do:
    #       - P from -1 to +1 MW
    #       - Q from -2 to 0 MVAr
    #    Another triangular shape
    poly2 = np.array([
        [-1.0, -2.0],
        [1.0, -2.0],
        [1.0, 0.0]
    ])

    bus_polygons = [poly1, poly2]

    # 4) Compute FOR with random sampling
    feasible_pts, hull = compute_FOR_via_random_sampling(
        net,
        flex_buses=flex_buses,
        bus_polygons=bus_polygons,
        n_samples_per_bus=10,
        show_plot=True
    )

    print(f"Found {feasible_pts.shape[0]} feasible points!")