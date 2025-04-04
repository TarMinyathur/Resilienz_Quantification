#indi_gt: indicators from graph theory

import networkx as nx
from initialize import add_indicator
from networkx.algorithms import community

def GraphenTheorieIndicator(graph, dfinalresults, printing= False):


    # Now calculate average shortest path length for the largest connected component
    if graph.number_of_nodes() > 1:
        avg_path_length = nx.average_shortest_path_length(graph)
        print(f"Average Shortest Path Length: {avg_path_length}")
    else:
        print("graph has only one node, cannot calculate average shortest path length.")

    # calculating of a way to normalize the average shortest path length
    num_nodes = graph.number_of_nodes()
    num_nodes = (num_nodes - 1) if num_nodes > 1 else 0
    norm_avg_pl = max(0, 1 - (avg_path_length / num_nodes))
    #print(f"Datatype of norm_avg_pl: {type(norm_avg_pl)}")
    print(f"Normalized Average Path Length: {norm_avg_pl}")
    dfinalresults = add_indicator(dfinalresults, 'Average Shortest Path Length', norm_avg_pl)

    # Calculate degree centrality
    degree_centrality = nx.degree_centrality(graph)

    # Calculate average degree centrality
    avg_degree_centrality = sum(degree_centrality.values()) / len(degree_centrality)

    if printing:
        print(f"Degree Centrality:")
        for node, centrality in degree_centrality.items():
            print(f"Bus {node}: {centrality}")

    print(f"\nAverage Degree Centrality: {avg_degree_centrality}")

    dfinalresults = add_indicator(dfinalresults, 'Average Degree Centrality', max(0, avg_degree_centrality))

    # Detect communities (optional): Using Louvain method
    communities = community.greedy_modularity_communities(graph)

    # Calculate modularity index
    modularity_index = calculate_modularity_index(graph, communities)

    print(f"Modularity Index (Q): {modularity_index}")

    dfinalresults = add_indicator(dfinalresults, 'Modularity Index', max(0, modularity_index))

    return dfinalresults

def calculate_modularity_index(G, communities):
    modularity_index = 0.0

    # Calculate total number of edges in the graph
    total_edges = G.number_of_edges()

    # Calculate modularity components for each community
    for community_nodes in communities:
        # Calculate e_ii: Fraction of edges within the community
        e_ii = sum(1 for u, v in G.edges(community_nodes) if v in community_nodes) / total_edges

        # Calculate a_i: Total fraction of edges from nodes in the community
        a_i = sum(1 for u, v in G.edges(community_nodes) if v not in community_nodes) / total_edges

        # Calculate (e_ii - a_i)^2 and accumulate to modularity index
        modularity_index += (e_ii - a_i) ** 2

    return modularity_index
