#indi_gt: indicators from graph theory

import networkx as nx
from initialize import add_indicator
from networkx.algorithms import community
from modularity import calculate_modularity_index

def GraphenTheorieIndicator(Graph, dfinalresults):


    # Now calculate average shortest path length for the largest connected component
    if Graph.number_of_nodes() > 1:
        avg_path_length = nx.average_shortest_path_length(Graph)
        print(f"Average Shortest Path Length: {avg_path_length}")
    else:
        print("Graph has only one node, cannot calculate average shortest path length.")

    # calculating of a way to normalize the average shortest path length
    num_nodes = Graph.number_of_nodes()
    num_nodes = (num_nodes - 1) if num_nodes > 1 else 0
    norm_avg_pl = max(0, 1 - (avg_path_length / num_nodes))
    print(f"Datatype of norm_avg_pl: {type(norm_avg_pl)}")
    print(f"Normalized Average Path Length: {norm_avg_pl}")
    dfinalresults = add_indicator(dfinalresults, 'Average Shortest Path Length', norm_avg_pl)

    # Calculate degree centrality
    degree_centrality = nx.degree_centrality(Graph)

    # Calculate average degree centrality
    avg_degree_centrality = sum(degree_centrality.values()) / len(degree_centrality)

    print(f"Degree Centrality:")
    for node, centrality in degree_centrality.items():
        print(f"Bus {node}: {centrality}")

    print(f"\nAverage Degree Centrality: {avg_degree_centrality}")

    dfinalresults = add_indicator(dfinalresults, 'Average Degree Centrality', max(0, avg_degree_centrality))

    # Detect communities (optional): Using Louvain method
    communities = community.greedy_modularity_communities(Graph)

    # Calculate modularity index
    modularity_index = calculate_modularity_index(Graph, communities)

    print(f"Modularity Index (Q): {modularity_index}")

    dfinalresults = add_indicator(dfinalresults, 'Modularity Index', max(0, modularity_index))
