# modularity.py

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