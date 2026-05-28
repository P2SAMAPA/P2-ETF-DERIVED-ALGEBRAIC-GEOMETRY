import numpy as np
import networkx as nx

def derived_ag_score(returns, max_dim=2):
    """
    Derived algebraic geometry obstruction via betweenness × clustering coefficient.
    High score = structurally important in the simplicial complex.
    """
    # Distance matrix from correlation
    corr = returns.corr().values
    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist, 0)
    # Build graph with a sparser threshold (75th percentile of distances)
    triu = np.triu_indices_from(dist, k=1)
    flat_dist = dist[triu]
    if len(flat_dist) == 0:
        return {ticker: 0.0 for ticker in returns.columns}
    threshold = np.percentile(flat_dist, 75)  # only keep 25% strongest edges
    nodes = returns.columns.tolist()
    G = nx.Graph()
    G.add_nodes_from(nodes)
    n = len(nodes)
    for i in range(n):
        for j in range(i+1, n):
            if dist[i,j] < threshold:
                G.add_edge(nodes[i], nodes[j])
    # If graph is disconnected, use largest connected component
    if nx.number_connected_components(G) > 1:
        # Take largest component
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        # For nodes not in largest, set score 0
        missing = set(nodes) - set(largest_cc)
    else:
        missing = set()
    # Compute betweenness centrality (normalized)
    try:
        betweenness = nx.betweenness_centrality(G, normalized=True)
    except:
        betweenness = {node: 0.0 for node in G.nodes()}
    # Compute local clustering coefficient (fraction of possible triangles)
    clustering = nx.clustering(G)
    # Score = betweenness * (1 + clustering) – varies across nodes
    scores = {}
    for node in nodes:
        if node in missing:
            scores[node] = 0.0
        else:
            b = betweenness.get(node, 0.0)
            c = clustering.get(node, 0.0)
            scores[node] = b * (1 + c)
    return scores
