import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigs

def massey_obstruction(returns, max_dim=2):
    """
    Compute per‑ETF derived obstruction using:
      1. Distance matrix = 1 - |corr|
      2. Build graph with edges where distance < median.
      3. Enumerate triangles (2‑simplices).
      4. Build triangle adjacency graph (nodes = ETFs, edges connect ETFs that share a triangle).
      5. Score = eigenvector centrality of that triangle‑adjacency graph.
    """
    corr = returns.corr().values
    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist, 0)
    n = dist.shape[0]
    nodes = returns.columns.tolist()
    
    # Build graph with threshold = median distance
    triu = np.triu_indices_from(dist, k=1)
    flat_dist = dist[triu]
    threshold = np.median(flat_dist) if len(flat_dist) > 0 else 0.5
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for i in range(n):
        for j in range(i+1, n):
            if dist[i,j] < threshold:
                G.add_edge(nodes[i], nodes[j])
    
    # Enumerate triangles (3‑cycles)
    triangles = list(nx.enumerate_all_cliques(G))
    triangles = [tri for tri in triangles if len(tri) == 3]
    
    # Build triangle‑adjacency: two ETFs share a triangle if they belong to same triangle?
    # Instead, build a graph where nodes are ETFs, and edge weight = number of triangles containing both.
    # Then compute eigenvector centrality of that weighted graph.
    tri_adj = np.zeros((n, n))
    for tri in triangles:
        idx = [nodes.index(t) for t in tri]
        for a in idx:
            for b in idx:
                if a != b:
                    tri_adj[a,b] += 1
    # Symmetrise
    tri_adj = (tri_adj + tri_adj.T) / 2
    # Create graph from adjacency
    H = nx.Graph()
    H.add_nodes_from(nodes)
    for i in range(n):
        for j in range(i+1, n):
            if tri_adj[i,j] > 0:
                H.add_edge(nodes[i], nodes[j], weight=tri_adj[i,j])
    
    # Compute eigenvector centrality
    if H.number_of_edges() == 0:
        scores = np.ones(n) / n
    else:
        centrality = nx.eigenvector_centrality_numpy(H, weight='weight')
        scores = np.array([centrality[t] for t in nodes])
        # Normalise to [0,1]
        if np.max(scores) > 0:
            scores = scores / np.max(scores)
    
    return {nodes[i]: scores[i] for i in range(n)}

def derived_ag_score(returns, max_dim=2):
    try:
        scores = massey_obstruction(returns, max_dim)
    except Exception as e:
        print(f"Derived AG error: {e}")
        scores = {ticker: 0.0 for ticker in returns.columns}
    return scores
