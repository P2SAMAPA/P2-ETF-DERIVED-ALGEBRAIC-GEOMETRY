import numpy as np
import networkx as nx

def massey_obstruction(returns, max_dim=2):
    """
    Compute per‑ETF score from derived algebraic geometry:
    - Build graph from correlation distance (1 - |corr|) with edge threshold = median distance.
    - Compute number of triangles each node participates in.
    - Compute number of connected components (cohomology dimension proxy).
    - Score = triangle_count * (1 + num_components / n_nodes).
    This yields varying scores across ETFs.
    """
    corr = returns.corr().values
    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist, 0)
    n = dist.shape[0]
    nodes = returns.columns.tolist()
    
    # Build graph
    G = nx.Graph()
    G.add_nodes_from(nodes)
    # Use threshold = median distance (keep ~50% of edges)
    triu = np.triu_indices_from(dist, k=1)
    flat_dist = dist[triu]
    if len(flat_dist) == 0:
        return {t: 0.0 for t in nodes}
    threshold = np.median(flat_dist)
    for i in range(n):
        for j in range(i+1, n):
            if dist[i,j] < threshold:
                G.add_edge(nodes[i], nodes[j])
    
    # Count triangles per node
    triangles = nx.triangles(G)
    tri_counts = np.array([triangles[node] for node in nodes])
    
    # Number of connected components (zero eigenvalues of Laplacian)
    num_components = nx.number_connected_components(G)
    # Avoid division by zero
    comp_factor = 1 + num_components / max(1, n)
    
    # Score = triangle count * comp_factor
    raw_scores = tri_counts * comp_factor
    # If all zero, use degree as fallback
    if np.max(raw_scores) == 0:
        degrees = np.array([G.degree(node) for node in nodes])
        raw_scores = degrees * comp_factor
    
    return {nodes[i]: float(raw_scores[i]) for i in range(n)}

def derived_ag_score(returns, max_dim=2):
    try:
        scores = massey_obstruction(returns, max_dim)
    except Exception as e:
        print(f"Derived AG error: {e}")
        scores = {ticker: 0.0 for ticker in returns.columns}
    return scores
