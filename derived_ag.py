import numpy as np
import networkx as nx

def massey_obstruction(returns, max_dim=2):
    """
    Compute derived obstruction via:
    1. Distance matrix from correlation: 1 - |corr|
    2. Build graph from edges with distance < median.
    3. Build 2‑simplices (triangles).
    4. For each node, score = (number of triangles containing node) × (1 + zero‑mode count)
    where zero‑modes = number of eigenvalues of Laplacian near zero.
    """
    corr = returns.corr().values
    n = corr.shape[0]
    if n < 3:
        return {t: 0.0 for t in returns.columns}
    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist, 0)
    # Build graph
    triu = np.triu_indices_from(dist, k=1)
    flat_dist = dist[triu]
    if len(flat_dist) == 0:
        return {t: 0.0 for t in returns.columns}
    threshold = np.median(flat_dist)
    G = nx.Graph()
    nodes = returns.columns.tolist()
    G.add_nodes_from(nodes)
    for i in range(n):
        for j in range(i+1, n):
            if dist[i,j] < threshold:
                G.add_edge(nodes[i], nodes[j])
    # Find triangles
    triangles = []
    for i in range(n):
        for j in range(i+1, n):
            if not G.has_edge(nodes[i], nodes[j]):
                continue
            for k in range(j+1, n):
                if G.has_edge(nodes[i], nodes[k]) and G.has_edge(nodes[j], nodes[k]):
                    triangles.append((i,j,k))
    # Count triangles per node
    node_tri_count = np.zeros(n)
    for (i,j,k) in triangles:
        node_tri_count[i] += 1
        node_tri_count[j] += 1
        node_tri_count[k] += 1
    # Avoid division by zero
    if np.max(node_tri_count) == 0:
        node_tri_score = np.zeros(n)
    else:
        node_tri_score = node_tri_count / np.max(node_tri_count)
    # Approximate cohomology dimension using Betti numbers from graph Laplacian
    # Number of connected components = zero eigenvalues of graph Laplacian
    if G.number_of_edges() == 0:
        cohom_dim = 0
    else:
        L = nx.laplacian_matrix(G).astype(float).todense()
        eigvals = np.linalg.eigvalsh(L)
        zero_modes = np.sum(np.abs(eigvals) < 1e-8)
        cohom_dim = zero_modes  # number of connected components
    # Final score: triangle participation times (1 + cohom_dim)
    score = node_tri_score * (1 + cohom_dim / max(1, n))
    return {nodes[i]: score[i] for i in range(n)}

def derived_ag_score(returns, max_dim=2):
    try:
        scores = massey_obstruction(returns, max_dim)
    except Exception as e:
        print(f"Derived AG error: {e}")
        scores = {ticker: 0.0 for ticker in returns.columns}
    return scores
