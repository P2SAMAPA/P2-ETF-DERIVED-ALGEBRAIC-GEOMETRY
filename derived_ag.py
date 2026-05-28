import numpy as np
import networkx as nx
from scipy.linalg import eigh

def massey_obstruction(returns, max_dim=2):
    """
    Compute per‑ETF Massey product obstruction using triangle centrality
    and the dimension of the graph's cycle space (a proxy for cohomology).
    """
    # Distance matrix
    corr = returns.corr().values
    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist, 0)
    n = dist.shape[0]
    nodes = returns.columns.tolist()

    # Build graph with edges where distance < median distance
    triu = np.triu_indices_from(dist, k=1)
    flat_dist = dist[triu]
    threshold = np.median(flat_dist)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for i in range(n):
        for j in range(i+1, n):
            if dist[i,j] < threshold:
                G.add_edge(nodes[i], nodes[j])
    
    # Compute triangle participation for each node
    triangles = [list(nx.enumerate_all_cliques(G)) if max_dim >= 2 else []]
    # Actually triangle enumeration is expensive; use built-in triangle count
    triangle_counts = nx.triangles(G)  # dict node -> count
    # Normalise
    max_count = max(triangle_counts.values()) if triangle_counts else 1
    triangle_score = {node: count / max_count for node, count in triangle_counts.items()}
    
    # Cohomology dimension approximated by number of cycles (cyclomatic number)
    # cyclomatic number = |E| - |V| + number_of_components
    n_edges = G.number_of_edges()
    n_components = nx.number_connected_components(G)
    cohom_dim = n_edges - n + n_components
    cohom_dim = max(0, cohom_dim)  # non-negative
    
    # Per‑ETF score = triangle participation * (1 + cohom_dim / max(1, n))
    score = {node: triangle_score.get(node, 0) * (1 + cohom_dim / max(1, n)) for node in nodes}
    return score

def derived_ag_score(returns, max_dim=2):
    """
    Wrapper for compatibility.
    """
    try:
        scores = massey_obstruction(returns, max_dim)
    except Exception as e:
        print(f"Derived AG error: {e}")
        scores = {ticker: 0.0 for ticker in returns.columns}
    return scores
