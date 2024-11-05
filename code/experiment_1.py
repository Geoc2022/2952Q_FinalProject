import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations, groupby
import random
import numpy as np

def generate_erdos_renyi_graph(n, p, draw=False):
    # Generate an Erdős-Rényi graph with n nodes and edge probability p
    G = nx.erdos_renyi_graph(n, p)
    L = np.array(nx.laplacian_matrix(G).toarray())

    # Draw the graph
    if draw:
        plt.figure(figsize=(8, 8))
        nx.draw(G, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500, font_size=10)
        plt.title(f"Erdős–Rényi Graph (n={n}, p={p})")
        plt.show()
    
    return G, L

def weight_one_random_graph(n, p):
    """
    Generates a random undirected graph, similarly to an Erdős-Rényi 
    graph, but enforcing that the resulting graph is conneted
    """
    edges = combinations(range(n), 2)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if p <= 0:
        return G
    if p >= 1:
        return nx.complete_graph(n, create_using=G)
    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        weight = 1
        G.add_edge(*random_edge, weight=weight)
        for e in node_edges:
            if random.random() < p:
                weight = 1
                G.add_edge(*e, weight=weight)
    return G

def compute_modified_laplacians(G, epsilon):
    n = len(G.nodes)
    
    # Step 1: Select subset B of size approximately epsilon * n
    B = set(random.sample(list(G.nodes), int(epsilon * n)))
    
    # Step 2: Generate edge sets for G' and G''
    G_prime = G.copy()
    G_double_prime = G.copy()
    
    # For G': Add edges (u, v) if u or v is in B
    for u in B:
        for v in G.nodes:
            if u != v and not G_prime.has_edge(u, v):
                G_prime.add_edge(u, v)
    
    # For G'': Remove edges (u, v) if u or v is in B
    for u in B:
        neighbors = list(G_double_prime.adj[u])  # Convert neighbors to list to avoid modification during iteration
        for v in neighbors:
            if G_double_prime.has_edge(u, v):
                G_double_prime.remove_edge(u, v)
    
    # Step 3: Compute Laplacians for G' and G''
    L_prime = nx.laplacian_matrix(G_prime).toarray()
    L_double_prime = nx.laplacian_matrix(G_double_prime).toarray()
    
    return L_prime, L_double_prime

# Example usage
n = 1000  # Number of nodes
p = 0.3  # Probability of an edge
epsilon = 0.1
G, L = generate_erdos_renyi_graph(n, p)
L_p, L_pp = compute_modified_laplacians(G, epsilon)

D = np.diag(L)
D_p = np.diag(L_p)
D_pp = np.diag(L_pp)

print(f"Mean of G: {np.mean(D) / n}")
print(f"Median of G: {np.median(D) / n}\n")

print(f"Mean of G': {np.mean(D_p) / n}")
print(f"Median of G': {np.median(D_p) / n}\n")

print(f"Mean of G'': {np.mean(D_pp) / n}")
print(f"Median of G'': {np.median(D_pp) / n}")