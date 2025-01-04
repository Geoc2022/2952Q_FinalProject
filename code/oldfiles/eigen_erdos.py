import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np


def generate_erdos_renyi_graph(n, p, draw=False):
    # Generate an Erdős-Rényi graph with n nodes and edge probability p
    G = nx.erdos_renyi_graph(n, p)

    # Draw the graph
    if draw:
        plt.figure(figsize=(8, 8))
        nx.draw(G, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500, font_size=10)
        plt.title(f"Erdős–Rényi Graph (n={n}, p={p})")
        plt.show()
    
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
    
    return G_prime, G_double_prime


def find_eigen(G, k):
    # L = nx.laplacian_matrix(G).toarray()
    A = nx.adjacency_matrix(G).toarray()
    A = np.linalg.matrix_power(A, k)
    eigvals, eigvecs = np.linalg.eig(A)
    return eigvals, eigvecs

def check_max_eigenvalue():
    def plot_eigenvalues(data):
        plt.figure(figsize=(8, 8))
        for p, values in data.items():
            plt.plot(range(1, 40), values, label=f"p={p}")
        plt.xlabel("k")
        plt.ylabel("Eigenvalue")
        plt.title("Eigenvalues of Erdős–Rényi Graphs")
        plt.legend()
        plt.show()

    n = 100
    egian_data = {
        p: [] for p in np.arange(0.1, 1, 0.1)
    }
    for p in np.arange(0.1, 1, 0.1):
        G = generate_erdos_renyi_graph(n, p)
        G_prime, G_double_prime = compute_modified_laplacians(G, 0.1)
        for k in range(1, 40):
            eigvals, eigvecs = find_eigen(G, k)
            print(f"p={p} k={k} Eigenvalues: {np.power(np.max(eigvals), 1 / k) / n}")
            egian_data[p].append(np.power(np.max(eigvals), 1 / k) / n)
    plot_eigenvalues(egian_data)

check_max_eigenvalue()

# def check_eigenvalues():
#     def plot_eigenvalues(data):
#         plt.figure(figsize=(8, 8))
#         for p, values in data.items():
#             plt.plot(range(0, 100), values, label=f"p={p}")
#         plt.xlabel("k")
#         plt.ylabel("Eigenvalue")
#         plt.title("Eigenvalues of Erdős–Rényi Graphs")
#         plt.legend()
#         plt.show()

#     n = 100
#     p = 0.5
#     G = generate_erdos_renyi_graph(n, p)
#     G_prime, G_double_prime = compute_modified_laplacians(G, 0.1)
#     data = np.array()
#     for k in range(1, 10):
#         eigvals, eigvecs = find_eigen(G, k)
#         print(f"p={p} k={k} Eigenvalues: {np.power(eigvals, 1 / k) / n}")
        

#     plot_eigenvalues(data)

# check_eigenvalues()