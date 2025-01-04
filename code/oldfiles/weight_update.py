import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations, groupby
import random
import numpy as np
import math
from scipy.optimize import curve_fit, minimize
from sklearn.metrics import r2_score

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

def graph_from_laplacian(L):
    """
    Creates a NetworkX graph from a given Laplacian matrix.
    
    Parameters:
        L (np.ndarray): The Laplacian matrix of the graph.
        
    Returns:
        nx.Graph: The reconstructed graph.
    """
    # Verify that L is a square matrix
    if L.shape[0] != L.shape[1]:
        raise ValueError("Laplacian matrix must be square.")
    
    # Recover the adjacency matrix A from the Laplacian L
    # A = D - L, where D is the degree matrix
    degrees = np.diag(L)  # Degrees are on the diagonal of L
    A = np.diag(degrees) - L  # Adjacency matrix
    
    # Create the graph from the adjacency matrix
    G = nx.from_numpy_array(A)
    
    return G

def perturb_graph(G, epsilon, p):
    """
    Perturbs the graph G by adding edges with probability p between an epsilon-fraction of nodes.

    Parameters:
    - G: networkx.Graph - The original graph.
    - epsilon: float - Fraction of nodes to perturb (0 < epsilon <= 1).
    - p: float - Probability of adding an edge between two nodes (0 <= p <= 1).

    Returns:
    - networkx.Graph - The perturbed graph.
    """
    # Create a copy of the graph to avoid modifying the original
    perturbed_G = G.copy()
    
    # Calculate the number of nodes to perturb
    n = len(G.nodes)
    perturb_count = int(epsilon * n)
    
    # Convert nodes to a list before sampling
    nodes_list = list(G.nodes)
    
    # Randomly sample epsilon-fraction of nodes
    nodes_to_perturb = random.sample(nodes_list, perturb_count)
    
    # For each selected node, decide to add edges with probability p
    for u in nodes_to_perturb:
        for v in G.nodes:
            if u != v:  # avoid self-loops and existing edges
                if random.random() < p:
                    if not perturbed_G.has_edge(u, v):
                        perturbed_G.add_edge(u, v)
                elif perturbed_G.has_edge(u, v):
                    perturbed_G.remove_edge(u, v)
    perturbed_L = np.array(nx.laplacian_matrix(perturbed_G).toarray())
    return perturbed_G, perturbed_L

def perturb_graph_general(G, epsilon):
    """
    Perturbs the graph G by adding edges with probability p between an epsilon-fraction of nodes.

    Parameters:
    - G: networkx.Graph - The original graph.
    - epsilon: float - Fraction of nodes to perturb (0 < epsilon <= 1).
    - p: float - Probability of adding an edge between two nodes (0 <= p <= 1).

    Returns:
    - networkx.Graph - The perturbed graph.
    """
    # Create a copy of the graph to avoid modifying the original
    perturbed_G = G.copy()
    
    # Calculate the number of nodes to perturb
    n = len(G.nodes)
    perturb_count = int(epsilon * n)
    
    # Convert nodes to a list before sampling
    nodes_list = list(G.nodes)
    
    # Randomly sample epsilon-fraction of nodes
    nodes_to_perturb = random.sample(nodes_list, perturb_count)
    
    # For each selected node, decide to add edges with probability p
    for u in nodes_to_perturb:
        for v in G.nodes:
            if u != v:  # avoid self-loops and existing edges
                p = random.random()
                if random.random() < p:
                    if not perturbed_G.has_edge(u, v):
                        perturbed_G.add_edge(u, v)
                elif perturbed_G.has_edge(u, v):
                    perturbed_G.remove_edge(u, v)
    perturbed_L = np.array(nx.laplacian_matrix(perturbed_G).toarray())
    return perturbed_G, perturbed_L

def robust_estimate_guess(G, epsilon):
    L = np.array(nx.laplacian_matrix(G).toarray())
    D = np.diag(L)
    n = len(D)
    mean = np.mean(D) / n
    med = np.median(D) / n
    y = (math.e ** epsilon) * abs(mean - med)
    # y = abs(mean - med) / (1 - epsilon)
    # y = abs(mean - med) / (math.e ** (-epsilon))
    if med <= mean:
        return med - y
    else:
        return med + y

def prune_then_mean_median(G, epsilon):
    epsilon = 1.0 * epsilon
    # Copy the graph to avoid modifying the original
    modified_G = G.copy()
    
    # Calculate the number of nodes to remove
    n = len(G.nodes)
    nodes_to_remove_count = int(epsilon * n)
    
    # Get the nodes sorted by degree
    degree_sorted_nodes = sorted(G.degree, key=lambda x: x[1])  # Sort by degree (ascending)
    
    # Identify the nodes to remove
    nodes_to_remove = (
        [node for node, _ in degree_sorted_nodes[:nodes_to_remove_count]] +  # Lowest degrees
        [node for node, _ in degree_sorted_nodes[-nodes_to_remove_count:]]   # Highest degrees
    )
    
    # Remove the identified nodes
    modified_G.remove_nodes_from(nodes_to_remove)
    L = np.array(nx.laplacian_matrix(modified_G).toarray())
    D = np.diag(L)
    mean = np.mean(D) / ((1 - 2 * epsilon) * n)
    med = np.median(D) / ((1 - 2 * epsilon) * n)
    return mean, med

def compute_sigma(w, n):
    """
    Compute the standard deviation of vertex degrees in the weighted graph.
    """
    degrees = np.zeros(n)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            degrees[i] += w[idx]
            degrees[j] += w[idx]
            idx += 1
    sigma = np.std(degrees)
    return sigma

# def compute_mean(w, n):
#     """
#     Compute the standard deviation of vertex degrees in the weighted graph.
#     """
#     degrees = np.zeros(n)
#     idx = 0
#     for i in range(n):
#         for j in range(i+1, n):
#             degrees[i] += w[idx]
#             degrees[j] += w[idx]
#             idx += 1
#     mean = np.mean(degrees)
#     return mean

# def gradient_sigma(w, n):
#     """
#     Compute the gradient of sigma(w) with respect to w.
#     """
#     # Compute degrees
#     degrees = np.zeros(n)
#     idx = 0
#     edge_to_vertices = []
#     for i in range(n):
#         for j in range(i+1, n):
#             degrees[i] += w[idx]
#             degrees[j] += w[idx]
#             edge_to_vertices.append((i, j))
#             idx += 1
#     mean_degree = np.mean(degrees)
#     # Compute gradient
#     grad = np.zeros(len(w))
#     idx = 0
#     for i, j in edge_to_vertices:
#         # Partial derivative of degree variance with respect to w[idx]
#         grad[idx] = (
#             (degrees[i] - mean_degree) * 2 +
#             (degrees[j] - mean_degree) * 2
#         ) / (2 * n)
#         idx += 1
#     # Adjust for standard deviation
#     sigma = np.std(degrees)
#     if sigma != 0:
#         grad /= (2 * sigma)
#     else:
#         grad = np.zeros_like(w)
#     return grad

# def project_onto_delta(w, n, epsilon):
#     """
#     Project w onto the feasible set Delta_{n, epsilon}.
#     """
#     # Projection onto the infinity norm constraint
#     w = np.clip(w, 0, 1 / ((1 - epsilon) * (n * (n - 1) / 2)))
#     # Projection onto the L1 norm equality constraint
#     total_weight = np.sum(w)
#     desired_total = n * (n - 1) / 2
#     if total_weight != desired_total:
#         w *= desired_total / total_weight
#     return w

# def gradient_descent(G, epsilon, sigma_hat, max_iter=1000, learning_rate=0.01):
#     """
#     Perform gradient descent to minimize f(w) over Delta_{n, epsilon}.
#     """
#     n = len(G.nodes)  # Number of vertices

#     # Initialize a numpy array of size n(n-1)/2
#     num_edges = n * (n - 1) // 2
#     w = np.zeros(num_edges, dtype=float)

#     # Get a mapping of edge indices
#     edge_index_map = {}
#     index = 0
#     for i in range(n):
#         for j in range(i + 1, n):
#             edge_index_map[(i, j)] = index
#             index += 1

#     # Fill the edge_array based on connections in G
#     for u, v in G.edges:
#         if u > v:  # Ensure (u, v) is ordered
#             u, v = v, u
#         w[edge_index_map[(u, v)]] = 1

#     for iteration in range(max_iter):
#         # Compute sigma(w)
#         sigma_w = compute_sigma(w, n)
#         # Compute the gradient of f(w)
#         grad_sigma = gradient_sigma(w, n)
#         grad_f = 2 * (sigma_w - sigma_hat) * grad_sigma
#         # Update w
#         w -= learning_rate * grad_f
#         # Project w back onto Delta_{n, epsilon}
#         w = project_onto_delta(w, n, epsilon)
#         # Optionally, print progress
#         if iteration % 100 == 0:
#             f_w = (sigma_w - sigma_hat) ** 2
#             print(f"Iteration {iteration}, f(w): {f_w:.6f}, sigma(w): {sigma_w:.6f}")
#     return w

def gradient_descent_graph(G, sigma_hat, epsilon, learning_rate=0.01, max_iters=1000, tol=1e-6):
    # Get the number of nodes and edges
    n = G.number_of_nodes()
    m = n * (n - 1) // 2  # Total possible edges in a complete graph

    # Initialize weights for all edges (1 for edges in G, 0 for non-edges)
    w = np.zeros((n, n))
    for u, v in G.edges():
        w[u, v] = w[v, u] = 1

    # Maximum weight per edge constraint
    max_weight = 1 / ((1 - epsilon) * m)

    # Helper functions
    def compute_sigma(weights):
        """Compute the standard deviation of weighted degrees."""
        degrees = np.sum(weights, axis=1)  # Sum of weights along rows gives the degrees
        return np.std(degrees)

    def project_weights(weights):
        """Project weights onto the feasible set."""
        weights = np.clip(weights, 0, max_weight)  # Ensure weights are in [0, max_weight]
        row_sums = np.sum(weights, axis=1)
        scale_factor = m / np.sum(row_sums)
        return weights * scale_factor

    # Gradient descent loop
    prev_loss = float('inf')
    for iteration in range(max_iters):
        # Compute current standard deviation
        sigma_w = compute_sigma(w)

        # Compute gradient (approximation)
        degrees = np.sum(w, axis=1)
        gradient = 2 * (sigma_w - sigma_hat) * (degrees - np.mean(degrees)).reshape(-1, 1)

        # Update weights using gradient descent
        w -= learning_rate * gradient

        # Project weights back to feasible set
        w = project_weights(w)

        # Compute loss
        loss = (sigma_w - sigma_hat) ** 2

        # Check for convergence
        if abs(prev_loss - loss) < tol:
            print(f"Converged after {iteration} iterations.")
            break
        prev_loss = loss

    return w, loss

def adjust_weights(G, epsilon, sigma_hat, max_iter=100, tol=1e-6, learning_rate=0.01):
    """
    Iteratively adjust weights to minimize the deviation from sigma_hat.

    Parameters:
        G (networkx.Graph): The input graph.
        epsilon (float): Fraction of vertices potentially perturbed by an adversary.
        sigma_hat (float): Target standard deviation.
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for convergence.
        learning_rate (float): Step size for gradient updates.

    Returns:
        np.ndarray: Optimized weights for edges.
    """
    n = len(G.nodes)
    m = len(G.edges)
    edges = list(G.edges)
    
    # Initialize weights: 1 for existing edges, 0 for non-edges
    w = np.array([1 if G[edge[0]][edge[1]]['weight'] == 1 else 0 for edge in edges], dtype=float)
    
    # Function to compute sigma(w)
    def compute_sigma(w):
        degrees = np.array([sum(w[i] for i, edge in enumerate(edges) if v in edge) for v in G.nodes])
        return np.std(degrees)

    # Gradient of f(w) = (sigma(w) - sigma_hat)^2
    def gradient(w):
        degrees = np.array([sum(w[i] for i, edge in enumerate(edges) if v in edge) for v in G.nodes])
        mean_deg = np.mean(degrees)
        std_deg = np.std(degrees)
        grad = np.zeros_like(w, dtype=float)  # Ensure floating-point type
        
        for i, edge in enumerate(edges):
            for v in edge:
                grad[i] += (degrees[v] - mean_deg) / (len(degrees) * std_deg)
        grad *= 2 * (std_deg - sigma_hat)
        return grad

    # Project weights onto Delta_{n, epsilon}
    def project(w):
        total_sum = w.sum()
        w = np.clip(w, 0, 1 / ((1 - epsilon) * m))
        scaling_factor = m / total_sum
        return w * scaling_factor

    # Gradient descent loop
    for i in range(max_iter):
        sigma = compute_sigma(w)
        f_val = (sigma - sigma_hat) ** 2

        # Compute gradient
        grad = gradient(w)
        
        # Update weights
        w -= learning_rate * grad
        
        # Project onto Delta_{n, epsilon}
        w = project(w)
        
        # Check for convergence
        if f_val < tol:
            print(f"Converged in {i + 1} steps")
            break

    return w

n = 100
p = 0.3
epsilon = 0.1
q = 0.8
G, L = generate_erdos_renyi_graph(n, p)
D = np.diag(L)
Gp, Lp = perturb_graph(G, epsilon, q)
complete_graph = nx.complete_graph(Gp.nodes)

# Step 2: Update edge weights based on G
for u, v in complete_graph.edges():
    if Gp.has_edge(u, v):
        complete_graph[u][v]['weight'] = 1  # Edge exists in G
    else:
        complete_graph[u][v]['weight'] = 0  # Edge does not exist in G
Gp = complete_graph
Dp = np.diag(Lp)
mean = np.mean(Dp)/n
var = np.var(Dp)  # Default: Population variance
p_hat = robust_estimate_guess(Gp, epsilon)

sigma_hat = np.sqrt(n * p_hat * (1 - p_hat))  # Estimated standard deviation
print(f"q: {q}\nTrue p: {p}\nMean: {mean}\nRobust Estimation: {robust_estimate_guess(Gp, epsilon)}")
print(f"Corrupted Std: {var ** 0.5}\nTrue Std: {(n - 1) * p * (1 - p)}")
# Run gradient descent
w_opt = adjust_weights(Gp, epsilon, sigma_hat, max_iter=1000)
print(w_opt)
edges = list(Gp.edges)
degrees = np.array([sum(w_opt[i] for i, edge in enumerate(edges) if v in edge) for v in Gp.nodes])
# Compute final sigma(w)
sigma_w_opt = compute_sigma(w_opt, n)
print(f"Weight Robust Estimation: {np.mean(degrees) / n}\nRobust Std: {sigma_w_opt}")
