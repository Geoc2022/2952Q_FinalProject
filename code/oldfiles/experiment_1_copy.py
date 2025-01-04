import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations, groupby
import random
import numpy as np
import math

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

def mean_adj_median_algo(G_prime):
    """
    Estimates p for a corrupted Erdős-Rényi graph using the Median-of-Means estimator.
    
    Parameters:
        G_prime (nx.Graph): The corrupted graph with an epsilon fraction of adversarial changes.
        epsilon (float): Fraction of nodes that have been changed by the adversary.
        
    Returns:
        float: Robust estimate of p.
    """
    n = G_prime.number_of_nodes()  # Number of nodes
    degrees = np.array([degree for _, degree in G_prime.degree()])  # Degree sequence
    
    # Set the number of groups for the Median-of-Means estimator
    k = int(np.sqrt(n))
    if k <= 1:
        raise ValueError("Graph is too small for robust estimation.")

    # Randomly split the degree sequence into k groups
    np.random.shuffle(degrees)
    groups = np.array_split(degrees, k)
    
    # Calculate the mean of each group
    group_means = [np.mean(group) for group in groups]
    
    # Take the median of the group means to get the MOM estimate for the average degree
    estimated_avg_degree = np.median(group_means)
    
    # Convert the average degree to an estimate for p
    # The expected degree in an Erdős-Rényi graph G(n, p) is (n - 1) * p
    p_estimate = estimated_avg_degree / (n - 1)
    
    return p_estimate

def random_walk_return_probability(adj_matrix, node, t):
    """
    Perform a random walk from the given node for t steps and
    calculate the return probability to the starting node.
    """
    n = adj_matrix.shape[0]
    current_node = node
    returns = 0
    
    for _ in range(t):
        neighbors = np.where(adj_matrix[current_node] == 1)[0]
        if len(neighbors) == 0:
            break
        next_node = np.random.choice(neighbors)
        if next_node == node:
            returns += 1
        current_node = next_node
    
    return returns / t

def robust_subset_estimation(graph, t=100, alpha=0.2, degree_threshold=0.1):
    """
    Estimate the robust subset S* based on random walk return probabilities.
    
    Parameters:
    - graph: networkx Graph object representing the perturbed graph G
    - t: number of steps for random walk
    - alpha: threshold for the return probability
    - degree_threshold: proportion of nodes with the highest and lowest degrees to trim
    
    Returns:
    - Estimated edge probability p_S for the robust subset S
    """
    S = set(graph.nodes)  # Start with all nodes in the set

    # Iteratively remove nodes with low return probabilities
    changed = True
    while changed:
        changed = False
        low_prob_nodes = set()
        
        # Calculate return probabilities for all nodes in S
        for node in list(S):
            return_prob = random_walk_return_probability(graph, node, t)
            if return_prob < alpha:
                low_prob_nodes.add(node)
        
        # Remove nodes with return probabilities below threshold
        if low_prob_nodes:
            S -= low_prob_nodes
            changed = True

    # Calculate edge probability for subset S
    subgraph = graph.subgraph(S)
    num_edges = subgraph.number_of_edges()
    num_nodes = len(subgraph.nodes)
    if num_nodes < 2:
        print("Subset is too small to estimate p.")
        return None
    p_S = num_edges / (num_nodes * (num_nodes - 1) / 2)

    # Refine by removing nodes with degree deviations
    degrees = np.array([subgraph.degree(node) for node in subgraph.nodes])
    mean_degree = np.mean(degrees)
    degree_diff = np.abs(degrees - mean_degree)
    threshold = np.percentile(degree_diff, 100 * (1 - degree_threshold))

    # Trim nodes with the highest degree deviations
    refined_S = [node for node, diff in zip(subgraph.nodes, degree_diff) if diff <= threshold]
    refined_subgraph = graph.subgraph(refined_S)
    refined_num_edges = refined_subgraph.number_of_edges()
    refined_num_nodes = len(refined_subgraph.nodes)
    
    if refined_num_nodes < 2:
        print("Refined subset is too small to estimate p.")
        return None
    p_S_refined = refined_num_edges / (refined_num_nodes * (refined_num_nodes - 1) / 2)

    return p_S_refined

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

def test(epsilon, n, p):
    G, L = generate_erdos_renyi_graph(n, p)
    D = np.diag(L)

    print(f"\n\n########################################################\nTest for p={p}, n={n}, epsilon={epsilon}")
    for q in [0.1, 0.3, 0.6, 0.9]:
        Gp, Lp = perturb_graph(G, epsilon, q)
        Dp = np.diag(Lp)
        mean = np.mean(Dp)/n
        med = np.median(Dp)/n
        robust_p = robust_estimate_guess(Gp, epsilon)
        # variance
        std = np.var(Dp) ** 0.5  # Default: Population variance
        sample_std = np.var(Dp, ddof=1) ** 0.5  # Sample variance with Bessel's correction
        robust_std = ((n-1) * robust_p * (1 - robust_p)) ** 0.5
        print(f"\nq: {q}\nMean: {mean}\nMedian: {med}\nRobust p: {robust_p}\nStd: {std}\nSample Std: {sample_std}\nRobust Std: {robust_std}\n")

for n in [1000, 2000, 3000]:
    for p in [0.2, 0.4, 0.6, 0.8]:
        for epsilon in [0.01, 0.05, 0.1]:
            test(epsilon, n, p)
# # Example usage
# n = 3000  # Number of nodes
# # N = 6000
# p = 0.3   # Probability of an edge
# epsilon = 0.1

# G, L = generate_erdos_renyi_graph(n, p)
# D = np.diag(L)
# # variance
# variance = np.var(D)  # Default: Population variance
# sample_variance = np.var(D, ddof=1)  # Sample variance with Bessel's correction

# # Interquartile Range (IQR)
# q1 = np.percentile(D, 25)  # 25th percentile (Q1)
# q3 = np.percentile(D, 75)  # 75th percentile (Q3)
# iqr = q3 - q1
# print(f"True Std: {variance ** 0.5}\nTrue Sample Std: {sample_variance ** 0.5}\nTrue IQR: {iqr}")

# Gp, Lp = perturb_graph(G, epsilon, 0.35)
# Dp = np.diag(Lp)
# mean = np.mean(Dp)/n
# med = np.median(Dp)/n
# # variance
# variance = np.var(Dp)  # Default: Population variance
# sample_variance = np.var(Dp, ddof=1)  # Sample variance with Bessel's correction

# # Interquartile Range (IQR)
# q1 = np.percentile(Dp, 25)  # 25th percentile (Q1)
# q3 = np.percentile(Dp, 75)  # 75th percentile (Q3)
# iqr = q3 - q1
# print(f"\nq: {0.35}\nTrue p: {p}\nMean: {mean}\nMedian: {med}\nPrune Then Mean and Median: {prune_then_mean_median(Gp, epsilon)}\nRobust Estimation: {robust_estimate_guess(Gp, epsilon)}")
# print(f"Std: {variance ** 0.5}\nSample Std: {sample_variance ** 0.5}\nIQR: {iqr}")


#########################################################################################
# n = 1000
# ps = np.arange(0, 1.05, 0.4)
# qs = np.arange(0, 1.05, 0.05)
# epsilons = np.arange(0.1, 1.05, 0.1)

# for epsilon in epsilons:
#     for p in ps:
#         G, L = generate_erdos_renyi_graph(n, p)
#         x = []
#         y = []
#         for q in qs:
#             G_q, L_q = perturb_graph(G, epsilon, q)
#             D_q = np.diag(L_q)
#             q_mean = np.mean(D_q) / n
#             q_med = np.median(D_q) / n

#             diff = np.around(min(abs(p - q_mean), abs(p - q_med)), 4)

#             # print(f"q: {np.around(q, 4)}")
#             # print(f"Mean of G_q: {q_mean}")
#             # print(f"Median of G_q: {q_med}")
#             # print(f"{diff, np.around(abs(q_mean - q_med), 4)}\n")
#             x.append(min(abs(p - q_mean), abs(p - q_med)))
#             y.append(abs(q_mean - q_med))

#         # Calculate the line of best fit (linear fit)
#         slope, intercept = np.polyfit(x, y, 1)  # 1 indicates a linear fit

#         # Generate y-values for the line of best fit
#         y_fit = [slope * xi + intercept for xi in x]

#         y_mean = np.mean(y)
#         ss_res = sum((yi - y_fit[i]) ** 2 for i, yi in enumerate(y))  # Sum of squares of residuals
#         ss_tot = sum((yi - y_mean) ** 2 for yi in y)  # Total sum of squares
#         r_squared = 1 - (ss_res / ss_tot)

#         print(f"p: {p}, epsilon: {epsilon}\nBest Fit Line: y={slope:.2f}x + {intercept:.2f}\nR^2: {r_squared}\n")

#         # # Plot the original data points
#         # plt.scatter(x, y, label='Data Points', color='blue')

#         # # Plot the line of best fit
#         # plt.plot(x, y_fit, label=f'Best Fit Line (y={slope:.2f}x + {intercept:.2f})', color='red')

#         # # Add labels and a title
#         # plt.xlabel('X-axis')
#         # plt.ylabel('Y-axis')
#         # plt.title('Line of Best Fit')
#         # plt.legend()

#         # # Show the plot
#         # plt.show()










########################################################################################


# print(np.around(np.sort(np.linalg.eigvals(L)), 4))
# L_p, L_pp = compute_modified_laplacians(G, epsilon)
# print(np.around(np.sort(np.linalg.eigvals(L_p)), 4))
# print(np.around(np.sort(np.linalg.eigvals(L_pp)), 4))
# print(np.around(np.sort(np.linalg.eigvals(LP)), 4))
# G_p = graph_from_laplacian(L_p)
# G_pp = graph_from_laplacian(L_pp)
# D = np.diag(L)
# D_p = np.diag(L_p)
# D_pp = np.diag(L_pp)
# DP = np.diag(LP)

# print(f"Mean of G: {np.mean(D) / n}")
# print(f"Median of G: {np.median(D) / n}\n")

# print(f"Mean of G': {np.mean(D_p) / n}")
# print(f"Median of G': {np.median(D_p) / n}\n")
# print(f"Robust Estimation of p using G': {robust_estimate_p_from_graph(graph_from_laplacian(L_p))}\n")
# print(f"Robust Estimation of p using G': {robust_subset_estimation(G_p)}\n")

# print(f"Mean of G'': {np.mean(D_pp) / n}")
# print(f"Median of G'': {np.median(D_pp) / n}\n")
# print(f"Robust Estimation of p using G'': {robust_estimate_p_from_graph(graph_from_laplacian(L_pp))}\n")
# print(f"Robust Estimation of p using G'': {robust_subset_estimation(G_pp)}\n")

# print(f"Mean of GP: {np.mean(DP) / n}")
# print(f"Median of GP: {np.median(DP) / n}\n")

# # Create a sample graph G (you can replace this with your own graph)
# G = nx.erdos_renyi_graph(1000, 0.3)

# # Calculate degree of each vertex
# degrees = [degree for node, degree in G.degree()]

# # Plot the histogram of vertex degrees
# plt.hist(degrees, bins=range(min(degrees), max(degrees) + 1), edgecolor='black')
# plt.xlabel("Degree")
# plt.ylabel("Frequency")
# plt.title("Histogram of Vertex Degrees")
# plt.show()

# degrees = np.sort(np.array(degrees))
# percentile = np.percentile(degrees, 5)
# print(percentile)
# print(degrees[0], degrees[-1])