import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations, groupby
from scipy.stats import binom
import random
import numpy as np
import math
from scipy.optimize import curve_fit
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

def robust_estimate_p_from_graph(G_prime):
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
    y = (1 / (1 - epsilon)) * abs(mean - med)
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

    print(f"\n\n########################################################\nTest for p={p}, n={n}, epsilon={epsilon}")
    for q in [0.1, 0.3, 0.6, 0.9]:
        Gp, Lp = perturb_graph(G, epsilon, q)
        Dp = np.diag(Lp)
        mean = np.mean(Dp)/n
        med = np.median(Dp)/n
        robust_p = robust_estimate_guess(Gp, epsilon)
        # variance
        # std = np.var(Dp) ** 0.5  # Default: Population variance
        sample_std = np.var(Dp, ddof=1) ** 0.5  # Sample variance with Bessel's correction
        robust_std = ((n-1) * robust_p * (1 - robust_p)) ** 0.5
        print(f"\nq: {q}\nMean: {mean}\nMedian: {med}\nRobust p: {robust_p}\nSample Std: {sample_std}\nRobust Std: {robust_std}\nStd Ratio: {sample_std / robust_std}\n")

def test2(epsilon, n, p, q):
    G, _ = generate_erdos_renyi_graph(n, p)
    
    Gp, Lp = perturb_graph(G, epsilon, q)
    Dp = np.diag(Lp)
    robust_p = robust_estimate_guess(Gp, epsilon)
    sample_std = np.var(Dp, ddof=1) ** 0.5
    robust_std = ((n-1) * robust_p * (1 - robust_p)) ** 0.5
    return sample_std/robust_std

def random_walk_estimate_p(G, epsilon, num_samples=10000):
    """
    Robustly estimate the edge probability p of an Erdős–Rényi graph G
    in the presence of adversarial perturbations.

    Parameters:
    - G: networkx.Graph, the perturbed graph
    - epsilon: float, the fraction of adversarially corrupted vertices
    - num_samples: int, the number of random walks to perform

    Returns:
    - p_estimated: float, the robust estimate of p
    """
    n = G.number_of_nodes()
    nodes = list(G.nodes)
    total_walks = 0
    connected_endpoints = 0

    for _ in range(num_samples):
        # Randomly choose the start node
        u = random.choice(nodes)
        
        # Ensure u has neighbors
        if not list(G.neighbors(u)):
            continue

        # Randomly select a neighbor v
        v = random.choice(list(G.neighbors(u)))
        
        # Ensure v has neighbors other than u
        neighbors_v = set(G.neighbors(v)) - {u}
        if not neighbors_v:
            continue

        # Randomly select a neighbor w of v
        w = random.choice(list(neighbors_v))
        
        # Check if u and w are connected
        total_walks += 1
        if G.has_edge(u, w):
            connected_endpoints += 1

    if total_walks == 0:
        raise ValueError("Not enough valid random walks found in the graph.")

    # Estimate raw probability
    raw_p = connected_endpoints / total_walks
    
    # Adjust for adversarial influence
    adjusted_p = (raw_p - epsilon) / ((1 - epsilon) ** 2)
    
    return max(0, min(1, adjusted_p))  # Ensure p is in [0, 1]


def robust_estimation_using_var(G, epsilon):
    n = G.number_of_nodes()
    num_removals = int(epsilon * n)

    modified_G = G.copy()
    for _ in range(num_removals):
        nodes = list(modified_G.nodes)
        sample_variances = {}
        L = nx.laplacian_matrix(modified_G).toarray()
        D = np.diag(L)
        for i, node in enumerate(nodes):
            modified_diag = np.copy(D)
            for j in range(len(D)):
                modified_diag[j] -= L[i][j]
            # p_hat = np.mean(modified_diag) / (len(L) - 1)
            # sample_variances[node] = abs(np.var(modified_diag, ddof=1) - p_hat * (1 - p_hat))
            sample_variances[node] = np.var(modified_diag, ddof=1)

        # Find the node with the highest sample variance
        node_to_remove = min(sample_variances, key=sample_variances.get)
        modified_G.remove_node(node_to_remove)
    
    L = np.array(nx.laplacian_matrix(modified_G).toarray())
    D = np.diag(L)
    return np.mean(D) / modified_G.number_of_nodes(), np.var(D, ddof=1)

p = 0.5
epsilon = 0.01
q = 0.8
n = 500
G, L = generate_erdos_renyi_graph(n, p)
D = np.diag(L)
Gp, Lp = perturb_graph(G, epsilon, q)
# Gp, Lp = perturb_graph_general(G, epsilon)
Dp = np.diag(Lp)
print(p*(1-p)*n, np.var(D), np.var(Dp))
print(f"True p: {p}\nPrevious Mean and Median: {np.mean(D) / n, np.median(D) / n}\nPerturbed Mean and Median: {np.mean(Dp) / n, np.median(Dp) / n}\nMedian - Mean: {robust_estimate_guess(Gp, epsilon)}")
print(f"Variance Method: {robust_estimation_using_var(Gp, epsilon)}")


# L_p, L_pp = compute_modified_laplacians(G, epsilon)
# D_p = np.diag(L_p)
# # Plot the histogram
# plt.figure(figsize=(10, 6))
# plt.hist(D, bins=range(min(D), max(D) + 1), density=True, alpha=0.7, label='Simulated Degrees')

# # Overlay the theoretical Binomial distribution
# x = np.arange(0, n)
# binomial_pmf = binom.pmf(x, n - 1, p)
# plt.plot(x, binomial_pmf, 'r-', lw=2, label='Theoretical Binomial PMF')
# plt.xlim(0, 600)
# plt.ylim(0, 0.04)

# # Add labels and legend
# plt.title(f'Degree Distribution for G(n={n}, p={p})')
# plt.xlabel('Degree')
# plt.ylabel('Probability')
# plt.legend()
# plt.show()

# plt.hist(D_p, bins=range(min(D_p), max(D_p) + 1), density=True, alpha=0.7, label='Simulated Degrees')
# plt.plot(x, binomial_pmf, 'r-', lw=2, label='Theoretical Binomial PMF')
# plt.xlim(0, 1200)
# plt.ylim(0, 0.04)

# # Add labels and legend
# plt.title(f"Degree Distribution for G'(n={n}, p={p})")
# plt.xlabel('Degree')
# plt.ylabel('Probability')
# plt.legend()
# plt.show()







N = 5400
x = np.array(list(range(200, N, 200)), dtype=float)
y = [0.305578, 0.305216, 0.305445, 0.3043872685185186, 0.3044542222222223,
     0.30356862139917695, 0.30371201814058957, 0.3033429976851852, 0.30285331504343854,
     0.30262994444444447, 0.30279632690541786, 0.302536612654321, 0.30224621959237347,
     0.3024572278911565, 0.30216239506172843, 0.3020440538194443, 0.30227099192618223,
     0.3023474108367627, 0.3019706832871653, 0.3021875, 0.30166250944822376, 0.301971533516988,
     0.3021168872085696, 0.30147376543209875, 0.3019624888888889, 0.3017241124260354] # q, epsilon oblivious
y = [0.30647407407407407, 0.30590185185185187, 0.30482716049382724, 0.30445925925925926,
     0.30420251851851854, 0.3041143004115226, 0.3032523053665911, 0.3033934027777778, 
     0.30333219021490626, 0.30287561111111116, 0.30269031221303944, 0.30245570987654324,
     0.3028632807363577, 0.3024440476190476, 0.3024630864197531, 0.30219014756944446,
     0.3023166089965398, 0.30203780864197527, 0.30206452754693747, 0.30215444444444445,
     0.30176329050138573, 0.30146430211202935, 0.30183049779458093, 0.30171662808641975,
     0.3019840888888889, 0.3018993261012492] # epsilon oblivious
y = []
# for n in range(200, N, 200):
#     if n < 2000:
#         tmp = 0
#         for _ in range(15):
#             G, L = generate_erdos_renyi_graph(n, p)
#             # Gp, Lp = perturb_graph(G, epsilon, q)
#             Gp, Lp = perturb_graph_general(G, epsilon)
#             tmp += robust_estimate_guess(Gp, epsilon)
#         print(n, tmp / 15)
#         y.append(tmp / 15)
#         print(" ")
#     elif n < 4000:
#         tmp = 0
#         for _ in range(10):
#             G, L = generate_erdos_renyi_graph(n, p)
#             Gp, Lp = perturb_graph(G, epsilon, q)
#             tmp += robust_estimate_guess(Gp, epsilon)
#         print(n, tmp / 10)
#         y.append(tmp / 10)
#         print(" ")
#     else:
#         G, L = generate_erdos_renyi_graph(n, p)
#         Gp, Lp = perturb_graph(G, epsilon, q)
#         p_hat = robust_estimate_guess(Gp, epsilon)
#         print(n, p_hat)
#         y.append(p_hat)
#         print(" ")

# plt.scatter(x, y, label="Data", color="red")

# def inverse_model(x, a, b):
#     return a + b * x ** (-0.5)

# params, covariance = curve_fit(inverse_model, x, y)
# a, b = params
# print(f"Fitted parameters: a = {a}, b = {b}")
# y_fit = inverse_model(x, a, b)
# R2 = r2_score(y, y_fit)
# print(f"R^2: {R2}")

# plt.plot(x, y_fit, label=f"Fit: y = {a:.2f} + {b:.2f}x^(-0.2)", color="blue")

# plt.xlabel("n")
# plt.ylabel("p_hat")
# plt.legend()
# plt.show()










#############################################################################################

# n = 1000
# p = 0.2
# q = 0.3
# x = np.arange(0.01, 0.91, 0.03)
# y = []
# for epsilon in x:
#     y.append(test2(epsilon, n, p, q))

# plt.plot(x, y, marker='o')  # Add markers for clarity (optional)
# plt.title("Y as a Function of X")
# plt.xlabel("X values")
# plt.ylabel("Y values")
# plt.grid(True)  # Add grid lines (optional)
# plt.show()

# from sklearn.metrics import r2_score

# coeffs = np.polyfit(x, y, deg=2)  # Try deg=2, deg=3, etc.
# poly_model = np.poly1d(coeffs)
# plt.scatter(x, y, label="Data", color="blue")
# plt.plot(x, poly_model(x), label=f"Polynomial Fit (deg=2)", color="red")
# plt.legend()
# plt.show()



# for n in [1000, 2000, 3000]:
#     for p in [0.2, 0.4, 0.6, 0.8]:
#         for epsilon in [0.01, 0.05, 0.1]:
#             test(epsilon, n, p)


# Example usage
# n = 100  # Number of nodes
# epsilons = np.concatenate([np.arange(0.01, 0.12, 0.01), np.arange(0.12, 0.82, 0.04)])
# ps = np.arange(0, 1.05, 0.1)
# z0 = []
# z1 = []
# z2 = []
# for epsilon in epsilons:
#     degree_2 = []
#     degree_1 = []
#     constant = []
#     for p in ps:
#         G, L = generate_erdos_renyi_graph(n, p)
#         D = np.diag(L)
#         x = []
#         y = []
#         for q in np.arange(0, 1.05, 0.05):
#             Gp, Lp = perturb_graph(G, epsilon, q)
#             Dp = np.diag(Lp)
#             mean = np.mean(Dp)/n
#             var = np.var(Dp)  # Default: Population variance
#             x.append(mean)
#             y.append(var)
#         coeffs = np.polyfit(x, y, deg=2)  # Try deg=2, deg=3, etc.
#         degree_2.append(coeffs[0])
#         degree_1.append(coeffs[1])
#         constant.append(coeffs[2])

#     # Fit a linear function
#     coeffs = np.polyfit(ps, degree_2, deg=2)
#     z0.append(coeffs[0])


# def inverse_model(x, a):
#     return a * x**(-1)

# def constant_model(x, a):
#     return a + x * 0

# # Perform curve fitting
# params, covariance = curve_fit(inverse_model, epsilons, z0)

# # Extract the fitted parameters
# a = params[0]
# print(f"Fitted parameters: a = {a}")

# # Generate fitted y-values
# y_fit = inverse_model(epsilons, a)

# # Plot the data and the fit
# plt.scatter(epsilons, z0, label="Data", color="red")
# plt.plot(epsilons, y_fit, label=f"Fit: y = {a:.2f} / x", color="red")

# plt.xlabel("epsilons")
# plt.ylabel("y")
# plt.legend()
# plt.show()


    # Format the linear equation for the legend
    # equation = f"y = {coeffs[0]:.2f}"

    # Plot data and polynomial fit
    # plt.scatter(ps, degree_2, label="Degree 2", color="orange")
    # plt.plot(ps, poly_model(ps), label=f"Constant Fit: {equation}", color="red")

    # coeffs = np.polyfit(ps, degree_1, deg=1)
    # poly_model = np.poly1d(coeffs)

    # # Format the linear equation for the legend
    # equation = f"y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}"

    # # Plot data and polynomial fit
    # plt.scatter(ps, degree_1, label="Degree 1", color="yellow")
    # plt.plot(ps, poly_model(ps), label=f"Linear Fit: {equation}", color="red")

    # coeffs = np.polyfit(ps, constant, deg=2)
    # poly_model = np.poly1d(coeffs)

    # # Format the linear equation for the legend
    # equation = f"y = {coeffs[0]:.2f}x^2 + {coeffs[1]:.2f}x + {coeffs[2]:.2f}"

    # # Plot data and polynomial fit
    # plt.scatter(ps, constant, label="Constant", color="green")
    # plt.plot(ps, poly_model(ps), label=f"Quadratic Fit: {equation}", color="red")


    # plt.legend()
    # plt.title("Y as a Function of X")
    # plt.xlabel("X values")
    # plt.ylabel("Y values")
    # plt.grid(True)  # Add grid lines (optional)
    # plt.show()


#####################################################################################

# G, L = generate_erdos_renyi_graph(n, p)
# D = np.diag(L)
# q = 0.4
# Gp, Lp = perturb_graph(G, epsilon, q)
# Dp = np.diag(Lp)
# mean = np.mean(Dp)/n
# med = np.median(Dp)/n
# std = np.var(Dp) ** 0.5  # Default: Population variance
# sample_std = np.var(Dp, ddof=1) ** 0.5  # Sample variance with Bessel's correction
# robust_mean = robust_estimate_guess(Gp, epsilon)
# prune_mean = prune_then_mean_median(Gp, epsilon)
# print(f"q: {q}\nTrue p: {p}\nMean: {mean}\nMedian: {med}")
# print(robust_mean)
# print(prune_mean)
# p_std_pos = (1 + (1 + 4 * std / (n ** 0.5)) ** 0.5) / 2
# p_std_neg = (1 - (1 + 4 * std / (n ** 0.5)) ** 0.5) / 2
# p_sample_std_pos = (1 + (1 + 4 * sample_std / (n - 1) ** 0.5) ** 0.5) / 2
# p_sample_std_neg = (1 - (1 + 4 * sample_std / (n - 1) ** 0.5) ** 0.5) / 2
# print(f"Std: {std}\nSample Std: {sample_std}\np std: {p_std_neg, p_std_pos}\np sample std: {p_sample_std_neg, p_sample_std_pos}")

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