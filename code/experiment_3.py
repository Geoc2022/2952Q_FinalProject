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
    
    return L_prime, L_double_prime, G_prime, G_double_prime

def sample_subgraphs(G, num_samples, subgraph_size_ratio):
    n = len(G.nodes)
    subgraph_size = int(subgraph_size_ratio * n)
    statistics = []

    for _ in range(num_samples):
        # Randomly select a subset of nodes
        sampled_nodes = random.sample(G.nodes, subgraph_size)
        subgraph = G.subgraph(sampled_nodes).copy()  # Create the subgraph

        # Compute Laplacian matrix for the subgraph
        L_sub = np.array(nx.laplacian_matrix(subgraph).toarray())

        # Calculate mean and median degree of the subgraph
        mean_degree, median_degree = np.mean(np.diag(L_sub)) / len(sampled_nodes), np.median(np.diag(L_sub)) / len(sampled_nodes)
        statistics.append((mean_degree, median_degree))

    return statistics

def aggregate_statistics(statistics):
    means = [stat[0] for stat in statistics]
    medians = [stat[1] for stat in statistics]

    mean_of_means = np.mean(means)
    median_of_means = np.median(means)
    mean_of_medians = np.mean(medians)
    median_of_medians = np.median(medians)
    
    return mean_of_means, median_of_means, mean_of_medians, median_of_medians

def print_statistics(mean_of_means, median_of_means, mean_of_medians, median_of_medians):
    print(f"Mean of means: {mean_of_means}")
    print(f"Median of means: {median_of_means}")
    print(f"Mean of medians: {mean_of_medians}")
    print(f"Median of medians: {median_of_medians}")


def create_plots(G):
    n = 100  # Number of nodes
    p = 0.3  # Probability of an edge

    all_xs = []
    all_ys = []

    for i in range(n * 2, n ** 2, n):
        num_samples = i  # Number of subgraphs to sample
        subgraph_size_ratio = random.uniform(0.6, 1) # Ratio of nodes in each subgraph (e.g., 80%)

        # Sample subgraphs and compute statistics
        subgraph_statistics = sample_subgraphs(G, num_samples, subgraph_size_ratio)

        all_means = [stat[0] for stat in subgraph_statistics]
        all_xs = all_xs + [i] * i
        all_ys = all_ys + all_means

    plt.scatter(all_xs, all_ys, color='blue', marker='o')
    plt.xlabel("Num Samples")
    plt.ylabel("Mean")
    plt.show()


n = 100  # Number of nodes
p = 0.3  # Probability of an edge
num_samples = 200  # Number of subgraphs to sample
subgraph_size_ratio = 0.6  # Ratio of nodes in each subgraph (e.g., 80%)
epsilon = 0.1

# Generate the main graph
G, _ = generate_erdos_renyi_graph(n, p)
L_p, L_pp, G_p, G_pp = compute_modified_laplacians(G, epsilon)

create_plots(G)

'''
# Example usage
n = 100  # Number of nodes
p = 0.3  # Probability of an edge
num_samples = 200  # Number of subgraphs to sample
subgraph_size_ratio = 0.6  # Ratio of nodes in each subgraph (e.g., 80%)
epsilon = 0.1

# Generate the main graph
G, L = generate_erdos_renyi_graph(n, p)
L_p, L_pp, G_p, G_pp = compute_modified_laplacians(G, epsilon)

# Sample subgraphs and compute statistics
subgraph_statistics = sample_subgraphs(G, num_samples, subgraph_size_ratio)
subgraph_statistics_G_p = sample_subgraphs(G_p, num_samples, subgraph_size_ratio)
subgraph_statistics_G_pp = sample_subgraphs(G_pp, num_samples, subgraph_size_ratio)

# Aggregate statistics across all subgraphs
mean_of_means, median_of_means, mean_of_medians, median_of_medians = aggregate_statistics(subgraph_statistics)
mean_of_means_G_p, median_of_means_G_p, mean_of_medians_G_p, median_of_medians_G_p = aggregate_statistics(subgraph_statistics_G_p)
mean_of_means_G_pp, median_of_means_G_pp, mean_of_medians_G_pp, median_of_medians_G_pp = aggregate_statistics(subgraph_statistics_G_pp)

print_statistics(mean_of_means, median_of_means, mean_of_medians, median_of_medians)
print_statistics(mean_of_means_G_p, median_of_means_G_p, mean_of_medians_G_p, median_of_medians_G_p)
print_statistics(mean_of_means_G_pp, median_of_means_G_pp, mean_of_medians_G_pp, median_of_medians_G_pp)

D = np.diag(L)
D_p = np.diag(L_p)
D_pp = np.diag(L_pp)



print(f"Mean of G: {np.mean(D) / n}")
print(f"Median of G: {np.median(D) / n}\n")

print(f"Mean of G': {np.mean(D_p) / n}")
print(f"Median of G': {np.median(D_p) / n}\n")

print(f"Mean of G'': {np.mean(D_pp) / n}")
print(f"Median of G'': {np.median(D_pp) / n}")

'''
