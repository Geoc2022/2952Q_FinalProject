import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt

from algos import variance_method, spectral_method
from basic import generate_erdos_renyi_graph

def var_adversary(G, p, epsilon, candidate_fraction, q):
    """
    Perturbs the graph G by minimizing the variance of the degree distribution and then increasing the variance to p(1-p)(n-1).

    Args:
        G (networkx.Graph): The input graph.
        p (float): The edge probability.
        epsilon (float): The perturbation parameter.
        candidate_fraction (float): The fraction of nodes to consider for edge redistribution.
        q (float): The edge probability for the second phase.

    Returns:
        networkx.Graph: The perturbed graph.
    """
    G_copy = G.copy()
    nodes_by_degree = sorted(G.degree, key=lambda x: x[1])  # List of (node, degree)
    n = len(G_copy.nodes)
    num_nodes = int(candidate_fraction * epsilon * n / 2)
    num_nodes2 = int(epsilon * n) - num_nodes

    V_small = [node for node, _ in nodes_by_degree[:num_nodes]]
    V_large = [node for node, _ in nodes_by_degree[-num_nodes:]]

    for i in range(num_nodes):
        v = V_small[i]
        u = V_large[i]
        d_u = G_copy.degree[u]
        num_edges_to_change = d_u - (n - 1) * p

        # Get neighbors of u and v
        neighbors_u = set(G_copy.neighbors(u))
        neighbors_v = set(G_copy.neighbors(v))

        # Find candidate nodes for edge redistribution
        candidate_nodes = list(neighbors_u - neighbors_v)
        random.shuffle(candidate_nodes)

        for _ in range(int(num_edges_to_change)):
            if not candidate_nodes:
                break
            w = candidate_nodes.pop()

            # Remove edge {u, w} and add edge {v, w}
            G_copy.remove_edge(u, w)
            G_copy.add_edge(v, w)

    nodes_by_degree = sorted(G_copy.degree, key=lambda x: x[1])  # List of (node, degree)
    num_nodes = int(candidate_fraction * epsilon * n / 2)
    num_nodes2 = int(epsilon * n) - num_nodes

    V_small = [node for node, _ in nodes_by_degree[:num_nodes2]]
    for i in range(num_nodes2):
        for v in G_copy.nodes:
            u = V_small[i]
            if v != u:
                if random.random() < q:
                    if not G_copy.has_edge(u, v):
                        G_copy.add_edge(u, v)
                elif G_copy.has_edge(u, v):
                    G_copy.remove_edge(u, v)

    return G_copy

def var_adversary_prime(G, e, draw=False, random_selection=False):
    """
    Perturbs the graph G by minimizing the variance of the degree distribution and then increasing the variance to p(1-p)(n-1).
    
    Args:
        G (networkx.Graph): The input graph.
        e (float): The perturbation parameter.
        draw (bool): Whether to draw the graph.
        random_selection (bool): Whether to randomly select the compromised nodes.
        
    Returns:
        networkx.Graph: The perturbed graph.
    """
    n = G.number_of_nodes()
    get_degree_list = lambda G: list((d for n, d in G.degree()))
    # Calculate the mean and variance of the degree distribution of the graph G    
    degrees = get_degree_list(G)
    p = np.mean(degrees) / (n - 1)
    print(f"Mean:\t{p}")
    var = np.var(degrees)
    print(f"Variance:\t{var}")
    print()
    # Draw the graph
    if draw:
        # plt.figure(figsize=(8, 8))
        # nx.draw(G, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500, font_size=10)
        # plt.title(f"Adversary Graph (n={n}, p={p}, e={e})")
        # plt.show()

        # show the distribution of the degrees
        plt.figure(figsize=(8, 6))
        plt.hist(get_degree_list(G), bins=range(n+1))
        plt.title("Degree Distribution")
        plt.xlabel("Degree")
        plt.ylabel("Count")
        plt.show()
    
    # Randomly select en nodes to be compromised by the adversary
    number_of_compromised_nodes = int(e * n)
    if random_selection:
        compromised_nodes = random.sample(range(n), number_of_compromised_nodes)
    else:
        # Select the nodes with the highest and lowest degrees
        sorted_degree = sorted(G.degree, key=lambda x: x[1])
        compromised_nodes = []
        for i in range(number_of_compromised_nodes // 2):
            compromised_nodes.append(sorted_degree[i][0])
            compromised_nodes.append(sorted_degree[-i-1][0])
        if number_of_compromised_nodes % 2 == 1:
            compromised_nodes.append(sorted_degree[n//2][0])


    for i, node in enumerate(compromised_nodes):
        print(f"Compromising ({i}/{number_of_compromised_nodes})", end="\r")
        # add/remove edges to/from the compromised node until the degree is exactly p(n-1)
        while G.degree(node) < p * (n - 1):
            # Randomly select a target node
            target_node = random.choice(list(G.nodes))
            # Add the edge
            G.add_edge(node, target_node)
            # print(f"dgree: {G.degree(node)}")
        
        while G.degree(node) > p * (n - 1):
            # Randomly select a neighbor of the compromised node
            neighbor = random.choice(list(G.neighbors(node)))
            # Remove the edge
            G.remove_edge(node, neighbor)
    print()

    # Draw the graph
    if draw:
        # plt.figure(figsize=(8, 8))
        # nx.draw(G, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500, font_size=10)
        # plt.title(f"Adversary Graph (n={n}, p={p}, e={e})")
        # plt.show()

        # show the distribution of the degrees
        plt.figure(figsize=(8, 6))
        plt.hist(get_degree_list(G), bins=range(n+1))
        plt.title("Degree Distribution")
        plt.xlabel("Degree")
        plt.ylabel("Count")
        plt.show()

    # Add edges to the compromised nodes until the variance of the degree distribution is equal to p(1-p)(n-1)
    node1 = None
    node2 = None
    while (var_hat := np.var(get_degree_list(G))) < var:
        print(f"Adding edges until variance = {var_hat:.3f} is {var:.3f}", end="\r")
        # Randomly select a pair of nodes with at least one compromised node
        node1 = random.choice(compromised_nodes)
        node2 = random.choice(list(G.nodes))
        while node2 == node1:
            node2 = random.choice(list(G.nodes))
        # Add an edge between the two nodes
        G.add_edge(node1, node2)
    if node1 is not None and node2 is not None:
        G.remove_edge(node1, node2)
    print()

    # Draw the graph
    if draw:
        # plt.figure(figsize=(8, 8))
        # nx.draw(G, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500, font_size=10)
        # plt.title(f"Adversary Graph (n={n}, p={p}, e={e})")
        # plt.show()

        # show the distribution of the degrees
        plt.figure(figsize=(8, 6))
        plt.hist(list((d for n, d in G.degree())), bins=range(n+1))
        plt.title("Degree Distribution")
        plt.xlabel("Degree")
        plt.ylabel("Count")
        plt.show()


    degrees = get_degree_list(G)
    print()
    print(f"Mean + adversary:\t{np.mean(degrees) / (n - 1)}")
    print(f"Variance + adversary:\t{np.var(degrees)}")

    return G, (p, np.mean(degrees) / (n - 1))

def test_adversary(n = 1000, p = 0.3, e = 0.01):
    print("Parameters:")
    print(f"n:\t{n}")
    print(f"p:\t{p}")
    print(f"e:\t{e}")
    print()

    # Generate an Erdős-Rényi graph
    G = generate_erdos_renyi_graph(n, p)

    p = nx.average_clustering(G)

    # Run the adversary algorithm
    # binary search for the optimal q
    var = p * (1 - p) * (n - 1)
    q = p
    G_prime = var_adversary(G, p, e, 0.8, q)
    G_prime_var = np.var([d for n, d in G_prime.degree()])
    while (np.abs(G_prime_var - var) / var) < .01:
        if G_prime_var < var:
            q -= 0.01
        else:
            q += 0.01
        G_prime = var_adversary(G, p, e, 0.8, q)
        G_prime_var = np.var([d for n, d in G_prime.degree()])


    print()
    p_hat, _ = variance_method(G, e)
    print(f"Var p:\t{p_hat}\tError:{abs(p_hat - p)}")
    print(f"Spectral p:\t{(p_hat := spectral_method(n, e, nx.adjacency_matrix(G).toarray(), G))}\tError:{abs(p_hat - p)}")


def main():
    test_adversary()

main()