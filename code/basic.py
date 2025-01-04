import networkx as nx
import random
import matplotlib.pyplot as plt

def generate_erdos_renyi_graph(n, p, draw=False):
    """
    Generates an Erdős–Rényi graph with n nodes and edge probability p using the networkx library.
    """
    G = nx.erdos_renyi_graph(n, p)

    if draw:
        plt.figure(figsize=(8, 8))
        nx.draw(G, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500, font_size=10)
        plt.title(f"Erdős–Rényi Graph (n={n}, p={p})")
        plt.show()
    
    return G

def compute_modified_laplacians(G, epsilon):
    """
    Creates two perturbed graphs G' and G'' from the original graph G with max pretrubations with epsilon fraction of nodes.
    
    Args:
        G (networkx.Graph): - The original graph.
        epsilon (float): - Fraction of nodes to perturb (0 < epsilon <= 1).
        
    Returns:
        networkx.Graph: - The perturbed graph G' which has all edges with a node in B.
        networkx.Graph: - The perturbed graph G'' which has no edges with a node in B.
    """
    n = len(G.nodes)
    
    # Create subset B of size epsilon * n
    B = set(random.sample(list(G.nodes), int(epsilon * n)))
    
    G_prime = G.copy()
    G_double_prime = G.copy()
    
    # For G': Add edges (u, v) if u or v is in B
    for u in B:
        for v in G.nodes:
            if u != v and not G_prime.has_edge(u, v):
                G_prime.add_edge(u, v)
    
    # For G'': Remove edges (u, v) if u or v is in B
    for u in B:
        neighbors = list(G_double_prime.adj[u])
        for v in neighbors:
            if G_double_prime.has_edge(u, v):
                G_double_prime.remove_edge(u, v)
    
    return G_prime, G_double_prime



def perturb_graph(G, epsilon, p):
    """
    Perturbs the graph G by adding edges with probability p between an epsilon-fraction of nodes.

    Args:
        G (networkx.Graph): - The original graph.
        epsilon (float): - Fraction of nodes to perturb (0 < epsilon <= 1).
        p (float): - Probability of adding an edge between two nodes (0 <= p <= 1).

    Returns:
        networkx.Graph: - The perturbed graph.
    """
    perturbed_G = G.copy()
    
    n = len(G.nodes)
    perturb_count = int(epsilon * n)
    
    nodes_list = list(G.nodes)
    
    nodes_to_perturb = random.sample(nodes_list, perturb_count)
    
    for u in nodes_to_perturb:
        for v in G.nodes:
            if u != v:
                if random.random() < p:
                    if not perturbed_G.has_edge(u, v):
                        perturbed_G.add_edge(u, v)
                elif perturbed_G.has_edge(u, v):
                    perturbed_G.remove_edge(u, v)
    return perturbed_G