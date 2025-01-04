import networkx as nx
import numpy as np
import scipy
import random
import matplotlib.pyplot as plt

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


def algo5(n, gamma, A, G):
    p_ast = algo4(n, gamma, A, G)
    if p_ast <= 0.5:
        p_hat = p_ast
    else:
        q_ast = algo4(n, gamma, np.ones_like(A, dtype=float) - np.eye(n) - A, G)
        p_hat = 1 - q_ast
    return p_hat 

def algo4(n, alpha_1, A, G):
    S_ast = algo2(n, alpha_1, A, G)
    p_Sf = algo3(n, alpha_1, A, S_ast, G)
    return p_Sf

def algo3(n, alpha_1, A, S_ast, G):
    S_subgraph = G.subgraph(S_ast)
    p_S = [S_subgraph.degree(i) / len(S_ast) for i in S_subgraph.nodes]
    p_S_ast = np.mean(p_S)
    
    scores = {i: np.abs(p_S[i] - p_S_ast) for i in range(S_subgraph.number_of_nodes())}
    sorted_scores = sorted(scores, key=lambda x: scores[x])

    S_f = sorted_scores[:int(3 * alpha_1 * n)]
    p_Sf = np.mean([p_S[i] for i in S_f])

    return p_Sf


def algo2(n, alpha_1, A, G):
    def get_p(S_subgraph):
        p_S = nx.average_clustering(S_subgraph)
        return p_S
    
    def metric(S_subgraph, A_S):
        p_S = get_p(S_subgraph)
        return np.linalg.norm(A_S - p_S, ord=2)
        
    S_comp = []
    S = np.arange(n)
    S_subgraph = G
    candidates = []
    for _ in range(1, int(9 * alpha_1 * n)):
        A_S = nx.adjacency_matrix(S_subgraph).toarray()
        _, eigenvectors = scipy.linalg.eigh(A_S, subset_by_index=[n-1, n-1])
        top_eigenvector = eigenvectors[:, -1]
        
        top_eigenvector_squared = top_eigenvector ** 2
        probabilities = top_eigenvector_squared / top_eigenvector_squared.sum()

        selected_index = np.random.choice(S_subgraph.nodes(), p=probabilities)
        
        S_subgraph.remove_node(selected_index)
        S_comp.append(selected_index)
        n -= 1

        candidates.append((S_comp, metric(S_subgraph, A_S)))
        
    S_comp, _ = min(candidates, key=lambda x: x[1])
    return np.delete(S, S_comp)


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

def test_q_adversary(n, gamma, p, q):
    G, _ = generate_erdos_renyi_graph(n, p)
    p_original = nx.average_clustering(G)

    preturbed_G, _ = perturb_graph(G, gamma, q)

    preturbed_A = nx.adjacency_matrix(preturbed_G).toarray()
    p_hat = algo5(n, gamma, preturbed_A, preturbed_G)
    # print(f"Original p: {p_original}")
    # print(f"Estimated p_hat: {p_hat}")
    return p_original, p_hat


def main():
    n = 200
    p = 0.4
    q = 0.8
    trials = 3

    errors = []
    gammas = np.arange(0.005, 0.1, 0.005)
    for gamma in gammas:
        error = 0
        for _ in range(trials):
            p_original, p_hat = test_q_adversary(n, gamma, p, q)
            error += np.abs(p_original - p_hat) / p_original
        print(f"Gamma: {gamma} - Relative error: {error / trials}")
        errors.append(error / trials)
    
    plt.plot(gammas, errors)
    plt.xlabel("Gamma")
    plt.ylabel("Relative Error")
    plt.title("Relative Error vs. Gamma")
    plt.show()

if __name__ == "__main__":
    main()