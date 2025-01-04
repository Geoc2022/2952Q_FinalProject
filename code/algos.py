import networkx as nx
import numpy as np
import scipy
import random
import matplotlib.pyplot as plt
import math

def mean_adjust_median(G, epsilon):
    """
    Estimates the edge probability p using the mean adjusted median algorithm.

    Args:
        G (networkx.Graph): The input graph.
        epsilon (float): The perturbation parameter.

    Returns:
        float: The estimated edge probability p.
    """
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

def mean_adjust_median_improved(G, epsilon):
    """
    Estimates the edge probability p using the mean adjusted median algorithm and then removes the bias.

    Args:
        G (networkx.Graph): The input graph.
        epsilon (float): The perturbation parameter.

    Returns:
        float: The improved estimated edge probability p.
    """
    n = G.number_of_nodes()
    Lp = nx.laplacian_matrix(G).toarray()
    Dp = np.diag(Lp)
    p1 = np.mean(Dp) / (n - 1)
    p_hat = mean_adjust_median(G, epsilon)
    q_hat = (p1 - p_hat * (1 - 2 * epsilon + epsilon ** 2 - (epsilon - epsilon ** 2) / (n - 1))) / (2 * epsilon - epsilon ** 2 + (epsilon - epsilon ** 2) / (n - 1))
    eta_hat = p_hat - q_hat
    muX = ((1 - epsilon) * n - 1) * p_hat + epsilon * n * q_hat
    sigmaX2 = ((1 - epsilon) * n - 1) * p_hat * (1 - p_hat) + epsilon * n * q_hat * (1 - q_hat)
    muY = (n - 1) * q_hat
    sigmaY2 = (n - 1) * q_hat * (1 - q_hat)
    a = (1 - epsilon) / (math.sqrt(2 * math.pi * sigmaX2)) + epsilon / (math.sqrt(2 * math.pi * sigmaY2)) * math.e ** (-(muX - muY) ** 2 / (2 * sigmaY2))
    fix = (epsilon * eta_hat + (2 * epsilon - epsilon ** 2) * np.sign(eta_hat) / (2 * a)) / ((1 - epsilon) * (n - 1))
    return p_hat + fix

def variance_method(G, epsilon, factor=1):
    """
    Estimate the edge probability p using a variance-based algorithm.

    Args:
        G (networkx.Graph): The input graph.
        epsilon (float): The perturbation parameter.
        factor (float, optional): The factor for the number of nodes to remove. Defaults to 1.

    Returns:
        tuple: The estimated edge probability p and the variance of the degree distribution.
    """
    nodes = set(G.nodes)
    n = len(nodes)
    num_removals = int(factor * epsilon * n)

    L = nx.laplacian_matrix(G).toarray()
    removed_indices = set()
    for _ in range(num_removals):
        sample_variances = {}
        D = np.diag(L)
        for node in nodes:
            if node in removed_indices:
                continue
            modified_diag = []
            for j in range(n):
                if j == node or j in removed_indices:
                    continue
                modified_diag.append(D[j] + L[node][j])
            # sample_variances[node] = np.var(modified_diag, ddof=1)
            p_hat = np.mean(modified_diag) / (n - len(removed_indices) - 1)
            sample_variances[node] = abs(np.var(modified_diag, ddof=1) - (n - 2 - len(removed_indices)) * p_hat * (1 - p_hat))

        node_to_remove = min(sample_variances, key=sample_variances.get) # type: ignore
        for j in range(n):
            if j == node_to_remove or j in removed_indices:
                continue
            else:
                L[j][j] += L[node_to_remove][j]
        for j in range(n):
            L[j][node_to_remove] = 0
            L[node_to_remove][j] = 0
        removed_indices.add(node_to_remove)
        nodes.remove(node_to_remove)
    
    D = np.diag(L)
    final_D = []
    for i in range(n):
        if i not in removed_indices:
            final_D.append(D[i])
    return np.mean(final_D) / (n - num_removals), np.var(final_D, ddof=1)

def prune_then_mean_median(G, epsilon):
    """
    Estimate the edge probability p using the prune-then-mean-median algorithm.

    Args:
        G (networkx.Graph): The input graph.
        epsilon (float): The perturbation parameter.

    Returns:
        float: The estimated edge probability p.
    """
    epsilon = 1.0 * epsilon
    modified_G = G.copy()
    
    n = len(G.nodes)
    nodes_to_remove_count = int(epsilon * n)
    
    degree_sorted_nodes = sorted(G.degree, key=lambda x: x[1])
    
    nodes_to_remove = (
        [node for node, _ in degree_sorted_nodes[:nodes_to_remove_count]] +
        [node for node, _ in degree_sorted_nodes[-nodes_to_remove_count:]]
    )
    
    modified_G.remove_nodes_from(nodes_to_remove)
    L = np.array(nx.laplacian_matrix(modified_G).toarray())
    D = np.diag(L)
    mean = np.mean(D) / ((1 - 2 * epsilon) * n)
    med = np.median(D) / ((1 - 2 * epsilon) * n)
    return mean, med

def spectral_method(n, gamma, A, G):
    """
    Estimate the edge probability p using the spectral method as described in "Robust Estimation for Random Graphs" (arXiv:2111.05320).

    Args:
        n (int): The number of nodes in the graph.
        gamma (float): The perturbation parameter.
        A (np.ndarray): The adjacency matrix of the graph.
        G (networkx.Graph): The input graph.

    Returns:
        float: The estimated edge probability p.
    """
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

    p_ast = algo4(n, gamma, A, G)
    if p_ast <= 0.5:
        p_hat = p_ast
    else:
        q_ast = algo4(n, gamma, np.ones_like(A, dtype=float) - np.eye(n) - A, G)
        p_hat = 1 - q_ast
    return p_hat 
